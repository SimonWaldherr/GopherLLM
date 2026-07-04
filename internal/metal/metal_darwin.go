//go:build darwin && cgo && metal

package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation -framework Metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	id<MTLBuffer> weights;
	id<MTLBuffer> x;
	id<MTLBuffer> out;
	int rows;
	int cols;
	int row_bytes;
} GLLMMetalWeight;

static id<MTLDevice> gllm_device = nil;
static id<MTLCommandQueue> gllm_queue = nil;
static id<MTLComputePipelineState> gllm_q6k_pipeline = nil;
static char gllm_error[1024];

static void gllm_set_error(NSString* prefix, NSError* error) {
	const char* p = prefix ? [prefix UTF8String] : "Metal error";
	const char* e = error ? [[error localizedDescription] UTF8String] : "";
	snprintf(gllm_error, sizeof(gllm_error), "%s%s%s", p, e[0] ? ": " : "", e);
}

static const char* gllm_q6k_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"static inline float gllm_f16(const device uchar* p) {\n"
"  ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);\n"
"  return float(as_type<half>(bits));\n"
"}\n"
"static inline int gllm_i8(uchar v) {\n"
"  int x = int(v);\n"
"  return x >= 128 ? x - 256 : x;\n"
"}\n"
"static inline int gllm_q6(const device uchar* ql, const device uchar* qh, int local, thread int& scale_idx) {\n"
"  int l;\n"
"  if (local < 32) {\n"
"    l = local;\n"
"    scale_idx = (l < 16) ? 0 : 1;\n"
"    return int((ql[l] & 0x0f) | ((qh[l] & 0x03) << 4));\n"
"  }\n"
"  if (local < 64) {\n"
"    l = local - 32;\n"
"    scale_idx = (l < 16) ? 2 : 3;\n"
"    return int((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 0x03) << 4));\n"
"  }\n"
"  if (local < 96) {\n"
"    l = local - 64;\n"
"    scale_idx = (l < 16) ? 4 : 5;\n"
"    return int((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4));\n"
"  }\n"
"  l = local - 96;\n"
"  scale_idx = (l < 16) ? 6 : 7;\n"
"  return int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4));\n"
"}\n"
"kernel void gllm_q6k_matvec(\n"
"    const device uchar* data [[buffer(0)]],\n"
"    const device float* x [[buffer(1)]],\n"
"    device float* out [[buffer(2)]],\n"
"    constant int& rows [[buffer(3)]],\n"
"    constant int& cols [[buffer(4)]],\n"
"    constant int& row_bytes [[buffer(5)]],\n"
"    uint tg [[threadgroup_position_in_grid]],\n"
"    uint lane [[thread_index_in_threadgroup]]) {\n"
"  constexpr int rows_per_tg = 1;\n"
"  int blocks = cols / 256;\n"
"  int row0 = int(tg) * rows_per_tg;\n"
"  for (int rr = 0; rr < rows_per_tg; rr++) {\n"
"    int row = row0 + rr;\n"
"    if (row >= rows) { return; }\n"
"    const device uchar* rowp = data + row * row_bytes;\n"
"    float acc = 0.0f;\n"
"    for (int b = 0; b < blocks; b++) {\n"
"      const device uchar* block = rowp + b * 210;\n"
"      const device uchar* ql = block;\n"
"      const device uchar* qh = block + 128;\n"
"      const device uchar* sc = block + 192;\n"
"      float d = gllm_f16(block + 208);\n"
"      for (int idx = int(lane); idx < 256; idx += 32) {\n"
"        int step = idx >= 128 ? 1 : 0;\n"
"        int local = idx - step * 128;\n"
"        int scale_local = 0;\n"
"        int q = gllm_q6(ql + step * 64, qh + step * 32, local, scale_local) - 32;\n"
"        int scale_i = step * 8 + scale_local;\n"
"        acc += d * float(gllm_i8(sc[scale_i])) * float(q) * x[b * 256 + idx];\n"
"      }\n"
"    }\n"
"    acc = simd_sum(acc);\n"
"    if (lane == 0) { out[row] = acc; }\n"
"  }\n"
"}\n";

static bool gllm_metal_init(void) {
	@autoreleasepool {
		if (gllm_device != nil && gllm_queue != nil) {
			return true;
		}
		gllm_device = MTLCreateSystemDefaultDevice();
		if (gllm_device == nil) {
			strncpy(gllm_error, "no Metal device available", sizeof(gllm_error) - 1);
			return false;
		}
		gllm_queue = [gllm_device newCommandQueue];
		if (gllm_queue == nil) {
			strncpy(gllm_error, "failed to create Metal command queue", sizeof(gllm_error) - 1);
			return false;
		}
		return true;
	}
}

static bool gllm_metal_available(void) {
	return gllm_metal_init();
}

static bool gllm_metal_init_q6k(void) {
	@autoreleasepool {
		if (!gllm_metal_init()) {
			return false;
		}
		if (gllm_q6k_pipeline != nil) {
			return true;
		}
		NSError* error = nil;
		NSString* source = [NSString stringWithUTF8String:gllm_q6k_source];
		id<MTLLibrary> library = [gllm_device newLibraryWithSource:source options:nil error:&error];
		if (library == nil) {
			gllm_set_error(@"failed to compile Q6_K Metal library", error);
			return false;
		}
		id<MTLFunction> fn = [library newFunctionWithName:@"gllm_q6k_matvec"];
		if (fn == nil) {
			strncpy(gllm_error, "failed to load Q6_K Metal function", sizeof(gllm_error) - 1);
			[library release];
			return false;
		}
		gllm_q6k_pipeline = [gllm_device newComputePipelineStateWithFunction:fn error:&error];
		[fn release];
		[library release];
		if (gllm_q6k_pipeline == nil) {
			gllm_set_error(@"failed to create Q6_K Metal pipeline", error);
			return false;
		}
		return true;
	}
}

static void* gllm_metal_new_q6k(const void* data, long len, int rows, int cols) {
	@autoreleasepool {
		if (data == NULL || len <= 0 || rows <= 0 || cols <= 0 || (cols % 256) != 0) {
			return NULL;
		}
		if (!gllm_metal_init_q6k()) {
			return NULL;
		}
		GLLMMetalWeight* w = (GLLMMetalWeight*)calloc(1, sizeof(GLLMMetalWeight));
		if (w == NULL) {
			strncpy(gllm_error, "failed to allocate Metal weight handle", sizeof(gllm_error) - 1);
			return NULL;
		}
		w->rows = rows;
		w->cols = cols;
		w->row_bytes = (cols / 256) * 210;
		w->weights = [gllm_device newBufferWithBytes:data length:(NSUInteger)len options:MTLResourceStorageModeShared];
		w->x = [gllm_device newBufferWithLength:(NSUInteger)cols * sizeof(float) options:MTLResourceStorageModeShared];
		w->out = [gllm_device newBufferWithLength:(NSUInteger)rows * sizeof(float) options:MTLResourceStorageModeShared];
		if (w->weights == nil || w->x == nil || w->out == nil) {
			strncpy(gllm_error, "failed to allocate Metal weight buffer", sizeof(gllm_error) - 1);
			if (w->weights != nil) [w->weights release];
			if (w->x != nil) [w->x release];
			if (w->out != nil) [w->out release];
			free(w);
			return NULL;
		}
		return w;
	}
}

static int gllm_metal_q6k_matvec(void* handle, const float* x, float* out) {
	@autoreleasepool {
		GLLMMetalWeight* w = (GLLMMetalWeight*)handle;
		if (w == NULL || w->weights == nil || x == NULL || out == NULL || !gllm_metal_init_q6k()) {
			return 0;
		}
		NSUInteger x_len = (NSUInteger)w->cols * sizeof(float);
		NSUInteger out_len = (NSUInteger)w->rows * sizeof(float);
		if (w->x == nil || w->out == nil) {
			strncpy(gllm_error, "missing Metal matvec buffers", sizeof(gllm_error) - 1);
			return 0;
		}
		memcpy([w->x contents], x, x_len);

		id<MTLCommandBuffer> cb = [gllm_queue commandBuffer];
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		[enc setComputePipelineState:gllm_q6k_pipeline];
		[enc setBuffer:w->weights offset:0 atIndex:0];
		[enc setBuffer:w->x offset:0 atIndex:1];
		[enc setBuffer:w->out offset:0 atIndex:2];
		int rows = w->rows;
		int cols = w->cols;
		int row_bytes = w->row_bytes;
		[enc setBytes:&rows length:sizeof(rows) atIndex:3];
		[enc setBytes:&cols length:sizeof(cols) atIndex:4];
		[enc setBytes:&row_bytes length:sizeof(row_bytes) atIndex:5];
		MTLSize groups = MTLSizeMake((NSUInteger)rows, 1, 1);
		MTLSize threads = MTLSizeMake(32, 1, 1);
		[enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
		[enc endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		int ok = [cb status] == MTLCommandBufferStatusCompleted;
		if (ok) {
			memcpy(out, [w->out contents], out_len);
		} else {
			strncpy(gllm_error, "Metal command buffer failed", sizeof(gllm_error) - 1);
		}
		return ok;
	}
}

static void gllm_metal_release_weight(void* handle) {
	@autoreleasepool {
		GLLMMetalWeight* w = (GLLMMetalWeight*)handle;
		if (w == NULL) {
			return;
		}
		if (w->weights != nil) {
			[w->weights release];
			w->weights = nil;
		}
		if (w->x != nil) {
			[w->x release];
			w->x = nil;
		}
		if (w->out != nil) {
			[w->out release];
			w->out = nil;
		}
		free(w);
	}
}
*/
import "C"

import (
	"runtime"
	"unsafe"
)

type Weight struct {
	ptr  unsafe.Pointer
	rows int
	cols int
}

func Available() bool {
	return bool(C.gllm_metal_available())
}

func PrepareQ6K(data []byte, rows, cols int) *Weight {
	if rows <= 0 || cols <= 0 || cols%256 != 0 || len(data) == 0 {
		return nil
	}
	ptr := C.gllm_metal_new_q6k(unsafe.Pointer(&data[0]), C.long(len(data)), C.int(rows), C.int(cols))
	if ptr == nil {
		return nil
	}
	w := &Weight{ptr: ptr, rows: rows, cols: cols}
	runtime.SetFinalizer(w, Release)
	return w
}

func MatvecQ6K(w *Weight, x, out []float32) bool {
	if w == nil || w.ptr == nil || len(x) < w.cols || len(out) < w.rows || w.rows == 0 {
		return false
	}
	ok := C.gllm_metal_q6k_matvec(w.ptr, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
	return ok != 0
}

func Release(w *Weight) {
	if w == nil || w.ptr == nil {
		return
	}
	C.gllm_metal_release_weight(w.ptr)
	w.ptr = nil
	runtime.SetFinalizer(w, nil)
}
