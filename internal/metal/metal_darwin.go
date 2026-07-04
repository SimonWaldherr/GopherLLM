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

typedef struct {
	uint32_t rows;
	uint32_t cols;
	uint32_t row_bytes;
	uint32_t n_blocks;
	uint32_t rows_per_group;
} GLLMMetalParams;

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
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"static inline int gllm_i8(uchar v) {\n"
"  int x = int(v);\n"
"  return x >= 128 ? x - 256 : x;\n"
"}\n"
"kernel void gllm_q6k_matvec(\n"
"    const device uchar* data [[buffer(0)]],\n"
"    const device float* x [[buffer(1)]],\n"
"    device float* out [[buffer(2)]],\n"
"    constant Params& p [[buffer(3)]],\n"
"    uint group [[threadgroup_position_in_grid]],\n"
"    uint sg [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]) {\n"
"  uint row_half = lane >> 4;\n"
"  uint sublane = lane & 15;\n"
"  uint row = group * p.rows_per_group + sg * 2 + row_half;\n"
"  if (row >= p.rows) { return; }\n"
"  const device uchar* row_base = data + row * p.row_bytes;\n"
"  float sum = 0.0f;\n"
"  for (uint b = sublane; b < p.n_blocks; b += 16) {\n"
"    const device uchar* block = row_base + b * 210;\n"
"    const device uchar* ql = block;\n"
"    const device uchar* qh = block + 128;\n"
"    const device uchar* sc = block + 192;\n"
"    ushort db = ushort(block[208]) | (ushort(block[209]) << 8);\n"
"    float d = float(as_type<half>(db));\n"
"    uint xoff = b * 256;\n"
"    #pragma unroll\n"
"    for (uint step = 0; step < 2; ++step) {\n"
"      const device uchar* ql_sub = ql + step * 64;\n"
"      const device uchar* qh_sub = qh + step * 32;\n"
"      const device uchar* sc_sub = sc + step * 8;\n"
"      uint y = xoff + step * 128;\n"
"      float dsc0 = d * float(gllm_i8(sc_sub[0]));\n"
"      float dsc2 = d * float(gllm_i8(sc_sub[2]));\n"
"      float dsc4 = d * float(gllm_i8(sc_sub[4]));\n"
"      float dsc6 = d * float(gllm_i8(sc_sub[6]));\n"
"      #pragma unroll(16)\n"
"      for (uint l = 0; l < 16; ++l) {\n"
"        uchar ql0 = ql_sub[l];\n"
"        uchar ql32 = ql_sub[l + 32];\n"
"        uchar qh0 = qh_sub[l];\n"
"        sum += dsc0 * float(int((ql0 & 15) | ((qh0 & 3) << 4)) - 32) * x[y + l];\n"
"        sum += dsc2 * float(int((ql32 & 15) | (((qh0 >> 2) & 3) << 4)) - 32) * x[y + 32 + l];\n"
"        sum += dsc4 * float(int((ql0 >> 4) | (((qh0 >> 4) & 3) << 4)) - 32) * x[y + 64 + l];\n"
"        sum += dsc6 * float(int((ql32 >> 4) | (((qh0 >> 6) & 3) << 4)) - 32) * x[y + 96 + l];\n"
"      }\n"
"      float dsc1 = d * float(gllm_i8(sc_sub[1]));\n"
"      float dsc3 = d * float(gllm_i8(sc_sub[3]));\n"
"      float dsc5 = d * float(gllm_i8(sc_sub[5]));\n"
"      float dsc7 = d * float(gllm_i8(sc_sub[7]));\n"
"      #pragma unroll(16)\n"
"      for (uint l = 16; l < 32; ++l) {\n"
"        uchar ql0 = ql_sub[l];\n"
"        uchar ql32 = ql_sub[l + 32];\n"
"        uchar qh0 = qh_sub[l];\n"
"        sum += dsc1 * float(int((ql0 & 15) | ((qh0 & 3) << 4)) - 32) * x[y + l];\n"
"        sum += dsc3 * float(int((ql32 & 15) | (((qh0 >> 2) & 3) << 4)) - 32) * x[y + 32 + l];\n"
"        sum += dsc5 * float(int((ql0 >> 4) | (((qh0 >> 4) & 3) << 4)) - 32) * x[y + 64 + l];\n"
"        sum += dsc7 * float(int((ql32 >> 4) | (((qh0 >> 6) & 3) << 4)) - 32) * x[y + 96 + l];\n"
"      }\n"
"    }\n"
"  }\n"
"  for (ushort offset = 8; offset > 0; offset >>= 1) {\n"
"    sum += simd_shuffle_xor(sum, offset);\n"
"  }\n"
"  if (sublane == 0) { out[row] = sum; }\n"
"}\n";

static int gllm_q6k_rows_per_group(int rows) {
	const char* value = getenv("GOPHERLLM_METAL_Q6K_ROWS_PER_GROUP");
	if (value != NULL && value[0] != '\0') {
		char* end = NULL;
		long parsed = strtol(value, &end, 10);
		if (end != value && parsed >= 2 && parsed <= 8 && (parsed % 2) == 0) {
			return (int)parsed;
		}
	}
	(void)rows;
	return 8;
}

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

static const char* gllm_metal_last_error(void) {
	return gllm_error;
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
		int rows_per_group = gllm_q6k_rows_per_group(w->rows);
		GLLMMetalParams params = {
			.rows = (uint32_t)w->rows,
			.cols = (uint32_t)w->cols,
			.row_bytes = (uint32_t)w->row_bytes,
			.n_blocks = (uint32_t)(w->cols / 256),
			.rows_per_group = (uint32_t)rows_per_group,
		};
		[enc setBytes:&params length:sizeof(params) atIndex:3];
		MTLSize groups = MTLSizeMake(((NSUInteger)w->rows + (NSUInteger)rows_per_group - 1) / (NSUInteger)rows_per_group, 1, 1);
		MTLSize threads = MTLSizeMake(32 * ((NSUInteger)rows_per_group / 2), 1, 1);
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

func LastError() string {
	return C.GoString(C.gllm_metal_last_error())
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
