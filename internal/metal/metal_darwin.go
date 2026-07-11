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
#include <unistd.h>

typedef struct {
	id<MTLBuffer> weights;
	id<MTLBuffer> x;
	id<MTLBuffer> out;
	int rows;
	int cols;
	int row_bytes;
	NSUInteger weight_offset;
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
static id<MTLComputePipelineState> gllm_q4k_pipeline = nil;
static id<MTLComputePipelineState> gllm_q6k_pipeline = nil;
static id<MTLComputePipelineState> gllm_silu_pipeline = nil;
static char gllm_error[1024];

static void gllm_set_error(NSString* prefix, NSError* error) {
	const char* p = prefix ? [prefix UTF8String] : "Metal error";
	const char* e = error ? [[error localizedDescription] UTF8String] : "";
	snprintf(gllm_error, sizeof(gllm_error), "%s%s%s", p, e[0] ? ": " : "", e);
}

static const char* gllm_q4k_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; };\n"
"kernel void gllm_q4k_matvec(\n"
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
"    const device uchar* block = row_base + b * 144;\n"
"    ushort db = ushort(block[0]) | (ushort(block[1]) << 8);\n"
"    ushort dmb = ushort(block[2]) | (ushort(block[3]) << 8);\n"
"    float d = float(as_type<half>(db));\n"
"    float dmin = float(as_type<half>(dmb));\n"
"    const device uchar* scales = block + 4;\n"
"    const device uchar* q = block + 16;\n"
"    uint xoff = b * 256;\n"
"    #pragma unroll\n"
"    for (uint step = 0; step < 4; ++step) {\n"
"      uint is = step * 2;\n"
"      uint sc1, m1, sc2, m2;\n"
"      if (is < 4) {\n"
"        sc1 = uint(scales[is] & 63);\n"
"        m1 = uint(scales[is + 4] & 63);\n"
"        sc2 = uint(scales[is + 1] & 63);\n"
"        m2 = uint(scales[is + 5] & 63);\n"
"      } else {\n"
"        sc1 = uint(scales[is + 4] & 15) | (uint(scales[is - 4] >> 6) << 4);\n"
"        m1 = uint(scales[is + 4] >> 4) | (uint(scales[is] >> 6) << 4);\n"
"        sc2 = uint(scales[is + 5] & 15) | (uint(scales[is - 3] >> 6) << 4);\n"
"        m2 = uint(scales[is + 5] >> 4) | (uint(scales[is + 1] >> 6) << 4);\n"
"      }\n"
"      const device uchar* qsub = q + step * 32;\n"
"      uint y = xoff + step * 64;\n"
"      float qd1 = 0.0f, qd2 = 0.0f, xs1 = 0.0f, xs2 = 0.0f;\n"
"      #pragma unroll(32)\n"
"      for (uint l = 0; l < 32; ++l) {\n"
"        uchar packed = qsub[l];\n"
"        float xv1 = x[y + l];\n"
"        float xv2 = x[y + 32 + l];\n"
"        qd1 += float(packed & 15) * xv1;\n"
"        qd2 += float(packed >> 4) * xv2;\n"
"        xs1 += xv1;\n"
"        xs2 += xv2;\n"
"      }\n"
"      sum += d * (float(sc1) * qd1 + float(sc2) * qd2)\n"
"           - dmin * (float(m1) * xs1 + float(m2) * xs2);\n"
"    }\n"
"  }\n"
"  for (ushort offset = 8; offset > 0; offset >>= 1) {\n"
"    sum += simd_shuffle_xor(sum, offset);\n"
"  }\n"
"  if (sublane == 0) { out[row] = sum; }\n"
"}\n"
"kernel void gllm_silu_mul(\n"
"    const device float* gate [[buffer(0)]],\n"
"    const device float* up [[buffer(1)]],\n"
"    device float* hidden [[buffer(2)]],\n"
"    constant uint& n [[buffer(3)]],\n"
"    uint i [[thread_position_in_grid]]) {\n"
"  if (i < n) {\n"
"    float g = gate[i];\n"
"    hidden[i] = (g / (1.0f + precise::exp(-g))) * up[i];\n"
"  }\n"
"}\n";

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

static int gllm_metal_rows_per_group(int rows) {
	const char* value = getenv("GOPHERLLM_METAL_ROWS_PER_GROUP");
	if (value == NULL || value[0] == '\0') {
		value = getenv("GOPHERLLM_METAL_Q6K_ROWS_PER_GROUP");
	}
	if (value != NULL && value[0] != '\0') {
		char* end = NULL;
		long parsed = strtol(value, &end, 10);
		if (end != value && parsed >= 2 && parsed <= 8 && (parsed % 2) == 0) {
			return (int)parsed;
		}
	}
	(void)rows;
	return 4;
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

static bool gllm_metal_init_q4k(void) {
	@autoreleasepool {
		if (!gllm_metal_init()) {
			return false;
		}
		if (gllm_q4k_pipeline != nil && gllm_silu_pipeline != nil) {
			return true;
		}
		NSError* error = nil;
		NSString* source = [NSString stringWithUTF8String:gllm_q4k_source];
		id<MTLLibrary> library = [gllm_device newLibraryWithSource:source options:nil error:&error];
		if (library == nil) {
			gllm_set_error(@"failed to compile Q4_K Metal library", error);
			return false;
		}
		id<MTLFunction> q4_fn = [library newFunctionWithName:@"gllm_q4k_matvec"];
		id<MTLFunction> silu_fn = [library newFunctionWithName:@"gllm_silu_mul"];
		if (q4_fn == nil || silu_fn == nil) {
			strncpy(gllm_error, "failed to load Q4_K/SiLU Metal functions", sizeof(gllm_error) - 1);
			if (q4_fn != nil) [q4_fn release];
			if (silu_fn != nil) [silu_fn release];
			[library release];
			return false;
		}
		id<MTLComputePipelineState> q4_pipeline = [gllm_device newComputePipelineStateWithFunction:q4_fn error:&error];
		id<MTLComputePipelineState> silu_pipeline = nil;
		if (q4_pipeline != nil) {
			silu_pipeline = [gllm_device newComputePipelineStateWithFunction:silu_fn error:&error];
		}
		[q4_fn release];
		[silu_fn release];
		[library release];
		if (q4_pipeline == nil || silu_pipeline == nil) {
			if (q4_pipeline != nil) [q4_pipeline release];
			if (silu_pipeline != nil) [silu_pipeline release];
			gllm_set_error(@"failed to create Q4_K/SiLU Metal pipelines", error);
			return false;
		}
		gllm_q4k_pipeline = q4_pipeline;
		gllm_silu_pipeline = silu_pipeline;
		return true;
	}
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

static id<MTLCommandBuffer> gllm_metal_new_command_buffer(void) {
	// Every matvec waits before returning and weight handles outlive the wait, so
	// retaining every referenced resource per dispatch is redundant. Rollback is
	// a one-line switch back to [gllm_queue commandBuffer].
	return [gllm_queue commandBufferWithUnretainedReferences];
}

static id<MTLBuffer> gllm_metal_new_weight_buffer(const void* data, long len, bool no_copy, NSUInteger* offset) {
	*offset = 0;
	if (no_copy) {
		NSUInteger page = (NSUInteger)getpagesize();
		uintptr_t address = (uintptr_t)data;
		uintptr_t base = address & ~((uintptr_t)page - 1);
		NSUInteger delta = (NSUInteger)(address - base);
		NSUInteger span = delta + (NSUInteger)len;
		NSUInteger rounded = (span + page - 1) & ~(page - 1);
		id<MTLBuffer> buffer = [gllm_device
			newBufferWithBytesNoCopy:(void*)base
			length:rounded
			options:MTLResourceStorageModeShared
			deallocator:nil];
		if (buffer != nil) {
			*offset = delta;
			return buffer;
		}
	}
	return [gllm_device newBufferWithBytes:data length:(NSUInteger)len options:MTLResourceStorageModeShared];
}

static void* gllm_metal_new_q4k(const void* data, long len, int rows, int cols, bool no_copy) {
	@autoreleasepool {
		if (data == NULL || len <= 0 || rows <= 0 || cols <= 0 || (cols % 256) != 0) {
			return NULL;
		}
		if (!gllm_metal_init_q4k()) {
			return NULL;
		}
		GLLMMetalWeight* w = (GLLMMetalWeight*)calloc(1, sizeof(GLLMMetalWeight));
		if (w == NULL) {
			strncpy(gllm_error, "failed to allocate Metal weight handle", sizeof(gllm_error) - 1);
			return NULL;
		}
		w->rows = rows;
		w->cols = cols;
		w->row_bytes = (cols / 256) * 144;
		w->weights = gllm_metal_new_weight_buffer(data, len, no_copy, &w->weight_offset);
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

static void* gllm_metal_new_q6k(const void* data, long len, int rows, int cols, bool no_copy) {
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
		w->weights = gllm_metal_new_weight_buffer(data, len, no_copy, &w->weight_offset);
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

static void gllm_metal_encode_q4k(id<MTLComputeCommandEncoder> enc, GLLMMetalWeight* w, id<MTLBuffer> x_buffer) {
	[enc setComputePipelineState:gllm_q4k_pipeline];
	[enc setBuffer:w->weights offset:w->weight_offset atIndex:0];
	[enc setBuffer:x_buffer offset:0 atIndex:1];
	[enc setBuffer:w->out offset:0 atIndex:2];
	int rows_per_group = gllm_metal_rows_per_group(w->rows);
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
}

static void gllm_metal_encode_q6k(id<MTLComputeCommandEncoder> enc, GLLMMetalWeight* w, id<MTLBuffer> x_buffer) {
	[enc setComputePipelineState:gllm_q6k_pipeline];
	[enc setBuffer:w->weights offset:w->weight_offset atIndex:0];
	[enc setBuffer:x_buffer offset:0 atIndex:1];
	[enc setBuffer:w->out offset:0 atIndex:2];
	int rows_per_group = gllm_metal_rows_per_group(w->rows);
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
}

static void gllm_metal_encode_silu(
	id<MTLComputeCommandEncoder> enc,
	id<MTLBuffer> gate,
	id<MTLBuffer> up,
	id<MTLBuffer> hidden,
	uint32_t n
) {
	[enc setComputePipelineState:gllm_silu_pipeline];
	[enc setBuffer:gate offset:0 atIndex:0];
	[enc setBuffer:up offset:0 atIndex:1];
	[enc setBuffer:hidden offset:0 atIndex:2];
	[enc setBytes:&n length:sizeof(n) atIndex:3];
	NSUInteger threads = MIN((NSUInteger)256, [gllm_silu_pipeline maxTotalThreadsPerThreadgroup]);
	MTLSize groups = MTLSizeMake(((NSUInteger)n + threads - 1) / threads, 1, 1);
	[enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static int gllm_metal_q4k_matvec(void* handle, const float* x, float* out) {
	@autoreleasepool {
		GLLMMetalWeight* w = (GLLMMetalWeight*)handle;
		if (w == NULL || w->weights == nil || x == NULL || out == NULL || !gllm_metal_init_q4k()) {
			return 0;
		}
		NSUInteger x_len = (NSUInteger)w->cols * sizeof(float);
		NSUInteger out_len = (NSUInteger)w->rows * sizeof(float);
		if (w->x == nil || w->out == nil) {
			strncpy(gllm_error, "missing Metal matvec buffers", sizeof(gllm_error) - 1);
			return 0;
		}
		memcpy([w->x contents], x, x_len);

		id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		gllm_metal_encode_q4k(enc, w, w->x);
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

static int gllm_metal_q4k_matvec2(void* a_handle, void* b_handle, const float* x, float* a_out, float* b_out) {
	@autoreleasepool {
		GLLMMetalWeight* a = (GLLMMetalWeight*)a_handle;
		GLLMMetalWeight* b = (GLLMMetalWeight*)b_handle;
		if (a == NULL || b == NULL || a->weights == nil || b->weights == nil ||
			x == NULL || a_out == NULL || b_out == NULL || a->cols != b->cols || !gllm_metal_init_q4k()) {
			return 0;
		}
		if (a->x == nil || a->out == nil || b->out == nil) {
			strncpy(gllm_error, "missing Metal matvec buffers", sizeof(gllm_error) - 1);
			return 0;
		}
		memcpy([a->x contents], x, (NSUInteger)a->cols * sizeof(float));

		id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		gllm_metal_encode_q4k(enc, a, a->x);
		gllm_metal_encode_q4k(enc, b, a->x);
		[enc endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		int ok = [cb status] == MTLCommandBufferStatusCompleted;
		if (ok) {
			memcpy(a_out, [a->out contents], (NSUInteger)a->rows * sizeof(float));
			memcpy(b_out, [b->out contents], (NSUInteger)b->rows * sizeof(float));
		} else {
			strncpy(gllm_error, "Metal command buffer failed", sizeof(gllm_error) - 1);
		}
		return ok;
	}
}

static int gllm_metal_q4k2_q6k_matvec3(
	void* q_handle,
	void* k_handle,
	void* v_handle,
	const float* x,
	float* q_out,
	float* k_out,
	float* v_out
) {
	@autoreleasepool {
		GLLMMetalWeight* q = (GLLMMetalWeight*)q_handle;
		GLLMMetalWeight* k = (GLLMMetalWeight*)k_handle;
		GLLMMetalWeight* v = (GLLMMetalWeight*)v_handle;
		if (q == NULL || k == NULL || v == NULL || q->weights == nil || k->weights == nil || v->weights == nil ||
			x == NULL || q_out == NULL || k_out == NULL || v_out == NULL ||
			q->cols != k->cols || q->cols != v->cols ||
			q->row_bytes != (q->cols / 256) * 144 || k->row_bytes != (k->cols / 256) * 144 ||
			v->row_bytes != (v->cols / 256) * 210 ||
			!gllm_metal_init_q4k() || !gllm_metal_init_q6k()) {
			return 0;
		}
		if (q->x == nil || q->out == nil || k->out == nil || v->out == nil) {
			strncpy(gllm_error, "missing Metal matvec buffers", sizeof(gllm_error) - 1);
			return 0;
		}
		memcpy([q->x contents], x, (NSUInteger)q->cols * sizeof(float));

		id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		gllm_metal_encode_q4k(enc, q, q->x);
		gllm_metal_encode_q4k(enc, k, q->x);
		gllm_metal_encode_q6k(enc, v, q->x);
		[enc endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		int ok = [cb status] == MTLCommandBufferStatusCompleted;
		if (ok) {
			memcpy(q_out, [q->out contents], (NSUInteger)q->rows * sizeof(float));
			memcpy(k_out, [k->out contents], (NSUInteger)k->rows * sizeof(float));
			memcpy(v_out, [v->out contents], (NSUInteger)v->rows * sizeof(float));
		} else {
			strncpy(gllm_error, "Metal command buffer failed", sizeof(gllm_error) - 1);
		}
		return ok;
	}
}

static int gllm_metal_q4k2_silu_q6k(
	void* gate_handle,
	void* up_handle,
	void* down_handle,
	const float* x,
	float* out
) {
	@autoreleasepool {
		GLLMMetalWeight* gate = (GLLMMetalWeight*)gate_handle;
		GLLMMetalWeight* up = (GLLMMetalWeight*)up_handle;
		GLLMMetalWeight* down = (GLLMMetalWeight*)down_handle;
		if (gate == NULL || up == NULL || down == NULL ||
			gate->weights == nil || up->weights == nil || down->weights == nil ||
			x == NULL || out == NULL || gate->cols != up->cols || gate->rows != up->rows ||
			down->cols != gate->rows ||
			gate->row_bytes != (gate->cols / 256) * 144 || up->row_bytes != (up->cols / 256) * 144 ||
			down->row_bytes != (down->cols / 256) * 210 ||
			!gllm_metal_init_q4k() || !gllm_metal_init_q6k()) {
			return 0;
		}
		if (gate->x == nil || gate->out == nil || up->out == nil || down->x == nil || down->out == nil) {
			strncpy(gllm_error, "missing Metal fused FFN buffers", sizeof(gllm_error) - 1);
			return 0;
		}
		memcpy([gate->x contents], x, (NSUInteger)gate->cols * sizeof(float));

		id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		gllm_metal_encode_q4k(enc, gate, gate->x);
		gllm_metal_encode_q4k(enc, up, gate->x);
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		gllm_metal_encode_silu(enc, gate->out, up->out, down->x, (uint32_t)gate->rows);
		[enc endEncoding];

		enc = [cb computeCommandEncoder];
		gllm_metal_encode_q6k(enc, down, down->x);
		[enc endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		int ok = [cb status] == MTLCommandBufferStatusCompleted;
		if (ok) {
			memcpy(out, [down->out contents], (NSUInteger)down->rows * sizeof(float));
		} else {
			strncpy(gllm_error, "Metal fused FFN command buffer failed", sizeof(gllm_error) - 1);
		}
		return ok;
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

		id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		gllm_metal_encode_q6k(enc, w, w->x);
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

func PrepareQ4K(data []byte, rows, cols int, noCopy bool) *Weight {
	if rows <= 0 || cols <= 0 || cols%256 != 0 || len(data) == 0 {
		return nil
	}
	ptr := C.gllm_metal_new_q4k(unsafe.Pointer(&data[0]), C.long(len(data)), C.int(rows), C.int(cols), C.bool(noCopy))
	if ptr == nil {
		return nil
	}
	w := &Weight{ptr: ptr, rows: rows, cols: cols}
	runtime.SetFinalizer(w, Release)
	return w
}

func Available() bool {
	return bool(C.gllm_metal_available())
}

func LastError() string {
	return C.GoString(C.gllm_metal_last_error())
}

func PrepareQ6K(data []byte, rows, cols int, noCopy bool) *Weight {
	if rows <= 0 || cols <= 0 || cols%256 != 0 || len(data) == 0 {
		return nil
	}
	ptr := C.gllm_metal_new_q6k(unsafe.Pointer(&data[0]), C.long(len(data)), C.int(rows), C.int(cols), C.bool(noCopy))
	if ptr == nil {
		return nil
	}
	w := &Weight{ptr: ptr, rows: rows, cols: cols}
	runtime.SetFinalizer(w, Release)
	return w
}

func MatvecQ4K(w *Weight, x, out []float32) bool {
	if w == nil || w.ptr == nil || len(x) < w.cols || len(out) < w.rows || w.rows == 0 {
		return false
	}
	ok := C.gllm_metal_q4k_matvec(w.ptr, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
	return ok != 0
}

func MatvecQ4K2(a, b *Weight, x, aOut, bOut []float32) bool {
	if a == nil || b == nil || a.ptr == nil || b.ptr == nil || a.cols != b.cols ||
		len(x) < a.cols || len(aOut) < a.rows || len(bOut) < b.rows || a.rows == 0 || b.rows == 0 {
		return false
	}
	ok := C.gllm_metal_q4k_matvec2(
		a.ptr,
		b.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&aOut[0])),
		(*C.float)(unsafe.Pointer(&bOut[0])),
	)
	return ok != 0
}

func MatvecQ4K2Q6K(q, k, v *Weight, x, qOut, kOut, vOut []float32) bool {
	if q == nil || k == nil || v == nil || q.ptr == nil || k.ptr == nil || v.ptr == nil ||
		q.cols != k.cols || q.cols != v.cols || len(x) < q.cols ||
		len(qOut) < q.rows || len(kOut) < k.rows || len(vOut) < v.rows ||
		q.rows == 0 || k.rows == 0 || v.rows == 0 {
		return false
	}
	ok := C.gllm_metal_q4k2_q6k_matvec3(
		q.ptr,
		k.ptr,
		v.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&qOut[0])),
		(*C.float)(unsafe.Pointer(&kOut[0])),
		(*C.float)(unsafe.Pointer(&vOut[0])),
	)
	return ok != 0
}

func MatvecQ4K2SwiGLUQ6K(gate, up, down *Weight, x, out []float32) bool {
	if gate == nil || up == nil || down == nil || gate.ptr == nil || up.ptr == nil || down.ptr == nil ||
		gate.cols != up.cols || gate.rows != up.rows || down.cols != gate.rows ||
		len(x) < gate.cols || len(out) < down.rows || gate.rows == 0 || down.rows == 0 {
		return false
	}
	ok := C.gllm_metal_q4k2_silu_q6k(
		gate.ptr,
		up.ptr,
		down.ptr,
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	return ok != 0
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
