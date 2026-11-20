// BF16 vocabulary projection using the mapped GGUF bytes directly. Four
// neighboring lanes per row mirror ARM64's four FP32 vector accumulators.
static id<MTLComputePipelineState> gllm_bf16_head_pipeline = nil;
static NSString* const gllm_bf16_head_source =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void bf16_head(\n"
"    const device ushort4* weights [[buffer(0)]],\n"
"    const device float4* x [[buffer(1)]], device float* out [[buffer(2)]],\n"
"    constant uint2& dims [[buffer(3)]], uint tid [[thread_position_in_grid]]) {\n"
"  uint row = tid >> 2;\n"
"  if (row >= dims.x) return;\n"
"  uint part = tid & 3u;\n"
"  uint vectors = dims.y / 4;\n"
"  const device ushort4* w = weights + ulong(row) * vectors;\n"
"  float4 sum = 0.0f;\n"
"  for (uint i = part; i < vectors; i += 4) {\n"
"    float4 values = as_type<float4>(uint4(w[i]) << 16);\n"
"    sum = fma(values, x[i], sum);\n"
"  }\n"
"  float4 a = simd_shuffle_down(sum, 1);\n"
"  float4 b = simd_shuffle_down(sum, 2);\n"
"  float4 c = simd_shuffle_down(sum, 3);\n"
"  if (part == 0) {\n"
"    float4 s = ((sum + a) + b) + c;\n"
"    out[row] = (s.x + s.y) + (s.z + s.w);\n"
"  }\n"
"}\n";

static bool gllm_metal_init_bf16_head(void) {
	if (!gllm_metal_init()) return false;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		NSError* error = nil;
		MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
		if (@available(macOS 15.0, *)) {
			[options setMathMode:MTLMathModeSafe];
		} else
#endif
		{
			#pragma clang diagnostic push
			#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			[options setFastMathEnabled:NO];
			#pragma clang diagnostic pop
		}
		id<MTLLibrary> library = [gllm_device newLibraryWithSource:gllm_bf16_head_source options:options error:&error];
		[options release];
		id<MTLFunction> function = [library newFunctionWithName:@"bf16_head"];
		id<MTLComputePipelineState> pipeline = function == nil ? nil :
			[gllm_device newComputePipelineStateWithFunction:function error:&error];
		[function release];
		[library release];
		if (pipeline == nil || [pipeline threadExecutionWidth] != 32 || [pipeline maxTotalThreadsPerThreadgroup] < 128) {
			[pipeline release];
			gllm_set_error(@"failed to initialize Metal BF16 head", error);
		} else {
			gllm_bf16_head_pipeline = pipeline;
		}
	});
	return gllm_bf16_head_pipeline != nil;
}

static bool gllm_metal_encode_bf16_head(id<MTLComputeCommandEncoder> enc, GLLMMetalWeight* w,
	id<MTLBuffer> x, id<MTLBuffer> out) {
	if (enc == nil || w == NULL || w->weights == nil || x == nil || out == nil ||
		w->rows <= 0 || w->cols <= 0 || w->cols % 16 != 0 || w->row_bytes != w->cols * 2 ||
		[x length] < (NSUInteger)w->cols * sizeof(float) ||
		[out length] < (NSUInteger)w->rows * sizeof(float) ||
		!gllm_metal_init_bf16_head()) return false;
	uint32_t dims[2] = {(uint32_t)w->rows, (uint32_t)w->cols};
	[enc setComputePipelineState:gllm_bf16_head_pipeline];
	[enc setBuffer:w->weights offset:w->weight_offset atIndex:0];
	[enc setBuffer:x offset:0 atIndex:1];
	[enc setBuffer:out offset:0 atIndex:2];
	[enc setBytes:dims length:sizeof(dims) atIndex:3];
	[enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)w->rows + 31) / 32, 1, 1)
		threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
	return true;
}

static void* gllm_metal_new_bf16_head(const void* data, long len, int rows, int cols) {
	@autoreleasepool {
		if (data == NULL || ((uintptr_t)data & 7u) != 0 || len <= 0 || rows <= 0 ||
			cols <= 0 || cols > INT32_MAX/2 || cols % 16 != 0 ||
			(NSUInteger)rows > NSUIntegerMax / ((NSUInteger)cols * 2) ||
			(NSUInteger)len != (NSUInteger)rows * (NSUInteger)cols * 2 ||
			!gllm_metal_init_bf16_head() || !gllm_metal_init_q6k()) return NULL;
		GLLMMetalWeight* w = calloc(1, sizeof(*w));
		if (w == NULL) return NULL;
		w->rows = rows;
		w->cols = cols;
		w->row_bytes = cols * 2;
		w->weights = gllm_metal_new_weight_buffer(data, len, true, &w->weight_offset);
		if (w->weights != nil) {
			w->x = [gllm_device newBufferWithLength:(NSUInteger)cols * sizeof(float) options:MTLResourceStorageModeShared];
			w->out = [gllm_device newBufferWithLength:(NSUInteger)rows * sizeof(float) options:MTLResourceStorageModeShared];
			w->argmax = [gllm_device newBufferWithLength:sizeof(GLLMArgmaxResult) options:MTLResourceStorageModeShared];
			w->recent = [gllm_device newBufferWithLength:GLLM_REPEAT_WINDOW * sizeof(uint32_t) options:MTLResourceStorageModeShared];
		}
		if (w->weights == nil || w->x == nil || w->out == nil || w->argmax == nil || w->recent == nil) {
			gllm_metal_release_weight(w);
			return NULL;
		}
		return w;
	}
}

static bool gllm_metal_bf16_head(void* handle, const float* x, float* logits,
	const uint32_t* recent, uint32_t recent_count, float penalty, uint32_t* token) {
	@autoreleasepool {
		GLLMMetalWeight* w = handle;
		if (w == NULL || x == NULL || (logits == NULL && token == NULL) ||
			recent_count > GLLM_REPEAT_WINDOW || (recent_count != 0 && recent == NULL) ||
			!isfinite(penalty) || penalty <= 0.0f) return false;
		@synchronized(gllm_queue) {
			memcpy([w->x contents], x, (NSUInteger)w->cols * sizeof(float));
			if (recent_count != 0) {
				memcpy([w->recent contents], recent, (NSUInteger)recent_count * sizeof(uint32_t));
			}
			id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			if (!gllm_metal_encode_bf16_head(enc, w, w->x, w->out)) {
				[enc endEncoding];
				return false;
			}
			if (token != NULL) gllm_metal_encode_argmax(enc, w, recent_count, penalty);
			[enc endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			if ([cb status] != MTLCommandBufferStatusCompleted) {
				gllm_set_error(@"Metal BF16 head failed", [cb error]);
				return false;
			}
			if (logits != NULL) memcpy(logits, [w->out contents], (NSUInteger)w->rows * sizeof(float));
			if (token != NULL) {
				const GLLMArgmaxResult* result = [w->argmax contents];
				if (result->index >= (uint32_t)w->rows || !isfinite(result->value)) return false;
				*token = result->index;
			}
			return true;
		}
	}
}

static int64_t gllm_metal_bf16_head_argmax(void* handle, const float* x,
	const uint32_t* recent, uint32_t recent_count, float penalty) {
	uint32_t token;
	return gllm_metal_bf16_head(handle, x, NULL, recent, recent_count, penalty, &token) ?
		(int64_t)token : -1;
}
