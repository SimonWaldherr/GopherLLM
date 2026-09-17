// Q8_0 offload for large FFN and vocabulary projections.
static const char* gllm_q8_0_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; uint batch; };\n"
"kernel void gllm_q8_0_matvec(const device uchar* data [[buffer(0)]], const device float* x [[buffer(1)]],\n"
" device float* out [[buffer(2)]], constant Params& p [[buffer(3)]],\n"
" uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],\n"
" uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row = group.x * p.rows_per_group + sg;\n"
" if (row >= p.rows) return;\n"
" uint token = group.y * 1;\n"
" const device uchar* r = data + ulong(row) * p.row_bytes;\n"
" float sum = 0.0f;\n"
" for (uint b = 0; b < p.n_blocks; ++b) {\n"
"   const device uchar* block = r + b * 34;\n"
"   ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);\n"
"   float scale = float(as_type<half>(bits));\n"
"   float value = scale * float(as_type<char>(block[2+lane]));\n"
"   ulong i = ulong(token) * p.cols + b*32 + lane;\n"
"   float activation = x[i];\n"
"   sum += value * activation;\n"
" }\n"
" for (ushort offset = 16; offset > 0; offset >>= 1) sum += simd_shuffle_xor(sum, offset);\n"
" if (lane == 0) {\n"
"   out[ulong(token)*p.rows+row] = sum;\n"
" }\n"
"}\n"
"kernel void gllm_q8_0_batch4(const device uchar* data [[buffer(0)]], const device float* x [[buffer(1)]],\n"
" device float* out [[buffer(2)]], constant Params& p [[buffer(3)]],\n"
" uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],\n"
" uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row = group.x * p.rows_per_group + sg;\n"
" if (row >= p.rows) return;\n"
" uint token = group.y * 4;\n"
" const device uchar* r = data + ulong(row) * p.row_bytes;\n"
" float4 sum = 0.0f;\n"
" for (uint b = 0; b < p.n_blocks; ++b) {\n"
"   const device uchar* block = r + b * 34;\n"
"   ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);\n"
"   float scale = float(as_type<half>(bits));\n"
"   float value = scale * float(as_type<char>(block[2+lane]));\n"
"   ulong i = ulong(token) * p.cols + b*32 + lane;\n"
"   float4 activation = float4(x[i], token+1 < p.batch ? x[i+p.cols] : 0.0f,\n"
"    token+2 < p.batch ? x[i+2*p.cols] : 0.0f, token+3 < p.batch ? x[i+3*p.cols] : 0.0f);\n"
"   sum += value * activation;\n"
" }\n"
" for (ushort offset = 16; offset > 0; offset >>= 1) sum += simd_shuffle_xor(sum, offset);\n"
" if (lane == 0) {\n"
"   out[ulong(token)*p.rows+row] = sum.x;\n"
"   if (token+1 < p.batch) out[ulong(token+1)*p.rows+row] = sum.y;\n"
"   if (token+2 < p.batch) out[ulong(token+2)*p.rows+row] = sum.z;\n"
"   if (token+3 < p.batch) out[ulong(token+3)*p.rows+row] = sum.w;\n"
" }\n"
"}\n";
static id<MTLComputePipelineState> gllm_q8_0_pipeline = nil;
static id<MTLComputePipelineState> gllm_q8_0_batch4_pipeline = nil;
static bool gllm_metal_init_q8_0(void) {
    if (!gllm_metal_init_q4k() || !gllm_metal_init_q6k()) return false;
    if (gllm_q8_0_pipeline != nil && gllm_q8_0_batch4_pipeline != nil) return true;
    NSError* error = nil;
    id<MTLLibrary> lib = [gllm_device newLibraryWithSource:[NSString stringWithUTF8String:gllm_q8_0_source] options:nil error:&error];
    if (lib == nil) { gllm_set_error(@"failed to compile Q8_0 Metal library", error); return false; }
    id<MTLFunction> single = [lib newFunctionWithName:@"gllm_q8_0_matvec"];
    id<MTLFunction> batch = [lib newFunctionWithName:@"gllm_q8_0_batch4"];
    id<MTLComputePipelineState> a = single != nil ? [gllm_device newComputePipelineStateWithFunction:single error:&error] : nil;
    id<MTLComputePipelineState> b = batch != nil ? [gllm_device newComputePipelineStateWithFunction:batch error:&error] : nil;
    [single release]; [batch release]; [lib release];
    if (a == nil || b == nil || [a threadExecutionWidth] != 32 || [b threadExecutionWidth] != 32) { [a release]; [b release]; gllm_set_error(@"failed to create Q8_0 pipelines", error); return false; }
    gllm_q8_0_pipeline = a; gllm_q8_0_batch4_pipeline = b;
    gllm_metal_matrix_pipeline(8);
    return true;
}
static void* gllm_metal_new_q8_0(const void* data, long len, int rows, int cols, bool no_copy) {
	@autoreleasepool {
		if (data == NULL || len <= 0 || rows <= 0 || cols <= 0 || (cols % 32) != 0) {
			return NULL;
		}
		if (!gllm_metal_init_q8_0()) {
			return NULL;
		}
		GLLMMetalWeight* w = (GLLMMetalWeight*)calloc(1, sizeof(GLLMMetalWeight));
		if (w == NULL) {
			strncpy(gllm_error, "failed to allocate Metal weight handle", sizeof(gllm_error) - 1);
			return NULL;
		}
		w->rows = rows;
		w->cols = cols;
		w->row_bytes = (cols / 32) * 34;
		w->weights = gllm_metal_new_weight_buffer(data, len, no_copy, &w->weight_offset);
		w->x = [gllm_device newBufferWithLength:(NSUInteger)cols * sizeof(float) options:MTLResourceStorageModeShared];
		w->out = [gllm_device newBufferWithLength:(NSUInteger)rows * sizeof(float) options:MTLResourceStorageModeShared];
		w->argmax = [gllm_device newBufferWithLength:sizeof(GLLMArgmaxResult) options:MTLResourceStorageModeShared];
		w->recent = [gllm_device newBufferWithLength:GLLM_REPEAT_WINDOW * sizeof(uint32_t) options:MTLResourceStorageModeShared];
		if (w->weights == nil || w->x == nil || w->out == nil || w->argmax == nil || w->recent == nil) {
			strncpy(gllm_error, "failed to allocate Metal weight buffer", sizeof(gllm_error) - 1);
			if (w->weights != nil) [w->weights release];
			if (w->x != nil) [w->x release];
			if (w->out != nil) [w->out release];
			if (w->argmax != nil) [w->argmax release];
			if (w->recent != nil) [w->recent release];
			free(w);
			return NULL;
		}
		return w;
	}
}

static void gllm_metal_encode_q8_0_to(id<MTLComputeCommandEncoder> enc, GLLMMetalWeight* w,
    id<MTLBuffer> x, id<MTLBuffer> out, int batch) {
    if (gllm_metal_encode_matrix(enc, w, x, out, batch, 8)) return;
    bool packed = batch >= 4;
    [enc setComputePipelineState:packed ? gllm_q8_0_batch4_pipeline : gllm_q8_0_pipeline];
    [enc setBuffer:w->weights offset:w->weight_offset atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1]; [enc setBuffer:out offset:0 atIndex:2];
    GLLMMetalParams p = { (uint32_t)w->rows, (uint32_t)w->cols, (uint32_t)w->row_bytes,
        (uint32_t)(w->cols / 32), 4, (uint32_t)batch };
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)w->rows+3)/4, packed ? (batch+3)/4 : batch, 1)
        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
static int gllm_metal_q8_0_matvec(void* handle, const float* x, float* out) {
	@autoreleasepool {
		GLLMMetalWeight* w = (GLLMMetalWeight*)handle;
		if (w == NULL || w->weights == nil || x == NULL || out == NULL || !gllm_metal_init_q8_0()) {
			return 0;
		}
		NSUInteger x_len = (NSUInteger)w->cols * sizeof(float);
		NSUInteger out_len = (NSUInteger)w->rows * sizeof(float);
		if (w->x == nil || w->out == nil) {
			strncpy(gllm_error, "missing Metal matvec buffers", sizeof(gllm_error) - 1);
			return 0;
		}
		@synchronized(w->weights) {
		memcpy([w->x contents], x, x_len);

		id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		gllm_metal_encode_q8_0_to(enc, w, w->x, w->out, 1);
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
		return 0;
	}
}

static int gllm_metal_q8_0_argmax(void* handle, const float* x, const uint32_t* recent, uint32_t recent_count, float repeat_penalty, uint32_t* token) {
	@autoreleasepool {
		GLLMMetalWeight* w = (GLLMMetalWeight*)handle;
		if (w == NULL || w->weights == nil || x == NULL || token == NULL || recent_count > GLLM_REPEAT_WINDOW ||
			(recent_count > 0 && recent == NULL) || !isfinite(repeat_penalty) || repeat_penalty <= 0.0f || !gllm_metal_init_q8_0()) {
			return 0;
		}
		if (w->x == nil || w->out == nil || w->argmax == nil || w->recent == nil || w->rows <= 0) {
			strncpy(gllm_error, "missing Metal argmax buffers", sizeof(gllm_error) - 1);
			return 0;
		}
		@synchronized(w->weights) {
		memcpy([w->x contents], x, (NSUInteger)w->cols * sizeof(float));
		if (recent_count > 0) {
			memcpy([w->recent contents], recent, (NSUInteger)recent_count * sizeof(uint32_t));
		}

		id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
		id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
		gllm_metal_encode_q8_0_to(enc, w, w->x, w->out, 1);
		// The reduction consumes w->out written by the preceding dispatch.
		// A compute encoder preserves that ordering, and avoiding a second
		// encoder matters on every greedy decode token for large vocabularies.
		gllm_metal_encode_argmax(enc, w, recent_count, repeat_penalty);
		[enc endEncoding];
		[cb commit];
		[cb waitUntilCompleted];
		if ([cb status] != MTLCommandBufferStatusCompleted) {
			strncpy(gllm_error, "Metal argmax command buffer failed", sizeof(gllm_error) - 1);
			return 0;
		}
		GLLMArgmaxResult result;
		memcpy(&result, [w->argmax contents], sizeof(result));
		if (result.index >= (uint32_t)w->rows || !isfinite(result.value)) {
			strncpy(gllm_error, "Metal argmax produced an invalid result", sizeof(gllm_error) - 1);
			return 0;
		}
		*token = result.index;
		return 1;
		}
		return 0;
	}
}

// Keep Q8_0 gate/up, SiLU, and down on device for decode and prefill.
static int gllm_metal_q8_0_silu_batch(
	void* gate_handle,
	void* up_handle,
	void* down_handle,
	const float* x,
	float* out,
	int batch
) {
	@autoreleasepool {
		GLLMMetalWeight* gate = (GLLMMetalWeight*)gate_handle;
		GLLMMetalWeight* up = (GLLMMetalWeight*)up_handle;
		GLLMMetalWeight* down = (GLLMMetalWeight*)down_handle;
		if (gate == NULL || up == NULL || down == NULL || batch <= 0 ||
			gate->weights == nil || up->weights == nil || down->weights == nil ||
			x == NULL || out == NULL || gate->cols != up->cols || gate->rows != up->rows ||
			down->cols != gate->rows ||
			gate->row_bytes != (gate->cols / 32) * 34 || up->row_bytes != (up->cols / 32) * 34 ||
			down->row_bytes != (down->cols / 32) * 34 ||
			!gllm_metal_init_q8_0()) {
			return 0;
		}
		// gllm_queue is process-global. Locking its small reusable workspace
		// avoids retaining five prompt slabs for every layer and protects two
		// concurrent runners from overlapping writes; the command buffer is
		// already synchronous, so this does not add a GPU synchronization point.
		@synchronized(gllm_queue) {
			if (!gllm_metal_ensure_batch_ffn_buffers(gate, up, down, batch)) {
				return 0;
			}
			NSUInteger count = (NSUInteger)batch;
			NSUInteger x_len = count * (NSUInteger)gate->cols * sizeof(float);
			NSUInteger out_len = count * (NSUInteger)down->rows * sizeof(float);
			memcpy([gllm_batch_workspace.x contents], x, x_len);

			id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

			gllm_metal_encode_q8_0_to(enc, gate, gllm_batch_workspace.x, gllm_batch_workspace.gate, batch);
			gllm_metal_encode_q8_0_to(enc, up, gllm_batch_workspace.x, gllm_batch_workspace.up, batch);
			gllm_metal_encode_silu(enc, gllm_batch_workspace.gate, gllm_batch_workspace.up, gllm_batch_workspace.hidden, (uint32_t)(count * (NSUInteger)gate->rows));
			gllm_metal_encode_q8_0_to(enc, down, gllm_batch_workspace.hidden, gllm_batch_workspace.out, batch);
			[enc endEncoding];
			[cb commit];
			[cb waitUntilCompleted];
			int ok = [cb status] == MTLCommandBufferStatusCompleted;
			if (ok) {
				memcpy(out, [gllm_batch_workspace.out contents], out_len);
			} else {
				strncpy(gllm_error, "Metal batched fused FFN command buffer failed", sizeof(gllm_error) - 1);
			}
			return ok;
		}
		return 0; // unreachable, keeps C's control-flow analysis explicit.
	}
}
