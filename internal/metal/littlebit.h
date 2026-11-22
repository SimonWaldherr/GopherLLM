// A LittleBit projection first writes the scaled latent values for both
// branches, then produces the output rows. Packed factors and original scale
// bytes stay resident. QKV and SwiGLU each submit a group in one command buffer.
typedef struct {
	uint32_t offset;
	uint32_t kind;
} GLLMLittleBitScale;

typedef struct {
	uint32_t rank;
	uint32_t u;
	uint32_t v;
	GLLMLittleBitScale u1;
	GLLMLittleBitScale u2;
	GLLMLittleBitScale v1;
	GLLMLittleBitScale v2;
} GLLMLittleBitFactors;

typedef struct {
	uint32_t rows;
	uint32_t cols;
	GLLMLittleBitFactors main;
	GLLMLittleBitFactors residual;
	GLLMLittleBitScale bias;
} GLLMLittleBitParams;

typedef struct {
	id<MTLBuffer> data;
	GLLMLittleBitParams params;
} GLLMLittleBitWeight;

static id<MTLComputePipelineState> gllm_littlebit_latent_pipeline;
static id<MTLComputePipelineState> gllm_littlebit_latent_two_tokens_pipeline;
static id<MTLComputePipelineState> gllm_littlebit_latent_bf16_decode_pipeline;
static id<MTLComputePipelineState> gllm_littlebit_output_pipeline;
static id<MTLComputePipelineState> gllm_littlebit_swiglu_pipeline;
static id<MTLBuffer> gllm_littlebit_x;
static id<MTLBuffer> gllm_littlebit_z;
static id<MTLBuffer> gllm_littlebit_y;

static NSString* const gllm_littlebit_source =
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Scale {uint offset; uint kind;};\n"
"struct Factors {uint rank; uint u; uint v; Scale u1; Scale u2; Scale v1; Scale v2;};\n"
"struct Params {uint rows; uint cols; Factors main; Factors residual; Scale bias;};\n"
"static float scale(const device uchar* data, Scale s, uint i) {\n"
"  if (s.kind == 1) return reinterpret_cast<const device float*>(data + s.offset)[i];\n"
"  ushort raw = reinterpret_cast<const device ushort*>(data + s.offset)[i];\n"
"  return s.kind == 2 ? float(as_type<half>(raw)) : as_type<float>(uint(raw) << 16);\n"
"}\n"
"static float signed_value(uint packed, uint i, float value) {\n"
"  return as_type<float>(as_type<uint>(value) ^ (((packed >> (i & 31)) & 1u) << 31));\n"
"}\n"
"static float signed_mask(uint sign, float value) {\n"
"  return as_type<float>(as_type<uint>(value) ^ (sign & 0x80000000u));\n"
"}\n"
"template <bool paired>\n"
"static void latent_rows(const device uchar* data, const device float* x, device float* z,\n"
"                        constant Params& p, uint batch, uint2 group, uint sg, uint lane) {\n"
"  // A SIMD group shares x*v2 between two adjacent V rows of one branch.\n"
"  uint pair = group.x * 4 + sg;\n"
"  uint rank_sum = p.main.rank + p.residual.rank;\n"
"  uint main_pairs = (p.main.rank >> 1) + (p.main.rank & 1);\n"
"  uint residual_pairs = (p.residual.rank >> 1) + (p.residual.rank & 1);\n"
"  if (pair >= main_pairs + residual_pairs) return;\n"
"  bool residual = pair >= main_pairs;\n"
"  Factors f = residual ? p.residual : p.main;\n"
"  uint row = 2 * (pair - (residual ? main_pairs : 0));\n"
"  bool second = row + 1 < f.rank;\n"
"  uint words = ((p.cols - 1) >> 5) + 1;\n"
"  const device uint* v0 = reinterpret_cast<const device uint*>(data + f.v) + row * words;\n"
"  // Repeat the last row of an odd branch so both accumulators stay in bounds.\n"
"  const device uint* v1 = v0 + (second ? words : 0);\n"
"  uint token = group.y * (paired ? 2 : 1);\n"
"  bool has_next = paired && token + 1 < batch;\n"
"  const device float* input = x + token * p.cols;\n"
"  // Reuse the last token in an odd batch, but never write a second result.\n"
"  const device float* next = input + (has_next ? p.cols : 0);\n"
"  uint sign_shift = 31 - lane;\n"
"  uint full = p.cols >> 5;\n"
"  float sum0 = 0.0f, sum1 = 0.0f, next0 = 0.0f, next1 = 0.0f;\n"
"  for (uint word = 0; word < full; ++word) {\n"
"    uint col = word * 32 + lane;\n"
"    float s = scale(data, f.v2, col), scaled = input[col] * s;\n"
"    uint mask0 = v0[word] << sign_shift, mask1 = v1[word] << sign_shift;\n"
"    sum0 += signed_mask(mask0, scaled);\n"
"    sum1 += signed_mask(mask1, scaled);\n"
"    if (paired) {\n"
"      float scaled_next = next[col] * s;\n"
"      next0 += signed_mask(mask0, scaled_next);\n"
"      next1 += signed_mask(mask1, scaled_next);\n"
"    }\n"
"  }\n"
"  if (lane < (p.cols & 31)) {\n"
"    uint col = full * 32 + lane;\n"
"    float s = scale(data, f.v2, col), scaled = input[col] * s;\n"
"    uint mask0 = v0[full] << sign_shift, mask1 = v1[full] << sign_shift;\n"
"    sum0 += signed_mask(mask0, scaled);\n"
"    sum1 += signed_mask(mask1, scaled);\n"
"    if (paired) {\n"
"      float scaled_next = next[col] * s;\n"
"      next0 += signed_mask(mask0, scaled_next);\n"
"      next1 += signed_mask(mask1, scaled_next);\n"
"    }\n"
"  }\n"
"  sum0 = simd_sum(sum0);\n"
"  sum1 = simd_sum(sum1);\n"
"  if (paired) { next0 = simd_sum(next0); next1 = simd_sum(next1); }\n"
"  if (lane == 0) {\n"
"    uint offset = token * rank_sum + (residual ? p.main.rank : 0) + row;\n"
"    float s0 = scale(data, f.v1, row) * scale(data, f.u2, row);\n"
"    z[offset] = sum0 * s0;\n"
"    if (has_next) z[offset + rank_sum] = next0 * s0;\n"
"    if (second) {\n"
"      float s1 = scale(data, f.v1, row + 1) * scale(data, f.u2, row + 1);\n"
"      z[offset + 1] = sum1 * s1;\n"
"      if (has_next) z[offset + rank_sum + 1] = next1 * s1;\n"
"    }\n"
"  }\n"
"}\n"
"kernel void littlebit_latent(\n"
"    const device uchar* data [[buffer(0)]], const device float* x [[buffer(1)]],\n"
"    device float* z [[buffer(2)]], constant Params& p [[buffer(3)]],\n"
"    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]) { latent_rows<false>(data,x,z,p,0,group,sg,lane); }\n"
"kernel void littlebit_latent_two_tokens(\n"
"    const device uchar* data [[buffer(0)]], const device float* x [[buffer(1)]],\n"
"    device float* z [[buffer(2)]], constant Params& p [[buffer(3)]],\n"
"    constant uint& batch [[buffer(4)]],\n"
"    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]) { latent_rows<true>(data,x,z,p,batch,group,sg,lane); }\n"
"kernel void littlebit_latent_bf16_decode(\n"
"    const device uchar* data [[buffer(0)]], const device float* x [[buffer(1)]],\n"
"    device float* z [[buffer(2)]], constant Params& p [[buffer(3)]],\n"
"    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],\n"
"    uint lane [[thread_index_in_simdgroup]]) {\n"
"  uint quad = group * 2 + sg;\n"
"  uint main_quads = (p.main.rank + 3) / 4;\n"
"  if (quad >= main_quads + (p.residual.rank + 3) / 4) return;\n"
"  bool residual = quad >= main_quads;\n"
"  Factors f = residual ? p.residual : p.main;\n"
"  uint row = 4 * (quad - (residual ? main_quads : 0));\n"
"  uint words = p.cols / 32;\n"
"  const device uint* packed = reinterpret_cast<const device uint*>(data + f.v);\n"
"  const device ushort* v2 = reinterpret_cast<const device ushort*>(data + f.v2.offset);\n"
"  uint4 rows = min(row + uint4(0,1,2,3), uint4(f.rank - 1));\n"
"  uint4 base = rows * words;\n"
"  uint shift = 31 - lane;\n"
"  float4 sum = 0;\n"
"  for (uint word = 0; word < words; ++word) {\n"
"    uint col = word * 32 + lane;\n"
"    float scaled = x[col] * as_type<float>(uint(v2[col]) << 16);\n"
"    uint4 masks = uint4(packed[base.x + word], packed[base.y + word],\n"
"                        packed[base.z + word], packed[base.w + word]) << shift;\n"
"    sum += as_type<float4>(uint4(as_type<uint>(scaled)) ^ (masks & 0x80000000u));\n"
"  }\n"
"  sum = float4(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z), simd_sum(sum.w));\n"
"  if (lane == 0) {\n"
"    uint offset = residual ? p.main.rank : 0;\n"
"    for (uint i = 0; i < 4 && row + i < f.rank; ++i) {\n"
"      uint r = row + i;\n"
"      z[offset + r] = sum[i] * (scale(data, f.v1, r) * scale(data, f.u2, r));\n"
"    }\n"
"  }\n"
"}\n"
"static float output_row(const device uchar* data, const device float* z,\n"
"                         Factors f, uint row, uint sublane) {\n"
"  uint words = ((f.rank - 1) >> 5) + 1;\n"
"  const device uint* u = reinterpret_cast<const device uint*>(data + f.u) + row * words;\n"
"  float sum = 0.0f;\n"
"  uint full = f.rank >> 5;\n"
"  // Read a packed word once per lane and keep the original summation order.\n"
"  for (uint word = 0; word < full; ++word) {\n"
"    uint bits = u[word] >> sublane;\n"
"    uint col = word * 32 + sublane;\n"
"    sum += signed_mask(bits << 31, z[col]);\n"
"    sum += signed_mask(bits << 23, z[col + 8]);\n"
"    sum += signed_mask(bits << 15, z[col + 16]);\n"
"    sum += signed_mask(bits << 7, z[col + 24]);\n"
"  }\n"
"  if ((f.rank & 31) != 0) {\n"
"    uint bits = u[full];\n"
"    for (uint col = full * 32 + sublane; col < f.rank; col += 8) {\n"
"      sum += signed_value(bits, col, z[col]);\n"
"    }\n"
"  }\n"
"  sum += simd_shuffle_xor(sum, 4);\n"
"  sum += simd_shuffle_xor(sum, 2);\n"
"  sum += simd_shuffle_xor(sum, 1);\n"
"  return sum * scale(data, f.u1, row);\n"
"}\n"
"static float output_value(const device uchar* data, const device float* z,\n"
"                          constant Params& p, uint row, uint sublane, uint token) {\n"
"  const device float* latent = z + token * (p.main.rank + p.residual.rank);\n"
"  float sum = output_row(data, latent, p.main, row, sublane);\n"
"  if (p.residual.rank != 0) sum += output_row(data, latent + p.main.rank, p.residual, row, sublane);\n"
"  if (sublane == 0 && p.bias.kind != 0) sum += scale(data, p.bias, row);\n"
"  return sum;\n"
"}\n"
"kernel void littlebit_output(\n"
"    const device uchar* data [[buffer(0)]], const device float* z [[buffer(1)]],\n"
"    device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],\n"
"    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {\n"
"  uint row = group.x * 16 + (tid >> 3);\n"
"  uint sublane = tid & 7;\n"
"  if (row >= p.rows) return;\n"
"  float sum = output_value(data, z, p, row, sublane, group.y);\n"
"  if (sublane == 0) y[group.y * p.rows + row] = sum;\n"
"}\n"
"kernel void littlebit_swiglu(\n"
"    const device uchar* gate [[buffer(0)]], const device uchar* up [[buffer(1)]],\n"
"    const device float* zg [[buffer(2)]], const device float* zu [[buffer(3)]],\n"
"    device float* hidden [[buffer(4)]], constant Params& pg [[buffer(5)]],\n"
"    constant Params& pu [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]]) {\n"
"  uint row = group.x * 16 + (tid >> 3);\n"
"  uint sublane = tid & 7;\n"
"  if (row >= pg.rows) return;\n"
"  float g = output_value(gate, zg, pg, row, sublane, group.y);\n"
"  float u = output_value(up, zu, pu, row, sublane, group.y);\n"
"  if (sublane == 0) hidden[group.y * pg.rows + row] = (g / (1.0f + precise::exp(-g))) * u;\n"
"}\n";

static bool gllm_metal_init_littlebit(void) {
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		if (!gllm_metal_init()) return;
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
		id<MTLLibrary> library = [gllm_device newLibraryWithSource:gllm_littlebit_source options:options error:&error];
		[options release];
		if (library == nil) {
			gllm_set_error(@"failed to compile Metal LittleBit kernels", error);
			return;
		}
		id<MTLFunction> latent = [library newFunctionWithName:@"littlebit_latent"];
		id<MTLFunction> output = [library newFunctionWithName:@"littlebit_output"];
		id<MTLFunction> swiglu = [library newFunctionWithName:@"littlebit_swiglu"];
		if (latent != nil && output != nil) {
			gllm_littlebit_latent_pipeline = [gllm_device newComputePipelineStateWithFunction:latent error:&error];
			if (gllm_littlebit_latent_pipeline != nil)
				gllm_littlebit_output_pipeline = [gllm_device newComputePipelineStateWithFunction:output error:&error];
		}
		[latent release];
		[output release];
		if (swiglu != nil)
			gllm_littlebit_swiglu_pipeline = [gllm_device newComputePipelineStateWithFunction:swiglu error:NULL];
		[swiglu release];
		id<MTLFunction> paired = [library newFunctionWithName:@"littlebit_latent_two_tokens"];
		if (paired != nil) {
			id<MTLComputePipelineState> pipe = [gllm_device newComputePipelineStateWithFunction:paired error:NULL];
			if ([pipe threadExecutionWidth] == 32 && [pipe maxTotalThreadsPerThreadgroup] >= 128)
				gllm_littlebit_latent_two_tokens_pipeline = pipe;
			else [pipe release];
			[paired release];
		}
		id<MTLFunction> decode = [library newFunctionWithName:@"littlebit_latent_bf16_decode"];
		if (decode != nil) {
			id<MTLComputePipelineState> pipe = [gllm_device newComputePipelineStateWithFunction:decode error:NULL];
			if ([pipe threadExecutionWidth] == 32 && [pipe maxTotalThreadsPerThreadgroup] >= 64)
				gllm_littlebit_latent_bf16_decode_pipeline = pipe;
			else [pipe release];
			[decode release];
		}
		if (gllm_littlebit_swiglu_pipeline != nil &&
		    ([gllm_littlebit_swiglu_pipeline threadExecutionWidth] != 32 ||
		     [gllm_littlebit_swiglu_pipeline maxTotalThreadsPerThreadgroup] < 128)) {
			[gllm_littlebit_swiglu_pipeline release];
			gllm_littlebit_swiglu_pipeline = nil;
		}
		[library release];
		if (gllm_littlebit_latent_pipeline == nil || gllm_littlebit_output_pipeline == nil) {
			gllm_set_error(@"failed to create Metal LittleBit pipelines", error);
		} else if ([gllm_littlebit_latent_pipeline threadExecutionWidth] == 32 &&
		           [gllm_littlebit_output_pipeline threadExecutionWidth] == 32 &&
		           [gllm_littlebit_latent_pipeline maxTotalThreadsPerThreadgroup] >= 128 &&
		           [gllm_littlebit_output_pipeline maxTotalThreadsPerThreadgroup] >= 128) {
			return;
		} else {
			gllm_set_error(@"unsupported Metal LittleBit SIMD group size", nil);
		}
		[gllm_littlebit_latent_pipeline release];
		[gllm_littlebit_latent_two_tokens_pipeline release];
		[gllm_littlebit_latent_bf16_decode_pipeline release];
		[gllm_littlebit_output_pipeline release];
		[gllm_littlebit_swiglu_pipeline release];
		gllm_littlebit_latent_pipeline = nil;
		gllm_littlebit_latent_two_tokens_pipeline = nil;
		gllm_littlebit_latent_bf16_decode_pipeline = nil;
		gllm_littlebit_output_pipeline = nil;
		gllm_littlebit_swiglu_pipeline = nil;
	});
	return gllm_littlebit_output_pipeline != nil;
}

static void* gllm_metal_new_littlebit(NSUInteger length, const GLLMLittleBitParams* params) {
	@autoreleasepool {
		if (!gllm_metal_init_littlebit()) return NULL;
		if (length > [gllm_device maxBufferLength]) {
			gllm_set_error(@"Metal LittleBit buffers exceed maxBufferLength", nil);
			return NULL;
		}
		GLLMLittleBitWeight* w = calloc(1, sizeof(*w));
		if (w == NULL) return NULL;
		w->data = [gllm_device newBufferWithLength:length options:MTLResourceStorageModeShared];
		if (w->data == nil) {
			gllm_set_error(@"failed to allocate Metal LittleBit factors", nil);
			free(w);
			return NULL;
		}
		w->params = *params;
		return w;
	}
}

static void* gllm_metal_littlebit_contents(void* ptr) {
	GLLMLittleBitWeight* w = ptr;
	return [w->data contents];
}

static void gllm_metal_release_littlebit(void* ptr) {
	GLLMLittleBitWeight* w = ptr;
	[w->data release];
	free(w);
}

static NSUInteger gllm_metal_littlebit_scratch_bytes(void) {
	@synchronized(gllm_queue) {
		return [gllm_littlebit_x length] + [gllm_littlebit_z length] + [gllm_littlebit_y length];
	}
}

// Commit a larger workspace only after every required allocation succeeds.
// The caller holds the same queue lock used by all LittleBit submissions.
static bool gllm_metal_littlebit_ensure_scratch(NSUInteger x, NSUInteger z, NSUInteger y) {
	id<MTLBuffer>* buffers[] = {&gllm_littlebit_x, &gllm_littlebit_z, &gllm_littlebit_y};
	NSUInteger lengths[] = {x, z, y};
	id<MTLBuffer> replacements[3] = {nil, nil, nil};
	NSUInteger limit = [gllm_device maxBufferLength];
	if (x > limit || z > limit || y > limit) {
		gllm_set_error(@"Metal LittleBit scratch exceeds maxBufferLength", nil);
		return false;
	}
	for (int i = 0; i < 3; ++i) {
		if ([*buffers[i] length] >= lengths[i]) continue;
		replacements[i] = [gllm_device newBufferWithLength:lengths[i] options:MTLResourceStorageModeShared];
		if (replacements[i] == nil) {
			for (int j = 0; j < i; ++j) [replacements[j] release];
			gllm_set_error(@"failed to allocate Metal LittleBit scratch buffers", nil);
			return false;
		}
	}
	for (int i = 0; i < 3; ++i) {
		if (replacements[i] == nil) continue;
		[*buffers[i] release];
		*buffers[i] = replacements[i];
	}
	return true;
}

static NSUInteger gllm_littlebit_rank(GLLMLittleBitWeight* w) {
	return (NSUInteger)w->params.main.rank + w->params.residual.rank;
}

static NSUInteger gllm_littlebit_latent_groups(GLLMLittleBitWeight* w) {
	// A threadgroup has four SIMD groups. Each branch rounds up to whole pairs.
	NSUInteger main = ((NSUInteger)w->params.main.rank + 1) / 2;
	NSUInteger residual = ((NSUInteger)w->params.residual.rank + 1) / 2;
	return (main + residual + 3) / 4;
}

static void gllm_encode_littlebit_latent(id<MTLComputeCommandEncoder> enc, GLLMLittleBitWeight* w,
                                         id<MTLBuffer> x, id<MTLBuffer> z, NSUInteger z_offset, NSUInteger batch) {
	bool paired = gllm_littlebit_latent_two_tokens_pipeline != nil && batch >= 16;
	// Whole packed words and BF16 V2 let each SIMD group reuse one scaled
	// activation across four rows. Two groups still cover eight rows per tile.
	bool decode = batch == 1 && w->params.cols % 32 == 0 && w->params.main.v2.kind == 3 &&
	              (w->params.residual.rank == 0 || w->params.residual.v2.kind == 3) &&
	              gllm_littlebit_latent_bf16_decode_pipeline != nil;
	id<MTLComputePipelineState> pipeline = gllm_littlebit_latent_pipeline;
	NSUInteger groups = gllm_littlebit_latent_groups(w), threads = 128;
	if (decode) {
		pipeline = gllm_littlebit_latent_bf16_decode_pipeline;
		NSUInteger main = ((NSUInteger)w->params.main.rank + 3) / 4;
		NSUInteger residual = ((NSUInteger)w->params.residual.rank + 3) / 4;
		groups = (main + residual + 1) / 2;
		threads = 64;
	} else if (paired) {
		pipeline = gllm_littlebit_latent_two_tokens_pipeline;
	}
	[enc setComputePipelineState:pipeline];
	[enc setBuffer:w->data offset:0 atIndex:0];
	[enc setBuffer:x offset:0 atIndex:1];
	[enc setBuffer:z offset:z_offset atIndex:2];
	[enc setBytes:&w->params length:sizeof(w->params) atIndex:3];
	if (paired) {
		uint32_t tokens = (uint32_t)batch;
		[enc setBytes:&tokens length:sizeof(tokens) atIndex:4];
	}
	[enc dispatchThreadgroups:MTLSizeMake(groups, paired ? (batch + 1) / 2 : batch, 1)
	        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

static void gllm_littlebit_encode_latent(id<MTLComputeCommandEncoder> enc, GLLMLittleBitWeight* w,
                                        NSUInteger batch, NSUInteger z_offset) {
	gllm_encode_littlebit_latent(enc, w, gllm_littlebit_x, gllm_littlebit_z, z_offset, batch);
}

static void gllm_littlebit_encode_output(id<MTLComputeCommandEncoder> enc, GLLMLittleBitWeight* w,
                                        NSUInteger batch, NSUInteger z_offset, NSUInteger y_offset) {
	[enc setComputePipelineState:gllm_littlebit_output_pipeline];
	[enc setBuffer:w->data offset:0 atIndex:0];
	[enc setBuffer:gllm_littlebit_z offset:z_offset atIndex:1];
	[enc setBuffer:gllm_littlebit_y offset:y_offset atIndex:2];
	[enc setBytes:&w->params length:sizeof(w->params) atIndex:3];
	[enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)w->params.rows + 15) / 16, batch, 1)
	        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static bool gllm_littlebit_complete(id<MTLCommandBuffer> cb, id<MTLComputeCommandEncoder> enc) {
	[enc endEncoding];
	[cb commit];
	[cb waitUntilCompleted];
	if ([cb status] == MTLCommandBufferStatusCompleted) return true;
	gllm_set_error(@"Metal LittleBit command buffer failed", [cb error]);
	return false;
}

static bool gllm_littlebit_project_group(GLLMLittleBitWeight* const* weights, float* const* outputs,
                                         NSUInteger count, const float* x, NSUInteger batch) {
	@autoreleasepool {
		@synchronized(gllm_queue) {
			NSUInteger step = batch * sizeof(float);
			NSUInteger x_bytes = step * weights[0]->params.cols;
			NSUInteger z_bytes = 0, y_bytes = 0;
			for (NSUInteger i = 0; i < count; ++i) {
				z_bytes += step * gllm_littlebit_rank(weights[i]);
				y_bytes += step * weights[i]->params.rows;
			}
			if (!gllm_metal_littlebit_ensure_scratch(x_bytes, z_bytes, y_bytes))
				return false;

			memcpy([gllm_littlebit_x contents], x, x_bytes);
			id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			if (enc == nil) return false;

			NSUInteger z_offset = 0, y_offset = 0;
			for (NSUInteger i = 0; i < count; ++i) {
				gllm_littlebit_encode_latent(enc, weights[i], batch, z_offset);
				z_offset += step * gllm_littlebit_rank(weights[i]);
			}
			[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
			z_offset = 0;
			for (NSUInteger i = 0; i < count; ++i) {
				gllm_littlebit_encode_output(enc, weights[i], batch, z_offset, y_offset);
				z_offset += step * gllm_littlebit_rank(weights[i]);
				y_offset += step * weights[i]->params.rows;
			}
			if (!gllm_littlebit_complete(cb, enc)) return false;

			const char* y = [gllm_littlebit_y contents];
			for (NSUInteger i = 0; i < count; ++i) {
				NSUInteger bytes = step * weights[i]->params.rows;
				memcpy(outputs[i], y, bytes);
				y += bytes;
			}
			return true;
		}
	}
}

static bool gllm_metal_littlebit_matvec(void* ptr, const float* x, float* out, NSUInteger batch) {
	GLLMLittleBitWeight* weights[] = {ptr};
	float* outputs[] = {out};
	return gllm_littlebit_project_group(weights, outputs, 1, x, batch);
}

static bool gllm_metal_littlebit_qkv(void* q, void* k, void* v, const float* x,
                                    float* oq, float* ok, float* ov, NSUInteger batch) {
	GLLMLittleBitWeight* weights[] = {q, k, v};
	float* outputs[] = {oq, ok, ov};
	return gllm_littlebit_project_group(weights, outputs, 3, x, batch);
}

static bool gllm_metal_littlebit_ffn(void* g, void* u, void* d, const float* x,
                                    float* out, NSUInteger batch) {
	@autoreleasepool {
		GLLMLittleBitWeight* gate = g;
		GLLMLittleBitWeight* up = u;
		GLLMLittleBitWeight* down = d;
		if (gllm_littlebit_swiglu_pipeline == nil) return false;
		@synchronized(gllm_queue) {
			NSUInteger step = batch * sizeof(float);
			NSUInteger x_bytes = step * gate->params.cols;
			NSUInteger hidden_bytes = step * gate->params.rows;
			NSUInteger gate_z = step * gllm_littlebit_rank(gate);
			NSUInteger gu_z = gate_z + step * gllm_littlebit_rank(up);
			NSUInteger down_z = step * gllm_littlebit_rank(down);
			NSUInteger y_bytes = step * down->params.rows;
			if (!gllm_metal_littlebit_ensure_scratch(MAX(x_bytes, hidden_bytes), MAX(gu_z, down_z), y_bytes))
				return false;

			memcpy([gllm_littlebit_x contents], x, x_bytes);
			id<MTLCommandBuffer> cb = gllm_metal_new_command_buffer();
			id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
			if (enc == nil) return false;
			gllm_littlebit_encode_latent(enc, gate, batch, 0);
			gllm_littlebit_encode_latent(enc, up, batch, gate_z);
			[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

			// Both projections have read the input. Reuse its storage for the
			// hidden activations produced by the fused Gate/Up output kernel.
			[enc setComputePipelineState:gllm_littlebit_swiglu_pipeline];
			[enc setBuffer:gate->data offset:0 atIndex:0];
			[enc setBuffer:up->data offset:0 atIndex:1];
			[enc setBuffer:gllm_littlebit_z offset:0 atIndex:2];
			[enc setBuffer:gllm_littlebit_z offset:gate_z atIndex:3];
			[enc setBuffer:gllm_littlebit_x offset:0 atIndex:4];
			[enc setBytes:&gate->params length:sizeof(gate->params) atIndex:5];
			[enc setBytes:&up->params length:sizeof(up->params) atIndex:6];
			[enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)gate->params.rows + 15) / 16, batch, 1)
			        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
			[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

			gllm_littlebit_encode_latent(enc, down, batch, 0);
			[enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
			gllm_littlebit_encode_output(enc, down, batch, 0, 0);
			if (!gllm_littlebit_complete(cb, enc)) return false;
			memcpy(out, [gllm_littlebit_y contents], y_bytes);
			return true;
		}
	}
}
