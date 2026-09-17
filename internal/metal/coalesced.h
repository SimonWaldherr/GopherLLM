// Adjacent SIMD lanes read adjacent quantized values, avoiding strided
// per-superblock loads. Four-token variants share weight decoding.
static const char* gllm_q4k_coalesced_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; uint batch; };\n"
"kernel void gllm_q4k_coalesced_batch4(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],\n"
" device float* out [[buffer(2)]],constant Params& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],\n"
" uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group.x*p.rows_per_group+sg;\n"
" if(row>=p.rows) return;\n"
" uint token=group.y*4;\n"
" const device uchar* r=data+ulong(row)*p.row_bytes;\n"
" float4 sum=0.0f;\n"
" for(uint b=0;b<p.n_blocks;++b) {\n"
"\n"
"  const device uchar* block = r + b * 144;\n"
"\n"
"  float d = float(as_type<half>(ushort(ushort(block[0]) | (ushort(block[1]) << 8))));\n"
"  float dm = float(as_type<half>(ushort(ushort(block[2]) | (ushort(block[3]) << 8))));\n"
"  const device uchar* sc = block + 4;\n"
"  for (uint step=0; step<4; ++step) {\n"
"    uint j=step*2;\n"
"    uint s0,m0,s1,m1;\n"
"    if (j<4) {s0=sc[j]&63; m0=sc[j+4]&63; s1=sc[j+1]&63; m1=sc[j+5]&63;}\n"
"    else {s0=(sc[j+4]&15)|((sc[j-4]>>6)<<4); m0=(sc[j+4]>>4)|((sc[j]>>6)<<4);\n"
"          s1=(sc[j+5]&15)|((sc[j-3]>>6)<<4); m1=(sc[j+5]>>4)|((sc[j+1]>>6)<<4);}\n"
"    uchar q=block[16+step*32+lane];\n"
"    float v0=d*float(s0)*float(q&15)-dm*float(m0);\n"
"    float v1=d*float(s1)*float(q>>4)-dm*float(m1);\n"
"    uint col=b*256+step*64+lane;\n"
"    sum += v0 * float4(x[ulong(token)*p.cols+col], token+1<p.batch ? x[ulong(token)*p.cols+col+p.cols] : 0.0f, token+2<p.batch ? x[ulong(token)*p.cols+col+2*p.cols] : 0.0f, token+3<p.batch ? x[ulong(token)*p.cols+col+3*p.cols] : 0.0f);\n"
"    sum += v1 * float4(x[ulong(token)*p.cols+col+32], token+1<p.batch ? x[ulong(token)*p.cols+col+32+p.cols] : 0.0f, token+2<p.batch ? x[ulong(token)*p.cols+col+32+2*p.cols] : 0.0f, token+3<p.batch ? x[ulong(token)*p.cols+col+32+3*p.cols] : 0.0f);\n"
"  }\n"
"\n"
" }\n"
" for(ushort offset=16;offset>0;offset>>=1) sum+=simd_shuffle_xor(sum,offset);\n"
" if(lane==0) {\n"
"out[ulong(token)*p.rows+row]=sum.x;\n"
" if(token+1<p.batch) out[ulong(token+1)*p.rows+row]=sum.y;\n"
" if(token+2<p.batch) out[ulong(token+2)*p.rows+row]=sum.z;\n"
" if(token+3<p.batch) out[ulong(token+3)*p.rows+row]=sum.w;\n"
" }\n"
"}\n";
static const char* gllm_q6k_coalesced_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; uint batch; };\n"
"kernel void gllm_q6k_coalesced_batch4(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],\n"
" device float* out [[buffer(2)]],constant Params& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],\n"
" uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group.x*p.rows_per_group+sg;\n"
" if(row>=p.rows) return;\n"
" uint token=group.y*4;\n"
" const device uchar* r=data+ulong(row)*p.row_bytes;\n"
" float4 sum=0.0f;\n"
" for(uint b=0;b<p.n_blocks;++b) {\n"
"\n"
"  const device uchar* block = r + b * 210;\n"
"\n"
"  float d=float(as_type<half>(ushort(ushort(block[208]) | (ushort(block[209])<<8))));\n"
"  for (uint h=0; h<2; ++h) {\n"
"    uchar lo=block[h*64+lane], lo2=block[h*64+lane+32], hi=block[128+h*32+lane];\n"
"    uint qs[4]={uint(lo&15)|((uint(hi)&3)<<4),uint(lo2&15)|(((uint(hi)>>2)&3)<<4),\n"
"                uint(lo>>4)|(((uint(hi)>>4)&3)<<4),uint(lo2>>4)|((uint(hi)>>6)<<4)};\n"
"    for (uint part=0; part<4; ++part) {\n"
"      float sc=float(as_type<char>(block[192+h*8+part*2+lane/16]));\n"
"      float value=d*sc*float(int(qs[part])-32);\n"
"      uint col=b*256+h*128+part*32+lane;\n"
"      sum += value * float4(x[ulong(token)*p.cols+col], token+1<p.batch ? x[ulong(token)*p.cols+col+p.cols] : 0.0f, token+2<p.batch ? x[ulong(token)*p.cols+col+2*p.cols] : 0.0f, token+3<p.batch ? x[ulong(token)*p.cols+col+3*p.cols] : 0.0f);\n"
"    }\n"
"  }\n"
"\n"
" }\n"
" for(ushort offset=16;offset>0;offset>>=1) sum+=simd_shuffle_xor(sum,offset);\n"
" if(lane==0) {\n"
"out[ulong(token)*p.rows+row]=sum.x;\n"
" if(token+1<p.batch) out[ulong(token+1)*p.rows+row]=sum.y;\n"
" if(token+2<p.batch) out[ulong(token+2)*p.rows+row]=sum.z;\n"
" if(token+3<p.batch) out[ulong(token+3)*p.rows+row]=sum.w;\n"
" }\n"
"}\n";
static id<MTLComputePipelineState> gllm_metal_coalesced_pipeline(bool q6) {
    static id<MTLComputePipelineState> pipelines[2] = {nil,nil};
    static bool attempted[2] = {false,false};
    @synchronized(gllm_queue) {
        int kind=q6?1:0;
        if (!attempted[kind]) {
            attempted[kind]=true;
            NSError* error=nil;
            id<MTLLibrary> lib=[gllm_device newLibraryWithSource:[NSString stringWithUTF8String:q6?gllm_q6k_coalesced_source:gllm_q4k_coalesced_source] options:nil error:&error];
            if (lib==nil) {gllm_set_error(@"failed to compile coalesced prefill kernels",error);return nil;}
            id<MTLFunction> fn=[lib newFunctionWithName:q6?@"gllm_q6k_coalesced_batch4":@"gllm_q4k_coalesced_batch4"];
            id<MTLComputePipelineState> pipeline=fn!=nil?[gllm_device newComputePipelineStateWithFunction:fn error:&error]:nil;
            if (pipeline!=nil && [pipeline threadExecutionWidth]==32) pipelines[kind]=pipeline;
            else { [pipeline release]; gllm_set_error(@"coalesced prefill requires a 32-lane pipeline",error); }
            [fn release]; [lib release];
        }
        return pipelines[kind];
    }
    return nil;
}
static bool gllm_metal_encode_coalesced(id<MTLComputeCommandEncoder> enc,GLLMMetalWeight* w,
 id<MTLBuffer> x,id<MTLBuffer> out,int batch,bool q6) {
    if(batch<4) return false; // Decode wins with the established superblock-per-lane kernel.
    const char* setting=getenv("GOPHERLLM_METAL_COALESCED");
    if(setting!=NULL && strcmp(setting,"0")==0) return false;
    id<MTLComputePipelineState> pipeline=gllm_metal_coalesced_pipeline(q6);
    if(pipeline==nil) return false;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:w->weights offset:w->weight_offset atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1]; [enc setBuffer:out offset:0 atIndex:2];
    GLLMMetalParams p={(uint32_t)w->rows,(uint32_t)w->cols,(uint32_t)w->row_bytes,(uint32_t)(w->cols/256),4,(uint32_t)batch};
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)w->rows+3)/4,(batch+3)/4,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    return true;
}
static bool gllm_metal_coalesced_available(void) {
 return gllm_metal_init() && gllm_metal_coalesced_pipeline(false)!=nil && gllm_metal_coalesced_pipeline(true)!=nil;
}
