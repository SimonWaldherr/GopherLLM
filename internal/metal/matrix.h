// Quantized 32x32 prefill tiles, with float32 SIMD-group matrix accumulation.
static const char* gllm_matrix_source =
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"constant uint quant [[function_constant(0)]];\n"
"struct Params { uint rows; uint cols; uint row_bytes; uint n_blocks; uint rows_per_group; uint batch; };\n"
"inline float scale16(const device uchar* b) {\n"
" return float(as_type<half>(ushort(ushort(b[0]) | (ushort(b[1])<<8))));\n"
"}\n"
"inline float weight_at(const device uchar* row, uint col) {\n"
" if(quant==8) {\n"
"  const device uchar* b=row+(col/32)*34;\n"
"  return scale16(b)*float(as_type<char>(b[2+col%32]));\n"
" }\n"
" if(quant==4) {\n"
"  const device uchar* b=row+(col/256)*144;\n"
"  uint c=col%256, j=c/32;\n"
"  const device uchar* sc=b+4;\n"
"  uint s,m;\n"
"  if(j<4) { s=sc[j]&63; m=sc[j+4]&63; }\n"
"  else { s=(sc[j+4]&15)|((sc[j-4]>>6)<<4); m=(sc[j+4]>>4)|((sc[j]>>6)<<4); }\n"
"  uchar q=b[16+(c/64)*32+c%32];\n"
"  return scale16(b)*float(s)*float((j&1)?q>>4:q&15)-scale16(b+2)*float(m);\n"
" }\n"
" const device uchar* b=row+(col/256)*210;\n"
" uint c=col%256,h=c/128,part=(c%128)/32,lane=c%32;\n"
" uchar lo=b[h*64+lane+(part&1)*32],hi=b[128+h*32+lane];\n"
" uint q=uint(part>=2?lo>>4:lo&15)|(((uint(hi)>>(part*2))&3)<<4);\n"
" float sc=float(as_type<char>(b[192+h*8+part*2+lane/16]));\n"
" return scale16(b+208)*sc*float(int(q)-32);\n"
"}\n"
"kernel void gllm_quant_matrix(const device uchar* weights [[buffer(0)]],\n"
" const device float* x [[buffer(1)]], device float* out [[buffer(2)]],\n"
" constant Params& p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],\n"
" uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {\n"
" threadgroup float tileA[1024];\n"
" threadgroup float tileB[1024];\n"
" uint row0=group.x*32,token0=group.y*32;\n"
" uint mr=(sg/2)*16,mc=(sg%2)*16;\n"
" simdgroup_float8x8 c00(0.0f),c01(0.0f),c10(0.0f),c11(0.0f);\n"
" for(uint k0=0;k0<p.cols;k0+=32) {\n"
"  for(uint i=tid;i<1024;i+=128) {\n"
"   uint r=i/32,k=i%32;\n"
"   tileA[i]=row0+r<p.rows?weight_at(weights+ulong(row0+r)*p.row_bytes,k0+k):0.0f;\n"
"   // Each lane loads adjacent columns from a token, then transposes in shared memory.\n"
"   uint t=i/32;\n"
"   tileB[k*32+t]=token0+t<p.batch?x[ulong(token0+t)*p.cols+k0+k]:0.0f;\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  for(uint k=0;k<32;k+=8) {\n"
"   simdgroup_float8x8 a0,a1,b0,b1;\n"
"   simdgroup_load(a0,tileA+mr*32+k,32);\n"
"   simdgroup_load(a1,tileA+(mr+8)*32+k,32);\n"
"   simdgroup_load(b0,tileB+k*32+mc,32);\n"
"   simdgroup_load(b1,tileB+k*32+mc+8,32);\n"
"   simdgroup_multiply_accumulate(c00,a0,b0,c00);\n"
"   simdgroup_multiply_accumulate(c01,a0,b1,c01);\n"
"   simdgroup_multiply_accumulate(c10,a1,b0,c10);\n"
"   simdgroup_multiply_accumulate(c11,a1,b1,c11);\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
" }\n"
" simdgroup_store(c00,tileA+mr*32+mc,32);\n"
" simdgroup_store(c01,tileA+mr*32+mc+8,32);\n"
" simdgroup_store(c10,tileA+(mr+8)*32+mc,32);\n"
" simdgroup_store(c11,tileA+(mr+8)*32+mc+8,32);\n"
" threadgroup_barrier(mem_flags::mem_threadgroup);\n"
" for(uint i=tid;i<1024;i+=128) {\n"
"  uint r=i/32,t=i%32;\n"
"  if(row0+r<p.rows && token0+t<p.batch) out[ulong(token0+t)*p.rows+row0+r]=tileA[i];\n"
" }\n"
"}\n"
;

static id<MTLComputePipelineState> gllm_metal_matrix_pipeline(uint32_t quant) {
 static id<MTLComputePipelineState> pipelines[3]={nil,nil,nil};
 static bool attempted[3]={false,false,false};
 @synchronized(gllm_queue) {
  int index=quant==4?0:quant==6?1:2;
  if(!attempted[index]) {
   attempted[index]=true;
   NSError* error=nil;
   id<MTLLibrary> lib=[gllm_device newLibraryWithSource:[NSString stringWithUTF8String:gllm_matrix_source] options:nil error:&error];
   if(lib==nil) { gllm_set_error(@"failed to compile matrix prefill",error); return nil; }
   MTLFunctionConstantValues* values=[[MTLFunctionConstantValues alloc] init];
   [values setConstantValue:&quant type:MTLDataTypeUInt atIndex:0];
   id<MTLFunction> fn=[lib newFunctionWithName:@"gllm_quant_matrix" constantValues:values error:&error];
   id<MTLComputePipelineState> pipe=fn!=nil?[gllm_device newComputePipelineStateWithFunction:fn error:&error]:nil;
   if(pipe!=nil && [pipe threadExecutionWidth]==32 && [pipe maxTotalThreadsPerThreadgroup]>=128) pipelines[index]=pipe;
   else { [pipe release]; gllm_set_error(@"matrix prefill pipeline unavailable",error); }
   [values release]; [fn release]; [lib release];
  }
  return pipelines[index];
 }
 return nil;
}
static bool gllm_metal_encode_matrix(id<MTLComputeCommandEncoder> enc,GLLMMetalWeight* w,
 id<MTLBuffer> x,id<MTLBuffer> out,int batch,uint32_t quant) {
 if(batch<16 || w->cols%32!=0) return false;
 const char* setting=getenv("GOPHERLLM_METAL_MATRIX");
 if(setting!=NULL && strcmp(setting,"0")==0) return false;
 id<MTLComputePipelineState> pipe=gllm_metal_matrix_pipeline(quant);
 if(pipe==nil) return false;
 [enc setComputePipelineState:pipe];
 [enc setBuffer:w->weights offset:w->weight_offset atIndex:0];
 [enc setBuffer:x offset:0 atIndex:1]; [enc setBuffer:out offset:0 atIndex:2];
 GLLMMetalParams p={(uint32_t)w->rows,(uint32_t)w->cols,(uint32_t)w->row_bytes,0,0,(uint32_t)batch};
 [enc setBytes:&p length:sizeof(p) atIndex:3];
 [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)w->rows+31)/32,(batch+31)/32,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
 return true;
}
static bool gllm_metal_matrix_available(void) {
 return gllm_metal_init() && gllm_metal_matrix_pipeline(4)!=nil && gllm_metal_matrix_pipeline(6)!=nil && gllm_metal_matrix_pipeline(8)!=nil;
}
