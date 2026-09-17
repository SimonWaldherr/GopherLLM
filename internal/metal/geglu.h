// GeGLU FFN: quantized gate/up -> tanh-GELU multiplication -> down.
// Shares the bounded prefill workspace under the same lock as SwiGLU.
static id<MTLComputePipelineState> gllm_geglu_pipeline=nil;
static bool gllm_geglu_init(void) {
 if(!gllm_decode_init())return false;
 @synchronized(gllm_queue) {
  if(gllm_geglu_pipeline!=nil)return true;
  NSString* source=@"#include <metal_stdlib>\nusing namespace metal;\nkernel void geglu(const device float* g [[buffer(0)]],const device float* u [[buffer(1)]],device float* h [[buffer(2)]],constant uint& n [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<n){float v=g[i];float z=1.5957691216057308f*(v+0.044715f*v*v*v);h[i]=(v/(1.0f+exp(-z)))*u[i];}}";
  NSError* error=nil;id<MTLLibrary> lib=[gllm_device newLibraryWithSource:source options:nil error:&error];
  id<MTLFunction> fn=lib!=nil?[lib newFunctionWithName:@"geglu"]:nil;
  id<MTLComputePipelineState> pipe=fn!=nil?[gllm_device newComputePipelineStateWithFunction:fn error:&error]:nil;
  [fn release];[lib release];
  if(pipe==nil){gllm_set_error(@"GeGLU shader initialization failed",error);return false;}
  gllm_geglu_pipeline=pipe;return true;
 }
}
static bool gllm_gelu_quant_valid(GLLMMetalWeight* w,uint32_t quant) {
 if(w==NULL || w->weights==nil || w->cols<=0 || w->rows<=0)return false;
 if(quant==4)return w->cols%256==0 && w->row_bytes==(w->cols/256)*144;
 if(quant==6)return w->cols%256==0 && w->row_bytes==(w->cols/256)*210;
 if(quant==8)return w->cols%32==0 && w->row_bytes==(w->cols/32)*34;
 return false;
}
static void gllm_gelu_matvec(id<MTLComputeCommandEncoder> e,GLLMMetalWeight* w,uint32_t quant,id<MTLBuffer> x,id<MTLBuffer> out,int batch) {
 if(batch==1){gllm_decode_quantized(e,w,quant,x,out);return;}
 if(quant==4)gllm_metal_encode_q4k_to(e,w,x,out,batch,4);
 else if(quant==6)gllm_metal_encode_q6k_to(e,w,x,out,batch,4);
 else gllm_metal_encode_q8_0_to(e,w,x,out,batch);
}
static bool gllm_geglu(void* gh,void* uh,void* dh,const uint32_t* quant,const float* x,float* out,int batch) {
 @autoreleasepool {
 GLLMMetalWeight *g=gh,*u=uh,*d=dh;
 if(!gllm_gelu_quant_valid(g,quant[0]) || !gllm_gelu_quant_valid(u,quant[1]) || !gllm_gelu_quant_valid(d,quant[2]) ||
    g->rows!=u->rows || g->cols!=u->cols || d->cols!=g->rows || batch<1 || batch>GLLM_BATCH_FFN_MAX_TOKENS || !gllm_geglu_init())return false;
 @synchronized(gllm_queue) {
  if(!gllm_metal_ensure_batch_ffn_buffers(g,u,d,batch))return false;
  memcpy([gllm_batch_workspace.x contents],x,(NSUInteger)batch*g->cols*sizeof(float));
  id<MTLCommandBuffer> cb=gllm_metal_new_command_buffer();
  id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  gllm_gelu_matvec(e,g,quant[0],gllm_batch_workspace.x,gllm_batch_workspace.gate,batch);
  gllm_gelu_matvec(e,u,quant[1],gllm_batch_workspace.x,gllm_batch_workspace.up,batch);
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  [e setComputePipelineState:gllm_geglu_pipeline];
  [e setBuffer:gllm_batch_workspace.gate offset:0 atIndex:0];[e setBuffer:gllm_batch_workspace.up offset:0 atIndex:1];[e setBuffer:gllm_batch_workspace.hidden offset:0 atIndex:2];
  uint32_t n=(uint32_t)((NSUInteger)batch*g->rows);[e setBytes:&n length:sizeof(n) atIndex:3];
  NSUInteger threads=MIN((NSUInteger)256,[gllm_geglu_pipeline maxTotalThreadsPerThreadgroup]);
  [e dispatchThreadgroups:MTLSizeMake((n+threads-1)/threads,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
  [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
  gllm_gelu_matvec(e,d,quant[2],gllm_batch_workspace.hidden,gllm_batch_workspace.out,batch);
  [e endEncoding];[cb commit];[cb waitUntilCompleted];
  if([cb status]!=MTLCommandBufferStatusCompleted){gllm_set_error(@"GeGLU command failed",[cb error]);return false;}
  memcpy(out,[gllm_batch_workspace.out contents],(NSUInteger)batch*d->rows*sizeof(float));return true;
 }
 }
}

static id<MTLBuffer> gllm_gemma_recent=nil;
static bool gllm_gemma_output(void* handle,uint32_t quant,const float* x,float* out,const uint32_t* recent,uint32_t count,float penalty,uint32_t* next) {
 @autoreleasepool {
 GLLMMetalWeight* w=handle;
 if(!gllm_gelu_quant_valid(w,quant) || count>GLLM_REPEAT_WINDOW || !gllm_decode_init() || !gllm_metal_init_q6k())return false;
 @synchronized(gllm_queue) {
  if(!gllm_metal_ensure_batch_argmax_buffers(w,1))return false;
  if(next!=NULL) {
   if(gllm_gemma_recent==nil)gllm_gemma_recent=gllm_dec_buffer(GLLM_REPEAT_WINDOW);
   if(gllm_gemma_recent==nil)return false;
   if(count>0)memcpy([gllm_gemma_recent contents],recent,count*sizeof(uint32_t));
  }
  memcpy([gllm_batch_workspace.x contents],x,(NSUInteger)w->cols*sizeof(float));
  id<MTLCommandBuffer> cb=gllm_metal_new_command_buffer();id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  gllm_decode_quantized(e,w,quant,gllm_batch_workspace.x,gllm_batch_workspace.out);
  if(next!=NULL) {
   GLLMMetalWeight temp=*w;temp.recent=gllm_gemma_recent;
   gllm_metal_encode_argmax_to(e,&temp,gllm_batch_workspace.out,0,gllm_batch_workspace.argmax,0,count,penalty);
  }
  [e endEncoding];[cb commit];[cb waitUntilCompleted];
  if([cb status]!=MTLCommandBufferStatusCompleted){gllm_set_error(@"Gemma output projection failed",[cb error]);return false;}
  if(next!=NULL){GLLMArgmaxResult r;memcpy(&r,[gllm_batch_workspace.argmax contents],sizeof(r));if(r.index>=(uint32_t)w->rows || !isfinite(r.value))return false;*next=r.index;}
  else memcpy(out,[gllm_batch_workspace.out contents],(NSUInteger)w->rows*sizeof(float));
  return true;
 }
 }
}
