// Bounded whole-chunk dense prefill. All intermediates remain on the GPU.
static const char* gllm_prefill_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct PR { uint heads,batch,interleaved,is_query; };\n"
"kernel void pf_rope(const device float* x [[buffer(0)]],device float* y [[buffer(1)]],const device float* sn [[buffer(2)]],const device float* cs [[buffer(3)]],const device float* temperature [[buffer(4)]],constant PR& p [[buffer(5)]],uint i [[thread_position_in_grid]]) {\n"
" if(i>=p.batch*p.heads*128)return;\n"
" uint token=i/(p.heads*128),d=i%128,base=i-d;\n"
" uint j=p.interleaved?d/2:d%64;\n"
" uint a=p.interleaved?2*j:j,b=p.interleaved?2*j+1:j+64;\n"
" float s=sn[token*64+j],c=cs[token*64+j];\n"
" float v=d==a?x[base+a]*c-x[base+b]*s:x[base+a]*s+x[base+b]*c;\n"
" y[i]=v*(p.is_query?temperature[token]:1.0f);\n"
"}\n"
"kernel void pf_copy(const device float* x [[buffer(0)]],device float* y [[buffer(1)]],constant uint& n [[buffer(2)]],uint i [[thread_position_in_grid]]) {\n"
" if(i<n)y[i]=x[i];\n"
"}\n"
"struct PA { uint heads,kvheads,stride,start_pos,window,chunks,base_token; float scale; };\n"
"inline float4 pf_kv(const device float* prefix,const device float* pending,uint t,uint start,uint stride,uint kv,uint lane) {\n"
" const device float* data=t<start?prefix:pending;\n"
" ulong o=ulong(t<start?t:t-start)*stride+kv*128+lane;\n"
" return float4(data[o],data[o+32],data[o+64],data[o+96]);\n"
"}\n"
"kernel void pf_attention(const device float* q [[buffer(0)]],const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],const device float* pendingK [[buffer(3)]],const device float* pendingV [[buffer(4)]],device float* out [[buffer(5)]],constant PA& p [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint token=group.x+p.base_token,head=group.y,kv=head/(p.heads/p.kvheads),pos=p.start_pos+token;\n"
" uint lo=p.window>0 && pos>p.window?pos-p.window:0;\n"
" ulong qo=ulong(token)*p.heads*128+head*128+lane;\n"
" float4 query=float4(q[qo],q[qo+32],q[qo+64],q[qo+96]);\n"
" float m=-INFINITY,s=0;float4 acc=0;\n"
" for(uint t=lo;t<=pos;t++) {\n"
"  float score=simd_sum(dot(query,pf_kv(k,pendingK,t,p.start_pos,p.stride,kv,lane)))*p.scale;\n"
"  float next=max(m,score),a=exp(m-next),b=exp(score-next);\n"
"  acc=acc*a+pf_kv(v,pendingV,t,p.start_pos,p.stride,kv,lane)*b;s=s*a+b;m=next;\n"
" }\n"
" out[qo]=acc.x/s;out[qo+32]=acc.y/s;out[qo+64]=acc.z/s;out[qo+96]=acc.w/s;\n"
"}\n"
"kernel void pf_attention_part(const device float* q [[buffer(0)]],const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],const device float* pendingK [[buffer(3)]],const device float* pendingV [[buffer(4)]],device float* partial [[buffer(5)]],constant PA& p [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint token=group.x+p.base_token,head=group.y,kv=head/(p.heads/p.kvheads),pos=p.start_pos+token;\n"
" uint lo=p.window>0 && pos>p.window?pos-p.window:0;\n"
" uint start=lo+group.z*32,end=min(start+32,pos+1);\n"
" ulong po=((ulong(group.x)*p.heads+head)*p.chunks+group.z)*130;\n"
" if(start>=end) {\n"
"  partial[po+lane]=0;partial[po+lane+32]=0;partial[po+lane+64]=0;partial[po+lane+96]=0;\n"
"  if(lane==0){partial[po+128]=-INFINITY;partial[po+129]=0;}return;\n"
" }\n"
" ulong qo=ulong(token)*p.heads*128+head*128+lane;\n"
" float4 query=float4(q[qo],q[qo+32],q[qo+64],q[qo+96]);\n"
" float m=-INFINITY,s=0;float4 acc=0;\n"
" for(uint t=start;t<end;t++) {\n"
"  float score=simd_sum(dot(query,pf_kv(k,pendingK,t,p.start_pos,p.stride,kv,lane)))*p.scale;\n"
"  float next=max(m,score),a=exp(m-next),b=exp(score-next);\n"
"  acc=acc*a+pf_kv(v,pendingV,t,p.start_pos,p.stride,kv,lane)*b;s=s*a+b;m=next;\n"
" }\n"
" partial[po+lane]=acc.x;partial[po+lane+32]=acc.y;partial[po+lane+64]=acc.z;partial[po+lane+96]=acc.w;\n"
" if(lane==0){partial[po+128]=m;partial[po+129]=s;}\n"
"}\n"
"kernel void pf_attention_merge(const device float* partial [[buffer(0)]],device float* out [[buffer(1)]],constant PA& p [[buffer(2)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint token=group.x+p.base_token,head=group.y;\n"
" float m=-INFINITY;\n"
" for(uint c=0;c<p.chunks;c++)m=max(m,partial[(ulong(group.x)*p.heads*p.chunks+head*p.chunks+c)*130+128]);\n"
" float s=0;float4 acc=0;\n"
" for(uint c=0;c<p.chunks;c++) {\n"
"  ulong po=(ulong(group.x)*p.heads*p.chunks+head*p.chunks+c)*130;\n"
"  float a=exp(partial[po+128]-m);s+=a*partial[po+129];\n"
"  acc+=a*float4(partial[po+lane],partial[po+lane+32],partial[po+lane+64],partial[po+lane+96]);\n"
" }\n"
" ulong o=ulong(token)*p.heads*128+head*128+lane;\n"
" out[o]=acc.x/s;out[o+32]=acc.y/s;out[o+64]=acc.z/s;out[o+96]=acc.w/s;\n"
"}\n"
;

// One serialized decoder owns bounded private prefill scratch. Pending KV rows
// never alias a shared Go cache: a failed/canceled submission publishes nothing.
enum {
 PF_X,PF_XN,PF_Q,PF_QR,PF_K,PF_KR,PF_V,PF_ATTN,PF_PROJ,PF_GATE,PF_UP,PF_HID,
 PF_SN,PF_CS,PF_TEMP,PF_PENDING_K,PF_PENDING_V,PF_PARTIAL,PF_BUFFERS,
};
#define GLLM_PREFILL_PARTIAL_BYTES ((NSUInteger)64*1024*1024)
#define GLLM_PREFILL_TOTAL_BYTES ((NSUInteger)256*1024*1024)
typedef struct { id<MTLBuffer> b[PF_BUFFERS]; } GLLMPrefill;
typedef struct { uint32_t heads,batch,interleaved,is_query; } GLLMPrefillRope;
typedef struct { uint32_t heads,kvheads,stride,start_pos,window,chunks,base_token;float scale; } GLLMPrefillAttention;
static id<MTLComputePipelineState> gllm_prefill_pipes[5]={nil};

static void gllm_prefill_release(void* ptr) {
 GLLMPrefill* p=ptr;if(p==NULL)return;
 for(int i=0;i<PF_BUFFERS;i++)[p->b[i] release];
 free(p);
}
static bool gllm_prefill_init(void) {
 if(!gllm_decode_init() || !gllm_metal_init_q6k() || !gllm_metal_init_q8_0())return false;
 @synchronized(gllm_queue) {
  if(gllm_prefill_pipes[0]!=nil)return true;
  NSError* error=nil;
  id<MTLLibrary> lib=[gllm_device newLibraryWithSource:[NSString stringWithUTF8String:gllm_prefill_source] options:nil error:&error];
  if(lib==nil){gllm_set_error(@"prefill shader compilation failed",error);return false;}
  NSString* names[5]={@"pf_rope",@"pf_copy",@"pf_attention",@"pf_attention_part",@"pf_attention_merge"};
  id<MTLComputePipelineState> pipes[5]={nil};bool ok=true;
  for(int i=0;i<5;i++){
   id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
   pipes[i]=fn!=nil?[gllm_device newComputePipelineStateWithFunction:fn error:&error]:nil;[fn release];
   if(pipes[i]==nil || [pipes[i] threadExecutionWidth]!=32 || [pipes[i] maxTotalThreadsPerThreadgroup]<256)ok=false;
  }
  [lib release];
  if(!ok){for(int i=0;i<5;i++)[pipes[i] release];gllm_set_error(@"prefill pipelines unavailable",error);return false;}
  for(int i=0;i<5;i++)gllm_prefill_pipes[i]=pipes[i];
  return true;
 }
 return false;
}
static bool gllm_prefill_buffers(GLLMDecoder* d,int batch,int start_pos,bool tiled) {
 NSUInteger p=(NSUInteger)batch,q=(NSUInteger)d->heads*128,kv=(NSUInteger)d->kvheads*128;
 NSUInteger chunks=((NSUInteger)start_pos+p+31)/32;
 NSUInteger partial=tiled?MIN(GLLM_PREFILL_PARTIAL_BYTES/sizeof(float),p*q/128*chunks*130):1;
 NSUInteger count[PF_BUFFERS]={
  p*d->dim,p*d->dim,p*q,p*q,p*kv,p*kv,p*kv,p*q,p*d->dim,p*d->hidden,p*d->hidden,p*d->hidden,
  p*64,p*64,p,p*kv*d->layers,p*kv*d->layers,partial,
 };
 GLLMPrefill* work=d->prefill;
 NSUInteger total=0;
 for(int i=0;i<PF_BUFFERS;i++){
  if(count[i]>GLLM_PREFILL_TOTAL_BYTES/sizeof(float))return false;
  NSUInteger bytes=count[i]*sizeof(float);
  if(work!=NULL)bytes=MAX(bytes,[work->b[i] length]);
  if(bytes>GLLM_PREFILL_TOTAL_BYTES-total)return false;total+=bytes;
 }
 if(work==NULL){work=calloc(1,sizeof(GLLMPrefill));if(work==NULL)return false;d->prefill=work;}
 for(int i=0;i<PF_BUFFERS;i++){
  NSUInteger bytes=count[i]*sizeof(float);
  if([work->b[i] length]>=bytes)continue;
  [work->b[i] release];work->b[i]=nil;
  work->b[i]=gllm_dec_buffer(count[i]);
  if(work->b[i]==nil)return false;
 }
 return true;
}
static NSUInteger gllm_prefill_bytes(void* ptr) {
 GLLMDecoder* d=ptr;if(d==NULL || d->prefill==NULL)return 0;
 GLLMPrefill* p=d->prefill;NSUInteger bytes=0;
 for(int i=0;i<PF_BUFFERS;i++)bytes+=[p->b[i] length];
 return bytes;
}
static void gllm_prefill_add(id<MTLComputeCommandEncoder> e,id<MTLBuffer> x,id<MTLBuffer> y,uint32_t n) {
 [e setComputePipelineState:gllm_dec_pipes[3]];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:y offset:0 atIndex:1];
 [e setBytes:&n length:sizeof(n) atIndex:2];[e dispatchThreadgroups:MTLSizeMake((n+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void gllm_prefill_rope(id<MTLComputeCommandEncoder> e,GLLMPrefill* p,id<MTLBuffer> x,id<MTLBuffer> y,NSUInteger offset,int heads,int batch,bool interleaved,bool query) {
 GLLMPrefillRope rp={(uint32_t)heads,(uint32_t)batch,(uint32_t)interleaved,(uint32_t)query};
 [e setComputePipelineState:gllm_prefill_pipes[0]];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:y offset:offset atIndex:1];
 [e setBuffer:p->b[PF_SN] offset:0 atIndex:2];[e setBuffer:p->b[PF_CS] offset:0 atIndex:3];[e setBuffer:p->b[PF_TEMP] offset:0 atIndex:4];
 [e setBytes:&rp length:sizeof(rp) atIndex:5];
 [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)heads*128*batch+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void gllm_prefill_attention(id<MTLComputeCommandEncoder> e,GLLMDecoder* d,GLLMDecodeLayer* l,GLLMPrefill* p,id<MTLBuffer> q,int layer,int batch,int start_pos,bool tiled) {
 uint32_t last=(uint32_t)(start_pos+batch-1),window=(uint32_t)MAX(l->window,0);
 uint32_t span=window>0 && last>window?window+1:last+1,chunks=(span+31)/32;
 NSUInteger one=(NSUInteger)d->heads*chunks*130*sizeof(float),capacity=[p->b[PF_PARTIAL] length]/one;
 NSUInteger offset=(NSUInteger)layer*batch*d->kvheads*128*sizeof(float);
 bool parts=tiled && capacity>0;
 int tile=parts?MIN((NSUInteger)batch,capacity):batch;
 for(int first=0;first<batch;first+=tile){
  int n=MIN(tile,batch-first);
  GLLMPrefillAttention ap={(uint32_t)d->heads,(uint32_t)d->kvheads,(uint32_t)d->kvheads*128,(uint32_t)start_pos,window,chunks,(uint32_t)first,d->scale};
  [e setComputePipelineState:gllm_prefill_pipes[parts?3:2]];
  [e setBuffer:q offset:0 atIndex:0];[e setBuffer:l->k offset:0 atIndex:1];[e setBuffer:l->v offset:0 atIndex:2];
  [e setBuffer:p->b[PF_PENDING_K] offset:offset atIndex:3];[e setBuffer:p->b[PF_PENDING_V] offset:offset atIndex:4];
  [e setBuffer:p->b[parts?PF_PARTIAL:PF_ATTN] offset:0 atIndex:5];[e setBytes:&ap length:sizeof(ap) atIndex:6];
  [e dispatchThreadgroups:MTLSizeMake(n,d->heads,parts?chunks:1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  if(parts){
   [e setComputePipelineState:gllm_prefill_pipes[4]];[e setBuffer:p->b[PF_PARTIAL] offset:0 atIndex:0];[e setBuffer:p->b[PF_ATTN] offset:0 atIndex:1];[e setBytes:&ap length:sizeof(ap) atIndex:2];
   [e dispatchThreadgroups:MTLSizeMake(n,d->heads,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  }
 }
}
// This function writes device-private scratch only. Publication is separate,
// after the Go caller has checked cancellation and the command completed.
static bool gllm_prefill_batch(void* ptr,const float* input,const float* sn,const float* cs,const float* temperature,int start_pos,int batch,bool interleaved,bool tiled) {
 @autoreleasepool {
 GLLMDecoder* d=ptr;
 if(d==NULL || batch<2 || batch>256 || start_pos<0 || start_pos>d->maxlen-batch || input==NULL || sn==NULL || cs==NULL || temperature==NULL)return false;
 if(!gllm_prefill_init() || !gllm_prefill_buffers(d,batch,start_pos,tiled))return false;
 GLLMPrefill* p=d->prefill;
 for(int layer=0;layer<d->layers;layer++)for(int i=0;i<7;i++)if(d->layer[layer].w[i]==NULL)return false;
 memcpy([p->b[PF_X] contents],input,(NSUInteger)batch*d->dim*sizeof(float));
 memcpy([p->b[PF_SN] contents],sn,(NSUInteger)batch*64*sizeof(float));
 memcpy([p->b[PF_CS] contents],cs,(NSUInteger)batch*64*sizeof(float));
 memcpy([p->b[PF_TEMP] contents],temperature,(NSUInteger)batch*sizeof(float));
 id<MTLCommandBuffer> cb=gllm_metal_new_command_buffer();
 id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
 if(cb==nil || e==nil)return false;
 for(int layer=0;layer<d->layers;layer++){
  GLLMDecodeLayer* l=&d->layer[layer];
  gllm_decode_norm(e,p->b[PF_X],l->norm,p->b[PF_XN],d->dim,batch,d->eps);
  gllm_metal_encode_quantized_to(e,l->w[0],l->quant[0],p->b[PF_XN],p->b[PF_Q],batch,4);
  gllm_metal_encode_quantized_to(e,l->w[1],l->quant[1],p->b[PF_XN],p->b[PF_K],batch,4);
  gllm_metal_encode_quantized_to(e,l->w[2],l->quant[2],p->b[PF_XN],p->b[PF_V],batch,4);
  id<MTLBuffer> q=p->b[PF_Q],k=p->b[PF_K];
  if(l->qNorm!=nil){gllm_decode_norm(e,q,l->qNorm,p->b[PF_QR],128,batch*d->heads,d->eps);q=p->b[PF_QR];}
  if(l->kNorm!=nil){gllm_decode_norm(e,k,l->kNorm,p->b[PF_KR],128,batch*d->kvheads,d->eps);k=p->b[PF_KR];}
  id<MTLBuffer> qr=q==p->b[PF_Q]?p->b[PF_QR]:p->b[PF_Q];
  gllm_prefill_rope(e,p,q,qr,0,d->heads,batch,interleaved,true);
  NSUInteger kvCount=(NSUInteger)batch*d->kvheads*128,offset=(NSUInteger)layer*kvCount*sizeof(float);
  gllm_prefill_rope(e,p,k,p->b[PF_PENDING_K],offset,d->kvheads,batch,interleaved,false);
  [e setComputePipelineState:gllm_prefill_pipes[1]];[e setBuffer:p->b[PF_V] offset:0 atIndex:0];[e setBuffer:p->b[PF_PENDING_V] offset:offset atIndex:1];
  uint32_t count=(uint32_t)kvCount;[e setBytes:&count length:sizeof(count) atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake((kvCount+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  gllm_prefill_attention(e,d,l,p,qr,layer,batch,start_pos,tiled);
  gllm_metal_encode_quantized_to(e,l->w[3],l->quant[3],p->b[PF_ATTN],p->b[PF_PROJ],batch,4);
  gllm_prefill_add(e,p->b[PF_X],p->b[PF_PROJ],(uint32_t)batch*d->dim);
  gllm_decode_norm(e,p->b[PF_X],l->ffnNorm,p->b[PF_XN],d->dim,batch,d->eps);
  gllm_metal_encode_quantized_to(e,l->w[4],l->quant[4],p->b[PF_XN],p->b[PF_GATE],batch,4);
  gllm_metal_encode_quantized_to(e,l->w[5],l->quant[5],p->b[PF_XN],p->b[PF_UP],batch,4);
  gllm_metal_encode_silu(e,p->b[PF_GATE],p->b[PF_UP],p->b[PF_HID],(uint32_t)batch*d->hidden);
  gllm_metal_encode_quantized_to(e,l->w[6],l->quant[6],p->b[PF_HID],p->b[PF_PROJ],batch,4);
  gllm_prefill_add(e,p->b[PF_X],p->b[PF_PROJ],(uint32_t)batch*d->dim);
 }
 gllm_decode_norm(e,p->b[PF_X],d->outputNorm,p->b[PF_XN],d->dim,batch,d->eps);
 [e endEncoding];[cb commit];[cb waitUntilCompleted];
 if([cb status]!=MTLCommandBufferStatusCompleted){gllm_set_error(@"whole prefill command failed",[cb error]);return false;}
 return true;
 }
}
static void gllm_prefill_publish_cache(void* ptr,int layer,int start_pos,int batch,float* k,float* v) {
 GLLMDecoder* d=ptr;GLLMPrefill* p=d->prefill;GLLMDecodeLayer* l=&d->layer[layer];
 NSUInteger stride=(NSUInteger)d->kvheads*128*sizeof(float),bytes=(NSUInteger)batch*stride,offset=(NSUInteger)layer*bytes,dst=(NSUInteger)start_pos*stride;
 memcpy((char*)k+dst,(char*)[p->b[PF_PENDING_K] contents]+offset,bytes);
 memcpy((char*)v+dst,(char*)[p->b[PF_PENDING_V] contents]+offset,bytes);
 if(!l->sharedKV){
  memcpy((char*)[l->k contents]+dst,(char*)[p->b[PF_PENDING_K] contents]+offset,bytes);
  memcpy((char*)[l->v contents]+dst,(char*)[p->b[PF_PENDING_V] contents]+offset,bytes);
 }
}
static void gllm_prefill_publish_output(void* ptr,int batch,float* residual,float* output) {
 GLLMDecoder* d=ptr;GLLMPrefill* p=d->prefill;NSUInteger bytes=(NSUInteger)batch*d->dim*sizeof(float);
 memcpy(residual,[p->b[PF_X] contents],bytes);memcpy(output,[p->b[PF_XN] contents],bytes);
}
