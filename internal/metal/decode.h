// Complete dense decode submission. All activations stay on device until final normalization.
static const char* gllm_decode_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct NP { uint n; float eps; };\n"
"kernel void dec_norm(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], device float* y [[buffer(2)]], constant NP& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {\n"
" threadgroup float sums[8];\n"
" float sum=0; uint base=group*p.n;\n"
" for(uint i=tid;i<p.n;i+=256) sum+=x[base+i]*x[base+i];\n"
" sum=simd_sum(sum); if(lane==0) sums[sg]=sum;\n"
" threadgroup_barrier(mem_flags::mem_threadgroup);\n"
" float total=simd_sum(lane<8?sums[lane]:0.0f);\n"
" float scale=rsqrt(total/float(p.n)+p.eps);\n"
" for(uint i=tid;i<p.n;i+=256) y[base+i]=x[base+i]*scale*w[i];\n"
"}\n"
"struct RP { uint heads; uint hd; uint pairs; uint pos; uint stride; uint interleaved; float temperature; };\n"
"kernel void dec_rope(const device float* x [[buffer(0)]],device float* y [[buffer(1)]],const device float* sn [[buffer(2)]],const device float* cs [[buffer(3)]],constant RP& p [[buffer(4)]],uint i [[thread_position_in_grid]]) {\n"
" if(i>=p.heads*p.hd)return;\n"
" uint d=i%p.hd,base=i-d;\n"
" float v=x[i];\n"
" if(d<2*p.pairs) {\n"
"  uint j=p.interleaved?d/2:d%p.pairs;\n"
"  uint a=p.interleaved?2*j:j,b=p.interleaved?2*j+1:j+p.pairs;\n"
"  v=d==a?x[base+a]*cs[j]-x[base+b]*sn[j]:x[base+a]*sn[j]+x[base+b]*cs[j];\n"
" }\n"
" y[ulong(p.pos)*p.stride+i]=v*p.temperature;\n"
"}\n"
"kernel void dec_copy_kv(const device float* x [[buffer(0)]],device float* y [[buffer(1)]],constant uint& n [[buffer(2)]],constant uint& pos [[buffer(3)]],uint i [[thread_position_in_grid]]) {\n"
" if(i<n)y[ulong(pos)*n+i]=x[i];\n"
"}\n"
"kernel void dec_add(device float* x [[buffer(0)]],const device float* y [[buffer(1)]],constant uint& n [[buffer(2)]],uint i [[thread_position_in_grid]]) { if(i<n)x[i]+=y[i]; }\n"
"struct AP { uint heads; uint kvheads; uint hd; uint stride; uint pos; uint start; uint chunks; float scale; };\n"
"kernel void dec_attention_part(const device float* q [[buffer(0)]],const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],device float* partial [[buffer(3)]],constant AP& p [[buffer(4)]],uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint job=group*4+sg,head=job/p.chunks,part=job%p.chunks;\n"
" if(head>=p.heads)return;\n"
" uint kv=head/(p.heads/p.kvheads),start=p.start+part*32,end=min(start+32,p.pos+1);\n"
" float4 query=float4(q[head*128+lane],q[head*128+lane+32],q[head*128+lane+64],q[head*128+lane+96]);\n"
" float m=-INFINITY,s=0;float4 acc=0;\n"
" for(uint t=start;t<end;t++) {\n"
"  ulong i=ulong(t)*p.stride+kv*128+lane;\n"
"  float4 key=float4(k[i],k[i+32],k[i+64],k[i+96]);\n"
"  float score=simd_sum(dot(query,key))*p.scale;\n"
"  float next=max(m,score),a=exp(m-next),b=exp(score-next);\n"
"  float4 value=float4(v[i],v[i+32],v[i+64],v[i+96]);\n"
"  acc=acc*a+value*b;s=s*a+b;m=next;\n"
" }\n"
" ulong o=ulong(job)*130;\n"
" partial[o+lane]=acc.x;partial[o+lane+32]=acc.y;partial[o+lane+64]=acc.z;partial[o+lane+96]=acc.w;\n"
" if(lane==0){partial[o+128]=m;partial[o+129]=s;}\n"
"}\n"
"kernel void dec_attention_merge(const device float* partial [[buffer(0)]],device float* out [[buffer(1)]],constant AP& p [[buffer(2)]],uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {\n"
" float m=-INFINITY;\n"
" for(uint c=0;c<p.chunks;c++)m=max(m,partial[(ulong(head)*p.chunks+c)*130+128]);\n"
" float s=0;float4 acc=0;\n"
" for(uint c=0;c<p.chunks;c++) {\n"
"  ulong o=(ulong(head)*p.chunks+c)*130;float a=exp(partial[o+128]-m);\n"
"  s+=a*partial[o+129];acc+=a*float4(partial[o+lane],partial[o+lane+32],partial[o+lane+64],partial[o+lane+96]);\n"
" }\n"
" out[head*128+lane]=acc.x/s;out[head*128+lane+32]=acc.y/s;out[head*128+lane+64]=acc.z/s;out[head*128+lane+96]=acc.w/s;\n"
"}\n"
"struct MV { uint rows,cols,row_bytes; };\n"
"kernel void dec_q4(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],device float* y [[buffer(2)]],constant MV& p [[buffer(3)]],uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group*4+sg; if(row>=p.rows)return;\n"
" float sum=0; uint sub=lane%8;\n"
" for(uint b=lane/8;b<p.cols/256;b+=4) {\n"
"  const device uchar* v=data+ulong(row)*p.row_bytes+b*144;\n"
"  float d=float(*((const device half*)v)), dm=float(*((const device half*)(v+2)));\n"
"  const device uchar* sc=v+4;\n"
"  for(uint step=0;step<4;step++) {\n"
"   uint j=step*2,s0,m0,s1,m1;\n"
"   if(j<4){s0=sc[j]&63;m0=sc[j+4]&63;s1=sc[j+1]&63;m1=sc[j+5]&63;}\n"
"   else{s0=(sc[j+4]&15)|((sc[j-4]>>6)<<4);m0=(sc[j+4]>>4)|((sc[j]>>6)<<4);s1=(sc[j+5]&15)|((sc[j-3]>>6)<<4);m1=(sc[j+5]>>4)|((sc[j+1]>>6)<<4);}\n"
"   uchar4 packed=*((const device uchar4*)(v+16+step*32+sub*4));\n"
"   float4 a=*((const device float4*)(x+b*256+step*64+sub*4));\n"
"   float4 z=*((const device float4*)(x+b*256+step*64+sub*4+32));\n"
"   sum+=dot(d*float(s0)*float4(packed&uchar4(15))-dm*float(m0),a);\n"
"   sum+=dot(d*float(s1)*float4(packed>>uchar4(4))-dm*float(m1),z);\n"
"  }\n"
" }\n"
" sum=simd_sum(sum);if(lane==0)y[row]=sum;\n"
"}\n"
"kernel void dec_q6(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],device float* y [[buffer(2)]],constant MV& p [[buffer(3)]],uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group*4+sg;if(row>=p.rows)return;\n"
" float sum=0;uint sub=lane%8;\n"
" for(uint b=lane/8;b<p.cols/256;b+=4) {\n"
"  const device uchar* v=data+ulong(row)*p.row_bytes+b*210;\n"
"  float d=float(*((const device half*)(v+208)));\n"
"  for(uint h=0;h<2;h++) {\n"
"   uint off=h*64+sub*4; uchar4 a=uchar4(v[off],v[off+1],v[off+2],v[off+3]);\n"
"   uchar4 z=uchar4(v[off+32],v[off+33],v[off+34],v[off+35]);\n"
"   uint ho=128+h*32+sub*4; uchar4 hi=uchar4(v[ho],v[ho+1],v[ho+2],v[ho+3]);\n"
"   uchar4 qs[4]={(a&uchar4(15))|((hi&uchar4(3))<<uchar4(4)),(z&uchar4(15))|(((hi>>uchar4(2))&uchar4(3))<<uchar4(4)),(a>>uchar4(4))|(((hi>>uchar4(4))&uchar4(3))<<uchar4(4)),(z>>uchar4(4))|((hi>>uchar4(6))<<uchar4(4))};\n"
"   for(uint part=0;part<4;part++) {\n"
"    float scale=d*float(as_type<char>(v[192+h*8+part*2+sub/4]));\n"
"    float4 a=*((const device float4*)(x+b*256+h*128+part*32+sub*4));\n"
"    sum+=scale*dot(float4(qs[part])-32.0f,a);\n"
"   }\n"
"  }\n"
" }\n"
" sum=simd_sum(sum);if(lane==0)y[row]=sum;\n"
"}\n"
"kernel void dec_q8(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],device float* y [[buffer(2)]],constant MV& p [[buffer(3)]],uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group*4+sg;if(row>=p.rows)return;\n"
" float sum=0;uint sub=lane%8;\n"
" for(uint b=lane/8;b<p.cols/32;b+=4) {\n"
"  const device uchar* v=data+ulong(row)*p.row_bytes+b*34;\n"
"  float scale=float(*((const device half*)v));uint off=2+sub*4;\n"
"  char4 q=char4(as_type<char>(v[off]),as_type<char>(v[off+1]),as_type<char>(v[off+2]),as_type<char>(v[off+3]));\n"
"  float4 a=*((const device float4*)(x+b*32+sub*4));\n"
"  sum+=scale*dot(float4(q),a);\n"
" }\n"
" sum=simd_sum(sum);if(lane==0)y[row]=sum;\n"
"}\n"
"struct BAP { uint heads; uint kvheads; uint stride; uint start_pos; uint window; float scale; };\n"
"kernel void dec_batch_attention(const device float* q [[buffer(0)]],const device float* k [[buffer(1)]],const device float* v [[buffer(2)]],device float* out [[buffer(3)]],constant BAP& p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint token=group.x,head=group.y;\n"
" uint kv=head/(p.heads/p.kvheads),pos=p.start_pos+token;\n"
" uint start=(p.window>0 && pos>p.window)?pos-p.window:0;\n"
" ulong qo=ulong(token)*p.heads*128+ulong(head)*128;\n"
" float4 query=float4(q[qo+lane],q[qo+lane+32],q[qo+lane+64],q[qo+lane+96]);\n"
" float m=-INFINITY;float s=0;float4 acc=0;\n"
" for(uint t=start;t<=pos;t++) {\n"
"  ulong i=ulong(t)*p.stride+kv*128+lane;\n"
"  float4 key=float4(k[i],k[i+32],k[i+64],k[i+96]);\n"
"  float score=simd_sum(dot(query,key))*p.scale;\n"
"  float nm=max(m,score);\n"
"  float corr=exp(m-nm);\n"
"  float w=exp(score-nm);\n"
"  float4 value=float4(v[i],v[i+32],v[i+64],v[i+96]);\n"
"  acc=acc*corr+w*value;s=s*corr+w;m=nm;\n"
" }\n"
" ulong o=qo+lane;\n"
" out[o]=acc.x/s;out[o+32]=acc.y/s;out[o+64]=acc.z/s;out[o+96]=acc.w/s;\n"
"}\n"
;
typedef struct { uint32_t n; float eps; } GLLMNormParams;
typedef struct { uint32_t heads,hd,pairs,pos,stride,interleaved;float temperature; } GLLMRopeParams;
typedef struct { uint32_t heads,kvheads,hd,stride,pos,start,chunks;float scale; } GLLMAttentionParams;
typedef struct { uint32_t heads,kvheads,stride,start_pos,window;float scale; } GLLMBatchAttnParams;
typedef struct {
 GLLMMetalWeight* w[7]; uint32_t quant[7];
 id<MTLBuffer> norm,ffnNorm,qNorm,kNorm,k,v;
 int window; bool sharedKV;
} GLLMDecodeLayer;
typedef struct {
 int dim,hidden,heads,kvheads,layers,maxlen;
 float eps,scale;
 GLLMMetalWeight* outputWeight; uint32_t outputQuant; id<MTLBuffer> logits,recent,argmax;
 GLLMDecodeLayer* layer;
 id<MTLBuffer> x,xn,q,qr,k,kr,v,attn,proj,gate,up,hid,sn,cs,partial,outputNorm;
} GLLMDecoder;
static id<MTLComputePipelineState> gllm_dec_pipes[10]={nil,nil,nil,nil,nil,nil,nil,nil,nil,nil};
static bool gllm_decode_init(void) {
 if(!gllm_metal_init())return false;
 @synchronized(gllm_queue) {
  // SiLU is shared with the Q4 library, even for a decoder with only Q6 weights.
  if(!gllm_metal_init_q4k())return false;
  if(gllm_dec_pipes[0]!=nil)return true;
  NSError* error=nil;
  id<MTLLibrary> lib=[gllm_device newLibraryWithSource:[NSString stringWithUTF8String:gllm_decode_source] options:nil error:&error];
  if(lib==nil){gllm_set_error(@"decode shader compilation failed",error);return false;}
  NSString* names[10]={@"dec_norm",@"dec_rope",@"dec_copy_kv",@"dec_add",@"dec_attention_part",@"dec_attention_merge",@"dec_q4",@"dec_q6",@"dec_q8",@"dec_batch_attention"};
  id<MTLComputePipelineState> pipes[10]={nil,nil,nil,nil,nil,nil,nil,nil,nil,nil};
  bool ok=true;
  for(int i=0;i<10;i++) {
   id<MTLFunction> fn=[lib newFunctionWithName:names[i]];
   pipes[i]=fn!=nil?[gllm_device newComputePipelineStateWithFunction:fn error:&error]:nil;
   [fn release];
   if(pipes[i]==nil || [pipes[i] threadExecutionWidth]!=32 || [pipes[i] maxTotalThreadsPerThreadgroup]<256)ok=false;
  }
  [lib release];
  if(!ok){for(int i=0;i<10;i++)[pipes[i] release];gllm_set_error(@"decode pipelines unavailable",error);return false;}
  for(int i=0;i<10;i++)gllm_dec_pipes[i]=pipes[i];
  return true;
 }
 return false;
}
static id<MTLBuffer> gllm_dec_buffer(NSUInteger n) {
 return [gllm_device newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
}
static void gllm_decode_release(void* ptr) {
 if(ptr==NULL)return;
 GLLMDecoder* d=ptr;
 if(d->layer!=NULL)for(int i=0;i<d->layers;i++) {
  GLLMDecodeLayer* l=&d->layer[i];
  [l->norm release];[l->ffnNorm release];[l->qNorm release];[l->kNorm release];[l->k release];[l->v release];
 }
 free(d->layer);
 [d->x release];[d->xn release];[d->q release];[d->qr release];[d->k release];[d->kr release];[d->v release];
 [d->attn release];[d->proj release];[d->gate release];[d->up release];[d->hid release];
 [d->sn release];[d->cs release];[d->partial release];[d->outputNorm release];[d->logits release];[d->recent release];[d->argmax release];free(d);
}
static void* gllm_decode_new(int dim,int hidden,int heads,int kvheads,int layers,int maxlen,float eps,float scale,const float* norm) {
 @autoreleasepool {
 if(!gllm_decode_init())return NULL;
 GLLMDecoder* d=calloc(1,sizeof(GLLMDecoder));if(d==NULL)return NULL;
 d->dim=dim;d->hidden=hidden;d->heads=heads;d->kvheads=kvheads;d->layers=layers;d->maxlen=maxlen;d->eps=eps;d->scale=scale;
 d->layer=calloc(layers,sizeof(GLLMDecodeLayer));
 d->x=gllm_dec_buffer(dim);d->xn=gllm_dec_buffer(dim);d->q=gllm_dec_buffer(heads*128);d->qr=gllm_dec_buffer(heads*128);
 d->k=gllm_dec_buffer(kvheads*128);d->kr=gllm_dec_buffer(kvheads*128);d->v=gllm_dec_buffer(kvheads*128);
 d->attn=gllm_dec_buffer(heads*128);d->proj=gllm_dec_buffer(dim);
 d->gate=gllm_dec_buffer(hidden);d->up=gllm_dec_buffer(hidden);d->hid=gllm_dec_buffer(hidden);
 d->sn=gllm_dec_buffer(64);d->cs=gllm_dec_buffer(64);d->partial=gllm_dec_buffer((NSUInteger)heads*((maxlen+31)/32)*130);
 d->outputNorm=gllm_dec_buffer(dim);
 if(d->layer==NULL || d->x==nil || d->xn==nil || d->q==nil || d->qr==nil || d->k==nil || d->kr==nil || d->v==nil || d->attn==nil || d->proj==nil || d->gate==nil || d->up==nil || d->hid==nil || d->sn==nil || d->cs==nil || d->partial==nil || d->outputNorm==nil) {gllm_decode_release(d);return NULL;}
 memcpy([d->outputNorm contents],norm,dim*sizeof(float));return d;
 }
}
static bool gllm_decode_layer(void* ptr,int index,void* q,void* k,void* v,void* o,void* gate,void* up,void* down,
 const uint32_t* quant,const float* norm,const float* ffn,const float* qnorm,const float* knorm,int window) {
 @autoreleasepool {
 GLLMDecoder* d=ptr;GLLMDecodeLayer* l=&d->layer[index];
 if(l->norm!=nil)return false;
 l->w[0]=q;l->w[1]=k;l->w[2]=v;l->w[3]=o;l->w[4]=gate;l->w[5]=up;l->w[6]=down;
 for(int i=0;i<7;i++)l->quant[i]=quant[i];
 l->window=window;l->norm=gllm_dec_buffer(d->dim);l->ffnNorm=gllm_dec_buffer(d->dim);
 l->k=gllm_dec_buffer((NSUInteger)d->maxlen*d->kvheads*128);l->v=gllm_dec_buffer((NSUInteger)d->maxlen*d->kvheads*128);
 if(l->norm==nil || l->ffnNorm==nil || l->k==nil || l->v==nil)return false;
 memcpy([l->norm contents],norm,d->dim*sizeof(float));memcpy([l->ffnNorm contents],ffn,d->dim*sizeof(float));
 if(qnorm!=NULL){l->qNorm=gllm_dec_buffer(128);if(l->qNorm==nil)return false;memcpy([l->qNorm contents],qnorm,128*sizeof(float));}
 if(knorm!=NULL){l->kNorm=gllm_dec_buffer(128);if(l->kNorm==nil)return false;memcpy([l->kNorm contents],knorm,128*sizeof(float));}
 return true;
 }
}
static bool gllm_decode_output(void* ptr,void* weight,uint32_t quant) {
 @autoreleasepool {
 GLLMDecoder* d=ptr; GLLMMetalWeight* w=weight;
 if(!gllm_metal_init_q6k())return false;
 if(d->recent==nil)d->recent=gllm_dec_buffer(GLLM_REPEAT_WINDOW);
 if(d->argmax==nil)d->argmax=gllm_dec_buffer(2);
 if(d->recent==nil || d->argmax==nil)return false;
 id<MTLBuffer> logits=gllm_dec_buffer(w->rows); if(logits==nil)return false;
 [d->logits release];d->logits=logits;d->outputWeight=w;d->outputQuant=quant;return true;
 }
}
// The Go caller pins these pointer-free allocations until decoder release.
static bool gllm_decode_bind_cache(void* ptr,int index,void* k,void* v,NSUInteger bytes) {
 @autoreleasepool {
 GLLMDecoder* d=ptr;GLLMDecodeLayer* l=&d->layer[index];
 id<MTLBuffer> kb=[gllm_device newBufferWithBytesNoCopy:k length:bytes options:MTLResourceStorageModeShared deallocator:nil];
 id<MTLBuffer> vb=[gllm_device newBufferWithBytesNoCopy:v length:bytes options:MTLResourceStorageModeShared deallocator:nil];
 if(kb==nil || vb==nil){[kb release];[vb release];return false;}
 [l->k release];[l->v release];l->k=kb;l->v=vb;l->sharedKV=true;return true;
 }
}
static void gllm_decode_cache(void* ptr,int layer,int pos,float* k,float* v,bool upload) {
 GLLMDecoder* d=ptr;GLLMDecodeLayer* l=&d->layer[layer];NSUInteger stride=d->kvheads*128*sizeof(float);
 if(l->sharedKV)return;
 if(upload){if(pos>0){memcpy([l->k contents],k,pos*stride);memcpy([l->v contents],v,pos*stride);}}
 else {memcpy(k,(char*)[l->k contents]+pos*stride,stride);memcpy(v,(char*)[l->v contents]+pos*stride,stride);}
}
// gllm_decode_cache_append uploads only the [from,pos) suffix instead of the
// whole [0,pos) prefix, letting chunked prefill amortize to O(n) total bytes
// copied per layer instead of O(n^2) when re-syncing every chunk boundary.
static bool gllm_decode_cache_append(void* ptr,int layer,int from,int pos,const float* k,const float* v) {
 GLLMDecoder* d=ptr;GLLMDecodeLayer* l=&d->layer[layer];NSUInteger stride=d->kvheads*128*sizeof(float);
 if(l->sharedKV)return true;
 if(pos>from){
  NSUInteger off=(NSUInteger)from*stride,len=(NSUInteger)(pos-from)*stride;
  memcpy((char*)[l->k contents]+off,(const char*)k+off,len);
  memcpy((char*)[l->v contents]+off,(const char*)v+off,len);
 }
 return true;
}
static bool gllm_decode_shift_cache(void* ptr,int length,int drop) {
 GLLMDecoder* d=ptr;NSUInteger stride=d->kvheads*128*sizeof(float);
 for(int i=0;i<d->layers;i++){if(d->layer[i].sharedKV)return false;}
 for(int i=0;i<d->layers;i++){
  GLLMDecodeLayer* l=&d->layer[i];
  memmove([l->k contents],(char*)[l->k contents]+drop*stride,(length-drop)*stride);
  memmove([l->v contents],(char*)[l->v contents]+drop*stride,(length-drop)*stride);
 }
 return true;
}
static void gllm_decode_norm(id<MTLComputeCommandEncoder> e,id<MTLBuffer> x,id<MTLBuffer> w,id<MTLBuffer> out,int n,int groups,float eps) {
 [e setComputePipelineState:gllm_dec_pipes[0]];[e setBuffer:x offset:0 atIndex:0];[e setBuffer:w offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];
 GLLMNormParams p={(uint32_t)n,eps};[e setBytes:&p length:sizeof(p) atIndex:3];
 [e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void gllm_decode_quantized(id<MTLComputeCommandEncoder> e,GLLMMetalWeight* w,uint32_t quant,id<MTLBuffer> x,id<MTLBuffer> out) {
 const char* setting=getenv("GOPHERLLM_METAL_DENSE_VECTOR");
 if(setting==NULL || strcmp(setting,"0")!=0) {
  [e setComputePipelineState:gllm_dec_pipes[quant==4?6:(quant==6?7:8)]];
  [e setBuffer:w->weights offset:w->weight_offset atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];
  uint32_t p[3]={(uint32_t)w->rows,(uint32_t)w->cols,(uint32_t)w->row_bytes};[e setBytes:p length:sizeof(p) atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake((w->rows+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
 } else if(quant==4)gllm_metal_encode_q4k_to(e,w,x,out,1,4);
 else if(quant==6)gllm_metal_encode_q6k_to(e,w,x,out,1,4);
 else gllm_metal_encode_q8_0_to(e,w,x,out,1);
}
static void gllm_decode_matvec(id<MTLComputeCommandEncoder> e,GLLMDecodeLayer* l,int i,id<MTLBuffer> x,id<MTLBuffer> out) {
 gllm_decode_quantized(e,l->w[i],l->quant[i],x,out);
}
static void gllm_decode_add(id<MTLComputeCommandEncoder> e,GLLMDecoder* d) {
 [e setComputePipelineState:gllm_dec_pipes[3]];[e setBuffer:d->x offset:0 atIndex:0];[e setBuffer:d->proj offset:0 atIndex:1];
 uint32_t n=d->dim;[e setBytes:&n length:sizeof(n) atIndex:2];[e dispatchThreadgroups:MTLSizeMake((n+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
// Concurrent dispatch is limited to independent Q/K/V, Q/K norms and gate/up.
// Every producer/consumer boundary has an explicit buffer barrier.
static void gllm_decode_barrier(id<MTLComputeCommandEncoder> e,bool concurrent) {
 if(concurrent)[e memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
static bool gllm_decode_step(void* ptr,const float* input,const float* sn,const float* cs,int pairs,int pos,bool interleaved,float temperature,float* residual,float* output,float* logits,const uint32_t* recent,uint32_t recent_count,float penalty,uint32_t* next) {
 @autoreleasepool {
 GLLMDecoder* d=ptr;
 if((logits!=NULL || next!=NULL) && d->outputWeight==NULL)return false;
 if(next!=NULL && recent_count>0)memcpy([d->recent contents],recent,recent_count*sizeof(uint32_t));
 memcpy([d->x contents],input,d->dim*sizeof(float));memcpy([d->sn contents],sn,pairs*sizeof(float));memcpy([d->cs contents],cs,pairs*sizeof(float));
 const char* setting=getenv("GOPHERLLM_METAL_DENSE_CONCURRENT");
 bool concurrent=setting==NULL || strcmp(setting,"0")!=0;
 id<MTLCommandBuffer> cb=gllm_metal_new_command_buffer();
 id<MTLComputeCommandEncoder> e=[cb computeCommandEncoderWithDispatchType:concurrent?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial];
 for(int layer=0;layer<d->layers;layer++) {
  GLLMDecodeLayer* l=&d->layer[layer];
  gllm_decode_norm(e,d->x,l->norm,d->xn,d->dim,1,d->eps);
  gllm_decode_barrier(e,concurrent);
  gllm_decode_matvec(e,l,0,d->xn,d->q);gllm_decode_matvec(e,l,1,d->xn,d->k);gllm_decode_matvec(e,l,2,d->xn,d->v);
  gllm_decode_barrier(e,concurrent);
  id<MTLBuffer> q=d->q,k=d->k;
  if(l->qNorm!=nil){gllm_decode_norm(e,q,l->qNorm,d->qr,128,d->heads,d->eps);q=d->qr;}
  if(l->kNorm!=nil){gllm_decode_norm(e,k,l->kNorm,d->kr,128,d->kvheads,d->eps);k=d->kr;}
  gllm_decode_barrier(e,concurrent);
  for(int isK=0;isK<2;isK++) {
   uint32_t heads=isK?d->kvheads:d->heads;
   GLLMRopeParams p={heads,128,(uint32_t)pairs,isK?(uint32_t)pos:0,isK?heads*128:0,(uint32_t)interleaved,isK?1.0f:temperature};
   [e setComputePipelineState:gllm_dec_pipes[1]];[e setBuffer:isK?k:q offset:0 atIndex:0];
   [e setBuffer:isK?l->k:(q==d->q?d->qr:d->q) offset:0 atIndex:1];
   [e setBuffer:d->sn offset:0 atIndex:2];[e setBuffer:d->cs offset:0 atIndex:3];[e setBytes:&p length:sizeof(p) atIndex:4];
   [e dispatchThreadgroups:MTLSizeMake((heads*128+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  }
  q=q==d->q?d->qr:d->q;
  uint32_t stride=d->kvheads*128,position=pos;
  [e setComputePipelineState:gllm_dec_pipes[2]];[e setBuffer:d->v offset:0 atIndex:0];[e setBuffer:l->v offset:0 atIndex:1];
  [e setBytes:&stride length:sizeof(stride) atIndex:2];[e setBytes:&position length:sizeof(position) atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake((stride+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  gllm_decode_barrier(e,concurrent);
  uint32_t start=l->window>0 && pos>l->window?pos-l->window:0;
  uint32_t chunks=(pos-start+32)/32;
  GLLMAttentionParams ap={(uint32_t)d->heads,(uint32_t)d->kvheads,128,stride,(uint32_t)pos,start,chunks,d->scale};
  [e setComputePipelineState:gllm_dec_pipes[4]];[e setBuffer:q offset:0 atIndex:0];[e setBuffer:l->k offset:0 atIndex:1];[e setBuffer:l->v offset:0 atIndex:2];[e setBuffer:d->partial offset:0 atIndex:3];[e setBytes:&ap length:sizeof(ap) atIndex:4];
  [e dispatchThreadgroups:MTLSizeMake((d->heads*chunks+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  gllm_decode_barrier(e,concurrent);
  [e setComputePipelineState:gllm_dec_pipes[5]];[e setBuffer:d->partial offset:0 atIndex:0];[e setBuffer:d->attn offset:0 atIndex:1];[e setBytes:&ap length:sizeof(ap) atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake(d->heads,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
  gllm_decode_barrier(e,concurrent);
  gllm_decode_matvec(e,l,3,d->attn,d->proj);
  gllm_decode_barrier(e,concurrent);
  gllm_decode_add(e,d);
  gllm_decode_barrier(e,concurrent);
  gllm_decode_norm(e,d->x,l->ffnNorm,d->xn,d->dim,1,d->eps);
  gllm_decode_barrier(e,concurrent);
  gllm_decode_matvec(e,l,4,d->xn,d->gate);gllm_decode_matvec(e,l,5,d->xn,d->up);
  gllm_decode_barrier(e,concurrent);
  gllm_metal_encode_silu(e,d->gate,d->up,d->hid,d->hidden);
  gllm_decode_barrier(e,concurrent);
  gllm_decode_matvec(e,l,6,d->hid,d->proj);
  gllm_decode_barrier(e,concurrent);
  gllm_decode_add(e,d);
  gllm_decode_barrier(e,concurrent);
 }
 gllm_decode_norm(e,d->x,d->outputNorm,d->xn,d->dim,1,d->eps);
 if(logits!=NULL || next!=NULL) {
  gllm_decode_barrier(e,concurrent);
  gllm_decode_quantized(e,d->outputWeight,d->outputQuant,d->xn,d->logits);
 }
 if(next!=NULL) {
  gllm_decode_barrier(e,concurrent);
  GLLMMetalWeight output=*d->outputWeight;output.recent=d->recent;
  gllm_metal_encode_argmax_to(e,&output,d->logits,0,d->argmax,0,recent_count,penalty);
 }
 [e endEncoding];[cb commit];[cb waitUntilCompleted];
 if([cb status]!=MTLCommandBufferStatusCompleted){gllm_set_error(@"dense decode command failed",[cb error]);return false;}
 if(next!=NULL) {
  GLLMArgmaxResult result;memcpy(&result,[d->argmax contents],sizeof(result));
  if(result.index>=(uint32_t)d->outputWeight->rows)return false;*next=result.index;
 }
 if(logits!=NULL)memcpy(logits,[d->logits contents],d->outputWeight->rows*sizeof(float));
 memcpy(residual,[d->x contents],d->dim*sizeof(float));memcpy(output,[d->xn contents],d->dim*sizeof(float));return true;
 }
}

// Bounded attention projections reuse the prepared decoder weights and the
// serialized prefill workspace. CPU RoPE/attention keep the public KV authoritative.
static bool gllm_decode_project(void* ptr,int layer,int matrix,const float* x,float* out,int batch) {
 @autoreleasepool {
 GLLMDecoder* d=ptr;
 if(d==NULL || layer<0 || layer>=d->layers || matrix<0 || matrix>3 || batch<16 || batch>256)return false;
 GLLMDecodeLayer* l=&d->layer[layer];GLLMMetalWeight* w=l->w[matrix];
 @synchronized(gllm_queue) {
  NSUInteger inBytes=(NSUInteger)batch*w->cols*sizeof(float),outBytes=(NSUInteger)batch*w->rows*sizeof(float);
  if(!gllm_metal_ensure_batch_buffer(&gllm_batch_workspace.x,inBytes) || !gllm_metal_ensure_batch_buffer(&gllm_batch_workspace.out,outBytes))return false;
  memcpy([gllm_batch_workspace.x contents],x,inBytes);
  id<MTLCommandBuffer> cb=gllm_metal_new_command_buffer();
  id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
  if(l->quant[matrix]==4)gllm_metal_encode_q4k_to(e,w,gllm_batch_workspace.x,gllm_batch_workspace.out,batch,4);
  else if(l->quant[matrix]==6)gllm_metal_encode_q6k_to(e,w,gllm_batch_workspace.x,gllm_batch_workspace.out,batch,4);
  else gllm_metal_encode_q8_0_to(e,w,gllm_batch_workspace.x,gllm_batch_workspace.out,batch);
  [e endEncoding];[cb commit];[cb waitUntilCompleted];
  bool ok=[cb status]==MTLCommandBufferStatusCompleted;
  if(ok)memcpy(out,[gllm_batch_workspace.out contents],outBytes);
  return ok;
 }
 }
}
