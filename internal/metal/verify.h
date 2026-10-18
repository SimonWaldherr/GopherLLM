// Register-resident short matvecs share each unpacked four-column weight
// vector across 2/4/8 activation rows. No shared-memory tensor padding or
// per-element metadata decoding is needed.
static const char* gllm_verify_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"constant uint span [[function_constant(0)]];\n"
"struct VP { uint rows,cols,row_bytes,n_blocks,rows_per_group,batch; };\n"
"inline float4 verify_dots(float4 w,const device float* x,uint col,uint token,constant VP& p) {\n"
" float4 s=0;\n"
" s.x=dot(w,*((const device float4*)(x+ulong(token)*p.cols+col)));\n"
" if(span>1 && token+1<p.batch)s.y=dot(w,*((const device float4*)(x+ulong(token+1)*p.cols+col)));\n"
" if(span>2 && token+2<p.batch)s.z=dot(w,*((const device float4*)(x+ulong(token+2)*p.cols+col)));\n"
" if(span>3 && token+3<p.batch)s.w=dot(w,*((const device float4*)(x+ulong(token+3)*p.cols+col)));\n"
" return s;\n"
"}\n"
"inline float4 verify_reduce(float4 s) {return float4(simd_sum(s.x),simd_sum(s.y),simd_sum(s.z),simd_sum(s.w));}\n"
"inline void verify_store(device float* out,float4 s,uint row,uint token,constant VP& p) {\n"
" out[ulong(token)*p.rows+row]=s.x;\n"
" if(span>1 && token+1<p.batch)out[ulong(token+1)*p.rows+row]=s.y;\n"
" if(span>2 && token+2<p.batch)out[ulong(token+2)*p.rows+row]=s.z;\n"
" if(span>3 && token+3<p.batch)out[ulong(token+3)*p.rows+row]=s.w;\n"
"}\n"
"inline void verify_accumulate(thread float4& s,thread float4& tail,float4 w,const device float* x,uint col,uint token,constant VP& p) {\n"
" s+=verify_dots(w,x,col,token,p);\n"
" if(span==8 && token+4<p.batch)tail+=verify_dots(w,x,col,token+4,p);\n"
"}\n"
"kernel void verify_q4(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],device float* out [[buffer(2)]],constant VP& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group.x*4+sg,token=group.y*span;if(row>=p.rows)return;\n"
" float4 s=0,tail=0;uint sub=lane%8;\n"
" for(uint b=lane/8;b<p.cols/256;b+=4) {\n"
"  const device uchar* v=data+ulong(row)*p.row_bytes+b*144;\n"
"  float d=float(*((const device half*)v)),dm=float(*((const device half*)(v+2)));\n"
"  const device uchar* sc=v+4;\n"
"  for(uint step=0;step<4;step++) {\n"
"   uint j=step*2,s0,m0,s1,m1;\n"
"   if(j<4){s0=sc[j]&63;m0=sc[j+4]&63;s1=sc[j+1]&63;m1=sc[j+5]&63;}\n"
"   else{s0=(sc[j+4]&15)|((sc[j-4]>>6)<<4);m0=(sc[j+4]>>4)|((sc[j]>>6)<<4);s1=(sc[j+5]&15)|((sc[j-3]>>6)<<4);m1=(sc[j+5]>>4)|((sc[j+1]>>6)<<4);}\n"
"   uchar4 packed=*((const device uchar4*)(v+16+step*32+sub*4));\n"
"   float4 a=d*float(s0)*float4(packed&uchar4(15))-dm*float(m0);\n"
"   float4 z=d*float(s1)*float4(packed>>uchar4(4))-dm*float(m1);\n"
"   uint col=b*256+step*64+sub*4;\n"
"   verify_accumulate(s,tail,a,x,col,token,p);verify_accumulate(s,tail,z,x,col+32,token,p);\n"
"  }\n"
" }\n"
" s=verify_reduce(s);tail=verify_reduce(tail);\n"
" if(lane==0){verify_store(out,s,row,token,p);if(span==8 && token+4<p.batch)verify_store(out,tail,row,token+4,p);}\n"
"}\n"
"kernel void verify_q6(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],device float* out [[buffer(2)]],constant VP& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group.x*4+sg,token=group.y*span;if(row>=p.rows)return;\n"
" float4 s=0,tail=0;uint sub=lane%8;\n"
" for(uint b=lane/8;b<p.cols/256;b+=4) {\n"
"  const device uchar* v=data+ulong(row)*p.row_bytes+b*210;float d=float(*((const device half*)(v+208)));\n"
"  for(uint h=0;h<2;h++) {\n"
"   uint off=h*64+sub*4,ho=128+h*32+sub*4;\n"
"   uchar4 a=uchar4(v[off],v[off+1],v[off+2],v[off+3]),z=uchar4(v[off+32],v[off+33],v[off+34],v[off+35]);\n"
"   uchar4 hi=uchar4(v[ho],v[ho+1],v[ho+2],v[ho+3]);\n"
"   uchar4 qs[4]={(a&uchar4(15))|((hi&uchar4(3))<<uchar4(4)),(z&uchar4(15))|(((hi>>uchar4(2))&uchar4(3))<<uchar4(4)),(a>>uchar4(4))|(((hi>>uchar4(4))&uchar4(3))<<uchar4(4)),(z>>uchar4(4))|((hi>>uchar4(6))<<uchar4(4))};\n"
"   for(uint part=0;part<4;part++) {\n"
"    float scale=d*float(as_type<char>(v[192+h*8+part*2+sub/4]));\n"
"    float4 w=scale*(float4(qs[part])-32.0f);\n"
"    verify_accumulate(s,tail,w,x,b*256+h*128+part*32+sub*4,token,p);\n"
"   }\n"
"  }\n"
" }\n"
" s=verify_reduce(s);tail=verify_reduce(tail);\n"
" if(lane==0){verify_store(out,s,row,token,p);if(span==8 && token+4<p.batch)verify_store(out,tail,row,token+4,p);}\n"
"}\n"
"kernel void verify_q8(const device uchar* data [[buffer(0)]],const device float* x [[buffer(1)]],device float* out [[buffer(2)]],constant VP& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {\n"
" uint row=group.x*4+sg,token=group.y*span;if(row>=p.rows)return;\n"
" float4 s=0,tail=0;uint sub=lane%8;\n"
" for(uint b=lane/8;b<p.cols/32;b+=4) {\n"
"  const device uchar* v=data+ulong(row)*p.row_bytes+b*34;float scale=float(*((const device half*)v));uint off=2+sub*4;\n"
"  char4 q=char4(as_type<char>(v[off]),as_type<char>(v[off+1]),as_type<char>(v[off+2]),as_type<char>(v[off+3]));\n"
"  verify_accumulate(s,tail,scale*float4(q),x,b*32+sub*4,token,p);\n"
" }\n"
" s=verify_reduce(s);tail=verify_reduce(tail);\n"
" if(lane==0){verify_store(out,s,row,token,p);if(span==8 && token+4<p.batch)verify_store(out,tail,row,token+4,p);}\n"
"}\n";

static id<MTLComputePipelineState> gllm_metal_verify_pipeline(uint32_t quant,uint32_t span) {
 static id<MTLComputePipelineState> pipelines[12]={nil};static id<MTLLibrary> library=nil;
 static bool attempted[12]={false};static bool libraryAttempted=false;
 @synchronized(gllm_queue) {
  int index=(quant==4?0:quant==6?4:8)+(span==1?0:span==2?1:span==4?2:3);
  if(!attempted[index]) {
   attempted[index]=true;
   NSError* error=nil;
   if(!libraryAttempted){libraryAttempted=true;library=[gllm_device newLibraryWithSource:[NSString stringWithUTF8String:gllm_verify_source] options:nil error:&error];}
   if(library==nil){gllm_set_error(@"failed to compile short verification matvec",error);return nil;}
   MTLFunctionConstantValues* values=[[MTLFunctionConstantValues alloc] init];[values setConstantValue:&span type:MTLDataTypeUInt atIndex:0];
   NSString* name=quant==4?@"verify_q4":quant==6?@"verify_q6":@"verify_q8";
   id<MTLFunction> fn=[library newFunctionWithName:name constantValues:values error:&error];
   id<MTLComputePipelineState> pipe=fn!=nil?[gllm_device newComputePipelineStateWithFunction:fn error:&error]:nil;
   if(pipe!=nil && [pipe threadExecutionWidth]==32 && [pipe maxTotalThreadsPerThreadgroup]>=128)pipelines[index]=pipe;
   else {[pipe release];gllm_set_error(@"short verification matvec unavailable",error);}
   [values release];[fn release];
  }
  return pipelines[index];
 }
 return nil;
}
static bool gllm_metal_encode_verify_rows(id<MTLComputeCommandEncoder> e,GLLMMetalWeight* w,
 id<MTLBuffer> x,id<MTLBuffer> out,int batch,uint32_t quant) {
 if(batch<1 || batch>32 || w->cols%256!=0)return false;
 uint32_t span=batch==1?1:batch<=2?2:batch<=4?4:8;
 id<MTLComputePipelineState> pipe=gllm_metal_verify_pipeline(quant,span);if(pipe==nil)return false;
 [e setComputePipelineState:pipe];[e setBuffer:w->weights offset:w->weight_offset atIndex:0];[e setBuffer:x offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];
 GLLMMetalParams p={(uint32_t)w->rows,(uint32_t)w->cols,(uint32_t)w->row_bytes,0,0,(uint32_t)batch};[e setBytes:&p length:sizeof(p) atIndex:3];
 [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)w->rows+3)/4,(batch+span-1)/span,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];return true;
}

static bool gllm_metal_encode_verify(id<MTLComputeCommandEncoder> e,GLLMMetalWeight* w,
 id<MTLBuffer> x,id<MTLBuffer> out,int batch,uint32_t quant) {
 if(batch<2 || batch>8)return false;
 const char* setting=getenv("GOPHERLLM_METAL_VERIFY_VECTOR");if(setting!=NULL && strcmp(setting,"0")==0)return false;
 return gllm_metal_encode_verify_rows(e,w,x,out,batch,quant);
}

static bool gllm_metal_encode_qwen_ffn_vector(id<MTLComputeCommandEncoder> e,GLLMMetalWeight* w,
 id<MTLBuffer> x,id<MTLBuffer> out,int batch,uint32_t quant) {
 const char* setting=getenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR");
 if(setting!=NULL && strcmp(setting,"0")==0)return false;
 // A partial pipeline failure must not give serial and batch different
 // arithmetic. Admit every span together, independently for each quant type.
 static int ready[3]={0};int kind=quant==4?0:quant==6?1:2;
 @synchronized(gllm_queue) {
  if(ready[kind]==0) {
   ready[kind]=1;
   for(uint32_t span=1;span<=8;span*=2)
    if(gllm_metal_verify_pipeline(quant,span)==nil)ready[kind]=-1;
  }
  if(ready[kind]<0)return false;
 }
 return gllm_metal_encode_verify_rows(e,w,x,out,batch,quant);
}
