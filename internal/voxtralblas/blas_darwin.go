//go:build darwin && cgo

package voxtralblas

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <math.h>
static void voxtral_sgemm(int n, int rows, int cols, const float *w, const float *x, float *out) {
 cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, rows, cols, 1, x, cols, w, cols, 0, out, rows);
}
static void voxtral_attention(int n,int total,int heads,int dim,int past,int window,float scale,const float *q,const float *k,const float *v,float *scores,float *out) {
 int width=heads*dim;
 for(int h=0;h<heads;h++) {
  const float *kh=k+h*total*dim,*vh=v+h*total*dim;
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,total,dim,scale,q+h*dim,width,kh,dim,0,scores,total);
  for(int t=0;t<n;t++) {
   int end=past+t,lo=window>0 && end-window+1>0?end-window+1:0;
   float *row=scores+t*total,largest=-INFINITY,sum=0;
   for(int j=lo;j<=end;j++)largest=fmaxf(largest,row[j]);
   for(int j=lo;j<=end;j++){row[j]=expf(row[j]-largest);sum+=row[j];}
   for(int j=0;j<lo;j++)row[j]=0;
   for(int j=lo;j<=end;j++)row[j]/=sum;
   for(int j=end+1;j<total;j++)row[j]=0;
  }
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,dim,total,1,scores,total,vh,dim,0,out+h*dim,width);
 }
}
*/
import "C"
import "unsafe"

func Mul(n, rows, cols int, w, x, out []float32) {
	C.voxtral_sgemm(C.int(n), C.int(rows), C.int(cols), (*C.float)(unsafe.Pointer(&w[0])), (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
}

func Attention(n, total, heads, dim, past, window int, scale float32, q, k, v, scores, out []float32) {
	C.voxtral_attention(C.int(n), C.int(total), C.int(heads), C.int(dim), C.int(past), C.int(window), C.float(scale), (*C.float)(unsafe.Pointer(&q[0])), (*C.float)(unsafe.Pointer(&k[0])), (*C.float)(unsafe.Pointer(&v[0])), (*C.float)(unsafe.Pointer(&scores[0])), (*C.float)(unsafe.Pointer(&out[0])))
}
