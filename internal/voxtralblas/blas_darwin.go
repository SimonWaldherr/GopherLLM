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
static void voxtral_attention(int n,int total,int heads,int dim,int past,int window,int stride,float scale,const float *q,const float *k,const float *v,float *scores,float *out) {
 int width=heads*dim;
 for(int h=0;h<heads;h++) {
  const float *kh=k+h*stride*dim,*vh=v+h*stride*dim;
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
static void voxtral_gemm_nn(int m, int n, int k, const float *a, const float *b, float *c) {
 cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 0, c, n);
}
*/
import "C"
import "unsafe"

func Mul(n, rows, cols int, w, x, out []float32) {
	C.voxtral_sgemm(C.int(n), C.int(rows), C.int(cols), (*C.float)(unsafe.Pointer(&w[0])), (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
}

// GemmNN computes c[m×n] = a[m×k] · b[k×n], all row-major. It is the plain
// matrix product the YOLO convolution path needs after im2col (weights times
// patch columns), as opposed to Mul's weight-transposed layout.
func GemmNN(m, n, k int, a, b, c []float32) {
	C.voxtral_gemm_nn(C.int(m), C.int(n), C.int(k), (*C.float)(unsafe.Pointer(&a[0])), (*C.float)(unsafe.Pointer(&b[0])), (*C.float)(unsafe.Pointer(&c[0])))
}

// stride is the per-head element stride within k/v (>= total); it lets a
// caller keep k/v allocated to a fixed per-head capacity across calls
// (growing it only when a chunk needs more) and just write new rows into it,
// instead of repacking every previously-cached position on every call.
func Attention(n, total, heads, dim, past, window, stride int, scale float32, q, k, v, scores, out []float32) {
	C.voxtral_attention(C.int(n), C.int(total), C.int(heads), C.int(dim), C.int(past), C.int(window), C.int(stride), C.float(scale), (*C.float)(unsafe.Pointer(&q[0])), (*C.float)(unsafe.Pointer(&k[0])), (*C.float)(unsafe.Pointer(&v[0])), (*C.float)(unsafe.Pointer(&scores[0])), (*C.float)(unsafe.Pointer(&out[0])))
}
