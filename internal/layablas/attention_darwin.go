//go:build darwin && cgo

// Package layablas supplies bounded-workspace bidirectional attention using Accelerate.
package layablas

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <math.h>

static void laya_project(int n, int rows, int cols, int stride, const float *w, const float *x, float *out) {
 cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,rows,cols,1,x,stride,w,cols,0,out,rows);
}

static void laya_attention_block(int n, int heads, int dim, int window,
 int start, int count, int lo, int hi, float scale,
 const float *qkv, float *scores, float *out) {
 int width = heads*dim, stride = 3*width, keys = hi-lo;
 for (int h=0; h<heads; h++) {
  const float *q = qkv + start*stride + h*dim;
  const float *k = qkv + lo*stride + width + h*dim;
  const float *v = qkv + lo*stride + 2*width + h*dim;
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,count,keys,dim,
   scale,q,stride,k,stride,0,scores,keys);
  for (int t=0; t<count; t++) {
   int p=start+t, first=lo, last=hi;
   if (window>=0) {
    if (p-window>first) first=p-window;
    if (p+window+1<last) last=p+window+1;
   }
   float *row=scores+t*keys, largest=-INFINITY;
   double sum=0;
   for (int j=first-lo; j<last-lo; j++) largest=fmaxf(largest,row[j]);
   for (int j=first-lo; j<last-lo; j++) {
    row[j]=(float)exp((double)(row[j]-largest)); sum+=(double)row[j];
   }
   for (int j=0; j<first-lo; j++) row[j]=0;
   for (int j=first-lo; j<last-lo; j++) row[j]/=(float)sum;
   for (int j=last-lo; j<keys; j++) row[j]=0;
  }
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,count,dim,keys,
   1,scores,keys,v,stride,0,out+start*width+h*dim,width);
 }
}
*/
import "C"

import (
	"context"
	"fmt"
	"unsafe"
)

const attentionBlock = 128

// Attention reads interleaved [sequence][Q,K,V][head][dim] data and overwrites
// out[sequence][head][dim]. A negative window means full bidirectional attention;
// otherwise each position sees its inclusive +/- window. Scratch is at most
// 128*sequence floats, shared across heads and reused by the caller across layers.
// Context cancellation is checked between query blocks. No Go pointers are retained.
func Attention(ctx context.Context, n, heads, dim, window int, scale float32, qkv, out []float32, scratch *[]float32) error {
	if n < 1 || n > 8192 || heads < 1 || dim < 1 || len(out)/n/heads != dim || len(out)%n != 0 || (len(out)/n)%heads != 0 || len(qkv) != 3*len(out) {
		return fmt.Errorf("laya attention: invalid dimensions")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	window = min(window, n)
	const block = attentionBlock
	size := min(n, block) * n
	if cap(*scratch) < size {
		*scratch = make([]float32, size)
	} else {
		*scratch = (*scratch)[:size]
	}
	for start := 0; start < n; start += block {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := min(block, n-start)
		lo, hi := 0, n
		if window >= 0 {
			lo = max(0, start-window)
			hi = min(n, start+count+window)
		}
		C.laya_attention_block(C.int(n), C.int(heads), C.int(dim), C.int(window), C.int(start), C.int(count), C.int(lo), C.int(hi), C.float(scale), (*C.float)(unsafe.Pointer(&qkv[0])), (*C.float)(unsafe.Pointer(&(*scratch)[0])), (*C.float)(unsafe.Pointer(&out[0])))
	}
	return ctx.Err()
}

// Project multiplies strided input rows by transposed weights without packing.
// The caller validates dimensions and the contiguous backing buffers.
func Project(n, rows, cols, stride int, w, x, out []float32) {
	C.laya_project(C.int(n), C.int(rows), C.int(cols), C.int(stride), (*C.float)(unsafe.Pointer(&w[0])), (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
}
