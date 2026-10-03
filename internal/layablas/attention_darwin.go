//go:build darwin && cgo

// Package layablas supplies bounded-workspace bidirectional attention using Accelerate.
package layablas

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <dispatch/dispatch.h>

static void laya_project(int n, int rows, int cols, int stride, const float *w, const float *x, float *out) {
 cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,n,rows,cols,1,x,stride,w,cols,0,out,rows);
}

typedef struct {
 int heads, dim, window, start, count, lo, hi, workers;
 float scale;
 const float *qkv;
 float *scores, *out;
} laya_attention_job;

static void laya_attention_worker(void *context, size_t worker) {
 laya_attention_job *job = (laya_attention_job *)context;
 int heads=job->heads, dim=job->dim, window=job->window;
 int start=job->start, count=job->count, lo=job->lo, hi=job->hi;
 float scale=job->scale;
 const float *qkv=job->qkv;
 float *scores=job->scores + worker*count*(hi-lo), *out=job->out;
 int width = heads*dim, stride = 3*width, keys = hi-lo;
 for (int h=(int)worker; h<heads; h+=job->workers) {
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

static void laya_attention_block(int heads, int dim, int window,
 int start, int count, int lo, int hi, float scale,
 const float *qkv, float *scores, float *out, int workers) {
 laya_attention_job job = {heads, dim, window, start, count, lo, hi, workers, scale, qkv, scores, out};
 // dispatch_apply_f waits for every worker before the Go buffers are reused.
 if (workers>1) {
  dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), &job, laya_attention_worker);
 } else {
  laya_attention_worker(&job,0);
 }
}
*/
import "C"

import (
	"context"
	"fmt"
	"runtime"
	"unsafe"
)

const attentionBlock = 128

// Attention reads interleaved [sequence][Q,K,V][head][dim] data and overwrites
// out[sequence][head][dim]. A negative window means full bidirectional attention;
// otherwise each position sees its inclusive +/- window. Workers have independent
// score blocks, each at most 128*sequence floats (128*(128+2*window) locally).
// Scratch is reused across layers and bounded by the configured worker count.
// Context cancellation is checked between query blocks. No Go pointers are retained.
func Attention(ctx context.Context, n, heads, dim, window int, scale float32, qkv, out []float32, scratch *[]float32) error {
	return AttentionWithThreads(ctx, n, heads, dim, window, scale, qkv, out, scratch, runtime.GOMAXPROCS(0))
}

// AttentionWithThreads caps parallel head processing at the caller's thread limit.
// Small workloads stay serial to avoid dispatch overhead.
func AttentionWithThreads(ctx context.Context, n, heads, dim, window int, scale float32, qkv, out []float32, scratch *[]float32, threads int) error {
	if n < 1 || n > 8192 || heads < 1 || dim < 1 || len(out)/n/heads != dim || len(out)%n != 0 || (len(out)/n)%heads != 0 || len(qkv) != 3*len(out) {
		return fmt.Errorf("laya attention: invalid dimensions")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	const block = attentionBlock
	window = min(window, n)
	keysPerQuery := n
	if window >= 0 {
		keysPerQuery = min(n, 2*window+1)
	}
	workers := min(heads, max(1, threads))
	if n < 64 || n*heads*keysPerQuery < 32768 {
		workers = 1
	}
	keys := n
	if window >= 0 {
		keys = min(n, block+2*window)
	}
	size := workers * min(n, block) * keys
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
		C.laya_attention_block(C.int(heads), C.int(dim), C.int(window), C.int(start), C.int(count), C.int(lo), C.int(hi), C.float(scale), (*C.float)(unsafe.Pointer(&qkv[0])), (*C.float)(unsafe.Pointer(&(*scratch)[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(workers))
	}
	return ctx.Err()
}

// Project multiplies strided input rows by transposed weights without packing.
// The caller validates dimensions and the contiguous backing buffers.
func Project(n, rows, cols, stride int, w, x, out []float32) {
	C.laya_project(C.int(n), C.int(rows), C.int(cols), C.int(stride), (*C.float)(unsafe.Pointer(&w[0])), (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&out[0])))
}
