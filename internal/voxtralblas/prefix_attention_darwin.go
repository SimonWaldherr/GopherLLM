//go:build darwin && cgo

package voxtralblas

/*
#cgo CFLAGS: -O2
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <dispatch/dispatch.h>

static void prefix_attention(int n, int total, int dim, int width, int kv_width, int position, int prefix,
 float scale, const float *q, const float *k, const float *v, float *scores, float *out) {
 cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, total, dim,
  scale, q, width, k, kv_width, 0, scores, total);
 for (int t = 0; t < n; t++) {
  int end = position + t < prefix ? position + t + 1 : total;
  float *row = scores + (size_t)t * total;
  float largest = -INFINITY, sum = 0;
  for (int j = 0; j < end; j++) largest = fmaxf(largest, row[j]);
  for (int j = 0; j < end; j++) row[j] -= largest;
  vvexpf(row, row, &end);
  for (int j = 0; j < end; j++) sum += row[j];
  for (int j = 0; j < end; j++) row[j] /= sum;
  for (int j = end; j < total; j++) row[j] = 0;
 }
 cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, dim, total,
  1, scores, total, v, kv_width, 0, out, width);
}
typedef struct {
 int n, total, dim, heads, kv_heads, position, prefix, workers;
 float scale;
 const float *q, *k, *v;
 float *scores, *out;
} prefix_attention_job;

static void prefix_attention_worker(void *context, size_t worker) {
 prefix_attention_job *job = (prefix_attention_job *)context;
 int width = job->heads * job->dim;
 int kv_width = job->kv_heads * job->dim, group = job->heads / job->kv_heads;
 float *scores = job->scores + worker * (size_t)job->n * job->total;
 for (int head = (int)worker; head < job->heads; head += job->workers) {
  size_t base = (size_t)head * job->dim;
  size_t kv = (size_t)(head / group) * job->dim;
  prefix_attention(job->n, job->total, job->dim, width, kv_width, job->position, job->prefix,
   job->scale, job->q + base, job->k + kv, job->v + kv, scores, job->out + base);
 }
}

static void prefix_attention_heads(int n, int total, int dim, int heads, int kv_heads, int position, int prefix,
 float scale, const float *q, const float *k, const float *v, float *scores, float *out, int workers) {
 prefix_attention_job job = {n, total, dim, heads, kv_heads, position, prefix, workers, scale, q, k, v, scores, out};
 if (workers > 1) dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job, prefix_attention_worker);
 else prefix_attention_worker(&job, 0);
}
*/
import "C"
import "unsafe"

// GroupedPrefixAttention computes token-major attention with query/output
// strides heads*dim and K/V strides kvHeads*dim; heads is divisible by kvHeads.
// Prefix queries are causal, subsequent queries attend to all keys. Caller
// supplies workers*n*total score elements and validated, disjoint views.
// Native workers finish before returning; no Go memory is retained.
func GroupedPrefixAttention(n, total, dim, heads, kvHeads, position, prefix int, scale float32, q, k, v, scores, out []float32, workers int) {
	C.prefix_attention_heads(C.int(n), C.int(total), C.int(dim), C.int(heads), C.int(kvHeads), C.int(position), C.int(prefix), C.float(scale),
		(*C.float)(unsafe.Pointer(&q[0])), (*C.float)(unsafe.Pointer(&k[0])), (*C.float)(unsafe.Pointer(&v[0])),
		(*C.float)(unsafe.Pointer(&scores[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(workers))
}
