// callbacks.h declares the streaming callback shape shared by every C source
// fragment cgo generates for this package. It carries ONLY declarations
// (typedefs): cgo forbids definitions in the preamble of any file that also
// contains //export directives, so the matching trampoline *bodies* live in
// callbacks.go's preamble instead, which has no //export lines of its own.
#ifndef GOPHERLLM_CALLBACKS_H
#define GOPHERLLM_CALLBACKS_H

// Called once per streamed text delta, in generation order, on a Go worker
// goroutine. userdata is whatever the caller passed to
// gopherllm_generate_stream, unmodified.
typedef void (*gopherllm_delta_cb)(const char *delta, void *userdata);

// Called exactly once, after the last delta, on success. result_json is
// {"text":..., "finish_reason":..., "generated_tokens":...}.
typedef void (*gopherllm_complete_cb)(const char *result_json, void *userdata);

// Called exactly once instead of gopherllm_complete_cb on failure.
typedef void (*gopherllm_error_cb)(const char *message, void *userdata);

#endif
