//go:build capi

package main

/*
#include <stdlib.h>
#include "callbacks.h"

// Go cannot call a C function pointer directly; each of these thin trampolines
// gives cgo a concrete call site to generate. They must live in a file with no
// //export directives of its own -- cgo forbids definitions (as opposed to
// declarations) in the preamble of an exported file, since that preamble is
// copied into more than one generated C translation unit.
static inline void gopherllm_invoke_delta(gopherllm_delta_cb cb, const char *s, void *ud) {
	cb(s, ud);
}
static inline void gopherllm_invoke_complete(gopherllm_complete_cb cb, const char *s, void *ud) {
	cb(s, ud);
}
static inline void gopherllm_invoke_error(gopherllm_error_cb cb, const char *s, void *ud) {
	cb(s, ud);
}
*/
import "C"
import (
	"unsafe"

	"github.com/SimonWaldherr/GopherLLM/mobile"
)

// cCallbackSink adapts a caller's three C function pointers to
// mobile.StreamSink. It owns none of the memory it is given: cb and userdata
// are the caller's for the lifetime of one gopherllm_generate_stream call.
type cCallbackSink struct {
	onDelta    C.gopherllm_delta_cb
	onComplete C.gopherllm_complete_cb
	onError    C.gopherllm_error_cb
	userdata   unsafe.Pointer
}

var _ mobile.StreamSink = (*cCallbackSink)(nil)

func (s *cCallbackSink) OnDelta(text string) {
	cs := C.CString(text)
	defer C.free(unsafe.Pointer(cs))
	C.gopherllm_invoke_delta(s.onDelta, cs, s.userdata)
}

func (s *cCallbackSink) OnComplete(resultJSON string) {
	cs := C.CString(resultJSON)
	defer C.free(unsafe.Pointer(cs))
	C.gopherllm_invoke_complete(s.onComplete, cs, s.userdata)
}

func (s *cCallbackSink) OnError(message string) {
	cs := C.CString(message)
	defer C.free(unsafe.Pointer(cs))
	C.gopherllm_invoke_error(s.onError, cs, s.userdata)
}
