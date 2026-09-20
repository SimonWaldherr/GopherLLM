// Command shim builds GopherLLM's C ABI: a small, stable surface over
// mobile.Engine (the same gomobile-friendly API the Swift/Obj-C binding
// uses) for any language with a C FFI. Build it with scripts/build-capi.sh,
// or directly:
//
//	go build -tags capi -buildmode=c-shared -o libgopherllm.so ./bindings/c/shim   # Linux
//	go build -tags capi -buildmode=c-shared -o libgopherllm.dylib ./bindings/c/shim # macOS
//
// which also emits a matching header (libgopherllm.h) that Rust and Python
// both consume (hand-mirrored, not bindgen-generated — see
// bindings/rust/gopherllm/src/ffi.rs and bindings/python/gopherllm/_ffi.py).
//
// Every function here does the minimum possible: convert C types to/from Go,
// forward to mobile.Engine (already tested independently), and convert the
// result back. There is deliberately no new business logic in this package.
//
// The "capi" build tag keeps this cgo-only package out of the module's
// default `go build ./...`/`go vet ./...`/`go test ./...` (and so out of the
// ordinary CI matrix, which includes a Windows runner with no C compiler
// configured for Go's cgo): without it, `go build ./...` alone would try to
// link this package's cgo runtime support and fail wherever a C toolchain
// isn't set up, even though nothing here is exercised by that build.
//
//go:build capi

package main

/*
#include <stdlib.h>
#include "callbacks.h"
*/
import "C"
import "unsafe"

func main() {}

// gopherllm_engine_new creates a new, unloaded engine and returns an opaque
// handle. The caller owns it and must eventually pass it to
// gopherllm_engine_free exactly once.
//
//export gopherllm_engine_new
func gopherllm_engine_new() uintptr {
	return newEngineHandle()
}

// gopherllm_engine_free releases the engine (unmapping any loaded model) and
// invalidates handle. Safe to call on a zero or already-freed handle.
//
//export gopherllm_engine_free
func gopherllm_engine_free(handle uintptr) {
	if e, ok := engineFromHandle(handle); ok {
		_ = e.Close()
	}
	deleteEngineHandle(handle)
}

// gopherllm_load loads a GGUF model file. optionsJSON is
// {"threads":int,"prepare_quantized":bool,"out_of_core":bool,"prefault":"none"|"core"|"all","metal":bool};
// an empty string ("" or NULL) uses the defaults. Returns NULL on success, or
// a newly allocated error string the caller must free with
// gopherllm_free_string.
//
//export gopherllm_load
func gopherllm_load(handle uintptr, path *C.char, optionsJSON *C.char) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return C.CString("invalid engine handle")
	}
	if err := e.Load(C.GoString(path), C.GoString(optionsJSON)); err != nil {
		return C.CString(err.Error())
	}
	return nil
}

// gopherllm_unload releases the current model, if any, without invalidating
// handle -- a new Load can reuse the same engine. Returns NULL on success or
// an error string the caller must free.
//
//export gopherllm_unload
func gopherllm_unload(handle uintptr) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return C.CString("invalid engine handle")
	}
	if err := e.Unload(); err != nil {
		return C.CString(err.Error())
	}
	return nil
}

// gopherllm_is_loaded returns 1 if handle currently owns a loaded model, 0
// otherwise (including an invalid handle).
//
//export gopherllm_is_loaded
func gopherllm_is_loaded(handle uintptr) C.int {
	e, ok := engineFromHandle(handle)
	if !ok || !e.IsLoaded() {
		return 0
	}
	return 1
}

// gopherllm_model_name returns the loaded model's display name, or an empty
// string if nothing is loaded or handle is invalid. The caller must free the
// result with gopherllm_free_string.
//
//export gopherllm_model_name
func gopherllm_model_name(handle uintptr) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return C.CString("")
	}
	return C.CString(e.ModelName())
}

// gopherllm_info_json returns stable, inexpensive model metadata as JSON
// ({"name":...,"architecture":...,"context_length":...,...}, or "{}" if
// nothing is loaded). The caller must free the result with
// gopherllm_free_string.
//
//export gopherllm_info_json
func gopherllm_info_json(handle uintptr) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return C.CString("{}")
	}
	return C.CString(e.InfoJSON())
}

// gopherllm_cancel interrupts handle's in-flight generate/generate_stream
// call, if any. Safe to call at any time, including with nothing running.
//
//export gopherllm_cancel
func gopherllm_cancel(handle uintptr) {
	if e, ok := engineFromHandle(handle); ok {
		e.Cancel()
	}
}

// gopherllm_generate runs a non-streaming chat completion. optionsJSON is
// {"max_tokens":int,"temperature":float,"top_p":float,"top_k":int,"min_p":float,
// "repeat_penalty":float,"seed":uint64,"system_prompt":string,"stop":[string]};
// an empty string uses the defaults.
//
// On success returns the generated text (caller frees) and, if errorOut is
// non-NULL, sets *errorOut to NULL. On failure returns NULL and, if errorOut
// is non-NULL, sets *errorOut to a newly allocated error string (caller
// frees). errorOut may be NULL if the caller only cares whether the call
// succeeded.
//
//export gopherllm_generate
func gopherllm_generate(handle uintptr, prompt *C.char, optionsJSON *C.char, errorOut **C.char) *C.char {
	setErr := func(msg string) {
		if errorOut != nil {
			*errorOut = C.CString(msg)
		}
	}
	e, ok := engineFromHandle(handle)
	if !ok {
		setErr("invalid engine handle")
		return nil
	}
	text, err := e.Generate(C.GoString(prompt), C.GoString(optionsJSON))
	if err != nil {
		setErr(err.Error())
		return nil
	}
	if errorOut != nil {
		*errorOut = nil
	}
	return C.CString(text)
}

// gopherllm_generate_stream runs a streaming chat completion, invoking
// onDelta once per text increment as it is generated and then exactly one of
// onComplete or onError, all on a Go worker goroutine before this function
// returns. userdata is passed through to every callback unmodified (a
// natural place for a language binding to stash its own context object).
//
// The C return value only reports a call that could not even start (a NULL
// callback, or an invalid handle); once generation begins, success or
// failure is reported exclusively via onComplete/onError, matching
// mobile.StreamSink's contract.
//
//export gopherllm_generate_stream
func gopherllm_generate_stream(handle uintptr, prompt *C.char, optionsJSON *C.char, onDelta C.gopherllm_delta_cb, onComplete C.gopherllm_complete_cb, onError C.gopherllm_error_cb, userdata unsafe.Pointer) *C.char {
	if onDelta == nil || onComplete == nil || onError == nil {
		return C.CString("onDelta, onComplete, and onError callbacks are all required")
	}
	e, ok := engineFromHandle(handle)
	if !ok {
		return C.CString("invalid engine handle")
	}
	sink := &cCallbackSink{onDelta: onDelta, onComplete: onComplete, onError: onError, userdata: userdata}
	// mobile.Engine.GenerateStream already reports failure through the sink
	// (OnError) before returning it here, so there is nothing left to convert
	// into the "call could not start" return value below.
	_ = e.GenerateStream(C.GoString(prompt), C.GoString(optionsJSON), sink)
	return nil
}

// gopherllm_free_string releases a string returned by any function above.
// Safe to call with NULL.
//
//export gopherllm_free_string
func gopherllm_free_string(s *C.char) {
	C.free(unsafe.Pointer(s))
}
