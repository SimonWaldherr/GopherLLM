// Command shim builds GopherLLM's C ABI, declared in
// bindings/c/include/gopherllm.h: a small, stable surface over mobile.Engine
// for any language with a C FFI. The Swift package (bindings/swift) links it
// as a static library inside an XCFramework (scripts/build-xcframework.sh);
// the Rust crate and Python package load it as a shared library
// (scripts/build-capi.sh). Direct builds:
//
//	go build -tags capi -buildmode=c-shared -o libgopherllm.so ./bindings/c/shim
//	go build -tags capi -buildmode=c-archive -o libgopherllm.a ./bindings/c/shim
//
// Every function does the minimum possible: convert C types to Go, forward
// to mobile.Engine (tested in its own package and end to end by
// bindings/c/test), and convert the result back. There is deliberately no
// business logic here. gopherllm.h documents the contract of each function;
// the Rust (bindings/rust/gopherllm/src/ffi.rs) and Python
// (bindings/python/gopherllm/_ffi.py) declarations mirror it by hand.
//
// The "capi" build tag keeps this cgo-only package out of the module's
// default `go build ./...`/`go vet ./...`/`go test ./...`, which also run
// where no C toolchain is configured (CI's Windows runner).
//
//go:build capi

package main

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#include <stdlib.h>
#define GOPHERLLM_CGO_EXPORTS
#include "gopherllm.h"
*/
import "C"
import (
	"errors"
	"unsafe"

	"github.com/SimonWaldherr/GopherLLM/mobile"
)

func main() {}

var errInvalidHandle = errors.New("invalid engine handle")

// cResult applies the error_out convention: the value on success, otherwise
// NULL with the error stored in *errorOut when the caller asked for it.
func cResult(value string, err error, errorOut **C.char) *C.char {
	if errorOut != nil {
		*errorOut = nil
	}
	if err != nil {
		if errorOut != nil {
			*errorOut = C.CString(err.Error())
		}
		return nil
	}
	return C.CString(value)
}

// cError returns NULL for success or the error message.
func cError(err error) *C.char {
	if err == nil {
		return nil
	}
	return C.CString(err.Error())
}

//export gopherllm_version
func gopherllm_version() *C.char { return C.CString(mobile.Version()) }

//export gopherllm_runtime_info_json
func gopherllm_runtime_info_json() *C.char { return C.CString(mobile.RuntimeInfoJSON()) }

//export gopherllm_inspect_model
func gopherllm_inspect_model(path *C.char, errorOut **C.char) *C.char {
	info, err := mobile.InspectModel(C.GoString(path))
	return cResult(info, err, errorOut)
}

//export gopherllm_free_string
func gopherllm_free_string(s *C.char) {
	C.free(unsafe.Pointer(s))
}

//export gopherllm_engine_new
func gopherllm_engine_new() uintptr {
	return newEngineHandle()
}

//export gopherllm_engine_free
func gopherllm_engine_free(handle uintptr) {
	if e, ok := engineFromHandle(handle); ok {
		_ = e.Close()
	}
	deleteEngineHandle(handle)
}

//export gopherllm_load
func gopherllm_load(handle uintptr, path *C.char, optionsJSON *C.char) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return cError(errInvalidHandle)
	}
	return cError(e.Load(C.GoString(path), C.GoString(optionsJSON)))
}

//export gopherllm_unload
func gopherllm_unload(handle uintptr) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return cError(errInvalidHandle)
	}
	return cError(e.Unload())
}

//export gopherllm_is_loaded
func gopherllm_is_loaded(handle uintptr) C.int {
	if e, ok := engineFromHandle(handle); ok && e.IsLoaded() {
		return 1
	}
	return 0
}

//export gopherllm_model_name
func gopherllm_model_name(handle uintptr) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return C.CString("")
	}
	return C.CString(e.ModelName())
}

//export gopherllm_info_json
func gopherllm_info_json(handle uintptr) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return C.CString("{}")
	}
	return C.CString(e.InfoJSON())
}

//export gopherllm_count_tokens
func gopherllm_count_tokens(handle uintptr, text *C.char, errorOut **C.char) C.int {
	e, ok := engineFromHandle(handle)
	if !ok {
		cResult("", errInvalidHandle, errorOut)
		return -1
	}
	n, err := e.CountTokens(C.GoString(text))
	if cResult("", err, errorOut); err != nil {
		return -1
	}
	return C.int(n)
}

//export gopherllm_cancel
func gopherllm_cancel(handle uintptr) {
	if e, ok := engineFromHandle(handle); ok {
		e.Cancel()
	}
}

//export gopherllm_generate
func gopherllm_generate(handle uintptr, prompt *C.char, optionsJSON *C.char, errorOut **C.char) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return cResult("", errInvalidHandle, errorOut)
	}
	text, err := e.Generate(C.GoString(prompt), C.GoString(optionsJSON))
	return cResult(text, err, errorOut)
}

//export gopherllm_chat
func gopherllm_chat(handle uintptr, messagesJSON *C.char, optionsJSON *C.char, errorOut **C.char) *C.char {
	e, ok := engineFromHandle(handle)
	if !ok {
		return cResult("", errInvalidHandle, errorOut)
	}
	result, err := e.Chat(C.GoString(messagesJSON), C.GoString(optionsJSON))
	return cResult(result, err, errorOut)
}

//export gopherllm_generate_stream
func gopherllm_generate_stream(handle uintptr, prompt *C.char, optionsJSON *C.char, onDelta C.gopherllm_delta_cb, onComplete C.gopherllm_complete_cb, onError C.gopherllm_error_cb, userdata unsafe.Pointer) *C.char {
	return startStream(handle, onDelta, onComplete, onError, userdata, func(e *mobile.Engine, sink mobile.StreamSink) {
		_ = e.GenerateStream(C.GoString(prompt), C.GoString(optionsJSON), sink)
	})
}

//export gopherllm_chat_stream
func gopherllm_chat_stream(handle uintptr, messagesJSON *C.char, optionsJSON *C.char, onDelta C.gopherllm_delta_cb, onComplete C.gopherllm_complete_cb, onError C.gopherllm_error_cb, userdata unsafe.Pointer) *C.char {
	return startStream(handle, onDelta, onComplete, onError, userdata, func(e *mobile.Engine, sink mobile.StreamSink) {
		_ = e.ChatStream(C.GoString(messagesJSON), C.GoString(optionsJSON), sink)
	})
}

// startStream validates what the C return value reports and hands every
// other outcome to the sink: mobile's streaming methods report their errors
// through OnError as well as their return value.
func startStream(handle uintptr, onDelta C.gopherllm_delta_cb, onComplete C.gopherllm_complete_cb, onError C.gopherllm_error_cb, userdata unsafe.Pointer, stream func(*mobile.Engine, mobile.StreamSink)) *C.char {
	if onDelta == nil || onComplete == nil || onError == nil {
		return C.CString("on_delta, on_complete and on_error callbacks are all required")
	}
	e, ok := engineFromHandle(handle)
	if !ok {
		return cError(errInvalidHandle)
	}
	stream(e, &cCallbackSink{onDelta: onDelta, onComplete: onComplete, onError: onError, userdata: userdata})
	return nil
}
