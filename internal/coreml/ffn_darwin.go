//go:build darwin && cgo && coreml

package coreml

/*
#cgo CFLAGS: -x objective-c -fblocks
#cgo LDFLAGS: -framework Foundation -framework CoreML
#include <stdlib.h>
#include "runtime.h"
#include "ffn_runtime.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

type nativeFFNBackend struct{ ptr unsafe.Pointer }

// OpenFFN loads a compiled, stateless float32 [T,D] feed-forward graph from a
// persistent .mlmodelc path. It accepts only hidden -> output with the same
// fixed shape, 1 <= T <= 128, D <= 16384 and T*D <= 1048576.
func OpenFFN(path string, units ComputeUnits) (*FFN, error) {
	if !bool(C.gllm_coreml_supported()) {
		return nil, ErrUnavailable
	}
	if units > All {
		return nil, fmt.Errorf("coreml: unknown compute units %d", units)
	}
	path, err := compiledModelPath(path)
	if err != nil {
		return nil, err
	}
	value := C.CString(path)
	defer C.free(unsafe.Pointer(value))
	var info C.GLLMCoreMLFFNInfo
	var message *C.char
	ptr := C.gllm_coreml_ffn_open(value, C.int(units), &info, &message)
	if ptr == nil {
		return nil, nativeError(message)
	}
	if message != nil {
		C.free(unsafe.Pointer(message))
	}
	manifest := ""
	if info.manifest != nil {
		manifest = C.GoString(info.manifest)
		C.free(unsafe.Pointer(info.manifest))
	}
	return &FFN{info: FFNInfo{Tokens: int(info.tokens), Dimension: int(info.dimension), Manifest: manifest}, backend: &nativeFFNBackend{ptr: ptr}}, nil
}

func (b *nativeFFNBackend) predict(input, output []float32) error {
	var message *C.char
	ok := C.gllm_coreml_ffn_predict(b.ptr, (*C.float)(unsafe.Pointer(&input[0])), (*C.float)(unsafe.Pointer(&output[0])), &message)
	runtime.KeepAlive(input)
	runtime.KeepAlive(output)
	if !bool(ok) {
		return nativeError(message)
	}
	if message != nil {
		C.free(unsafe.Pointer(message))
	}
	return nil
}

func (b *nativeFFNBackend) close() {
	C.gllm_coreml_ffn_release(b.ptr)
	b.ptr = nil
}
