//go:build darwin && cgo && coreml

package coreml

/*
#cgo CFLAGS: -x objective-c -fblocks
#cgo LDFLAGS: -framework Foundation -framework CoreML
#include <stdlib.h>
#include "runtime.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type nativeBackend struct {
	ptr       unsafe.Pointer
	tokenType bool
}

func Open(path string, c Config) (*Model, error) {
	if !bool(C.gllm_coreml_supported()) {
		return nil, ErrUnavailable
	}
	if err := validateConfig(c); err != nil {
		return nil, err
	}
	path, err := compiledModelPath(path)
	if err != nil {
		return nil, err
	}
	values := []*C.char{C.CString(path), C.CString(c.InputIDs), C.CString(c.AttentionMask), C.CString(c.TokenTypeIDs), C.CString(c.Output)}
	defer func() {
		for _, value := range values {
			C.free(unsafe.Pointer(value))
		}
	}()
	var info C.GLLMCoreMLInfo
	var message *C.char
	ptr := C.gllm_coreml_open(values[0], C.int(c.ComputeUnits), values[1], values[2], values[3], values[4], &info, &message)
	if ptr == nil {
		return nil, nativeError(message)
	}
	if message != nil {
		C.free(unsafe.Pointer(message))
	}
	return &Model{
		info: Info{BatchSize: int(info.batch_size), SequenceLength: int(info.sequence_length), Dimension: int(info.dimension), ComputeUnits: c.ComputeUnits,
			NeuralEngineAvailable: bool(info.neural_engine_available), NeuralEngineCores: int(info.neural_engine_cores)},
		backend: &nativeBackend{ptr: ptr, tokenType: c.TokenTypeIDs != ""},
	}, nil
}

func nativeError(message *C.char) error {
	if message == nil {
		return errors.New("coreml: native operation failed")
	}
	defer C.free(unsafe.Pointer(message))
	return errors.New("coreml: " + C.GoString(message))
}

func (b *nativeBackend) predict(ids, mask, types []int32, out []float32) error {
	if b.tokenType != (len(types) != 0) {
		return errors.New("coreml: token types do not match the configured model inputs")
	}
	var tokenTypes *C.int32_t
	if len(types) > 0 {
		tokenTypes = (*C.int32_t)(unsafe.Pointer(&types[0]))
	}
	var message *C.char
	ok := C.gllm_coreml_predict(b.ptr, (*C.int32_t)(unsafe.Pointer(&ids[0])), (*C.int32_t)(unsafe.Pointer(&mask[0])), tokenTypes,
		(*C.float)(unsafe.Pointer(&out[0])), &message)
	runtime.KeepAlive(ids)
	runtime.KeepAlive(mask)
	runtime.KeepAlive(types)
	runtime.KeepAlive(out)
	if !bool(ok) {
		return nativeError(message)
	}
	if message != nil {
		C.free(unsafe.Pointer(message))
	}
	return nil
}

func (b *nativeBackend) close() {
	C.gllm_coreml_release(b.ptr)
	b.ptr = nil
}

// Inspect loads only the diagnostic compute plan. It does not create a Model,
// run prediction, or validate the embedding feature interface. Feature names
// in Config are ignored; compute units must match the configuration to inspect.
// The API requires macOS 14.4/iOS 17.4 or newer.
func Inspect(path string, c Config) (Plan, error) {
	if !bool(C.gllm_coreml_inspect_supported()) {
		return Plan{}, fmt.Errorf("coreml: compute plans require macOS 14.4/iOS 17.4 or newer: %w", ErrUnavailable)
	}
	if c.ComputeUnits > All {
		return Plan{}, fmt.Errorf("coreml: unknown compute units %d", c.ComputeUnits)
	}
	path, err := compiledModelPath(path)
	if err != nil {
		return Plan{}, err
	}
	value := C.CString(path)
	defer C.free(unsafe.Pointer(value))
	var plan C.GLLMCoreMLPlan
	var message *C.char
	if !bool(C.gllm_coreml_inspect(value, C.int(c.ComputeUnits), &plan, &message)) {
		return Plan{}, nativeError(message)
	}
	if message != nil {
		C.free(unsafe.Pointer(message))
	}
	return Plan{
		Operations: int(plan.operations), CPUOperations: int(plan.cpu_operations), GPUOperations: int(plan.gpu_operations),
		NeuralEngineOperations: int(plan.neural_engine_operations), UnassignedOperations: int(plan.unassigned_operations),
	}, nil
}
