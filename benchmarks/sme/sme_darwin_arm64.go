//go:build gopherllm_sme_experiment && darwin && arm64 && cgo && !ios

package sme

/*
#cgo CFLAGS: -O3
#include "tile.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"syscall"
	"unsafe"
)

func initSME() (tileKernel, error) {
	sme2, err := featureGate(syscall.SysctlUint32)
	if err != nil {
		return tileKernel{}, err
	}
	// Even the VL query enters streaming mode. It must follow the OS feature
	// checks, rather than serve as an instruction-based feature probe.
	lanes := int(C.gopherllm_sme_lanes())
	if _, err := packedSize(1, 1, lanes); err != nil {
		return tileKernel{}, fmt.Errorf("unsupported streaming vector length: %d int32 lanes", lanes)
	}
	return tileKernel{lanes: lanes, sme2: sme2}, nil
}

func runSME(a, b []int8, out []int32, rows, cols, depth, lanes int) error {
	// C keeps no Go pointers and calls no Go code. Its ordinary C interface
	// lets Clang manage streaming mode and private ZA state across cgo.
	status := C.gopherllm_sme_tile(
		(*C.int8_t)(unsafe.Pointer(&a[0])),
		(*C.int8_t)(unsafe.Pointer(&b[0])),
		(*C.int32_t)(unsafe.Pointer(&out[0])),
		C.size_t(rows), C.size_t(cols), C.size_t((depth+3)/4), C.size_t(lanes))
	if status != 0 {
		return errors.New("streaming vector length changed since packing; output was not written")
	}
	return nil
}
