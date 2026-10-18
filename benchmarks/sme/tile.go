// Package sme contains an opt-in SME microkernel experiment, independent of
// model inference. Normal builds contain only the packing and scalar reference.
package sme

import (
	"errors"
	"fmt"
	"os"
)

// The worst signed-int8 product is (-128)*(-128). Bound depth so every
// intermediate sum fits int32, including inputs that use the full int8 range.
const maxDepth = (1<<31 - 1) / (128 * 128)

func packedSize(rows, depth, lanes int) (int, error) {
	// Streaming VL is 128..2048 bits, in multiples of 128 bits. Each SMOPA
	// int32 lane consumes four adjacent int8 values from the depth dimension.
	if lanes < 4 || lanes > 64 || lanes%4 != 0 || rows < 1 || rows > lanes || depth < 1 || depth > maxDepth {
		return 0, errors.New("invalid SME tile dimensions")
	}
	return ((depth + 3) / 4) * lanes * 4, nil
}

// packRows changes row-major [rows][depth] to [ceil(depth/4)][lanes][4].
// It clears both the depth tail and unused rows, including when dst is reused.
// The source and destination must not overlap.
func packRows(dst, src []int8, rows, depth, lanes int) error {
	size, err := packedSize(rows, depth, lanes)
	if err != nil {
		return err
	}
	if len(dst) < size || len(src) < rows*depth {
		return errors.New("short SME packing buffers")
	}
	clear(dst[:size])
	for k := 0; k < depth; k += 4 {
		width := min(4, depth-k)
		for row := range rows {
			start := (k/4*lanes + row) * 4
			copy(dst[start:start+width], src[row*depth+k:row*depth+k+width])
		}
	}
	return nil
}

func validateTile(a, b []int8, out []int32, rows, cols, depth, lanes int) error {
	aSize, err := packedSize(rows, depth, lanes)
	if err != nil {
		return err
	}
	bSize, err := packedSize(cols, depth, lanes)
	if err != nil {
		return err
	}
	if len(a) < aSize || len(b) < bSize || len(out) < rows*cols {
		return errors.New("short SME tile buffers")
	}
	return nil
}

// packedReference implements the SMOPA grouping using ordinary Go arithmetic.
// out is overwritten, not accumulated. It includes zero-padded depth tails.
// Both inputs must come from packRows with the supplied dimensions.
func packedReference(a, b []int8, out []int32, rows, cols, depth, lanes int) error {
	if err := validateTile(a, b, out, rows, cols, depth, lanes); err != nil {
		return err
	}
	for row := range rows {
		for col := range cols {
			var sum int32
			for group := range (depth + 3) / 4 {
				aBase, bBase := (group*lanes+row)*4, (group*lanes+col)*4
				for k := range 4 {
					sum += int32(a[aBase+k]) * int32(b[bBase+k])
				}
			}
			out[row*cols+col] = sum
		}
	}
	return nil
}

// featureGate is independently testable without executing an SME instruction.
// SME2 is reported for future experiments, but the SMOPA kernel needs only SME
// and its optional signed-int8-to-int32 arithmetic support.
func featureGate(read func(string) (uint32, error)) (sme2 bool, err error) {
	for _, name := range []string{"hw.optional.arm.FEAT_SME", "hw.optional.arm.SME_I8I32"} {
		value, err := read(name)
		if err != nil || value == 0 {
			return false, fmt.Errorf("SME unavailable: %s is missing, disabled or unreadable", name)
		}
	}
	value, err := read("hw.optional.arm.FEAT_SME2")
	return err == nil && value != 0, nil
}

type tileKernel struct {
	lanes int
	sme2  bool
}

func newSMEKernel() (tileKernel, error) {
	if os.Getenv("GOPHERLLM_SME_EXPERIMENT") != "1" {
		return tileKernel{}, errors.New("set GOPHERLLM_SME_EXPERIMENT=1 to opt in")
	}
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" {
		return tileKernel{}, errors.New("SME disabled by GOPHERLLM_DISABLE_SIMD")
	}
	return initSME()
}

func (k tileKernel) run(a, b []int8, out []int32, rows, cols, depth int) error {
	if err := validateTile(a, b, out, rows, cols, depth, k.lanes); err != nil {
		return err
	}
	return runSME(a, b, out, rows, cols, depth, k.lanes)
}
