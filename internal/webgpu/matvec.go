//go:build js

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"
)

// MatvecKernel is a compiled dequantizing matvec compute pipeline (Q4_K or
// Q6_K -- see shader_q4k.go/shader_q6k.go), reusable across many Run calls
// against the same Device so the (relatively expensive) shader compilation
// happens once.
type MatvecKernel struct {
	dev      *Device
	pipeline *ComputePipeline
}

func newMatvecKernel(dev *Device, wgsl string) (*MatvecKernel, error) {
	p, err := dev.CreateComputePipeline(wgsl, "main")
	if err != nil {
		return nil, err
	}
	return &MatvecKernel{dev: dev, pipeline: p}, nil
}

// NewQ4KMatvecKernel compiles the Q4_K dequantizing matvec kernel.
func NewQ4KMatvecKernel(dev *Device) (*MatvecKernel, error) { return newMatvecKernel(dev, Q4KMatvecWGSL) }

// NewQ6KMatvecKernel compiles the Q6_K dequantizing matvec kernel.
func NewQ6KMatvecKernel(dev *Device) (*MatvecKernel, error) { return newMatvecKernel(dev, Q6KMatvecWGSL) }

// Run computes out[i] = dot(row_i, x) for `rows` contiguous rows of
// weightBytes (raw on-disk quantized bytes, `cols` columns each) against
// activation x (length cols), returning one f32 per row.
//
// rows is capped by the device's maxComputeWorkgroupsPerDimension (one
// workgroup per row, dispatched along the x dimension) -- callers with a
// larger matrix must split it into multiple Run calls themselves; this is
// deliberately not done automatically here since the right chunking
// strategy (by row range vs. by byte-size, and how results from separate
// chunks get placed into one shared output buffer) depends on how the
// caller is using the result, not on anything this package can decide.
func (k *MatvecKernel) Run(weightBytes []byte, x []float32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("webgpu: rows and cols must be positive, got rows=%d cols=%d", rows, cols)
	}
	if limit := k.dev.Limits().MaxComputeWorkgroupsPerDimension; limit > 0 && rows > limit {
		return nil, fmt.Errorf("webgpu: rows (%d) exceeds this device's maxComputeWorkgroupsPerDimension (%d); split into multiple Run calls", rows, limit)
	}
	if cols%256 != 0 {
		return nil, fmt.Errorf("webgpu: cols (%d) must be a multiple of 256 (the Q4_K/Q6_K block size)", cols)
	}

	xBuf, err := k.dev.CreateStorageBuffer(len(x)*4, f32ToBytes(x))
	if err != nil {
		return nil, fmt.Errorf("webgpu: uploading activation: %w", err)
	}
	wBuf, err := k.dev.CreateStorageBuffer(len(weightBytes), weightBytes)
	if err != nil {
		return nil, fmt.Errorf("webgpu: uploading weights: %w", err)
	}
	outBuf, err := k.dev.CreateStorageBuffer(rows*4, nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: creating output buffer: %w", err)
	}
	params := make([]byte, 8)
	binary.LittleEndian.PutUint32(params[0:], uint32(cols))
	binary.LittleEndian.PutUint32(params[4:], 0) // rowOffset: this Run call always writes out[0:rows]
	paramsBuf, err := k.dev.CreateStorageBuffer(8, params)
	if err != nil {
		return nil, fmt.Errorf("webgpu: uploading params: %w", err)
	}

	bindGroup := k.dev.CreateBindGroup(k.pipeline, wBuf, xBuf, outBuf, paramsBuf)
	enc := k.dev.NewEncoder()
	enc.BeginCompute()
	enc.Dispatch(k.pipeline, bindGroup, rows, 1, 1)
	enc.EndCompute()
	enc.Submit()

	outBytes, err := k.dev.ReadBuffer(outBuf, 0, rows*4)
	if err != nil {
		return nil, fmt.Errorf("webgpu: reading output: %w", err)
	}
	return bytesToF32(outBytes), nil
}

func f32ToBytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}

func bytesToF32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}
