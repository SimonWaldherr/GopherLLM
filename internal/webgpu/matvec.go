//go:build js

package webgpu

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// MatvecKernel is a compiled dequantizing matvec compute pipeline (Q4_K or
// Q6_K -- see shader_q4k.go/shader_q6k.go), reusable across many Run/
// RunPrepared calls against the same Device so the (relatively expensive)
// shader compilation happens once.
type MatvecKernel struct {
	dev                    *Device
	pipeline               *ComputePipeline
	mu                     sync.Mutex
	xBuf                   *Buffer
	outBuf                 *Buffer
	paramsBuf              *Buffer
	xCapacity, outCapacity int
}

func newMatvecKernel(dev *Device, wgsl string) (*MatvecKernel, error) {
	p, err := dev.CreateComputePipeline(wgsl, "main")
	if err != nil {
		return nil, err
	}
	return &MatvecKernel{dev: dev, pipeline: p}, nil
}

// NewQ4KMatvecKernel compiles the Q4_K dequantizing matvec kernel.
func NewQ4KMatvecKernel(dev *Device) (*MatvecKernel, error) {
	return newMatvecKernel(dev, Q4KMatvecWGSL)
}

// NewQ6KMatvecKernel compiles the Q6_K dequantizing matvec kernel.
func NewQ6KMatvecKernel(dev *Device) (*MatvecKernel, error) {
	return newMatvecKernel(dev, Q6KMatvecWGSL)
}

// Destroy releases reusable per-dispatch buffers. PreparedWeight values own
// their own buffers and must be released separately.
func (k *MatvecKernel) Destroy() {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.xBuf != nil {
		k.xBuf.Destroy()
		k.xBuf = nil
	}
	if k.outBuf != nil {
		k.outBuf.Destroy()
		k.outBuf = nil
	}
	if k.paramsBuf != nil {
		k.paramsBuf.Destroy()
		k.paramsBuf = nil
	}
	k.xCapacity = 0
	k.outCapacity = 0
}

// PreparedWeight is one tensor's raw quantized bytes already resident in a
// GPU buffer, created once (see Device.PrepareWeight) and reused for every
// subsequent RunPrepared call. This distinction matters a lot in practice:
// decode calls a matvec fresh for every generated token, so re-uploading a
// multi-megabyte weight matrix's bytes on every single call (as the
// one-shot Run convenience method below does) would make the GPU path
// slower than the CPU path it's meant to replace, not faster.
type PreparedWeight struct {
	buf        *Buffer
	rows, cols int
}

// Destroy releases the GPU buffer backing this prepared weight. No method
// may use w afterward.
func (w *PreparedWeight) Destroy() {
	w.buf.Destroy()
}

// PrepareWeight uploads data (one tensor's raw on-disk quantized bytes, in
// the on-disk Q4_K/Q6_K block layout the shaders expect) into a new GPU
// storage buffer once. Returns an error if data exceeds this device's
// maxStorageBufferBindingSize -- splitting an oversized tensor across
// multiple buffers (needed for very large tensors like a tied output/
// embedding projection) is not implemented yet; callers should catch this
// and fall back to the CPU path for that specific tensor rather than
// failing the whole model load.
func (d *Device) PrepareWeight(data []byte, rows, cols int) (*PreparedWeight, error) {
	if len(data) == 0 || rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("webgpu: invalid weight shape rows=%d cols=%d len(data)=%d", rows, cols, len(data))
	}
	if limit := d.limits.MaxStorageBufferBindingSize; limit > 0 && len(data) > limit {
		return nil, fmt.Errorf("webgpu: tensor size %d exceeds this device's maxStorageBufferBindingSize %d (chunking not implemented)", len(data), limit)
	}
	buf, err := d.CreateStorageBuffer(len(data), data)
	if err != nil {
		return nil, err
	}
	return &PreparedWeight{buf: buf, rows: rows, cols: cols}, nil
}

// RunPrepared computes out[i] = dot(row_i, x) against an already-uploaded
// weight (see PrepareWeight), uploading only the small activation vector x
// fresh on every call -- the cheap per-token cost real decode should pay.
func (k *MatvecKernel) RunPrepared(w *PreparedWeight, x []float32) ([]float32, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	if w.rows <= 0 || w.cols <= 0 {
		return nil, fmt.Errorf("webgpu: prepared weight has invalid shape rows=%d cols=%d", w.rows, w.cols)
	}
	if len(x) != w.cols {
		return nil, fmt.Errorf("webgpu: activation length %d does not match weight cols %d", len(x), w.cols)
	}
	if w.cols%256 != 0 {
		return nil, fmt.Errorf("webgpu: cols (%d) must be a multiple of 256 (the Q4_K/Q6_K block size)", w.cols)
	}
	if limit := k.dev.limits.MaxComputeWorkgroupsPerDimension; limit > 0 && w.rows > limit {
		return nil, fmt.Errorf("webgpu: rows (%d) exceeds this device's maxComputeWorkgroupsPerDimension (%d)", w.rows, limit)
	}

	if err := k.ensureWorkBuffers(len(x)*4, w.rows*4); err != nil {
		return nil, err
	}
	if err := k.dev.WriteBuffer(k.xBuf, 0, f32ToBytes(x)); err != nil {
		return nil, fmt.Errorf("webgpu: uploading activation: %w", err)
	}
	params := make([]byte, 8)
	binary.LittleEndian.PutUint32(params[0:], uint32(w.cols))
	binary.LittleEndian.PutUint32(params[4:], 0) // rowOffset: always 0 -- one PreparedWeight is one whole (unchunked) tensor
	if err := k.dev.WriteBuffer(k.paramsBuf, 0, params); err != nil {
		return nil, fmt.Errorf("webgpu: uploading params: %w", err)
	}

	bindGroup := k.dev.CreateBindGroup(k.pipeline, w.buf, k.xBuf, k.outBuf, k.paramsBuf)
	enc := k.dev.NewEncoder()
	enc.BeginCompute()
	enc.Dispatch(k.pipeline, bindGroup, w.rows, 1, 1)
	enc.EndCompute()
	enc.Submit()

	outBytes, err := k.dev.ReadBuffer(k.outBuf, 0, w.rows*4)
	if err != nil {
		return nil, fmt.Errorf("webgpu: reading output: %w", err)
	}
	return bytesToF32(outBytes), nil
}

// ensureWorkBuffers grows reusable activation/output buffers on demand. The
// decode loop normally keeps dimensions fixed, turning every later matvec
// into writes plus one dispatch/readback rather than three GPU allocations.
func (k *MatvecKernel) ensureWorkBuffers(xSize, outSize int) error {
	if k.xBuf == nil || k.xCapacity < xSize {
		if k.xBuf != nil {
			k.xBuf.Destroy()
		}
		buf, err := k.dev.CreateStorageBuffer(xSize, nil)
		if err != nil {
			return fmt.Errorf("webgpu: creating activation buffer: %w", err)
		}
		k.xBuf, k.xCapacity = buf, xSize
	}
	if k.outBuf == nil || k.outCapacity < outSize {
		if k.outBuf != nil {
			k.outBuf.Destroy()
		}
		buf, err := k.dev.CreateStorageBuffer(outSize, nil)
		if err != nil {
			return fmt.Errorf("webgpu: creating output buffer: %w", err)
		}
		k.outBuf, k.outCapacity = buf, outSize
	}
	if k.paramsBuf == nil {
		buf, err := k.dev.CreateStorageBuffer(8, nil)
		if err != nil {
			return fmt.Errorf("webgpu: creating parameter buffer: %w", err)
		}
		k.paramsBuf = buf
	}
	return nil
}

// Run is a one-shot convenience wrapper for testing/benchmarking: prepares
// weightBytes as a fresh buffer, runs once, and discards it. Real decode
// should call PrepareWeight once at load time and reuse the result across
// many RunPrepared calls instead (see PreparedWeight's doc comment).
func (k *MatvecKernel) Run(weightBytes []byte, x []float32, rows, cols int) ([]float32, error) {
	prepared, err := k.dev.PrepareWeight(weightBytes, rows, cols)
	if err != nil {
		return nil, err
	}
	defer prepared.Destroy()
	return k.RunPrepared(prepared, x)
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
