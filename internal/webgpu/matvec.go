//go:build js

package webgpu

import (
	"encoding/binary"
	"fmt"
	"sync"
	"syscall/js"
	"unsafe"
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
	readbackBuf            *Buffer
	xCapacity, outCapacity int
	readbackCapacity       int
	xUploadView            js.Value
	paramsUploadView       js.Value
	params                 [8]byte
	workGeneration         uint64
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
	if k.readbackBuf != nil {
		k.readbackBuf.Destroy()
		k.readbackBuf = nil
	}
	k.xCapacity = 0
	k.outCapacity = 0
	k.readbackCapacity = 0
	k.xUploadView = js.Value{}
	k.paramsUploadView = js.Value{}
	k.workGeneration++
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

	// A bind group only changes when the kernel replaces one of its reusable
	// activation/output/parameter buffers. Caching it with the prepared
	// weight avoids rebuilding four JS descriptor objects for every decoded
	// token while still invalidating it correctly after a buffer grows.
	boundKernel    *MatvecKernel
	bindGroup      js.Value
	bindGeneration uint64
}

// Destroy releases the GPU buffer backing this prepared weight. No method
// may use w afterward.
func (w *PreparedWeight) Destroy() {
	if w == nil {
		return
	}
	if w.buf != nil {
		w.buf.Destroy()
		w.buf = nil
	}
	w.boundKernel = nil
	w.bindGroup = js.Value{}
	w.bindGeneration = 0
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
// It is the allocating convenience form of RunPreparedInto.
func (k *MatvecKernel) RunPrepared(w *PreparedWeight, x []float32) ([]float32, error) {
	if w == nil {
		return nil, fmt.Errorf("webgpu: prepared weight is nil")
	}
	out := make([]float32, w.rows)
	if err := k.RunPreparedInto(w, x, out); err != nil {
		return nil, err
	}
	return out, nil
}

// RunPreparedInto is RunPrepared without a result allocation. out must have
// room for w.rows values. Real model decode should use this method because
// its destination is already a reusable activation scratch slice; avoiding
// the intermediate []float32 and copy removes a full output-vector
// allocation from every WebGPU matvec.
func (k *MatvecKernel) RunPreparedInto(w *PreparedWeight, x, out []float32) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if w == nil || w.buf == nil || w.rows <= 0 || w.cols <= 0 {
		return fmt.Errorf("webgpu: prepared weight has invalid shape")
	}
	if len(x) != w.cols {
		return fmt.Errorf("webgpu: activation length %d does not match weight cols %d", len(x), w.cols)
	}
	if len(out) < w.rows {
		return fmt.Errorf("webgpu: output length %d is smaller than weight rows %d", len(out), w.rows)
	}
	if w.cols%256 != 0 {
		return fmt.Errorf("webgpu: cols (%d) must be a multiple of 256 (the Q4_K/Q6_K block size)", w.cols)
	}
	if limit := k.dev.limits.MaxComputeWorkgroupsPerDimension; limit > 0 && w.rows > limit {
		return fmt.Errorf("webgpu: rows (%d) exceeds this device's maxComputeWorkgroupsPerDimension (%d)", w.rows, limit)
	}

	if err := k.ensureWorkBuffers(len(x)*4, w.rows*4); err != nil {
		return err
	}
	// js.CopyBytesToJS performs the required WASM-memory -> JS copy
	// synchronously. Reinterpreting the float slice avoids first serializing
	// every value into a separate Go []byte; WASM memory and WebGPU buffer
	// mappings use the same little-endian IEEE-754 representation.
	js.CopyBytesToJS(k.xUploadView, float32SliceBytes(x))
	k.dev.WriteBufferView(k.xBuf, 0, k.xUploadView, len(x)*4)
	binary.LittleEndian.PutUint32(k.params[0:], uint32(w.cols))
	binary.LittleEndian.PutUint32(k.params[4:], 0) // rowOffset: always 0 -- one PreparedWeight is one whole (unchunked) tensor
	js.CopyBytesToJS(k.paramsUploadView, k.params[:])
	k.dev.WriteBufferView(k.paramsBuf, 0, k.paramsUploadView, len(k.params))

	bindGroup := k.bindGroupFor(w)
	enc := k.dev.NewEncoder()
	enc.BeginCompute()
	enc.Dispatch(k.pipeline, bindGroup, w.rows, 1, 1)
	enc.EndCompute()
	// The old path submitted compute, then Device.ReadBuffer created a second
	// command buffer and a fresh staging allocation. Keeping the copy in this
	// command buffer reduces every matvec to one queue submission and reuses a
	// mapped staging buffer across all decode steps.
	outputBytes := w.rows * 4
	enc.CopyBuffer(k.outBuf, 0, k.readbackBuf, 0, outputBytes)
	enc.Submit()

	if err := k.dev.ReadMappedBufferInto(k.readbackBuf, 0, float32SliceBytes(out[:w.rows])); err != nil {
		return fmt.Errorf("webgpu: reading output: %w", err)
	}
	return nil
}

// ensureWorkBuffers grows reusable activation/output buffers on demand. The
// decode loop normally keeps dimensions fixed, turning every later matvec
// into writes plus one dispatch/readback rather than three GPU allocations.
func (k *MatvecKernel) ensureWorkBuffers(xSize, outSize int) error {
	buffersChanged := false
	if k.xBuf == nil || k.xCapacity < xSize {
		if k.xBuf != nil {
			k.xBuf.Destroy()
		}
		buf, err := k.dev.CreateStorageBuffer(xSize, nil)
		if err != nil {
			return fmt.Errorf("webgpu: creating activation buffer: %w", err)
		}
		k.xBuf, k.xCapacity = buf, xSize
		k.xUploadView = js.Global().Get("Uint8Array").New(xSize)
		buffersChanged = true
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
		buffersChanged = true
	}
	if k.readbackBuf == nil || k.readbackCapacity < outSize {
		if k.readbackBuf != nil {
			k.readbackBuf.Destroy()
		}
		buf, err := k.dev.CreateReadbackBuffer(outSize)
		if err != nil {
			return fmt.Errorf("webgpu: creating readback buffer: %w", err)
		}
		k.readbackBuf, k.readbackCapacity = buf, outSize
		buffersChanged = true
	}
	if k.paramsBuf == nil {
		buf, err := k.dev.CreateStorageBuffer(8, nil)
		if err != nil {
			return fmt.Errorf("webgpu: creating parameter buffer: %w", err)
		}
		k.paramsBuf = buf
		k.paramsUploadView = js.Global().Get("Uint8Array").New(len(k.params))
		buffersChanged = true
	}
	if buffersChanged {
		k.workGeneration++
	}
	return nil
}

func (k *MatvecKernel) bindGroupFor(w *PreparedWeight) js.Value {
	if w.boundKernel == k && w.bindGeneration == k.workGeneration {
		return w.bindGroup
	}
	bindGroup := k.dev.CreateBindGroup(k.pipeline, w.buf, k.xBuf, k.outBuf, k.paramsBuf)
	w.boundKernel = k
	w.bindGroup = bindGroup
	w.bindGeneration = k.workGeneration
	return bindGroup
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

// float32SliceBytes exposes a float32 slice's existing IEEE-754 bytes for
// one synchronous JS bridge call. It does not retain the Go backing array or
// hand a Go pointer to JavaScript; js.CopyBytesToJS/js.CopyBytesToGo copy the
// bytes before returning. Empty input is supported for completeness, though
// matvec activations and outputs are always non-empty.
func float32SliceBytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(v))), len(v)*4)
}
