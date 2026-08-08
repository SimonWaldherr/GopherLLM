//go:build js

package webgpu

import (
	"fmt"
	"syscall/js"
)

// Buffer wraps one GPUBuffer.
type Buffer struct {
	value js.Value
	size  int
}

// Size returns the buffer's byte size.
func (b *Buffer) Size() int { return b.size }

// Destroy releases the underlying GPUBuffer's memory. No method may use b
// afterward.
func (b *Buffer) Destroy() {
	b.value.Call("destroy")
}

// gpuBufferUsage reads GPUBufferUsage bitmask constants live off the global
// object and ORs them together, rather than transcribing the W3C spec's
// numeric values -- sidesteps any question of copying spec constants and
// stays correct if a browser ever changes them.
func gpuBufferUsage(names ...string) int {
	u := js.Global().Get("GPUBufferUsage")
	total := 0
	for _, n := range names {
		total |= u.Get(n).Int()
	}
	return total
}

func newObject() js.Value { return js.Global().Get("Object").New() }

// roundUpTo4 rounds n up to the next multiple of 4. WebGPU requires both
// GPUBuffer sizes and GPUQueue.writeBuffer's byte count to be multiples of
// 4 -- a real, not hypothetical, constraint here: GGUF's Q6_K block is 210
// bytes, so plenty of real Q6_K tensors (e.g. 37 rows x 210B = 7770B) have
// a total byte size that is not 4-aligned, confirmed by an actual browser
// panic ("Number of bytes to write must be a multiple of 4") during
// development. Rounding up (zero-padding) rather than rejecting keeps
// every caller ignorant of this WebGPU-specific alignment rule.
func roundUpTo4(n int) int {
	if rem := n % 4; rem != 0 {
		return n + (4 - rem)
	}
	return n
}

// CreateStorageBuffer allocates a GPUBuffer readable by compute shaders
// (STORAGE usage) plus COPY_DST/COPY_SRC so it can be uploaded to and, if
// ever needed, copied from directly. If data is non-nil its bytes are
// uploaded immediately via WriteBuffer. The underlying GPU allocation is
// rounded up to a multiple of 4 bytes (see roundUpTo4); Buffer.Size still
// reports the caller's logical (unrounded) size, and the shaders in this
// package never read past it, so the extra padding is inert.
func (d *Device) CreateStorageBuffer(size int, data []byte) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("webgpu: buffer size must be positive, got %d", size)
	}
	desc := newObject()
	desc.Set("size", roundUpTo4(size))
	desc.Set("usage", gpuBufferUsage("STORAGE", "COPY_DST", "COPY_SRC"))
	buf := &Buffer{value: d.device.Call("createBuffer", desc), size: size}
	if data != nil {
		if err := d.WriteBuffer(buf, 0, data); err != nil {
			return nil, err
		}
	}
	return buf, nil
}

// CreateUniformBuffer allocates a small GPUBuffer with UNIFORM usage, for
// per-dispatch scalar parameters (row offsets, dimensions) a WGSL shader
// reads via a uniform binding. See CreateStorageBuffer's doc comment for
// why the allocation is rounded up to a multiple of 4.
func (d *Device) CreateUniformBuffer(size int) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("webgpu: buffer size must be positive, got %d", size)
	}
	desc := newObject()
	desc.Set("size", roundUpTo4(size))
	desc.Set("usage", gpuBufferUsage("UNIFORM", "COPY_DST"))
	return &Buffer{value: d.device.Call("createBuffer", desc), size: size}, nil
}

// WriteBuffer uploads data into buf at byteOffset via the device queue.
// queue.writeBuffer is synchronous from the caller's perspective (the
// browser copies immediately) -- no map/unmap dance needed for uploads,
// only for reading results back (see ReadBuffer). data is zero-padded up
// to a multiple of 4 bytes before the call if needed (see roundUpTo4);
// buf's own allocation is already padded to match by CreateStorageBuffer/
// CreateUniformBuffer, so the padded write always stays in bounds.
func (d *Device) WriteBuffer(buf *Buffer, byteOffset int, data []byte) error {
	if padded := roundUpTo4(len(data)); padded != len(data) {
		p := make([]byte, padded)
		copy(p, data)
		data = p
	}
	arr := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(arr, data)
	d.queue.Call("writeBuffer", buf.value, byteOffset, arr)
	return nil
}

// ReadBuffer copies size bytes starting at byteOffset out of src (which
// must have been created with COPY_SRC usage) via a temporary staging
// buffer with MAP_READ usage -- a compute-shader-writable STORAGE buffer
// cannot itself be mapped for reading under WebGPU, so results must first
// be copied into a MAP_READ-capable buffer. Every size WebGPU touches here
// (buffer/copy/map/range size) must be a multiple of 4 (see
// CreateStorageBuffer's doc comment); the caller's requested size need not
// be, so the extra padding is read into the staging buffer and trimmed
// back off before returning.
func (d *Device) ReadBuffer(src *Buffer, byteOffset, size int) ([]byte, error) {
	if size <= 0 {
		return nil, fmt.Errorf("webgpu: read size must be positive, got %d", size)
	}
	paddedSize := roundUpTo4(size)
	desc := newObject()
	desc.Set("size", paddedSize)
	desc.Set("usage", gpuBufferUsage("MAP_READ", "COPY_DST"))
	staging := d.device.Call("createBuffer", desc)
	defer staging.Call("destroy")

	encoder := d.device.Call("createCommandEncoder")
	encoder.Call("copyBufferToBuffer", src.value, byteOffset, staging, 0, paddedSize)
	cmdBuf := encoder.Call("finish")
	d.queue.Call("submit", js.ValueOf([]any{cmdBuf}))

	mapMode := js.Global().Get("GPUMapMode").Get("READ")
	if _, err := await(staging.Call("mapAsync", mapMode, 0, paddedSize)); err != nil {
		return nil, fmt.Errorf("webgpu: mapAsync: %w", err)
	}
	rangeVal := staging.Call("getMappedRange", 0, paddedSize)
	view := js.Global().Get("Uint8Array").New(rangeVal)
	out := make([]byte, paddedSize)
	js.CopyBytesToGo(out, view)
	staging.Call("unmap")
	return out[:size], nil
}
