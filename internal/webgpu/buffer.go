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

// CreateReadbackBuffer allocates a MAP_READ staging buffer. Compute outputs
// cannot be mapped directly, so callers copy into one of these buffers after
// ending their compute pass and then call ReadMappedBufferInto. Keeping a
// staging buffer alive across decode steps is materially cheaper than
// allocating/destroying a GPUBuffer for every matvec result.
func (d *Device) CreateReadbackBuffer(size int) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("webgpu: buffer size must be positive, got %d", size)
	}
	desc := newObject()
	desc.Set("size", roundUpTo4(size))
	desc.Set("usage", gpuBufferUsage("MAP_READ", "COPY_DST"))
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

// WriteBufferView writes size bytes from an existing Uint8Array view to buf
// without allocating another JS typed array. It is intended for hot paths
// that keep a capacity-sized upload view around (activation vectors and
// scalar parameters in MatvecKernel). Because the view is Uint8Array,
// WebGPU interprets its dataOffset and size arguments in bytes. Supplying
// size is important after a buffer has grown for a large FFN activation: a
// later smaller projection must not re-upload the unused tail capacity.
func (d *Device) WriteBufferView(buf *Buffer, byteOffset int, view js.Value, size int) {
	d.queue.Call("writeBuffer", buf.value, byteOffset, view, 0, size)
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

	out := make([]byte, paddedSize)
	if err := d.ReadMappedBufferInto(&Buffer{value: staging, size: paddedSize}, 0, out); err != nil {
		return nil, err
	}
	return out[:size], nil
}

// ReadMappedBufferInto maps an already-submitted MAP_READ buffer and copies
// its bytes into dst. The caller is responsible for recording and submitting
// the copy into src first. dst must be 4-byte aligned in length, matching the
// WebGPU map range requirement. This deliberately accepts a caller-owned
// destination so hot paths can reuse Go memory instead of allocating a fresh
// byte slice per GPU dispatch.
func (d *Device) ReadMappedBufferInto(src *Buffer, byteOffset int, dst []byte) error {
	if src == nil {
		return fmt.Errorf("webgpu: read source is nil")
	}
	if byteOffset < 0 || byteOffset%4 != 0 {
		return fmt.Errorf("webgpu: read offset must be a non-negative multiple of 4, got %d", byteOffset)
	}
	if len(dst) <= 0 || len(dst)%4 != 0 {
		return fmt.Errorf("webgpu: mapped read size must be a positive multiple of 4, got %d", len(dst))
	}
	if byteOffset+len(dst) > src.size {
		return fmt.Errorf("webgpu: mapped read range [%d,%d) exceeds buffer size %d", byteOffset, byteOffset+len(dst), src.size)
	}

	mapMode := js.Global().Get("GPUMapMode").Get("READ")
	if _, err := await(src.value.Call("mapAsync", mapMode, byteOffset, len(dst))); err != nil {
		return fmt.Errorf("webgpu: mapAsync: %w", err)
	}
	defer src.value.Call("unmap")
	rangeVal := src.value.Call("getMappedRange", byteOffset, len(dst))
	view := js.Global().Get("Uint8Array").New(rangeVal)
	js.CopyBytesToGo(dst, view)
	return nil
}
