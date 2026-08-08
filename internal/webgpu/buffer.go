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

// CreateStorageBuffer allocates a GPUBuffer readable by compute shaders
// (STORAGE usage) plus COPY_DST/COPY_SRC so it can be uploaded to and, if
// ever needed, copied from directly. If data is non-nil its bytes are
// uploaded immediately via WriteBuffer.
func (d *Device) CreateStorageBuffer(size int, data []byte) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("webgpu: buffer size must be positive, got %d", size)
	}
	desc := newObject()
	desc.Set("size", size)
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
// reads via a uniform binding.
func (d *Device) CreateUniformBuffer(size int) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("webgpu: buffer size must be positive, got %d", size)
	}
	desc := newObject()
	desc.Set("size", size)
	desc.Set("usage", gpuBufferUsage("UNIFORM", "COPY_DST"))
	return &Buffer{value: d.device.Call("createBuffer", desc), size: size}, nil
}

// WriteBuffer uploads data into buf at byteOffset via the device queue.
// queue.writeBuffer is synchronous from the caller's perspective (the
// browser copies immediately) -- no map/unmap dance needed for uploads,
// only for reading results back (see ReadBuffer).
func (d *Device) WriteBuffer(buf *Buffer, byteOffset int, data []byte) error {
	arr := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(arr, data)
	d.queue.Call("writeBuffer", buf.value, byteOffset, arr)
	return nil
}

// ReadBuffer copies size bytes starting at byteOffset out of src (which
// must have been created with COPY_SRC usage) via a temporary staging
// buffer with MAP_READ usage -- a compute-shader-writable STORAGE buffer
// cannot itself be mapped for reading under WebGPU, so results must first
// be copied into a MAP_READ-capable buffer.
func (d *Device) ReadBuffer(src *Buffer, byteOffset, size int) ([]byte, error) {
	if size <= 0 {
		return nil, fmt.Errorf("webgpu: read size must be positive, got %d", size)
	}
	desc := newObject()
	desc.Set("size", size)
	desc.Set("usage", gpuBufferUsage("MAP_READ", "COPY_DST"))
	staging := d.device.Call("createBuffer", desc)

	encoder := d.device.Call("createCommandEncoder")
	encoder.Call("copyBufferToBuffer", src.value, byteOffset, staging, 0, size)
	cmdBuf := encoder.Call("finish")
	d.queue.Call("submit", js.ValueOf([]any{cmdBuf}))

	mapMode := js.Global().Get("GPUMapMode").Get("READ")
	if _, err := await(staging.Call("mapAsync", mapMode, 0, size)); err != nil {
		return nil, fmt.Errorf("webgpu: mapAsync: %w", err)
	}
	rangeVal := staging.Call("getMappedRange", 0, size)
	view := js.Global().Get("Uint8Array").New(rangeVal)
	out := make([]byte, size)
	js.CopyBytesToGo(out, view)
	staging.Call("unmap")
	return out, nil
}
