//go:build js

package webgpu

import "syscall/js"

// Encoder records one or more compute dispatches into a single
// GPUCommandEncoder, so a whole transformer layer's matmuls can be
// submitted to the GPU queue once instead of once per matrix -- the
// CPU<->GPU round-trip latency this avoids, not raw compute time, is what
// dominates a naive per-matrix-submit design (see the project plan's
// CPU/GPU split rationale).
type Encoder struct {
	device  *Device
	encoder js.Value
	pass    js.Value // the open GPUComputePassEncoder, or the zero Value if none
}

// NewEncoder starts recording a new command buffer.
func (d *Device) NewEncoder() *Encoder {
	return &Encoder{device: d, encoder: d.device.Call("createCommandEncoder")}
}

// BeginCompute opens a compute pass. Must be paired with EndCompute before
// Submit (or before another BeginCompute) -- WebGPU allows only one open
// pass per encoder at a time.
func (e *Encoder) BeginCompute() {
	e.pass = e.encoder.Call("beginComputePass")
}

// Dispatch binds pipeline and bindGroup, then dispatches an x*y*z workgroup
// grid. Must be called between BeginCompute and EndCompute.
func (e *Encoder) Dispatch(pipeline *ComputePipeline, bindGroup js.Value, x, y, z int) {
	e.pass.Call("setPipeline", pipeline.pipeline)
	e.pass.Call("setBindGroup", 0, bindGroup)
	e.pass.Call("dispatchWorkgroups", x, y, z)
}

// EndCompute closes the currently open compute pass.
func (e *Encoder) EndCompute() {
	e.pass.Call("end")
	e.pass = js.Value{}
}

// CopyBuffer records a buffer-to-buffer copy in this encoder. Call it after
// EndCompute when copying a compute output into a MAP_READ staging buffer;
// keeping both operations in one command buffer avoids an extra queue submit
// and an avoidable CPU/GPU synchronization point per matvec.
func (e *Encoder) CopyBuffer(src *Buffer, srcOffset int, dst *Buffer, dstOffset, size int) {
	e.encoder.Call("copyBufferToBuffer", src.value, srcOffset, dst.value, dstOffset, size)
}

// Submit finishes recording and submits the command buffer to the device
// queue. The Encoder must not be reused afterward.
func (e *Encoder) Submit() {
	cmdBuf := e.encoder.Call("finish")
	e.device.queue.Call("submit", js.ValueOf([]any{cmdBuf}))
}

// CreateBindGroup builds a group-0 bind group for pipeline's auto-inferred
// layout from buffers, bound at consecutive binding indices starting at 0
// in the given order -- callers must match this order to their WGSL
// shader's @binding declarations.
func (d *Device) CreateBindGroup(pipeline *ComputePipeline, buffers ...*Buffer) js.Value {
	layout := pipeline.pipeline.Call("getBindGroupLayout", 0)
	entries := make([]any, len(buffers))
	for i, b := range buffers {
		resource := newObject()
		resource.Set("buffer", b.value)
		entry := newObject()
		entry.Set("binding", i)
		entry.Set("resource", resource)
		entries[i] = entry
	}
	desc := newObject()
	desc.Set("layout", layout)
	desc.Set("entries", js.ValueOf(entries))
	return d.device.Call("createBindGroup", desc)
}
