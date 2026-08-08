//go:build js

package webgpu

import (
	"context"
	"fmt"
	"syscall/js"
)

// Device wraps a GPUDevice and its default GPUQueue.
type Device struct {
	device js.Value
	queue  js.Value
	limits DeviceLimits
}

// DeviceLimits mirrors the subset of GPUSupportedLimits this package's
// callers need, read from the real device at acquisition time -- never
// hardcoded, since these vary by hardware/driver (see RequestAdapterAndDevice).
type DeviceLimits struct {
	MaxStorageBufferBindingSize      int
	MaxStorageBuffersPerShaderStage  int
	MinStorageBufferOffsetAlignment  int
	MaxComputeWorkgroupsPerDimension int
	MaxComputeInvocationsPerWorkgroup int
}

// Available reports whether navigator.gpu exists at all -- a cheap
// synchronous check, safe to call before attempting the async
// RequestAdapterAndDevice.
func Available() bool {
	nav := js.Global().Get("navigator")
	if nav.IsUndefined() {
		return false
	}
	return !nav.Get("gpu").IsUndefined()
}

// RequestAdapterAndDevice acquires a GPU adapter and device. ctx cancels
// the wait (WebGPU's own APIs have no native cancellation, so this races
// the result against ctx.Done(); the underlying JS Promise keeps running to
// completion regardless, same as any other unawaited Promise -- acceptable
// since adapter/device acquisition is normally a few milliseconds).
func RequestAdapterAndDevice(ctx context.Context) (*Device, error) {
	if !Available() {
		return nil, fmt.Errorf("webgpu: navigator.gpu is not available in this browser")
	}
	gpu := js.Global().Get("navigator").Get("gpu")

	type result struct {
		dev *Device
		err error
	}
	ch := make(chan result, 1)
	go func() {
		adapterVal, err := await(gpu.Call("requestAdapter"))
		if err != nil {
			ch <- result{err: fmt.Errorf("webgpu: requestAdapter: %w", err)}
			return
		}
		if adapterVal.IsNull() || adapterVal.IsUndefined() {
			ch <- result{err: fmt.Errorf("webgpu: no adapter available")}
			return
		}
		deviceVal, err := await(adapterVal.Call("requestDevice"))
		if err != nil {
			ch <- result{err: fmt.Errorf("webgpu: requestDevice: %w", err)}
			return
		}
		dev := &Device{
			device: deviceVal,
			queue:  deviceVal.Get("queue"),
			limits: readLimits(deviceVal.Get("limits")),
		}
		ch <- result{dev: dev}
	}()
	select {
	case r := <-ch:
		return r.dev, r.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func readLimits(l js.Value) DeviceLimits {
	get := func(name string) int {
		v := l.Get(name)
		if v.IsUndefined() || v.IsNull() {
			return 0
		}
		return v.Int()
	}
	return DeviceLimits{
		MaxStorageBufferBindingSize:       get("maxStorageBufferBindingSize"),
		MaxStorageBuffersPerShaderStage:   get("maxStorageBuffersPerShaderStage"),
		MinStorageBufferOffsetAlignment:   get("minStorageBufferOffsetAlignment"),
		MaxComputeWorkgroupsPerDimension:  get("maxComputeWorkgroupsPerDimension"),
		MaxComputeInvocationsPerWorkgroup: get("maxComputeInvocationsPerWorkgroup"),
	}
}

// Limits returns the device's real reported limits.
func (d *Device) Limits() DeviceLimits { return d.limits }

// Destroy releases the underlying GPUDevice. No method may be called on
// d or anything created from it afterward.
func (d *Device) Destroy() {
	d.device.Call("destroy")
}
