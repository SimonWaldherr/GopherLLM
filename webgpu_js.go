//go:build js

package gopherllm

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/SimonWaldherr/GopherLLM/internal/webgpu"
)

// GPUWeight is one tensor's WebGPU-resident copy (see Weight.GPU), uploaded
// once at load time by prepareWebGPUWeight and reused for every subsequent
// matvec -- re-uploading a multi-megabyte weight matrix on every decode
// step would make this slower than the CPU path it replaces, not faster.
type GPUWeight struct {
	prepared *webgpu.PreparedWeight
	typ      GGMLType
}

var (
	webgpuOnce    sync.Once
	webgpuDevice  *webgpu.Device
	webgpuQ4K     *webgpu.MatvecKernel
	webgpuQ6K     *webgpu.MatvecKernel
	webgpuInitErr error
)

func webgpuInit() {
	webgpuOnce.Do(func() {
		if !webgpu.Available() {
			webgpuInitErr = fmt.Errorf("webgpu: navigator.gpu is not available in this browser")
			return
		}
		dev, err := webgpu.RequestAdapterAndDevice(context.Background())
		if err != nil {
			webgpuInitErr = err
			return
		}
		q4k, err := webgpu.NewQ4KMatvecKernel(dev)
		if err != nil {
			webgpuInitErr = err
			return
		}
		q6k, err := webgpu.NewQ6KMatvecKernel(dev)
		if err != nil {
			webgpuInitErr = err
			return
		}
		webgpuDevice, webgpuQ4K, webgpuQ6K = dev, q4k, q6k
	})
}

// webgpuForceDisabled lets a host application (or the wasm demo harness'
// benchmark page) opt back out of GPU acceleration for an A/B comparison
// against the pure-CPU scalar path on the exact same loaded bytes, without
// needing a second wasm build. Checked before the (otherwise permanent,
// sync.Once-cached) device acquisition, so it must be set before the first
// WebGPUAvailable/model-load call to have any effect on that load.
var webgpuForceDisabled atomic.Bool

// SetWebGPUForceDisabled lets a caller opt out of GPU acceleration (e.g.
// for an A/B throughput comparison against the CPU path) before the next
// model load. Has no effect on weights already prepared by an earlier load.
func SetWebGPUForceDisabled(disabled bool) {
	webgpuForceDisabled.Store(disabled)
}

// WebGPUAvailable reports whether a WebGPU device was successfully
// acquired. The first call performs the (one-time, cached) device
// acquisition and shader compilation; every subsequent call, and every
// prepareWebGPUWeight call, reuses the cached result -- one wasm binary
// transparently runs GPU-accelerated when a device is available and falls
// back to the existing pure-Go scalar path otherwise, with no build-time
// flag or LoadOptions field required.
func WebGPUAvailable() bool {
	if webgpuForceDisabled.Load() {
		return false
	}
	webgpuInit()
	return webgpuInitErr == nil
}

// prepareWebGPUWeight uploads data (Q4_K/Q6_K only -- every other quant
// format has no WGSL kernel yet, see internal/webgpu) to a GPU-resident
// buffer once. It returns nil (falling through to the existing CPU/Metal/
// prepared-quantized path in Weight.MatvecInto) on any failure: no WebGPU
// device available, a tensor too large for this device's
// maxStorageBufferBindingSize (chunking an oversized tensor -- e.g. a tied
// 300MB+ output/embedding projection -- across multiple buffers is not
// implemented yet), or any other error. This mirrors the existing Metal
// precedent's own silent-fallback discipline (matvecMetalQ4KInto etc.).
func prepareWebGPUWeight(data []byte, typ GGMLType, rows, cols int) *GPUWeight {
	if typ != GGMLTypeQ4_K && typ != GGMLTypeQ6_K {
		return nil
	}
	if !WebGPUAvailable() {
		return nil
	}
	if cols%256 != 0 {
		return nil
	}
	prepared, err := webgpuDevice.PrepareWeight(data, rows, cols)
	if err != nil {
		return nil
	}
	return &GPUWeight{prepared: prepared, typ: typ}
}

func matvecWebGPUQ4KInto(w *GPUWeight, x []float32, rows, cols int, out *[]float32) bool {
	return matvecWebGPU(w, GGMLTypeQ4_K, webgpuQ4K, x, rows, out)
}

func matvecWebGPUQ6KInto(w *GPUWeight, x []float32, rows, cols int, out *[]float32) bool {
	return matvecWebGPU(w, GGMLTypeQ6_K, webgpuQ6K, x, rows, out)
}

func matvecWebGPU(w *GPUWeight, wantType GGMLType, kernel *webgpu.MatvecKernel, x []float32, rows int, out *[]float32) bool {
	if w == nil || w.typ != wantType || kernel == nil {
		return false
	}
	result, err := kernel.RunPrepared(w.prepared, x)
	if err != nil {
		return false
	}
	ensureLenNoClear(out, rows)
	copy(*out, result)
	return true
}
