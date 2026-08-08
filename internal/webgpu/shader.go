//go:build js

package webgpu

import (
	"fmt"
	"strings"
	"syscall/js"
)

// ComputePipeline wraps one GPUComputePipeline.
type ComputePipeline struct {
	pipeline js.Value
}

// CreateComputePipeline compiles wgslSource (hand-written WGSL -- see the
// root package's *.wgsl.go files for the actual kernels) and creates a
// compute pipeline with the given entry point, using WebGPU's "auto"
// pipeline layout (bind group layouts are inferred from the shader's own
// @group/@binding declarations). Shader compilation errors are collected
// via getCompilationInfo and returned as a Go error rather than left to
// silently produce a pipeline that does nothing -- with no external WGSL
// reference to check against, this is the primary debugging signal while
// authoring new kernels.
func (d *Device) CreateComputePipeline(wgslSource, entryPoint string) (*ComputePipeline, error) {
	moduleDesc := newObject()
	moduleDesc.Set("code", wgslSource)
	module := d.device.Call("createShaderModule", moduleDesc)

	if err := checkCompilationInfo(module); err != nil {
		return nil, err
	}

	compute := newObject()
	compute.Set("module", module)
	compute.Set("entryPoint", entryPoint)
	pipelineDesc := newObject()
	pipelineDesc.Set("layout", "auto")
	pipelineDesc.Set("compute", compute)
	pipeline := d.device.Call("createComputePipeline", pipelineDesc)
	return &ComputePipeline{pipeline: pipeline}, nil
}

func checkCompilationInfo(module js.Value) error {
	infoVal, err := await(module.Call("getCompilationInfo"))
	if err != nil {
		return fmt.Errorf("webgpu: getCompilationInfo: %w", err)
	}
	messages := infoVal.Get("messages")
	n := messages.Get("length").Int()
	var errs []string
	for i := 0; i < n; i++ {
		m := messages.Index(i)
		if m.Get("type").String() == "error" {
			errs = append(errs, fmt.Sprintf("line %d:%d: %s",
				m.Get("lineNum").Int(), m.Get("linePos").Int(), m.Get("message").String()))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("webgpu: shader compilation failed:\n%s", strings.Join(errs, "\n"))
	}
	return nil
}
