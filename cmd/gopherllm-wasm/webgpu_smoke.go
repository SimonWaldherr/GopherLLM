//go:build js

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/SimonWaldherr/GopherLLM/internal/webgpu"
)

// smokeShaderWGSL is a trivial hand-written compute shader (double every
// element) used only to validate the internal/webgpu plumbing itself
// (device acquisition, buffer upload, shader compilation, bind group,
// dispatch, readback) before attempting the much harder Q4_K/Q6_K
// dequantizing matvec kernels.
const smokeShaderWGSL = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&input)) {
    output[i] = input[i] * 2.0;
  }
}
`

func runWebGPUSmokeTest(ctx context.Context) (string, error) {
	if !webgpu.Available() {
		return "", fmt.Errorf("navigator.gpu is not available in this browser")
	}
	dev, err := webgpu.RequestAdapterAndDevice(ctx)
	if err != nil {
		return "", err
	}
	limits := dev.Limits()

	const n = 256
	input := make([]float32, n)
	inputBytes := make([]byte, n*4)
	for i := range input {
		input[i] = float32(i)
		binary.LittleEndian.PutUint32(inputBytes[i*4:], math.Float32bits(input[i]))
	}

	inBuf, err := dev.CreateStorageBuffer(len(inputBytes), inputBytes)
	if err != nil {
		return "", fmt.Errorf("creating input buffer: %w", err)
	}
	outBuf, err := dev.CreateStorageBuffer(len(inputBytes), nil)
	if err != nil {
		return "", fmt.Errorf("creating output buffer: %w", err)
	}

	pipeline, err := dev.CreateComputePipeline(smokeShaderWGSL, "main")
	if err != nil {
		return "", fmt.Errorf("compiling shader: %w", err)
	}
	bindGroup := dev.CreateBindGroup(pipeline, inBuf, outBuf)

	enc := dev.NewEncoder()
	enc.BeginCompute()
	enc.Dispatch(pipeline, bindGroup, (n+63)/64, 1, 1)
	enc.EndCompute()
	enc.Submit()

	outBytes, err := dev.ReadBuffer(outBuf, 0, len(inputBytes))
	if err != nil {
		return "", fmt.Errorf("reading back result: %w", err)
	}

	for i := 0; i < n; i++ {
		got := math.Float32frombits(binary.LittleEndian.Uint32(outBytes[i*4:]))
		want := input[i] * 2
		if got != want {
			return "", fmt.Errorf("mismatch at index %d: got %v, want %v", i, got, want)
		}
	}
	return fmt.Sprintf("PASS: %d elements doubled correctly on the GPU; maxStorageBufferBindingSize=%d maxComputeWorkgroupsPerDimension=%d",
		n, limits.MaxStorageBufferBindingSize, limits.MaxComputeWorkgroupsPerDimension), nil
}
