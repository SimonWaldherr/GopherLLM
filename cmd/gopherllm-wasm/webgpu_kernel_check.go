//go:build js

package main

import (
	"context"
	"fmt"
	"math"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/webgpu"
)

// syntheticRow returns a deterministic, non-degenerate float32 row (no
// third-party test-vector source -- a fixed formula, same idea as the
// existing native-Go test suite's smallWeights helper).
func syntheticRow(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(((i*7+seed*13)%23)-11) / 6
	}
	return out
}

// runWebGPUKernelTest quantizes synthetic rows through this project's own
// Q4_K/Q6_K quantizer, computes the reference matvec with the existing
// portable Go kernel, runs the same bytes through the new WGSL kernels, and
// compares. This is the real correctness gate for the hand-written GPU
// shaders: unlike the Phase A wasm harness (which could diff against a
// byte-identical native run), GPU float32 summation order differs from the
// CPU reduction order, so comparison uses a relative-error tolerance, not
// bit-exact equality.
func runWebGPUKernelTest(ctx context.Context) (string, error) {
	dev, err := webgpu.RequestAdapterAndDevice(ctx)
	if err != nil {
		return "", err
	}

	const rows, cols = 8, 512 // 2 Q4_K/Q6_K blocks per row
	x := syntheticRow(cols, 1)

	q4kSummary, err := testQ4K(dev, rows, cols, x)
	if err != nil {
		return "", fmt.Errorf("Q4_K: %w", err)
	}
	q6kSummary, err := testQ6K(dev, rows, cols, x)
	if err != nil {
		return "", fmt.Errorf("Q6_K: %w", err)
	}
	return "PASS: " + q4kSummary + "; " + q6kSummary, nil
}

func testQ4K(dev *webgpu.Device, rows, cols int, x []float32) (string, error) {
	var data []byte
	for r := 0; r < rows; r++ {
		data = append(data, gopherllm.QuantizeRowQ4K(syntheticRow(cols, r+100), cols)...)
	}
	var cpuOut []float32
	gopherllm.MatvecQ4KInto(data, x, rows, cols, &cpuOut)

	kernel, err := webgpu.NewQ4KMatvecKernel(dev)
	if err != nil {
		return "", fmt.Errorf("compiling kernel: %w", err)
	}
	defer kernel.Destroy()
	prepared, err := dev.PrepareWeight(data, rows, cols)
	if err != nil {
		return "", fmt.Errorf("preparing weight: %w", err)
	}
	defer prepared.Destroy()
	gpuOut, err := kernel.RunPrepared(prepared, x)
	if err != nil {
		return "", fmt.Errorf("running kernel: %w", err)
	}
	// The second call is intentional: it exercises the decode-time fast path
	// where the same bind group, GPU readback buffer, and JS upload views are
	// reused after mapAsync/unmap has completed once.
	gpuOutReuse, err := kernel.RunPrepared(prepared, x)
	if err != nil {
		return "", fmt.Errorf("rerunning prepared kernel: %w", err)
	}
	if _, err := compareRows(gpuOut, gpuOutReuse); err != nil {
		return "", fmt.Errorf("prepared reuse mismatch: %w", err)
	}
	// Grow the shared activation buffer, then return to this smaller matrix.
	// This specifically verifies WriteBufferView's explicit size argument: the
	// capacity-sized Uint8Array must not write its stale FFN-sized tail into a
	// smaller projection's input buffer.
	largeCols := cols * 2
	largeX := syntheticRow(largeCols, 1001)
	var largeData []byte
	for r := 0; r < rows; r++ {
		largeData = append(largeData, gopherllm.QuantizeRowQ4K(syntheticRow(largeCols, r+1000), largeCols)...)
	}
	largePrepared, err := dev.PrepareWeight(largeData, rows, largeCols)
	if err != nil {
		return "", fmt.Errorf("preparing larger weight: %w", err)
	}
	defer largePrepared.Destroy()
	if _, err := kernel.RunPrepared(largePrepared, largeX); err != nil {
		return "", fmt.Errorf("running larger prepared kernel: %w", err)
	}
	gpuOutAfterGrow, err := kernel.RunPrepared(prepared, x)
	if err != nil {
		return "", fmt.Errorf("rerunning smaller prepared kernel after growth: %w", err)
	}
	if _, err := compareRows(cpuOut, gpuOutAfterGrow); err != nil {
		return "", fmt.Errorf("smaller prepared result after growth: %w", err)
	}
	maxRelErr, err := compareRows(cpuOut, gpuOut)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("Q4_K max relative error over %d rows = %.6g", rows, maxRelErr), nil
}

func testQ6K(dev *webgpu.Device, rows, cols int, x []float32) (string, error) {
	var data []byte
	for r := 0; r < rows; r++ {
		data = append(data, gopherllm.QuantizeRowQ6K(syntheticRow(cols, r+200), cols)...)
	}
	var cpuOut []float32
	gopherllm.MatvecQ6KInto(data, x, rows, cols, &cpuOut)

	kernel, err := webgpu.NewQ6KMatvecKernel(dev)
	if err != nil {
		return "", fmt.Errorf("compiling kernel: %w", err)
	}
	defer kernel.Destroy()
	prepared, err := dev.PrepareWeight(data, rows, cols)
	if err != nil {
		return "", fmt.Errorf("preparing weight: %w", err)
	}
	defer prepared.Destroy()
	gpuOut, err := kernel.RunPrepared(prepared, x)
	if err != nil {
		return "", fmt.Errorf("running kernel: %w", err)
	}
	gpuOutReuse, err := kernel.RunPrepared(prepared, x)
	if err != nil {
		return "", fmt.Errorf("rerunning prepared kernel: %w", err)
	}
	if _, err := compareRows(gpuOut, gpuOutReuse); err != nil {
		return "", fmt.Errorf("prepared reuse mismatch: %w", err)
	}
	largeCols := cols * 2
	largeX := syntheticRow(largeCols, 2001)
	var largeData []byte
	for r := 0; r < rows; r++ {
		largeData = append(largeData, gopherllm.QuantizeRowQ6K(syntheticRow(largeCols, r+2000), largeCols)...)
	}
	largePrepared, err := dev.PrepareWeight(largeData, rows, largeCols)
	if err != nil {
		return "", fmt.Errorf("preparing larger weight: %w", err)
	}
	defer largePrepared.Destroy()
	if _, err := kernel.RunPrepared(largePrepared, largeX); err != nil {
		return "", fmt.Errorf("running larger prepared kernel: %w", err)
	}
	gpuOutAfterGrow, err := kernel.RunPrepared(prepared, x)
	if err != nil {
		return "", fmt.Errorf("rerunning smaller prepared kernel after growth: %w", err)
	}
	if _, err := compareRows(cpuOut, gpuOutAfterGrow); err != nil {
		return "", fmt.Errorf("smaller prepared result after growth: %w", err)
	}
	maxRelErr, err := compareRows(cpuOut, gpuOut)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("Q6_K max relative error over %d rows = %.6g", rows, maxRelErr), nil
}

func compareRows(cpu, gpu []float32) (float64, error) {
	if len(cpu) != len(gpu) {
		return 0, fmt.Errorf("length mismatch: cpu=%d gpu=%d", len(cpu), len(gpu))
	}
	var maxRelErr float64
	for i := range cpu {
		diff := math.Abs(float64(cpu[i] - gpu[i]))
		denom := math.Max(1e-6, math.Abs(float64(cpu[i])))
		relErr := diff / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > 1e-2 {
			return maxRelErr, fmt.Errorf("row %d: cpu=%v gpu=%v relative error=%v exceeds tolerance", i, cpu[i], gpu[i], relErr)
		}
	}
	return maxRelErr, nil
}
