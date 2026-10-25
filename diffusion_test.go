package gopherllm

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDiffusionRejectsUnsupportedConfig(t *testing.T) {
	for _, tc := range []struct {
		name, key string
		value     any
	}{
		{"resolution", "sample_size", 2048},
		{"attention_heads", "attention_head_dim", 8},
		{"conditioning", "class_embed_type", "timestep"},
		{"unknown_graph_option", "cross_attention_dim", 128},
		{"missing_required", "norm_eps", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var config map[string]any
			if err := json.Unmarshal([]byte(diffusionCIFARConfig), &config); err != nil {
				t.Fatal(err)
			}
			if tc.value == nil {
				delete(config, tc.key)
			} else {
				config[tc.key] = tc.value
			}
			data, err := json.Marshal(config)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "config.json")
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			if err := diffusionValidateJSON(path, diffusionCIFARConfig, diffusionConfigDefaults); err == nil || !strings.Contains(err.Error(), tc.key) {
				t.Fatalf("unsupported graph option was accepted: %v", err)
			}
		})
	}
}

func TestDiffusionRejectsNonRegularConfig(t *testing.T) {
	if err := diffusionValidateJSON(t.TempDir(), diffusionCIFARConfig, diffusionConfigDefaults); err == nil {
		t.Fatal("accepted a directory as model configuration")
	}
}

func TestDiffusionValidatesTensorBeforeAllocation(t *testing.T) {
	for _, info := range []SafetensorsTensorInfo{
		{DType: SafetensorsF32, Shape: []int{1 << 30, 1 << 30}, Numel: 0},
		{DType: SafetensorsF32, Shape: []int{1}, Numel: -1, End: 4},
		{DType: SafetensorsF32, Shape: []int{1}, Numel: 1, Begin: -4, End: 0},
		{DType: SafetensorsF32, Shape: []int{1}, Numel: 1, End: 8},
		{DType: SafetensorsF16, Shape: []int{1}, Numel: 1, End: 2},
	} {
		loader := diffusionLoader{ctx: context.Background(), file: &SafetensorsFile{
			Tensors: map[string]SafetensorsTensorInfo{"weight": info}, Data: make([]byte, 4)}, used: make(map[string]bool)}
		if got := loader.tensor("weight", 1); got != nil || loader.err == nil {
			t.Fatalf("invalid tensor accepted: %+v", info)
		}
	}
	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data, math.Float32bits(float32(math.Inf(1))))
	loader := diffusionLoader{ctx: context.Background(), file: &SafetensorsFile{
		Tensors: map[string]SafetensorsTensorInfo{"weight": {DType: SafetensorsF32, Shape: []int{1}, Numel: 1, End: 4}}, Data: data}, used: make(map[string]bool)}
	if got := loader.tensor("weight", 1); got != nil || loader.err == nil {
		t.Fatal("non-finite checkpoint accepted")
	}
}

func TestDiffusionRejectsNonFiniteTensorTails(t *testing.T) {
	// Exercise both the vector body and scalar tail of the shared finite scan.
	const count = 33
	data := make([]byte, count*4)
	for _, position := range []int{0, 15, 16, 31, 32} {
		for _, bits := range []uint32{0x7f800000, 0xff800000, 0x7fc00001} {
			clear(data)
			binary.LittleEndian.PutUint32(data[position*4:], bits)
			loader := diffusionLoader{ctx: context.Background(), file: &SafetensorsFile{
				Tensors: map[string]SafetensorsTensorInfo{"weight": {DType: SafetensorsF32, Shape: []int{count}, Numel: count, End: len(data)}}, Data: data}, used: make(map[string]bool)}
			if got := loader.tensor("weight", count); got != nil || loader.err == nil || loader.used["weight"] {
				t.Fatalf("non-finite value accepted at %d (%08x)", position, bits)
			}
		}
	}
}

func TestDiffusionDownsamplePadding(t *testing.T) {
	x := diffusionTensor{c: 1, h: 4, w: 4, data: make([]float32, 16)}
	for i := range x.data {
		x.data[i] = float32(i + 1)
	}
	conv := diffusionConv{in: 1, out: 1, kernel: 3, weight: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}, bias: []float32{0}}
	workspace := diffusionWorkspace{ctx: context.Background()}
	got := conv.forward(&workspace, x, 2, 0)
	if got.h != 2 || got.w != 2 {
		t.Fatalf("downsample shape: %dx%d", got.h, got.w)
	}
	for i, want := range []float32{54, 45, 72, 54} {
		if got.data[i] != want {
			t.Fatalf("downsample[%d]=%g, want %g (right/bottom padding only)", i, got.data[i], want)
		}
	}
}

func TestDiffusionDDIMClipsAndFinishesAtOriginalSample(t *testing.T) {
	sample := []float32{2, -2, 0.25}
	diffusionDDIMStep(sample, []float32{0, 0, 0}, 0.25, 1)
	for i, want := range []float32{1, -1, 0.5} {
		if sample[i] != want {
			t.Fatalf("final sample[%d]=%g, want %g", i, sample[i], want)
		}
	}
}

func TestDiffusionCancellationAndClosedModel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := OpenDiffusion(ctx, "missing-model"); !errors.Is(err, context.Canceled) {
		t.Fatalf("load did not honor cancellation: %v", err)
	}
	model := &DiffusionModel{gate: make(chan struct{}, 1)}
	model.gate <- struct{}{}
	if _, err := model.Generate(ctx, DiffusionOptions{Steps: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("queued generation did not honor cancellation: %v", err)
	}
	<-model.gate
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Generate(context.Background(), DiffusionOptions{}); err == nil {
		t.Fatal("closed model accepted generation")
	}
	for _, steps := range []int{-1, 1001} {
		if _, err := model.Generate(context.Background(), DiffusionOptions{Steps: steps}); err == nil || !strings.Contains(err.Error(), "steps") {
			t.Fatalf("invalid steps accepted: %d, %v", steps, err)
		}
	}
}

func diffusionReference(t testing.TB) *SafetensorsFile {
	t.Helper()
	data, err := os.ReadFile("testdata/diffusion_cifar10.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	file, err := ParseSafetensors(data)
	if err != nil {
		t.Fatal(err)
	}
	return file
}

func TestDiffusionScheduleMatchesDiffusers(t *testing.T) {
	want, err := diffusionReference(t).F32("alphas")
	if err != nil {
		t.Fatal(err)
	}
	got := diffusionAlphas()
	for i, value := range got {
		if math.Abs(float64(value-want[i])) > 2e-7 {
			t.Fatalf("alpha[%d]=%.9g, reference %.9g", i, value, want[i])
		}
	}
}

func diffusionTestModel(t testing.TB) *DiffusionModel {
	t.Helper()
	path := os.Getenv("GOPHERLLM_DIFFUSION_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_DIFFUSION_MODEL to a local google/ddpm-cifar10-32 directory")
	}
	model, err := OpenDiffusion(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = model.Close() })
	return model
}

func diffusionCompare(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("sample length %d, want %d", len(got), len(want))
	}
	maxError, squaredError := float64(0), float64(0)
	for i := range got {
		error := math.Abs(float64(got[i] - want[i]))
		if math.IsNaN(error) || math.IsInf(error, 0) {
			t.Fatalf("non-finite output[%d]", i)
		}
		maxError = max(maxError, error)
		squaredError += error * error
	}
	t.Logf("max error %.7g, RMSE %.7g", maxError, math.Sqrt(squaredError/float64(len(got))))
	if maxError > tolerance {
		t.Fatalf("maximum error %g exceeds %g", maxError, tolerance)
	}
}

// This opt-in test uses the original 143 MB checkpoint, not a tiny synthetic
// graph. The checked-in fixture was generated by scripts/diffusion-reference.py
// using PyTorch 2.8.0 and Diffusers 0.35.1 on CPU. It includes complete UNet
// outputs and complete DDIM trajectories from identical input noise, including
// step counts which do not divide the 1000 training timesteps.
func TestDiffusionCIFARReference(t *testing.T) {
	model := diffusionTestModel(t)
	fixture := diffusionReference(t)
	input, err := fixture.F32("input")
	if err != nil {
		t.Fatal(err)
	}
	for _, timestep := range []int{0, 500, 999} {
		t.Run(fmt.Sprintf("noise_%d", timestep), func(t *testing.T) {
			workspace := diffusionWorkspace{ctx: context.Background()}
			got, err := model.net.forward(&workspace, input, timestep)
			if err != nil {
				t.Fatal(err)
			}
			want, err := fixture.F32(fmt.Sprintf("noise_%d", timestep))
			if err != nil {
				t.Fatal(err)
			}
			diffusionCompare(t, got, want, 3e-4)
			bytes := len(workspace.columns) * 4
			for _, buffer := range workspace.buffers {
				bytes += cap(buffer) * 4
			}
			t.Logf("workspace %.2f MiB", float64(bytes)/(1<<20))
		})
	}
	for _, steps := range []int{4, 7, 23, 50} {
		t.Run(fmt.Sprintf("ddim_%d", steps), func(t *testing.T) {
			got := append([]float32(nil), input...)
			if err := model.net.sample(context.Background(), got, steps); err != nil {
				t.Fatal(err)
			}
			want, err := fixture.F32(fmt.Sprintf("ddim_%d", steps))
			if err != nil {
				t.Fatal(err)
			}
			diffusionCompare(t, got, want, 1e-3)
		})
	}
}

func TestDiffusionCIFARGenerate(t *testing.T) {
	model := diffusionTestModel(t)
	first, err := model.Generate(context.Background(), DiffusionOptions{Seed: 7, Steps: 2})
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.Generate(context.Background(), DiffusionOptions{Seed: 7, Steps: 2})
	if err != nil {
		t.Fatal(err)
	}
	if first.Bounds() != image.Rect(0, 0, 32, 32) || !bytes.Equal(first.(*image.NRGBA).Pix, second.(*image.NRGBA).Pix) {
		t.Fatal("sampling is not reproducible or returned the wrong image dimensions")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if _, err := model.Generate(ctx, DiffusionOptions{Steps: 1000}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("active generation ignored cancellation: %v", err)
	}
}

func BenchmarkDiffusionCIFARStep(b *testing.B) {
	model := diffusionTestModel(b)
	input, err := diffusionReference(b).F32("input")
	if err != nil {
		b.Fatal(err)
	}
	workspace := diffusionWorkspace{ctx: context.Background()}
	if _, err := model.net.forward(&workspace, input, 500); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := model.net.forward(&workspace, input, 500); err != nil {
			b.Fatal(err)
		}
	}
}
