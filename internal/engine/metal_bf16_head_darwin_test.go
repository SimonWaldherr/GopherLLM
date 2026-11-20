//go:build darwin && arm64 && cgo && metal

package engine

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMetalBF16HeadMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols = 137, 1024
	x, raw, _ := benchmarkBF16WeightFixture(rows, cols)
	path := filepath.Join(t.TempDir(), "head.bf16")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	check := func(x, want []float32, path string, histories [][]uint32) {
		t.Helper()
		mapping, err := OpenMmap(path)
		if err != nil {
			t.Fatal(err)
		}
		defer mapping.Close()
		if !mapping.IsMapped() {
			t.Fatal("BF16 Metal head requires an OS mapping")
		}
		w := Weight{Raw: mapping.Bytes(), Type: GGMLTypeBF16, Rows: rows, Cols: cols}
		w.Metal = prepareMetalBF16Head(w)
		if w.Metal == nil {
			t.Fatalf("prepare BF16 Metal head: %s", MetalError())
		}
		defer releaseMetalWeight(w.Metal)
		var got []float32
		if !matvecMetalBF16HeadInto(w.Metal, x, &got) {
			t.Fatalf("BF16 projection failed: %s", MetalError())
		}
		for i := range want {
			if math.IsNaN(float64(want[i])) && math.IsNaN(float64(got[i])) {
				continue
			}
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("row %d: Metal %g, CPU %g", i, got[i], want[i])
			}
		}
		for _, h := range histories {
			penalty := float32(1)
			if len(h) > 0 {
				penalty = 1.2
			}
			scored := append([]float32(nil), want...)
			applyRepeatPenalty(scored, h, penalty)
			expected := argmaxFiniteToken(scored)
			token, ok := argmaxMetalBF16Head(w.Metal, x, h, penalty)
			if !ok || token != expected {
				t.Fatalf("history %v: got %d (%t), want %d (%s)", h, token, ok, expected, MetalError())
			}
		}
		cpuWeight := w
		cpuWeight.Metal = nil
		bias := make([]float32, rows)
		bias[12] = 2000
		for _, test := range []struct {
			bias    []float32
			softcap float32
		}{{bias: bias}, {softcap: 4}} {
			config := Config{LogitScale: 1, FinalLogitSoftcap: test.softcap}
			cpu, ok := argmaxOutputTokenPenalizedInto(config, ModelWeights{Output: cpuWeight, OutputBias: test.bias},
				&DecodeBuffer{XN: x}, []uint32{12}, 1.2, nil)
			gpu, gpuOK := argmaxOutputTokenPenalizedInto(config, ModelWeights{Output: w, OutputBias: test.bias},
				&DecodeBuffer{XN: x}, []uint32{12}, 1.2, nil)
			if !ok || !gpuOK || cpu != gpu {
				t.Fatalf("bias/softcap fallback: Metal %d (%t), CPU %d (%t)", gpu, gpuOK, cpu, ok)
			}
		}
	}
	cpu := Weight{Raw: raw, Type: GGMLTypeBF16, Rows: rows, Cols: cols}
	check(x, cpu.Matvec(x), path, [][]uint32{nil, {0, 3, 8, 12, 12, ^uint32(0)}})

	// Equal maxima at rows 1 and 3 select the smaller token unless penalized.
	// A nonfinite row is excluded even when it is the first row.
	for i := range x {
		x[i] = 1
		binary.LittleEndian.PutUint16(raw[2*i:], 0x7f80)
		binary.LittleEndian.PutUint16(raw[2*(cols+i):], 0x3f80)
		binary.LittleEndian.PutUint16(raw[2*(3*cols+i):], 0x3f80)
	}
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	check(x, cpu.Matvec(x), path, [][]uint32{nil, {1, 1}, {1, 3, 1, ^uint32(0)}})
}

func BenchmarkMetalBF16HeadQwen3(b *testing.B) {
	path := os.Getenv("GOPHERLLM_LITTLEBIT_TEST_GGUF")
	if path == "" || !MetalAvailable() {
		b.Skip("set GOPHERLLM_LITTLEBIT_TEST_GGUF and enable Metal")
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true})
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	head := r.standard.Output
	if head.Type != GGMLTypeBF16 || head.Metal == nil || head.Metal.bf16 == nil {
		b.Fatalf("expected borrowed Metal BF16 head: %s", MetalError())
	}
	x := make([]float32, head.Cols)
	for i := range x {
		x[i] = float32(i%17-8) / 8
	}
	want, ok := head.rawScalarArgmaxMatvec(x)
	if !ok {
		b.Fatal("CPU output unavailable")
	}
	got, ok := argmaxMetalBF16Head(head.Metal, x, nil, 1)
	if !ok || got != want {
		b.Fatalf("Metal token %d (%t), CPU %d", got, ok, want)
	}
	var cpuLogits []float32
	if !head.rawScalarMatvecInto(x, &cpuLogits) {
		b.Fatal("CPU logits unavailable")
	}
	gpuLogits := head.Matvec(x)
	for i := range cpuLogits {
		if math.Float32bits(gpuLogits[i]) != math.Float32bits(cpuLogits[i]) {
			b.Fatalf("logit %d: Metal %g, CPU %g", i, gpuLogits[i], cpuLogits[i])
		}
	}
	b.Run("CPU", func(b *testing.B) {
		for b.Loop() {
			head.rawScalarArgmaxMatvec(x)
		}
		b.SetBytes(int64(len(head.Raw)))
		b.ReportAllocs()
	})
	b.Run("Metal", func(b *testing.B) {
		for b.Loop() {
			if _, ok := argmaxMetalBF16Head(head.Metal, x, nil, 1); !ok {
				b.Fatal("Metal failed")
			}
		}
		b.SetBytes(int64(len(head.Raw)))
		b.ReportAllocs()
	})
}
