package engine

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestQwen35DeltaActivations(t *testing.T) {
	lengths := make([]int, 66)
	for i := range lengths {
		lengths[i] = i
	}
	lengths = append(lengths, 6144, 10240)
	special := []float32{0, math.Float32frombits(1 << 31), 17, -87.33654,
		math.Nextafter32(17, 0), math.Nextafter32(-87.33654, 0),
		math.Float32frombits(1), -math.Float32frombits(1), math.MaxFloat32, -math.MaxFloat32,
		float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	for _, n := range lengths {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			gate, values := randomExpertInput(n, 43907), make([]float32, n+2)
			for i := range gate {
				gate[i] *= 100
				if i%3 == 0 {
					gate[i] = special[(i/3)%len(special)]
				}
				values[i+1] = float32(i%7-3) * .25
			}
			values[0], values[n+1] = 123, 456
			want := make([]float32, n)
			for i, g := range gate {
				want[i] = values[i+1] * (g * fastSigmoidF32(g))
			}
			check := func() {
				t.Helper()
				for i, x := range want {
					got := values[i+1]
					if math.Float32bits(got) != math.Float32bits(x) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(x))) {
						t.Fatalf("element %d got %g, want %g", i, got, x)
					}
				}
				if values[0] != 123 || values[n+1] != 456 {
					t.Fatal("activation guard overwritten")
				}
			}
			qwen35DeltaGate(gate, values[1:n+1])
			check()
			copy(values[1:n+1], gate)
			for i, g := range gate {
				want[i] = g * fastSigmoidF32(g)
			}
			qwen35DeltaSilu(values[1 : n+1])
			check()
		})
	}
}

func TestQwen35DeltaHeadExactStateAndOutput(t *testing.T) {
	for _, n := range []int{0, 1, 7, 16, 32, 64, 127, 128, 129, 256} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			// Unaligned views and guards exercise vector accesses at both ends.
			storage := make([]float32, n*n+2)
			storage[0], storage[len(storage)-1] = 123, 456
			state := storage[1 : len(storage)-1]
			copy(state, randomExpertInput(len(state), 43900))
			want := slices.Clone(state)
			q, k := randomExpertInput(n, 43901), randomExpertInput(n, 43902)
			qwen35L2NormalizeInPlace(q, 1e-6, .125)
			qwen35L2NormalizeInPlace(k, 1e-6, 1)
			v := randomExpertInput(n, 43903)
			qBefore, kBefore, vBefore := slices.Clone(q), slices.Clone(k), slices.Clone(v)
			outputs := make([]float32, n+2)
			outputs[0], outputs[n+1] = 789, 987
			out, reference, recall := outputs[1:n+1], make([]float32, n), make([]float32, n)
			for step := range 12 {
				decay, beta := float32(.97), float32(.43)
				if step >= 8 {
					edge := step - 8
					decay, beta = float32(edge/2), float32(edge%2)
				}
				qwen35DeltaHeadFallback(want, q, k, v, reference, recall, decay, beta)
				qwen35DeltaHead(state, q, k, v, out, recall, decay, beta)
				for i := range state {
					if math.Float32bits(state[i]) != math.Float32bits(want[i]) {
						t.Fatalf("step=%d state[%d]=%g, want %g", step, i, state[i], want[i])
					}
				}
				for i := range out {
					if math.Float32bits(out[i]) != math.Float32bits(reference[i]) {
						t.Fatalf("step=%d output[%d]=%g, want %g", step, i, out[i], reference[i])
					}
				}
			}
			if storage[0] != 123 || storage[len(storage)-1] != 456 || outputs[0] != 789 || outputs[n+1] != 987 {
				t.Fatal("state or output guard overwritten")
			}
			if !slices.Equal(q, qBefore) || !slices.Equal(k, kBefore) || !slices.Equal(v, vBefore) {
				t.Fatal("read-only input changed")
			}
		})
	}
}

func BenchmarkQwen35DeltaHead(b *testing.B) {
	for _, fused := range []bool{false, true} {
		b.Run(fmt.Sprintf("fused=%t", fused), func(b *testing.B) {
			setPerformanceThreads(b, 1)
			const n = 128
			state, q, k := make([]float32, n*n), randomExpertInput(n, 43904), randomExpertInput(n, 43905)
			qwen35L2NormalizeInPlace(q, 1e-6, .125)
			qwen35L2NormalizeInPlace(k, 1e-6, 1)
			v, out, recall := randomExpertInput(n, 43906), make([]float32, n), make([]float32, n)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if fused {
					qwen35DeltaHead(state, q, k, v, out, recall, .97, .43)
				} else {
					qwen35DeltaHeadFallback(state, q, k, v, out, recall, .97, .43)
				}
			}
		})
	}
}
