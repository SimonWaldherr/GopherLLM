package gopherllm

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestDiffusionNormalizationWorkers(t *testing.T) {
	for _, shape := range [][2]int{{32, 16}, {128, 256}, {256, 1024}} {
		channels, spatial := shape[0], shape[1]
		norm := diffusionNorm{weight: make([]float32, channels), bias: make([]float32, channels)}
		for i := range norm.weight {
			norm.weight[i] = .5 + float32(i%7)/10
			norm.bias[i] = float32(i%11-5) / 8
		}
		input := make([]float32, channels*spatial)
		for i := range input {
			input[i] = float32(math.Sin(float64(i)*.1)) + float32(i%13)/9
		}
		for _, activate := range []bool{false, true} {
			want := make([]float32, len(input))
			task := diffusionNormTask{norm, want, input, spatial, activate}
			task.runRows(0, 32)
			for _, workers := range []int{1, 3, 8} {
				for _, inPlace := range []bool{false, true} {
					t.Run(fmt.Sprintf("c%d_s%d_silu%t_workers%d_inplace%t", channels, spatial, activate, workers, inPlace), func(t *testing.T) {
						setPerformanceThreads(t, workers)
						src := slices.Clone(input)
						dst := make([]float32, len(input))
						if inPlace {
							dst = src
						}
						work := diffusionWorkspace{ctx: context.Background()}
						norm.apply(&work, dst, src, spatial, activate)
						for i, v := range want {
							if math.Float32bits(dst[i]) != math.Float32bits(v) {
								t.Fatalf("value %d: %g, want %g", i, dst[i], v)
							}
						}
						if !inPlace && !slices.Equal(src, input) {
							t.Fatal("normalization changed the input")
						}
						if work.norm.src != nil || work.norm.dst != nil {
							t.Fatal("worker retained activation views")
						}
					})
				}
			}
		}
	}
}
