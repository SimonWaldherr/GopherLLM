//go:build darwin && cgo

package engine

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestVoxtralConvActivationExact(t *testing.T) {
	const channels = 1280
	for _, threads := range []int{1, 8} {
		setPerformanceThreads(t, threads)
		for _, frames := range []int{1, 31, 32} {
			for _, alias := range []bool{false, true} {
				for _, biasLen := range []int{0, 17, channels} {
					t.Run(fmt.Sprintf("threads=%d/frames=%d/alias=%t/bias=%d", threads, frames, alias, biasLen), func(t *testing.T) {
						start, outLen := 3, frames+7
						if alias {
							start, outLen = 0, frames
						}
						out := make([]float32, channels*outLen)
						for i := range out {
							out[i] = -77
						}
						projected := make([]float32, channels*frames)
						if alias {
							projected = out
						}
						values := []float32{-3, -0.1, 0, 0.1, 3, math.SmallestNonzeroFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x80000000), math.Float32frombits(0x7fc01234)}
						for i := range projected {
							projected[i] = values[i%len(values)]
						}
						bias := randomExpertInput(biasLen, 7183)
						want := slices.Clone(out)
						for i, v := range projected {
							channel, frame := i/frames, i%frames
							if channel < len(bias) {
								v += bias[channel]
							}
							want[channel*outLen+start+frame] = geluTanh(v)
						}
						task := voxtralConvActivation{projected: projected, bias: bias, out: out, frames: frames, outLen: outLen, start: start}
						parallelRowsBatchedTask(channels, &task)
						for i, v := range want {
							if math.Float32bits(out[i]) != math.Float32bits(v) {
								t.Fatalf("output[%d]=%08x, want %08x", i, math.Float32bits(out[i]), math.Float32bits(v))
							}
						}
					})
				}
			}
		}
	}
}
