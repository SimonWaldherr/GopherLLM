//go:build darwin && cgo && metal

package gopherllm

import (
	"math"
	"strconv"
	"sync"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func TestMetalGemmaParallelArgmax(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	for _, rows := range []int{16383, 16384, 16417, 262144} {
		t.Run(strconv.Itoa(rows), func(t *testing.T) {
			w := sharedFFNWeight(t, GGMLTypeQ6_K, rows, 256)
			for _, zero := range []bool{false, true} {
				x := randomExpertInput(256, 112)
				if zero {
					clear(x)
				}
				for _, recent := range [][]uint32{nil, {1, 1, 31, 32, 4096, uint32(rows - 1), uint32(rows + 17)}} {
					for _, penalty := range []float32{1, 1.1, 0.9} {
						var expected uint32
						for _, mode := range []string{"0", "1"} {
							t.Setenv("GOPHERLLM_METAL_GEMMA_PARALLEL_ARGMAX", mode)
							var next uint32
							if !metalbackend.GemmaOutput(w, 6, x, nil, recent, penalty, &next) {
								t.Fatal(MetalError())
							}
							if mode == "0" {
								expected = next
							} else if expected != next {
								t.Fatalf("winner=%d, want %d", next, expected)
							}
							if zero && next != 0 {
								t.Fatal("lowest-index zero tie lost")
							}
						}
					}
				}
			}
			x := make([]float32, 256)
			for i := range x {
				x[i] = float32(math.NaN())
			}
			var next uint32
			if metalbackend.GemmaOutput(w, 6, x, nil, nil, 1, &next) {
				t.Fatal("nonfinite winner accepted")
			}
		})
	}
}

func TestMetalGemmaParallelArgmaxConcurrent(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	t.Setenv("GOPHERLLM_METAL_GEMMA_PARALLEL_ARGMAX", "1")
	var workers sync.WaitGroup
	for _, rows := range []int{16384, 16417} {
		w := sharedFFNWeight(t, GGMLTypeQ6_K, rows, 256)
		x := randomExpertInput(256, int64(rows))
		var expected uint32
		if !metalbackend.GemmaOutput(w, 6, x, nil, nil, 1, &expected) {
			t.Fatal(MetalError())
		}
		workers.Add(1)
		go func() {
			defer workers.Done()
			for range 16 {
				var next uint32
				if !metalbackend.GemmaOutput(w, 6, x, nil, nil, 1, &next) || next != expected {
					t.Errorf("concurrent selection got %d, want %d", next, expected)
					return
				}
			}
		}()
	}
	workers.Wait()
}
