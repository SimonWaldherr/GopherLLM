package yolo

import (
	"math"
	"sync"

	"github.com/SimonWaldherr/GopherLLM/internal/threads"
)

// The YOLO executor currently owns these small float32 helpers. They keep the
// vision package independent of the root inference runtime; shared kernel
// extraction can replace them with internal/compute in a later migration.
func parallelChunks(n int, fn func(start, end int)) {
	if n <= 0 {
		fn(0, n)
		return
	}
	workers := min(threads.Count(), n)
	if workers <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	wg.Add(workers - 1)
	for worker := 1; worker < workers; worker++ {
		start, end := n*worker/workers, n*(worker+1)/workers
		go func() {
			defer wg.Done()
			fn(start, end)
		}()
	}
	fn(0, n/workers)
	wg.Wait()
}

func yoloAxpy(out []float32, alpha float32, x []float32) {
	for i := 0; i < min(len(out), len(x)); i++ {
		out[i] += alpha * x[i]
	}
}

const (
	yoloInvLn2 = 1.4426950408889634
	yoloLn2    = 0.6931471805599453
	yoloExpHi  = 88.02969
	yoloExpLo  = -87.33654
)

func yoloFastExp(x float32) float32 {
	if x > yoloExpHi {
		return float32(math.Inf(1))
	}
	if x < yoloExpLo {
		return 0
	}
	t := x * yoloInvLn2
	var k int32
	if t >= 0 {
		k = int32(t + 0.5)
	} else {
		k = int32(t - 0.5)
	}
	r := float32(float64(x) - float64(k)*yoloLn2)
	p := 1 + r*(1+r*(0.5+r*(1.0/6+r*(1.0/24+r*(1.0/120+r*(1.0/720))))))
	return p * math.Float32frombits(uint32(127+k)<<23)
}

func yoloFastSigmoid(x float32) float32 {
	if x >= 17 {
		return 1
	}
	if x <= yoloExpLo {
		return 0
	}
	return 1 / (1 + yoloFastExp(-x))
}
