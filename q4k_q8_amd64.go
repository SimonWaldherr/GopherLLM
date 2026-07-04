//go:build amd64

package main

import (
	"math"
	"os"
	"sync"
)

// useQ8Activations enables the opt-in int8-activation Q4_K matvec path. It
// quantizes activations to int8 per 32-element block and uses VPMADDUBSW integer
// dot products, which are ~1.6x faster than the float kernel at the cost of a
// small quantization error. Off by default; enable with GOPHERLLM_Q8_ACTIVATIONS.
var useQ8Activations = hasAVX2 && os.Getenv("GOPHERLLM_Q8_ACTIVATIONS") != ""

var (
	q8ScratchPool     = sync.Pool{New: func() any { s := make([]int8, 0, 4096); return &s }}
	xscaleScratchPool = sync.Pool{New: func() any { s := make([]float32, 0, 128); return &s }}
)

// quantizeQ8Into quantizes x to int8 per 32-element block (symmetric absmax)
// into q8Scratch, writing the per-block scales into scaleScratch.
func quantizeQ8Into(x []float32, cols int, q8Scratch *[]int8, scaleScratch *[]float32) ([]int8, []float32) {
	blocks := cols / 32
	ensureLenNoClear(q8Scratch, cols)
	ensureLenNoClear(scaleScratch, blocks)
	q8 := *q8Scratch
	sc := *scaleScratch
	for b := 0; b < blocks; b++ {
		base := b * 32
		xb := x[base : base+32]
		_ = xb[31]
		var amax float32
		for i := 0; i < 32; i++ {
			v := xb[i]
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}
		if amax == 0 {
			sc[b] = 0
			for i := 0; i < 32; i++ {
				q8[base+i] = 0
			}
			continue
		}
		scale := amax / 127
		sc[b] = scale
		inv := 1 / scale
		for i := 0; i < 32; i++ {
			q8[base+i] = int8(math.Round(float64(xb[i] * inv)))
		}
	}
	return q8, sc
}

// acquireQ8 quantizes x once (shared across all rows/matrices of a matvec) and
// returns the int8 activations, per-block scales, and a release func that
// returns the scratch buffers to their pools.
func acquireQ8(x []float32, cols int) (q8 []int8, xscale []float32, release func()) {
	q8s := q8ScratchPool.Get().(*[]int8)
	scs := xscaleScratchPool.Get().(*[]float32)
	q8, xscale = quantizeQ8Into(x, cols, q8s, scs)
	return q8, xscale, func() {
		*q8s = q8
		q8ScratchPool.Put(q8s)
		*scs = xscale
		xscaleScratchPool.Put(scs)
	}
}

// dotQ4KF32Q8WithXSums is the int8-activation analogue of
// dotQ4KF32SIMDWithXSums. The min term keeps the exact float xsums; only the
// (weight nibble · activation) term uses int8, rescaled per block by xscale.
func dotQ4KF32Q8WithXSums(row []byte, q8 []int8, xscale, xsums []float32, cols int) float32 {
	var intqdots [8]int32
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		block := row[base : base+144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q4kQDots8Q8(&block[16], &q8[b*256], &intqdots[0])
		for step := 0; step < 4; step++ {
			is := step * 2
			var sc1, m1, sc2, m2 byte
			if is < 4 {
				sc1, m1 = scales[is]&63, scales[is+4]&63
				sc2, m2 = scales[is+1]&63, scales[is+5]&63
			} else {
				sc1, m1 = (scales[is+4]&0x0f)|((scales[is-4]>>6)<<4), (scales[is+4]>>4)|((scales[is]>>6)<<4)
				sc2, m2 = (scales[is+5]&0x0f)|((scales[is-3]>>6)<<4), (scales[is+5]>>4)|((scales[is+1]>>6)<<4)
			}
			xb := b*8 + is
			sum += d*float32(sc1)*xscale[xb]*float32(intqdots[is]) - dmin*float32(m1)*xsums[xb]
			sum += d*float32(sc2)*xscale[xb+1]*float32(intqdots[is+1]) - dmin*float32(m2)*xsums[xb+1]
		}
	}
	return sum
}

func dotQ4KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = dotQ4KF32Q8WithXSums(data[off:off+rowBytes], q8, xscale, xsums, cols)
	}
}
