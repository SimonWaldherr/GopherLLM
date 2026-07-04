//go:build amd64

package gopherllm

import (
	"math/rand"
	"testing"
)

func BenchmarkQuantizeQ8_4096(b *testing.B) {
	if !hasAVX2 {
		b.Skip("AVX2 required")
	}
	x := benchFloatSlice(4096)
	var q8s []int8
	var scs []float32
	b.ReportAllocs()
	b.SetBytes(int64(len(x) * 4))
	for b.Loop() {
		_, _ = quantizeQ8Into(x, 4096, &q8s, &scs)
	}
}

func BenchmarkDotQ4KQ8Row_4096(b *testing.B) {
	if !hasAVX2 {
		b.Skip("AVX2 required")
	}
	rng := rand.New(rand.NewSource(7))
	row := randomQ4KRow(rng, 4096)
	x := randomVec(rng, 4096)
	scratch := []float32{}
	xs := fillQ4KXSums(x, 4096, &scratch)
	var q8s []int8
	var scs []float32
	q8, xsc := quantizeQ8Into(x, 4096, &q8s, &scs)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = dotQ4KF32Q8WithXSums(row, q8, xsc, xs, 4096)
	}
}
