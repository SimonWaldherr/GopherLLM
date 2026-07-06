//go:build amd64

package gopherllm

import (
	"math/rand"
	"testing"
)

func BenchmarkQuantizeQ8K_4096(b *testing.B) {
	if !hasAVX2 || !hasF16C {
		b.Skip("AVX2+F16C required")
	}
	x := benchFloatSlice(4096)
	q8 := make([]int8, 4096)
	sc := make([]float32, 4096/256)
	b.ReportAllocs()
	b.SetBytes(int64(len(x) * 4))
	for b.Loop() {
		q8kQuantize(&x[0], &q8[0], &sc[0], 4096/256)
	}
}

func BenchmarkDotQ4KQ8KRow_4096(b *testing.B) {
	if !hasAVX2 || !hasF16C {
		b.Skip("AVX2+F16C required")
	}
	rng := rand.New(rand.NewSource(7))
	row := randomQ4KRow(rng, 4096)
	x := randomVec(rng, 4096)
	scratch := []float32{}
	xs := fillQ4KXSums(x, 4096, &scratch)
	q8 := make([]int8, 4096)
	sc := make([]float32, 4096/256)
	q8kQuantize(&x[0], &q8[0], &sc[0], 4096/256)
	b.ReportAllocs()
	b.SetBytes(int64(len(row)))
	for b.Loop() {
		_ = q4kDotQ8KRow(&row[0], &q8[0], &sc[0], &xs[0], 4096/256)
	}
}

func BenchmarkDotQ6KQ8KRow_4096(b *testing.B) {
	if !hasAVX2 || !hasF16C {
		b.Skip("AVX2+F16C required")
	}
	rng := rand.New(rand.NewSource(8))
	row := randomQ6KRow(rng, 4096)
	x := randomVec(rng, 4096)
	scratch := []float32{}
	xs := fillQ6KXSums16(x, 4096, &scratch)
	ScaleF32(xs, 32)
	q8 := make([]int8, 4096)
	sc := make([]float32, 4096/256)
	q8kQuantize(&x[0], &q8[0], &sc[0], 4096/256)
	b.ReportAllocs()
	b.SetBytes(int64(len(row)))
	for b.Loop() {
		_ = q6kDotQ8KRow(&row[0], &q8[0], &sc[0], &xs[0], 4096/256)
	}
}
