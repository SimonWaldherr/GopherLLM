package numeric

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestAllFiniteF32Boundaries(t *testing.T) {
	finiteBits := []uint32{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff}
	nonfiniteBits := []uint32{0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000, 0xff800001, 0xffc12345}
	for _, size := range []int{0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 97, 257} {
		for offset := 0; offset < 4; offset++ {
			storage := make([]float32, size+32)
			for i := range storage {
				storage[i] = float32(math.Inf(1))
			}
			values := storage[16+offset : 16+offset+size]
			for i := range values {
				values[i] = math.Float32frombits(finiteBits[i%len(finiteBits)])
			}
			if !AllFiniteF32(values) {
				t.Fatalf("finite values rejected or slice boundary crossed: size=%d offset=%d", size, offset)
			}
			for i := range values {
				original := values[i]
				for _, bits := range nonfiniteBits {
					values[i] = math.Float32frombits(bits)
					if AllFiniteF32(values) {
						t.Fatalf("non-finite value accepted: size=%d offset=%d index=%d bits=%08x", size, offset, i, bits)
					}
					if math.Float32bits(values[i]) != bits {
						t.Fatal("validation changed its input")
					}
				}
				values[i] = original
			}
		}
	}
}

func TestAllFiniteF32RandomBits(t *testing.T) {
	rng := rand.New(rand.NewSource(81734))
	for range 1000 {
		values := make([]float32, rng.Intn(2049))
		want := true
		for i := range values {
			values[i] = math.Float32frombits(rng.Uint32())
			v := float64(values[i])
			want = want && !math.IsNaN(v) && !math.IsInf(v, 0)
		}
		if got := AllFiniteF32(values); got != want {
			t.Fatalf("random bit classification: got %v, want %v (length %d)", got, want, len(values))
		}
		if got := allFiniteF32Generic(values); got != want {
			t.Fatalf("portable classification: got %v, want %v", got, want)
		}
	}
}

func BenchmarkAllFiniteF32(b *testing.B) {
	for _, size := range []int{768, 32 * 3072, 32 * 5120} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			values := make([]float32, size)
			for i := range values {
				values[i] = float32(i%31-15) * .125
			}
			b.ReportAllocs()
			b.SetBytes(int64(size * 4))
			b.ResetTimer()
			for range b.N {
				if !AllFiniteF32(values) {
					b.Fatal("finite input rejected")
				}
			}
		})
	}
}
