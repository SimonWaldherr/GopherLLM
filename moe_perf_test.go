package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// quantExpertWeightForTest builds a rank-3 expert tensor in GGUF's
// [input, output, expert] order. Weight.Rows intentionally remains Output,
// matching loadExpertWeight: expertMatvecInto must select the correct raw
// plane itself.
func quantExpertWeightForTest(typ GGMLType, input, output, experts int, seed int64) ExpertWeight {
	rowBytes, ok := typ.DataSize(input)
	if !ok {
		panic("unsupported quantized expert test type")
	}
	rng := rand.New(rand.NewSource(seed))
	raw := make([]byte, 0, experts*output*rowBytes)
	for range experts * output {
		row := make([]byte, rowBytes)
		for i := range row {
			row[i] = byte(rng.Intn(256))
		}
		for off := 0; off < len(row); {
			switch typ {
			case GGMLTypeQ4_K:
				// Keep both f16 scales finite and non-zero.
				row[off+1] = 0x2c
				row[off+3] = 0x1c
				off += 144
			case GGMLTypeQ6_K:
				// Q6_K stores its f16 block scale at the end.
				row[off+209] = 0x2c
				off += 210
			default:
				panic("unsupported quantized expert test type")
			}
		}
		raw = append(raw, row...)
	}
	return ExpertWeight{
		Weight:  Weight{Raw: raw, Type: typ, Rows: output, Cols: input},
		Input:   input,
		Output:  output,
		Experts: experts,
	}
}

func randomExpertInput(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	return x
}

func expertMatvecRowDotReference(w ExpertWeight, expert int, x []float32) []float32 {
	out := make([]float32, w.Output)
	row := make([]float32, w.Input)
	for r := range w.Output {
		w.Weight.RowInto(expert*w.Output+r, w.Input, &row)
		out[r] = DotF32(row, x)
	}
	return out
}

func TestExpertMatvecQuantPlanesMatchRowDot(t *testing.T) {
	const (
		input, output, experts = 512, 128, 3
	)
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		t.Run(typ.String(), func(t *testing.T) {
			w := quantExpertWeightForTest(typ, input, output, experts, int64(100+typ))
			x := randomExpertInput(input, int64(200+typ))
			for _, expert := range []int{0, experts - 1} {
				t.Run(map[bool]string{true: "first", false: "last"}[expert == 0], func(t *testing.T) {
					want := expertMatvecRowDotReference(w, expert, x)
					var got, row []float32
					// The reference uses exact f32 activations, so do the same on
					// AMD64 rather than comparing against its approximate Q8 path.
					withQ8Activations(false, func() {
						expertMatvecInto(w, expert, x, &got, &row)
					})
					if len(got) != len(want) {
						t.Fatalf("expert %d output length = %d, want %d", expert, len(got), len(want))
					}
					for i := range want {
						diff := math.Abs(float64(got[i] - want[i]))
						limit := 1e-2 * math.Max(1, math.Abs(float64(want[i])))
						if diff > limit {
							t.Fatalf("expert %d row %d = %v, want %v (diff %g, limit %g)", expert, i, got[i], want[i], diff, limit)
						}
					}
				})
			}
		})
	}
}

func benchmarkExpertMatvec(b *testing.B, typ GGMLType) {
	const (
		input, output, experts = 1024, 1024, 8
	)
	w := quantExpertWeightForTest(typ, input, output, experts, int64(300+typ))
	x := randomExpertInput(input, int64(400+typ))
	rowBytes, ok := typ.DataSize(input)
	if !ok {
		b.Fatal("missing row size")
	}
	var out, row []float32
	b.ReportAllocs()
	b.SetBytes(int64(output*rowBytes + input*4))
	b.ResetTimer()
	for b.Loop() {
		expertMatvecInto(w, experts-1, x, &out, &row)
	}
}

func BenchmarkExpertMatvecQ4K_1024x1024(b *testing.B) {
	benchmarkExpertMatvec(b, GGMLTypeQ4_K)
}

func BenchmarkExpertMatvecQ6K_1024x1024(b *testing.B) {
	benchmarkExpertMatvec(b, GGMLTypeQ6_K)
}
