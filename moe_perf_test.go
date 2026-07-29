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
			case GGMLTypeMXFP4:
				// MXFP4's exponent is the last byte of each 32-value block.
				// Keep scale at 2^(127-127)=1 so random mantissas stay finite.
				row[off+16] = 127
				off += 17
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
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeMXFP4} {
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

func TestExpertMatvecPairQuantPlanesMatchRowDot(t *testing.T) {
	const (
		input, output, experts = 512, 96, 3
	)
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		t.Run(typ.String(), func(t *testing.T) {
			gate := quantExpertWeightForTest(typ, input, output, experts, int64(700+typ))
			up := quantExpertWeightForTest(typ, input, output, experts, int64(800+typ))
			x := randomExpertInput(input, int64(900+typ))
			for _, expert := range []int{0, experts - 1} {
				wantGate := expertMatvecRowDotReference(gate, expert, x)
				wantUp := expertMatvecRowDotReference(up, expert, x)
				var gotGate, gotUp, sums []float32
				withQ8Activations(false, func() {
					if !expertMatvec2Into(gate, up, expert, x, &sums, &gotGate, &gotUp) {
						t.Fatalf("expert pair path declined %s", typ)
					}
				})
				for _, pair := range []struct {
					name string
					got  []float32
					want []float32
				}{
					{"gate", gotGate, wantGate},
					{"up", gotUp, wantUp},
				} {
					if len(pair.got) != len(pair.want) {
						t.Fatalf("expert %d %s length = %d, want %d", expert, pair.name, len(pair.got), len(pair.want))
					}
					for i := range pair.want {
						diff := math.Abs(float64(pair.got[i] - pair.want[i]))
						limit := 1e-2 * math.Max(1, math.Abs(float64(pair.want[i])))
						if diff > limit {
							t.Fatalf("expert %d %s row %d = %v, want %v (diff %g, limit %g)", expert, pair.name, i, pair.got[i], pair.want[i], diff, limit)
						}
					}
				}
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

func BenchmarkExpertMatvecMXFP4_1024x1024(b *testing.B) {
	benchmarkExpertMatvec(b, GGMLTypeMXFP4)
}

func benchmarkSparseMoEWeights(dim, hidden, experts, used int) *SparseMoEWeights {
	rng := rand.New(rand.NewSource(501))
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = (rng.Float32()*2 - 1) * 0.05
		}
		return v
	}
	return &SparseMoEWeights{
		Router:        Weight{F32: fill(experts * dim)},
		Gate:          ExpertWeight{Weight: Weight{F32: fill(experts * hidden * dim)}, Input: dim, Output: hidden, Experts: experts},
		Up:            ExpertWeight{Weight: Weight{F32: fill(experts * hidden * dim)}, Input: dim, Output: hidden, Experts: experts},
		Down:          ExpertWeight{Weight: Weight{F32: fill(experts * dim * hidden)}, Input: hidden, Output: dim, Experts: experts},
		NormalizeTopK: true,
		Scale:         1,
		ExpertUsed:    used,
	}
}

// BenchmarkSparseMoEForward measures the complete decode-time MoE branch:
// router top-k, expert SwiGLUs, and weighted aggregation. 128/6 matches the
// common high-expert deployment shape while the 256-wide synthetic matrices
// keep the benchmark suitable for normal developer machines.
func BenchmarkSparseMoEForward_E128_K6_256x256(b *testing.B) {
	const dim, hidden, experts, used = 256, 256, 128, 6
	w := benchmarkSparseMoEWeights(dim, hidden, experts, used)
	x := randomExpertInput(dim, 502)
	buf := NewDecodeBuffer(Config{Dim: dim, HiddenDim: hidden, ExpertCount: experts}, dim, 1, dim)
	sparseMoEForward(w, x, buf) // warm scratch capacity before allocation metrics
	b.ReportAllocs()
	// Three selected expert matrices are streamed for each active expert.
	b.SetBytes(int64(3 * used * dim * hidden * 4))
	b.ResetTimer()
	for b.Loop() {
		sparseMoEForward(w, x, buf)
	}
}

// BenchmarkSparseMoEForward_DeepSeekV3 exercises the group-limited noaux
// route at V3's production router geometry (256 experts, 8 groups, 4 kept,
// 8 active). The matrices stay compact so it measures routing overhead and
// reusable group-router scratch on ordinary developer machines.
func BenchmarkSparseMoEForward_DeepSeekV3_E256_G8_K8_256x256(b *testing.B) {
	const dim, hidden, experts, used, groups, groupsUsed = 256, 256, 256, 8, 8, 4
	w := benchmarkSparseMoEWeights(dim, hidden, experts, used)
	w.RoutingSigmoid = true
	w.RouterCorrectionBias = make([]float32, experts)
	w.GroupCount = groups
	w.GroupUsed = groupsUsed
	w.Scale = 2.5
	x := randomExpertInput(dim, 503)
	buf := NewDecodeBuffer(Config{
		Dim: dim, HiddenDim: hidden, ExpertCount: experts, ExpertUsedCount: used,
		ExpertGroupCount: groups, ExpertGroupUsedCount: groupsUsed,
	}, dim, 1, dim)
	sparseMoEForward(w, x, buf) // warm scratch capacity before allocation metrics
	b.ReportAllocs()
	b.SetBytes(int64(3 * used * dim * hidden * 4))
	b.ResetTimer()
	for b.Loop() {
		sparseMoEForward(w, x, buf)
	}
}
