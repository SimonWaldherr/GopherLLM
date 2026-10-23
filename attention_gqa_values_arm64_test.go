//go:build arm64

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func gqa4WeightedValuesPrior(out, values, scores []float32, dim, stride, span, rows int) {
	for i := range rows {
		axpyF32x4(&out[0], &out[dim], &out[2*dim], &out[3*dim],
			scores[i], scores[span+i], scores[2*span+i], scores[3*span+i], &values[i*stride], dim)
	}
}

func TestGQA4WeightedValuesBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(9437))
	for _, dim := range []int{128} {
		for _, rows := range []int{16, 17, 32, 129, 255, 256, 257, 513, 4096} {
			for _, padding := range []int{0, 17, 896} {
				span, stride := rows+3, dim+padding
				out := make([]float32, 4*dim+2)
				values := make([]float32, (rows-1)*stride+dim+2)
				scores := make([]float32, 4*span+2)
				for _, buf := range [][]float32{out, values, scores} {
					for i := range buf {
						buf[i] = float32(rng.NormFloat64())
					}
				}
				originalValues := append([]float32(nil), values...)
				originalScores := append([]float32(nil), scores...)
				want := append([]float32(nil), out...)
				gqa4WeightedValuesPrior(want[1:len(want)-1], values[1:len(values)-1], scores[1:len(scores)-1], dim, stride, span, rows)
				if !gqa4WeightedValuesF32(out[1:len(out)-1], values[1:len(values)-1], scores[1:len(scores)-1], dim, stride, span, 0, rows) {
					t.Fatal("valid shape rejected")
				}
				for b, pair := range [][2][]float32{{out, want}, {values, originalValues}, {scores, originalScores}} {
					for i := range pair[0] {
						if math.Float32bits(pair[0][i]) != math.Float32bits(pair[1][i]) {
							t.Fatalf("dim=%d rows=%d padding=%d buffer=%d i=%d: got %g want %g", dim, rows, padding, b, i, pair[0][i], pair[1][i])
						}
					}
				}
			}
		}
	}
}

func TestGQA4WeightedValuesBounds(t *testing.T) {
	const dim, stride, span = 128, 1024, 7
	for _, tc := range []struct {
		name                           string
		dim, stride, span, start, rows int
		outLen, valuesLen, scoresLen   int
	}{
		{"dimension", 127, stride, span, 0, span, 4 * dim, span * stride, 4 * span},
		{"negative_dimension", -1, stride, span, 0, span, 4 * dim, span * stride, 4 * span},
		{"stride", dim, dim - 1, span, 0, span, 4 * dim, span * stride, 4 * span},
		{"negative_stride", dim, -1, span, 0, span, 4 * dim, span * stride, 4 * span},
		{"span", dim, stride, span - 1, 0, span, 4 * dim, span * stride, 4 * span},
		{"start", dim, stride, span, -1, span, 4 * dim, span * stride, 4 * span},
		{"start_past_values", dim, stride, span, span + 1, span, 4 * dim, span * stride, 4 * span},
		{"start_overflow", dim, stride, span, int(^uint(0) >> 1), span, 4 * dim, span * stride, 4 * span},
		{"empty_rows", dim, stride, span, 0, 0, 4 * dim, span * stride, 4 * span},
		{"output", dim, stride, span, 0, span, 4*dim - 1, span * stride, 4 * span},
		{"scores", dim, stride, span, 0, span, 4 * dim, span * stride, 4*span - 1},
		{"long_compact_layout", dim, dim, 4097, 0, 4097, 4 * dim, 4097 * dim, 4 * 4097},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := make([]float32, tc.outLen)
			for i := range out {
				out[i] = -17
			}
			if gqa4WeightedValuesF32(out, make([]float32, tc.valuesLen), make([]float32, tc.scoresLen), tc.dim, tc.stride, tc.span, tc.start, tc.rows) {
				t.Fatal("unsupported shape accepted")
			}
			for i, v := range out {
				if v != -17 {
					t.Fatalf("rejected shape wrote output[%d]", i)
				}
			}
		})
	}
}

func TestGQA4WeightedValuesSpecialBits(t *testing.T) {
	const dim, stride, span = 128, 1024, 17
	patterns := [...]uint32{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
		0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00123, 0xffc00456}
	for _, specialScores := range []bool{false, true} {
		out, values, scores := make([]float32, dim*4), make([]float32, span*stride), make([]float32, span*4)
		for i := range out {
			out[i] = math.Float32frombits(patterns[i%len(patterns)])
		}
		for i := range values {
			values[i] = math.Float32frombits(patterns[(i+i/stride)%len(patterns)])
		}
		for i := range scores {
			if specialScores {
				scores[i] = math.Float32frombits(patterns[(i+3)%len(patterns)])
			} else {
				scores[i] = float32(i%9-4) * .125
			}
		}
		want := append([]float32(nil), out...)
		gqa4WeightedValuesPrior(want, values, scores, dim, stride, span, span)
		if !gqa4WeightedValuesF32(out, values, scores, dim, stride, span, 0, span) {
			t.Fatal("valid shape rejected")
		}
		for i := range out {
			if math.Float32bits(out[i]) != math.Float32bits(want[i]) {
				t.Fatalf("specialScores=%t i=%d: got %08x want %08x", specialScores, i, math.Float32bits(out[i]), math.Float32bits(want[i]))
			}
		}
	}
}

func TestGQA4WeightedValuesAdmissionBoundaries(t *testing.T) {
	const dim = 128
	for _, tc := range []struct {
		rows, stride int
		want         bool
	}{
		{1, 1024, false}, {15, 1024, false}, {16, 1024, true},
		{4096, 128, true}, {4097, 128, false},
		{4096, 1024, true}, {4097, 1024, true},
		{4097, 512, false}, {4097, 1025, false},
	} {
		t.Run(fmt.Sprintf("rows=%d/stride=%d", tc.rows, tc.stride), func(t *testing.T) {
			values := benchFloatSlice((tc.rows-1)*tc.stride + dim)
			scores := benchFloatSlice(4 * tc.rows)
			out := benchFloatSlice(4 * dim)
			want := append([]float32(nil), out...)
			if accepted := gqa4WeightedValuesF32(out, values, scores, dim, tc.stride, tc.rows, 0, tc.rows); accepted != tc.want {
				t.Fatalf("accepted=%t, want %t", accepted, tc.want)
			}
			if tc.want {
				gqa4WeightedValuesPrior(want, values, scores, dim, tc.stride, tc.rows, tc.rows)
			}
			for i := range out {
				if math.Float32bits(out[i]) != math.Float32bits(want[i]) {
					t.Fatalf("output[%d] differs: got %g want %g", i, out[i], want[i])
				}
			}
		})
	}
}

func gqa4PriorAttention(queries, keys, values []float32, stride, dim, startT, endT int, scale, softcap float32, out []float32) {
	span := endT - startT + 1
	scores := make([]float32, 4*span)
	n := 0
	for t := startT; t <= endT; t++ {
		offset := t * stride
		if offset+dim > len(keys) {
			break
		}
		a, b, c, d := dotF32x4(&queries[0], &queries[dim], &queries[2*dim], &queries[3*dim], &keys[offset], dim)
		scores[n], scores[span+n], scores[2*span+n], scores[3*span+n] = a*scale, b*scale, c*scale, d*scale
		n++
	}
	if n == 0 {
		return
	}
	var denoms [4]float32
	for h := range 4 {
		denoms[h] = attentionWeightsInPlace(scores[h*span:h*span+n], softcap)
	}
	for i := range n {
		offset := (startT + i) * stride
		if offset+dim > len(values) {
			break
		}
		axpyF32x4(&out[0], &out[dim], &out[2*dim], &out[3*dim], scores[i], scores[span+i], scores[2*span+i], scores[3*span+i], &values[offset], dim)
	}
	for h := range 4 {
		ScaleF32(out[h*dim:(h+1)*dim], 1/denoms[h])
	}
}

func TestGQA4WeightedValuesAttentionParity(t *testing.T) {
	rng := rand.New(rand.NewSource(9438))
	const end = 66
	for _, dim := range []int{31, 64, 128, 129, 256} {
		for _, stride := range []int{dim, 8 * dim} {
			queries := randomVec(rng, 4*dim)
			keys := randomVec(rng, (end+1)*stride)
			values := randomVec(rng, (end+1)*stride)
			for _, start := range []int{0, 3, end} {
				for _, dropRows := range []int{0, 1, 5, end + 1} {
					for _, softcap := range []float32{0, 2.5} {
						got := randomVec(rng, 4*dim+2)
						want := append([]float32(nil), got...)
						v := values[:max(0, len(values)-dropRows*stride-1)]
						k := keys[:len(keys)-stride-1] // Short keys leave n smaller than span.
						gqa4PriorAttention(queries, k, v, stride, dim, start, end, .08838, softcap, want[1:len(want)-1])
						onlineAttentionGroup4(queries, k, v, stride, stride, dim, dim, start, end, .08838, softcap, got[1:len(got)-1])
						for i := range got {
							if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
								t.Fatalf("dim=%d stride=%d start=%d dropRows=%d softcap=%g i=%d: got %g want %g", dim, stride, start, dropRows, softcap, i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

func BenchmarkGQA4WeightedValues(b *testing.B) {
	const dim = 128
	for _, rows := range []int{16, 128, 1024, 4096, 32768} {
		for _, stride := range []int{128, 1024} {
			values := benchFloatSlice(rows * stride)
			scores := benchFloatSlice(rows * 4)
			out := make([]float32, dim*4)
			for _, bulk := range []bool{false, true} {
				b.Run(fmt.Sprintf("rows=%d/stride=%d/bulk=%t", rows, stride, bulk), func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						clear(out)
						if !bulk || !gqa4WeightedValuesF32(out, values, scores, dim, stride, rows, 0, rows) {
							gqa4WeightedValuesPrior(out, values, scores, dim, stride, rows, rows)
						}
					}
				})
			}
		}
	}
}
