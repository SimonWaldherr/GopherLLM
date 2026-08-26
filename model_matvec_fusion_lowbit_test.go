package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// TestLowBitFusionYieldsToQ8Matvecs protects the dispatch rule used by a
// uniformly Q2_K/Q3_K-compressed Ministral. Generic same-type fusion is still
// useful for exact float activation dots, but must not hide the much faster
// Q8-activation SIMD kernels while they are available.
func TestLowBitFusionYieldsToQ8Matvecs(t *testing.T) {
	const cols = 256
	formats := []struct {
		name string
		typ  GGMLType
		row  func(*rand.Rand, int) []byte
	}{
		{"Q2_K", GGMLTypeQ2_K, randomQ2KRow},
		{"Q3_K", GGMLTypeQ3_K, randomQ3KRow},
	}
	for _, format := range formats {
		t.Run(format.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(71))
			makeWeight := func(rows int) Weight {
				rowBytes, ok := format.typ.DataSize(cols)
				if !ok {
					t.Fatal("missing row size")
				}
				raw := make([]byte, 0, rows*rowBytes)
				for range rows {
					raw = append(raw, format.row(rng, cols)...)
				}
				return Weight{Raw: raw, Type: format.typ, Rows: rows, Cols: cols}
			}
			wq, wk, wv := makeWeight(5), makeWeight(3), makeWeight(3)
			gate, up := makeWeight(7), makeWeight(7)
			x := randomVec(rng, cols)

			withQ8Activations(true, func() {
				// useQ8Activations can be forced on even where its implementation is
				// scalar. Only a real Q8 SIMD backend is worth trading one float
				// projection fusion for separate per-matrix Q8 preprocessing.
				simdQ8 := lowBitQ8SIMDEnabled()
				exactMatvec := func(w Weight) []float32 {
					out := []float32{}
					withQ8Activations(false, func() { w.MatvecInto(x, &out) })
					return out
				}

				q, k, v, sums := []float32{}, []float32{}, []float32{}, []float32{}
				qkvFused := tryMatvec3Into(wq, wk, wv, x, &sums, &q, &k, &v)
				if simdQ8 && qkvFused {
					t.Fatal("SIMD Q8 low-bit QKV fusion must yield to individual matvecs")
				}
				if !simdQ8 && !qkvFused {
					t.Fatal("low-bit QKV fusion must remain active without Q8 SIMD")
				}

				// Exercise the caller's fallback as well as the direct dispatch check.
				q, k, v = []float32{}, []float32{}, []float32{}
				tryMatvecAttentionInto(wq, wk, wv, x, &sums, &q, &k, &v)
				if simdQ8 {
					requireMatvecOutputsClose(t, "q8 q", q, wq.Matvec(x))
					requireMatvecOutputsClose(t, "q8 k", k, wk.Matvec(x))
					requireMatvecOutputsClose(t, "q8 v", v, wv.Matvec(x))
				} else {
					requireMatvecOutputsClose(t, "float q", q, exactMatvec(wq))
					requireMatvecOutputsClose(t, "float k", k, exactMatvec(wk))
					requireMatvecOutputsClose(t, "float v", v, exactMatvec(wv))
				}

				gateOut, upOut := []float32{}, []float32{}
				gateUpFused := tryMatvec2Into(gate, up, x, &sums, &gateOut, &upOut)
				if simdQ8 && gateUpFused {
					t.Fatal("SIMD Q8 low-bit gate/up fusion must yield to individual matvecs")
				}
				if !simdQ8 && !gateUpFused {
					t.Fatal("low-bit gate/up fusion must remain active without Q8 SIMD")
				}
				if simdQ8 {
					gate.MatvecInto(x, &gateOut)
					up.MatvecInto(x, &upOut)
					requireMatvecOutputsClose(t, "q8 gate", gateOut, gate.Matvec(x))
					requireMatvecOutputsClose(t, "q8 up", upOut, up.Matvec(x))
				} else {
					requireMatvecOutputsClose(t, "float gate", gateOut, exactMatvec(gate))
					requireMatvecOutputsClose(t, "float up", upOut, exactMatvec(up))
				}

				// Q8 activation quantization deliberately changes arithmetic. Pin its
				// quality envelope on the actual SIMD path rather than merely proving
				// that the dispatcher calls the same approximation on both sides.
				if simdQ8 {
					for _, projection := range []struct {
						name string
						w    Weight
					}{
						{"q", wq},
						{"k", wk},
						{"v", wv},
						{"gate", gate},
						{"up", up},
					} {
						requireCosine(t, "q8 "+projection.name, exactMatvec(projection.w), projection.w.Matvec(x))
					}
				}
			})

			withQ8Activations(false, func() {
				q, k, v, sums := []float32{}, []float32{}, []float32{}, []float32{}
				if !tryMatvec3Into(wq, wk, wv, x, &sums, &q, &k, &v) {
					t.Fatal("float low-bit QKV fusion declined valid weights")
				}
				requireMatvecOutputsClose(t, "float q", q, wq.Matvec(x))
				requireMatvecOutputsClose(t, "float k", k, wk.Matvec(x))
				requireMatvecOutputsClose(t, "float v", v, wv.Matvec(x))

				gateOut, upOut := []float32{}, []float32{}
				if !tryMatvec2Into(gate, up, x, &sums, &gateOut, &upOut) {
					t.Fatal("float low-bit gate/up fusion declined valid weights")
				}
				requireMatvecOutputsClose(t, "float gate", gateOut, gate.Matvec(x))
				requireMatvecOutputsClose(t, "float up", upOut, up.Matvec(x))
			})
		})
	}
}

func requireMatvecOutputsClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-5*math.Max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}
