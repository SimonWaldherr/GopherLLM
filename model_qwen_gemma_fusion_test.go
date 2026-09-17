package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func projectionTestWeight(rng *rand.Rand, typ GGMLType, rows, cols int) Weight {
	var raw []byte
	for range rows {
		switch typ {
		case GGMLTypeQ8_0:
			raw = append(raw, randomQ8_0Row(rng, cols)...)
		case GGMLTypeQ4_K:
			raw = append(raw, randomQ4KRow(rng, cols)...)
		case GGMLTypeQ6_K:
			raw = append(raw, randomQ6KRow(rng, cols)...)
		}
	}
	return Weight{Type: typ, Raw: raw, Rows: rows, Cols: cols}
}

func TestQ8ProjectionFusion(t *testing.T) {
	for _, cols := range []int{256, 2560} {
		rng := rand.New(rand.NewSource(int64(cols)))
		weights := []Weight{projectionTestWeight(rng, GGMLTypeQ8_0, 65, cols), projectionTestWeight(rng, GGMLTypeQ8_0, 17, cols), projectionTestWeight(rng, GGMLTypeQ8_0, 17, cols)}
		x := randomVec(rng, cols)
		for _, enabled := range []bool{false, true} {
			withQ8Activations(enabled, func() {
				var a, b, c, sums []float32
				if !tryMatvec3Into(weights[0], weights[1], weights[2], x, &sums, &a, &b, &c) {
					t.Fatal("valid QKV declined")
				}
				for i, got := range [][]float32{a, b, c} {
					requireMatvecOutputsClose(t, "QKV", got, weights[i].Matvec(x))
					if enabled {
						var exact []float32
						withQ8Activations(false, func() { exact = weights[i].Matvec(x) })
						requireCosine(t, "Q8 versus float", exact, got)
					}
				}
				if !tryMatvec2Into(weights[0], weights[1], x, &sums, &a, &b) {
					t.Fatal("valid gate/up declined")
				}
				requireMatvecOutputsClose(t, "gate", a, weights[0].Matvec(x))
				requireMatvecOutputsClose(t, "up", b, weights[1].Matvec(x))
			})
		}
	}
}

func TestQ8ProjectionFusionDeclinesWithoutWrites(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	good := projectionTestWeight(rng, GGMLTypeQ8_0, 3, 256)
	for _, kind := range []string{"truncated", "shape", "float", "type", "rows"} {
		bad := good
		switch kind {
		case "truncated":
			bad.Raw = bad.Raw[:len(bad.Raw)-1]
		case "shape":
			bad.Cols = 512
		case "float":
			bad.F32 = []float32{1}
		case "type":
			bad.Type = GGMLTypeQ4_K
		case "rows":
			bad.Rows = -1
		}
		a, b := []float32{42}, []float32{43}
		if matvecQ8_0GroupInto([]Weight{good, bad}, make([]float32, 256), []*[]float32{&a, &b}) {
			t.Fatalf("accepted %s", kind)
		}
		if len(a) != 1 || a[0] != 42 || len(b) != 1 || b[0] != 43 {
			t.Fatalf("wrote outputs for %s", kind)
		}
	}
	// Q8_0 widths below a Q8_K activation block keep exact float fusion.
	withQ8Activations(true, func() {
		w := projectionTestWeight(rng, GGMLTypeQ8_0, 3, 32)
		x := randomVec(rng, 32)
		var a, b, sums []float32
		if !tryMatvec2Into(w, w, x, &sums, &a, &b) {
			t.Fatal("unaligned fallback declined")
		}
		requireMatvecOutputsClose(t, "unaligned", a, w.Matvec(x))
	})
}

func TestGemma4ProjectionFusion(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const cols = 256
	x := randomVec(rng, cols)
	for _, kvType := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		for _, mode := range []string{"separate-v", "k-as-v", "shared-kv"} {
			layer := Gemma4LayerWeights{
				AttnQ:   projectionTestWeight(rng, GGMLTypeQ4_K, 32, cols),
				AttnK:   projectionTestWeight(rng, GGMLTypeQ4_K, 16, cols),
				AttnV:   projectionTestWeight(rng, kvType, 16, cols),
				FFNGate: projectionTestWeight(rng, GGMLTypeQ4_K, 64, cols),
				FFNUp:   projectionTestWeight(rng, kvType, 64, cols),
				HasKV:   mode != "shared-kv", UsesKAsV: mode == "k-as-v",
			}
			buf := DecodeBuffer{K: []float32{42}, V: []float32{43}}
			projectGemma4QKV(layer, x, &buf)
			requireMatvecOutputsClose(t, "gemma q", buf.Q, layer.AttnQ.Matvec(x))
			if layer.HasKV {
				requireMatvecOutputsClose(t, "gemma k", buf.K, layer.AttnK.Matvec(x))
			} else if len(buf.K) != 1 || buf.K[0] != 42 {
				t.Fatal("shared KV overwritten")
			}
			if layer.HasKV && !layer.UsesKAsV {
				requireMatvecOutputsClose(t, "gemma v", buf.V, layer.AttnV.Matvec(x))
			} else if len(buf.V) != 1 || buf.V[0] != 43 {
				t.Fatal("unused V projection written")
			}
			projectGemma4GateUp(layer, x, &buf)
			requireMatvecOutputsClose(t, "gemma gate", buf.Gate, layer.FFNGate.Matvec(x))
			requireMatvecOutputsClose(t, "gemma up", buf.Up, layer.FFNUp.Matvec(x))
			for _, v := range buf.Q {
				if math.IsNaN(float64(v)) {
					t.Fatal("NaN projection")
				}
			}
		}
	}
}
