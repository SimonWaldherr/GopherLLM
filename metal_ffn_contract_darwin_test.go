//go:build darwin && cgo && metal

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// Qwen's native 4B MTP checkpoint contracts 9216 FFN channels to a 2560-wide
// model. Preparing that Q6_K tensor must enable fusion without lowering the
// standalone attention-projection crossover.
func TestMetalQ6ContractingFFNFusion(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const dim, hidden = 2560, 9216
	rng := rand.New(rand.NewSource(4_2560_9216))
	weight := func(rows, cols int, typ GGMLType) ([]byte, *MetalWeight) {
		t.Helper()
		patterns := make([][]byte, 32)
		for i := range patterns {
			row := make([]float32, cols)
			for j := range row {
				row[j] = float32(rng.NormFloat64()) * .004
			}
			if typ == GGMLTypeQ4_K {
				patterns[i] = QuantizeRowQ4K(row, cols)
			} else {
				patterns[i] = QuantizeRowQ6K(row, cols)
			}
		}
		raw := make([]byte, rows*len(patterns[0]))
		for i := range rows {
			copy(raw[i*len(patterns[0]):], patterns[i%len(patterns)])
		}
		w := prepareMetalWeight(raw, typ, rows, cols, false)
		if w == nil {
			t.Fatalf("prepare %s %dx%d: %s", typ, rows, cols, MetalError())
		}
		t.Cleanup(func() { releaseMetalWeight(w) })
		return raw, w
	}
	_, gate := weight(hidden, dim, GGMLTypeQ4_K)
	_, up := weight(hidden, dim, GGMLTypeQ4_K)
	raw, down := weight(dim, hidden, GGMLTypeQ6_K)
	if metalWeightUsesDirect(down) || metalWeightMayUseDirect(GGMLTypeQ6_K, dim) {
		t.Fatal("contracting fusion lowered the standalone Q6 threshold")
	}
	if !metalQ6DownSwiGLUWeightsReady(gate, up, down) ||
		!metalSwiGLUBatchWeightsReady(gate, up, down) || !metalQwen35FFNBatchReady(gate, up, down) {
		t.Fatal("prepared contracting Q6 FFN was not eligible")
	}
	input := make([]float32, 8*dim)
	for i := range input {
		input[i] = float32(math.Sin(float64(i)*.271) + .3*math.Cos(float64(i)*.037))
	}
	var direct []float32
	if matvecMetalQ6KInto(down, input[:hidden], down.rows, down.cols, &direct) {
		t.Fatal("standalone Q6 down projection bypassed its direct threshold")
	}
	var legacy []float32
	if !matvecMetalSwiGLUInto(gate, up, down, input[:dim], &legacy) {
		t.Fatal("legacy scalar fusion declined the prepared contracting FFN")
	}
	var generic []float32
	if !matvecMetalSwiGLUBatchInto(gate, up, down, input[:2*dim], 2, &generic) {
		t.Fatal("generic batch fusion declined the prepared contracting FFN")
	}
	for _, flag := range []string{"0", "1"} {
		t.Run("vector"+flag, func(t *testing.T) {
			t.Setenv("GOPHERLLM_METAL_QWEN_FFN_VECTOR", flag)
			for _, n := range []int{1, 2, 3, 4, 8} {
				var got []float32
				if !matvecMetalSwiGLUDecodeBatchInto(gate, up, down, input[:n*dim], n, &got) {
					t.Fatalf("P%d contracting decode batch declined", n)
				}
				for i := range n {
					var want []float32
					if !matvecMetalQwen35SwiGLUInto(gate, up, down, input[i*dim:(i+1)*dim], &want) {
						t.Fatal("contracting scalar FFN declined")
					}
					for j, value := range want {
						if !finite32(got[i*dim+j]) || math.Float32bits(value) != math.Float32bits(got[i*dim+j]) {
							t.Fatalf("P%d position%d output%d differs from scalar FFN", n, i, j)
						}
					}
				}
			}
		})
	}
	stride := len(raw) / dim
	minimum := prepareMetalWeight(raw[:2048*stride], GGMLTypeQ6_K, 2048, hidden, false)
	if minimum == nil {
		t.Fatal("minimum contracting Q6 width was not prepared")
	}
	releaseMetalWeight(minimum)
	for _, shape := range [][2]int{{1792, hidden}, {dim, dim}} {
		if w := prepareMetalWeight(raw, GGMLTypeQ6_K, shape[0], shape[1], false); w != nil {
			releaseMetalWeight(w)
			t.Fatalf("unsupported fusion-only Q6 shape prepared: %v", shape)
		}
	}
	narrowGate, narrowUp, narrowDown := *gate, *up, *down
	narrowGate.rows, narrowUp.rows, narrowDown.cols = 8191, 8191, 8191
	if metalQ6DownSwiGLUWeightsReady(&narrowGate, &narrowUp, &narrowDown) {
		t.Fatal("FFN gate/up direct threshold was lowered")
	}
	t.Log(fmt.Sprintf("prepared %d→%d→%d; standalone Q6 remains CPU", dim, hidden, dim))
}
