package engine

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

// Independent oracle for the original, separately dispatched shared branch.
func serialMoESharedForward(w *SparseMoEWeights, x []float32, buf *DecodeBuffer) {
	w.SharedGate.MatvecInto(x, &buf.Gate)
	w.SharedUp.MatvecInto(x, &buf.Up)
	ensureLenNoClear(&buf.Hidden, len(buf.Gate))
	siluMulF32(buf.Gate, buf.Up, buf.Hidden)
	w.SharedDown.MatvecInto(buf.Hidden, &buf.MOE)
	if !w.SharedAlways {
		w.SharedGateIn.MatvecInto(x, &buf.RouterLogits)
		ScaleF32(buf.MOE, nemotronSigmoid(buf.RouterLogits[0]))
	}
	addInPlace(buf.Proj[:w.Down.Output], buf.MOE)
}

func TestMoESharedProjectionPreservesMixture(t *testing.T) {
	setPerformanceThreads(t, 3)
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
		for _, q8 := range []bool{false, true} {
			for _, always := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/q8=%t/always=%t", typ, q8, always), func(t *testing.T) {
					if q8 && !q8ActivationsAvailable() {
						t.Skip("Q8 activation instructions unavailable")
					}
					withQ8Activations(q8, func() {
						weight := func(input, output int) Weight { return moeDownWeightForTest(typ, input, output, 1).Weight }
						gate, up, down := weight(256, 512), weight(256, 512), weight(512, 256)
						gateIn := Weight{F32: randomExpertInput(256, 41200)}
						w := &SparseMoEWeights{SharedGate: &gate, SharedUp: &up, SharedDown: &down, SharedGateIn: &gateIn, SharedAlways: always, Down: ExpertWeight{Output: 256}}
						x, prior := randomExpertInput(256, 41201), randomExpertInput(256, 41202)
						got, want := &DecodeBuffer{Proj: slices.Clone(prior)}, &DecodeBuffer{Proj: slices.Clone(prior)}
						sparseMoESharedForward(w, x, got)
						serialMoESharedForward(w, x, want)
						for i := range prior {
							if math.Float32bits(got.Proj[i]) != math.Float32bits(want.Proj[i]) {
								t.Fatalf("output %d differs: %g vs %g", i, got.Proj[i], want.Proj[i])
							}
						}
					})
				})
			}
		}
	}
}

func TestMoESharedFusionPreservesPreparedPath(t *testing.T) {
	w, _ := expertPlaneWeight(moeDownWeightForTest(GGMLTypeQ4_K, 256, 512, 1), 0)
	prepared := w
	prepared.Prepared = &PreparedQuantizedWeight{}
	for _, pair := range [][2]Weight{{prepared, w}, {w, prepared}} {
		buf := &DecodeBuffer{Gate: []float32{1}, Up: []float32{2}}
		if moeSharedGateUpInto(pair[0], pair[1], make([]float32, 256), buf) || !slices.Equal(buf.Gate, []float32{1}) || !slices.Equal(buf.Up, []float32{2}) {
			t.Fatal("fusion changed prepared projection path")
		}
	}
}

func TestMoEConfiguredSharedParity(t *testing.T) {
	r := loadMoEPerformanceModel(t)
	if r.kind != loadedQwen35 {
		t.Skip("requires Qwen3.5-MoE")
	}
	checked, fused := 0, 0
	for index, layer := range r.qwen35.Layers {
		w := layer.FFN.MoE
		if w == nil || w.SharedGate == nil {
			continue
		}
		x := randomExpertInput(w.Gate.Input, int64(41300+index))
		buf := &DecodeBuffer{}
		if moeSharedGateUpInto(*w.SharedGate, *w.SharedUp, x, buf) {
			fused++
		}
		sparseMoEForward(w, x, buf)
		// Keep the routed branch, then add the shared expert using its old path.
		routed := *w
		routed.SharedGateIn, routed.SharedGate, routed.SharedUp, routed.SharedDown = nil, nil, nil, nil
		oracle := &DecodeBuffer{}
		sparseMoEForward(&routed, x, oracle)
		serialMoESharedForward(w, x, oracle)
		if !slices.Equal(buf.Proj, oracle.Proj) {
			t.Fatalf("layer %d differs from separately dispatched shared expert", index)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("configured checkpoint has no shared expert to compare")
	}
	t.Logf("shared expert parity: layers=%d fused_cpu_pairs=%d", checked, fused)
}
