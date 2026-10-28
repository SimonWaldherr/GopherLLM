//go:build darwin && cgo && metal

package engine

import (
	"slices"
	"testing"
)

func TestMoESharedMetalKeepsProjectionBackend(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	setPerformanceThreads(t, 8)
	const dim, hidden = 256, metalQ4KDirectMinRows
	gate, _ := expertPlaneWeight(moeDownWeightForTest(GGMLTypeQ4_K, dim, hidden, 1), 0)
	up := gate
	down, _ := expertPlaneWeight(moeDownWeightForTest(GGMLTypeQ6_K, hidden, dim, 1), 0)
	gate.Metal = prepareMetalWeight(gate.Raw, gate.Type, gate.Rows, gate.Cols, false)
	if !metalWeightUsesDirect(gate.Metal) {
		t.Fatal("test gate did not qualify for direct Metal")
	}
	defer releaseMetalWeight(gate.Metal)
	x := randomExpertInput(dim, 41400)
	// Include a mixed CPU/GPU pair: fusion must not move either projection.
	for _, metalUp := range []bool{false, true} {
		if metalUp {
			up.Metal = gate.Metal
		}
		w := &SparseMoEWeights{SharedGate: &gate, SharedUp: &up, SharedDown: &down, SharedAlways: true, Down: ExpertWeight{Output: dim}}
		got, want := &DecodeBuffer{Proj: make([]float32, dim)}, &DecodeBuffer{Proj: make([]float32, dim)}
		if moeSharedGateUpInto(gate, up, x, got) {
			t.Fatal("CPU fusion bypassed a direct Metal projection")
		}
		sparseMoESharedForward(w, x, got)
		serialMoESharedForward(w, x, want)
		if !slices.Equal(got.Proj, want.Proj) {
			t.Fatal("Metal shared expert changed output")
		}
	}
}
