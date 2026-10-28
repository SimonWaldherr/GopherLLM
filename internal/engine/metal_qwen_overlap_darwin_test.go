//go:build darwin && cgo && metal

package engine

import (
	"slices"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func TestMetalQ6KOverlapLifecycle(t *testing.T) {
	w := metalQ6ProjectionFixture(t, metalQ6KDirectMinRows+1, 256)
	x := metalQ6ProjectionInputs(1, w.cols, 31)
	want := metalQ6ProjectionSerial(t, w, x, 1)
	out := make([]float32, w.rows)
	calls := 0
	if !metalbackend.MatvecQ6KWhile(w.q6, x, out, func() { calls++ }) || calls != 1 {
		t.Fatal("CPU work did not run during submission")
	}
	requireMoEPrefillExact(t, out, want)
	// A panic must still drain and release the submission before the weight
	// can be reused. Recover outside the API, then immediately submit again.
	func() {
		defer func() {
			if recover() != "CPU work" {
				t.Error("callback panic lost")
			}
		}()
		metalbackend.MatvecQ6KWhile(w.q6, x, out, func() { panic("CPU work") })
	}()
	if !metalbackend.MatvecQ6KWhile(w.q6, x, out, func() {}) {
		t.Fatal("weight not reusable after panic")
	}
	requireMoEPrefillExact(t, out, want)
	for _, bad := range []struct {
		w      *metalbackend.Weight
		x, out []float32
	}{{nil, x, out}, {w.q6, nil, out}, {w.q6, x, out[:len(out)-1]}} {
		before := slices.Clone(out)
		if metalbackend.MatvecQ6KWhile(bad.w, bad.x, bad.out, func() { t.Error("invalid input ran work") }) {
			t.Error("invalid input accepted")
		}
		requireMoEPrefillExact(t, out, before)
	}
}

func TestQwenMetalOverlapInputs(t *testing.T) {
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activations unavailable")
	}
	withQ8Activations(true, func() {
		q := metalQ6ProjectionFixture(t, metalQ6KDirectMinRows+1, 256)
		w := Qwen35DeltaNetWeights{
			QKVConv:   Weight{Type: GGMLTypeQ6_K, Rows: q.rows, Cols: q.cols, Metal: q},
			Gate:      qwen35InputWeight(t, GGMLTypeQ4_K, 1024, q.cols),
			AlphaProj: qwen35InputWeight(t, GGMLTypeQ4_K, 17, q.cols),
			BetaProj:  qwen35InputWeight(t, GGMLTypeQ4_K, 17, q.cols),
		}
		x := metalQ6ProjectionInputs(1, q.cols, 13)
		var want, got DecodeBuffer
		w.QKVConv.MatvecInto(x, &want.MambaIn)
		w.Gate.MatvecInto(x, &want.MambaZ)
		if !tryMatvec2Into(w.AlphaProj, w.BetaProj, x, &want.Q4KXSums, &want.MambaDT, &want.MambaBeta) {
			t.Fatal("fixture did not fuse alpha/beta")
		}
		t.Setenv("GOPHERLLM_QWEN_OVERLAP_INPUTS", "1")
		if !qwen35OverlapDeltaInputs(w, x, &got) {
			t.Fatal("overlap declined")
		}
		for i, v := range [][]float32{want.MambaIn, want.MambaZ, want.MambaDT, want.MambaBeta} {
			requireMoEPrefillExact(t, [][]float32{got.MambaIn, got.MambaZ, got.MambaDT, got.MambaBeta}[i], v)
		}
		for _, edit := range []func(*Qwen35DeltaNetWeights){
			func(w *Qwen35DeltaNetWeights) { w.QKVConv.F32 = []float32{1} },
			func(w *Qwen35DeltaNetWeights) { w.QKVConv.Type = GGMLTypeQ4_K },
			func(w *Qwen35DeltaNetWeights) { w.QKVConv.Metal = nil },
			func(w *Qwen35DeltaNetWeights) { w.Gate.Prepared = &PreparedQuantizedWeight{} },
			func(w *Qwen35DeltaNetWeights) { w.AlphaProj.Raw = nil },
		} {
			copyW := w
			edit(&copyW)
			var rejected DecodeBuffer
			if qwen35OverlapDeltaInputs(copyW, x, &rejected) || rejected.MambaIn != nil {
				t.Fatal("ineligible projection changed the buffer")
			}
		}
		t.Setenv("GOPHERLLM_QWEN_OVERLAP_INPUTS", "0")
		if qwen35OverlapDeltaInputs(w, x, &got) {
			t.Fatal("overlap opt-out ignored")
		}
	})
}
