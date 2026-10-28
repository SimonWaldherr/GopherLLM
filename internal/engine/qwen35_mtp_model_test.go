//go:build darwin && cgo && metal

package engine

import (
	"io"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"
)

// BenchmarkQwen35MTPBodyConfiguredHead evaluates only a real checkpoint's
// one-layer draft body, without its vocabulary projection or target trunk.
// Run with -tags metal, -benchtime=200ms -count=3 and a local model supplied by
// GOPHERLLM_VALIDATE_MODEL (or GOPHERLLM_SPEC_AB_RAW_MODEL). This is a backend
// microbenchmark, not evidence of complete-model speculative speed.
func BenchmarkQwen35MTPBodyConfiguredHead(b *testing.B) {
	path := os.Getenv("GOPHERLLM_VALIDATE_MODEL")
	if path == "" {
		path = os.Getenv("GOPHERLLM_SPEC_AB_RAW_MODEL")
	}
	if path == "" {
		b.Skip("set GOPHERLLM_VALIDATE_MODEL to a local Qwen checkpoint with MTP")
	}
	if !MetalAvailable() {
		b.Fatal(MetalError())
	}
	setPerformanceThreads(b, 8)
	previousQ8 := useQ8Activations.Load()
	useQ8Activations.Store(q8ActivationsAvailable())
	b.Cleanup(func() {
		useQ8Activations.Store(previousQ8)
	})
	r, info, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: true, PrepareQuantized: false, Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = r.Close() })
	if r.kind != loadedQwen35 || r.qwen35.MTP == nil {
		b.Fatal("a native Qwen checkpoint with a loaded MTP head is required")
	}
	cfg, selected := r.config, qwen35MTPExecutionWeights(*r.qwen35.MTP)
	for name, w := range map[string]Weight{"Q": selected.Attention.Q,
		"gate": selected.FFN.Gate, "up": selected.FFN.Up, "down": selected.FFN.Down} {
		if w.Metal == nil {
			b.Fatalf("MTP %s has no Metal view; this would not compare the requested backend", name)
		}
	}
	cpu := mtpHeadCPUWeights(selected)
	var bodyBytes int64
	for _, w := range [...]Weight{selected.EHProj, selected.Attention.Q, selected.Attention.K, selected.Attention.V,
		selected.Attention.O, selected.FFN.Gate, selected.FFN.Up, selected.FFN.Down} {
		bodyBytes += int64(len(w.Raw)) + int64(len(w.F32))*4
	}
	b.Logf("model=%s layers_loaded=%d layers_evaluated=0(target)+1(MTP) dim=%d body_projection_bytes=%d load=%v mapped=%t Q8=%t prepare=false prefault=none threads=8 no_vocab=true",
		path, cfg.NLayers, cfg.Dim, bodyBytes, info.LoadTime, r.IsMapped(), useQ8Activations.Load())
	for name, w := range map[string]Weight{"EH": selected.EHProj, "Q": selected.Attention.Q, "K": selected.Attention.K,
		"V": selected.Attention.V, "O": selected.Attention.O, "gate": selected.FFN.Gate, "up": selected.FFN.Up, "down": selected.FFN.Down} {
		b.Logf("projection=%s rows=%d cols=%d type=%d metal=%t direct=%t", name, w.Rows, w.Cols, w.Type, w.Metal != nil, metalWeightUsesDirect(w.Metal))
	}
	const prefixLen, cacheLen = 8, 16
	prefix := newQwen35MTPState(cfg, cacheLen, kvF32)
	rng := rand.New(rand.NewSource(3517))
	hidden := make([][]float32, prefixLen+1)
	for i := range hidden {
		hidden[i] = make([]float32, cfg.Dim)
		for d := range hidden[i] {
			hidden[i][d] = float32(rng.NormFloat64()) * .2
		}
	}
	for pos := range prefixLen {
		token := uint32((pos*997 + 313) % cfg.VocabSize)
		qwen35MTPProcess(cfg, &cpu, prefix, token, hidden[pos], pos)
	}
	hPrev := hidden[prefixLen-1]
	token := uint32(811 % cfg.VocabSize)
	newState := func() *Qwen35MTPState {
		s := newQwen35MTPState(cfg, cacheLen, kvF32)
		if copyQwen35MTPPrefix(s, prefix, prefixLen) != prefixLen {
			b.Fatal("cannot initialize the common MTP prefix")
		}
		return s
	}
	reference, metalState := newState(), newState()
	forwardQwen35MTPBodyInto(cfg, cpu, reference, token, hPrev, prefixLen)
	forwardQwen35MTPBodyInto(cfg, selected, metalState, token, hPrev, prefixLen)
	// CPU Q8 activation rounding and Metal float reductions intentionally use
	// different arithmetic. Bound their hidden-state difference directly; the
	// selected execution path must still retain the exact CPU K/V below.
	const maxHiddenRelativeL2 = .02
	hiddenError := mtpHeadRelativeL2(b, metalState.Scratch.XN, reference.Scratch.XN)
	if hiddenError > maxHiddenRelativeL2 {
		b.Fatalf("selected Metal hidden relative L2=%g exceeds CPU comparison bound %g", hiddenError, maxHiddenRelativeL2)
	}
	positions := prefixLen + 1
	k, v := positions*reference.KV.PerPosKDim, positions*reference.KV.PerPosVDim
	if !slices.Equal(metalState.KV.K[0][:k], reference.KV.K[0][:k]) || !slices.Equal(metalState.KV.V[0][:v], reference.KV.V[0][:v]) {
		b.Fatal("selected Metal Q/O/FFN offload changed the CPU K/V contract")
	}
	catchup := newState()
	copy(catchup.PendingHidden, hPrev)
	qwen35MTPProcess(cfg, &cpu, catchup, token, hidden[prefixLen], prefixLen)
	if !slices.Equal(catchup.KV.K[0][:k], reference.KV.K[0][:k]) || !slices.Equal(catchup.KV.V[0][:v], reference.KV.V[0][:v]) {
		b.Fatal("real MTP KV-only catch-up differs from the full CPU body")
	}
	b.Logf("first_body_hidden_relative_L2 selected_metal=%g bound=%g selected_KV=bit_exact", hiddenError, maxHiddenRelativeL2)
	for _, mode := range []struct {
		name    string
		weights Qwen35MTPWeights
	}{{"cpu", cpu}, {"selected_metal", selected}} {
		b.Run(mode.name, func(b *testing.B) {
			setPerformanceThreads(b, 8)
			state := newState()
			run := func() { forwardQwen35MTPBodyInto(cfg, mode.weights, state, token, hPrev, prefixLen) }
			run()
			b.ReportAllocs()
			b.SetBytes(bodyBytes)
			b.ResetTimer()
			for b.Loop() {
				run()
			}
		})
	}
}

func mtpHeadCPUWeights(w Qwen35MTPWeights) Qwen35MTPWeights {
	w.TokenEmbd.Metal, w.EHProj.Metal = nil, nil
	w.Attention.Q.Metal, w.Attention.K.Metal, w.Attention.V.Metal, w.Attention.O.Metal = nil, nil, nil, nil
	w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal = nil, nil, nil
	return w
}

func mtpHeadRelativeL2(t testing.TB, got, want []float32) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("MTP comparison lengths differ: %d/%d", len(got), len(want))
	}
	var error2, norm2 float64
	for i, expected := range want {
		if !finite32(got[i]) || !finite32(expected) {
			t.Fatalf("nonfinite MTP comparison at %d: %g/%g", i, got[i], expected)
		}
		delta := float64(got[i]) - float64(expected)
		error2 += delta * delta
		norm2 += float64(expected) * float64(expected)
	}
	return math.Sqrt(error2 / max(norm2, 1e-30))
}
