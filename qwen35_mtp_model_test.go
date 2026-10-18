//go:build darwin && cgo && metal

package gopherllm

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
	cfg, original := r.config, *r.qwen35.MTP
	for name, w := range map[string]Weight{"Q": original.Attention.Q,
		"gate": original.FFN.Gate, "up": original.FFN.Up, "down": original.FFN.Down} {
		if w.Metal == nil {
			b.Fatalf("MTP %s has no Metal view; this would not compare the requested backend", name)
		}
	}
	cpu := mtpHeadCPUWeights(original)
	hybrid := original
	hybrid.TokenEmbd.Metal, hybrid.EHProj.Metal = nil, nil
	hybrid.Attention.K.Metal, hybrid.Attention.V.Metal = nil, nil
	var bodyBytes int64
	for _, w := range [...]Weight{original.EHProj, original.Attention.Q, original.Attention.K, original.Attention.V,
		original.Attention.O, original.FFN.Gate, original.FFN.Up, original.FFN.Down} {
		bodyBytes += int64(len(w.Raw)) + int64(len(w.F32))*4
	}
	b.Logf("model=%s layers_loaded=%d layers_evaluated=0(target)+1(MTP) dim=%d body_projection_bytes=%d load=%v mapped=%t Q8=%t prepare=false prefault=none threads=8 no_vocab=true",
		path, cfg.NLayers, cfg.Dim, bodyBytes, info.LoadTime, r.IsMapped(), useQ8Activations.Load())
	for name, w := range map[string]Weight{"EH": original.EHProj, "Q": original.Attention.Q, "K": original.Attention.K,
		"V": original.Attention.V, "O": original.Attention.O, "gate": original.FFN.Gate, "up": original.FFN.Up, "down": original.FFN.Down} {
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
	reference, metalState, hybridState := newState(), newState(), newState()
	forwardQwen35MTPBodyInto(cfg, cpu, reference, token, hPrev, prefixLen)
	forwardQwen35MTPBodyInto(cfg, original, metalState, token, hPrev, prefixLen)
	forwardQwen35MTPBodyInto(cfg, hybrid, hybridState, token, hPrev, prefixLen)
	originalError := mtpHeadRelativeL2(b, metalState.Scratch.XN, reference.Scratch.XN)
	hybridError := mtpHeadRelativeL2(b, hybridState.Scratch.XN, reference.Scratch.XN)
	bound := max(.02, 4*originalError)
	if hybridError > bound {
		b.Fatalf("hybrid hidden relative L2=%g exceeds CPU-Q8/GPU comparison envelope %g (original Metal=%g)", hybridError, bound, originalError)
	}
	positions := prefixLen + 1
	k, v := positions*reference.KV.PerPosKDim, positions*reference.KV.PerPosVDim
	if !slices.Equal(hybridState.KV.K[0][:k], reference.KV.K[0][:k]) || !slices.Equal(hybridState.KV.V[0][:v], reference.KV.V[0][:v]) {
		b.Fatal("hybrid Q/O/FFN offload changed the CPU K/V contract")
	}
	catchup := newState()
	copy(catchup.PendingHidden, hPrev)
	qwen35MTPProcess(cfg, &cpu, catchup, token, hidden[prefixLen], prefixLen)
	if !slices.Equal(catchup.KV.K[0][:k], reference.KV.K[0][:k]) || !slices.Equal(catchup.KV.V[0][:v], reference.KV.V[0][:v]) {
		b.Fatal("real MTP KV-only catch-up differs from the full CPU body")
	}
	b.Logf("first_body_hidden_relative_L2 originalMetal=%g hybridQOFFN=%g hybrid_envelope=%g hybrid_KV=bit_exact original_K_L2=%g original_V_L2=%g",
		originalError, hybridError, bound, mtpHeadRelativeL2(b, metalState.KV.K[0][:k], reference.KV.K[0][:k]), mtpHeadRelativeL2(b, metalState.KV.V[0][:v], reference.KV.V[0][:v]))
	for _, mode := range []struct {
		name    string
		weights Qwen35MTPWeights
	}{{"cpu", cpu}, {"original_metal", original}, {"hybrid_qo_ffn", hybrid}} {
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
