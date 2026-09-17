package gopherllm

import (
	"math"
	"testing"
)

func TestVoxtralRealtimeTimeCondFixedAcrossPositions(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	a, err := voxtralRealtimeTimeCond(cfg.Time, w.Decoder.TimeEmbedInvFreq, cfg.Time.DefaultNumDelayTokens)
	if err != nil {
		t.Fatalf("voxtralRealtimeTimeCond: %v", err)
	}
	b, err := voxtralRealtimeTimeCond(cfg.Time, w.Decoder.TimeEmbedInvFreq, cfg.Time.DefaultNumDelayTokens)
	if err != nil {
		t.Fatalf("voxtralRealtimeTimeCond (2nd call): %v", err)
	}
	if len(a) != cfg.Time.EmbedDim {
		t.Fatalf("t_cond length %d, want EmbedDim=%d", len(a), cfg.Time.EmbedDim)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("t_cond is not deterministic at index %d: %v vs %v", i, a[i], b[i])
		}
	}
	// A different delay must produce a different conditioning vector,
	// otherwise the ada gate could never distinguish checkpoints/configs
	// with different built-in streaming delays.
	c, err := voxtralRealtimeTimeCond(cfg.Time, w.Decoder.TimeEmbedInvFreq, cfg.Time.DefaultNumDelayTokens+3)
	if err != nil {
		t.Fatalf("voxtralRealtimeTimeCond (different delay): %v", err)
	}
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("t_cond identical for different numDelayTokens")
	}
}

func TestForwardVoxtralRealtimeDecoderStepShapesAndFinite(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	state, err := NewVoxtralRealtimeDecoderState(cfg, w, cfg.Time.DefaultNumDelayTokens)
	if err != nil {
		t.Fatalf("NewVoxtralRealtimeDecoderState: %v", err)
	}
	if len(state.AdaScale) != len(w.Decoder.Layers) {
		t.Fatalf("AdaScale has %d layers, want %d", len(state.AdaScale), len(w.Decoder.Layers))
	}

	embed, err := EmbedVoxtralRealtimeToken(cfg, w, 3)
	if err != nil {
		t.Fatalf("EmbedVoxtralRealtimeToken: %v", err)
	}
	if len(embed) != cfg.Decoder.HiddenSize {
		t.Fatalf("embedding has %d elements, want HiddenSize=%d", len(embed), cfg.Decoder.HiddenSize)
	}

	// Simulate a few autoregressive steps, each fusing a (fake) audio
	// embedding into the token embedding as voxtral_realtime.go's doc
	// comment describes (elementwise sum, no interleaving).
	for pos := 0; pos < 5; pos++ {
		tok, err := EmbedVoxtralRealtimeToken(cfg, w, pos%cfg.Decoder.VocabSize)
		if err != nil {
			t.Fatalf("pos %d: EmbedVoxtralRealtimeToken: %v", pos, err)
		}
		fused := make([]float32, cfg.Decoder.HiddenSize)
		for i := range fused {
			fused[i] = tok[i] + 0.01*float32(pos+1)
		}
		logits, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, fused, pos)
		if err != nil {
			t.Fatalf("pos %d: ForwardVoxtralRealtimeDecoderStep: %v", pos, err)
		}
		if len(logits) != cfg.Decoder.VocabSize {
			t.Fatalf("pos %d: logits length %d, want VocabSize=%d", pos, len(logits), cfg.Decoder.VocabSize)
		}
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("pos %d: logit %d is %v", pos, i, v)
			}
		}
		for li := range w.Decoder.Layers {
			if len(state.KHistory[li]) != pos+1 || len(state.VHistory[li]) != pos+1 {
				t.Fatalf("pos %d layer %d: K/V history length %d/%d, want %d", pos, li, len(state.KHistory[li]), len(state.VHistory[li]), pos+1)
			}
		}
	}
}

func TestForwardVoxtralRealtimeDecoderStepRejectsWrongInputWidth(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	state, err := NewVoxtralRealtimeDecoderState(cfg, w, cfg.Time.DefaultNumDelayTokens)
	if err != nil {
		t.Fatalf("NewVoxtralRealtimeDecoderState: %v", err)
	}
	if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, make([]float32, cfg.Decoder.HiddenSize+1), 0); err == nil {
		t.Fatalf("expected an error for a mis-sized input embedding")
	}
}

// Compare recycled storage with independently allocated, explicitly cropped
// histories. Absolute RoPE positions must remain correct over many rollovers.
func TestVoxtralDecoderSlidingWindowMatchesCroppedHistory(t *testing.T) {
	for _, window := range []int{1, 3, 16} {
		cfg, w := buildTinyVoxtralRealtimeWeights()
		cfg.Decoder.SlidingWindow = window
		state, err := NewVoxtralRealtimeDecoderState(cfg, w, 2)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := NewVoxtralRealtimeDecoderState(cfg, w, 2)
		if err != nil {
			t.Fatal(err)
		}
		refCfg := cfg
		refCfg.Decoder.SlidingWindow = 0
		var saved, savedCopy []float32
		for pos := 0; pos < 65; pos++ {
			for li := range ref.KHistory {
				if len(ref.KHistory[li]) >= window {
					ref.KHistory[li] = ref.KHistory[li][1:]
					ref.VHistory[li] = ref.VHistory[li][1:]
				}
			}
			input, err := EmbedVoxtralRealtimeToken(cfg, w, pos%cfg.Decoder.VocabSize)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, pos)
			if err != nil {
				t.Fatal(err)
			}
			want, err := ForwardVoxtralRealtimeDecoderStep(refCfg, w, ref, input, pos)
			if err != nil {
				t.Fatal(err)
			}
			for i := range got {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 1e-6 {
					t.Fatalf("window=%d pos=%d logit=%d: got %g want %g", window, pos, i, got[i], want[i])
				}
			}
			for li := range state.KHistory {
				if len(state.KHistory[li]) != min(pos+1, window) || len(state.VHistory[li]) != min(pos+1, window) {
					t.Fatalf("window=%d pos=%d: unbounded history", window, pos)
				}
			}
			if pos == 0 {
				saved = got
				savedCopy = append([]float32(nil), got...)
			}
		}
		for i := range saved {
			if saved[i] != savedCopy[i] {
				t.Fatal("later steps modified returned logits")
			}
		}
	}
}

func TestVoxtralDecoderRejectsInvalidStateAndPosition(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	input := make([]float32, cfg.Decoder.HiddenSize)
	if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, nil, input, 0); err == nil {
		t.Fatal("nil state accepted")
	}
	state, err := NewVoxtralRealtimeDecoderState(cfg, w, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, pos := range []int{-1, 1, 20} {
		if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, pos); err == nil {
			t.Fatalf("position %d accepted", pos)
		}
	}
	if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, 0); err == nil {
		t.Fatal("duplicate position accepted")
	}
	if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, 1); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkVoxtralRealtimeDecoderStep(b *testing.B) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	state, err := NewVoxtralRealtimeDecoderState(cfg, w, 2)
	if err != nil {
		b.Fatal(err)
	}
	input := make([]float32, cfg.Decoder.HiddenSize)
	for pos := range cfg.Decoder.SlidingWindow {
		if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, pos); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, i+cfg.Decoder.SlidingWindow); err != nil {
			b.Fatal(err)
		}
	}
}

func TestVoxtralDecoderUsesGGUFSplitHalfKeys(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	state, err := NewVoxtralRealtimeDecoderState(cfg, w, 2)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, cfg.Decoder.HiddenSize)
	for i := range input {
		input[i] = float32(i+1) / 10
	}
	const pos = 3
	for p := 0; p <= pos; p++ {
		if _, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, p); err != nil {
			t.Fatal(err)
		}
	}
	// Independently rotate each first-layer key by pairing the first and second
	// head halves, as required by the row-permuted GGUF (not adjacent elements).
	var normed, expected []float32
	rmsNormInto(input, w.Decoder.Layers[0].AttnNorm, cfg.Decoder.Epsilon, &normed)
	w.Decoder.Layers[0].K.MatvecInto(normed, &expected)
	half := cfg.Decoder.HeadDim / 2
	for head := 0; head < cfg.Decoder.NKVHeads; head++ {
		for i := 0; i < half; i++ {
			angle := float64(pos) / math.Pow(float64(cfg.Decoder.RopeTheta), float64(2*i)/float64(cfg.Decoder.HeadDim))
			sine, cosine := math.Sincos(angle)
			a, b := head*cfg.Decoder.HeadDim+i, head*cfg.Decoder.HeadDim+i+half
			x, y := expected[a], expected[b]
			expected[a] = x*float32(cosine) - y*float32(sine)
			expected[b] = x*float32(sine) + y*float32(cosine)
		}
	}
	for i, want := range expected {
		if got := state.KHistory[0][pos][i]; math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("key %d got %g want %g", i, got, want)
		}
	}
}
