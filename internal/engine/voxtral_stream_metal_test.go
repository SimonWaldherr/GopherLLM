//go:build darwin && cgo && metal

package engine

import (
	"math"
	"math/rand"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func voxtralMetalDecoderFixture(t *testing.T) (VoxtralRealtimeConfig, VoxtralRealtimeWeights, *VoxtralRealtimeDecoderState, *rand.Rand) {
	t.Helper()
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	rng := rand.New(rand.NewSource(7351))
	weight := func(rows, cols int) Weight {
		raw := make([]byte, 0, rows*cols/256*210)
		for range rows {
			row := randomQ6KRow(rng, cols)
			for b := 0; b < len(row); b += 210 {
				row[b+209] = 0x04
			}
			raw = append(raw, row...)
		}
		return Weight{Raw: raw, Rows: rows, Cols: cols, Type: GGMLTypeQ6_K}
	}
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1 + float32(rng.NormFloat64())*.02
		}
		return v
	}
	cfg := VoxtralRealtimeConfig{Decoder: VoxtralRealtimeDecoderConfig{HiddenSize: 256, IntermediateSize: 512, HeadDim: 128, NHeads: 4, NKVHeads: 1, NLayers: 2, VocabSize: 32, SlidingWindow: 6, RopeTheta: 10000, Epsilon: 1e-5}}
	w := VoxtralRealtimeWeights{Decoder: VoxtralRealtimeDecoderWeights{TokenEmbd: weight(32, 256), OutputNorm: norm(256)}}
	state := &VoxtralRealtimeDecoderState{KHistory: make([][][]float32, 2), VHistory: make([][][]float32, 2), AdaScale: make([][]float32, 2)}
	for i := range 2 {
		w.Decoder.Layers = append(w.Decoder.Layers, VoxtralRealtimeDecoderLayer{AttnNorm: norm(256), FFNNorm: norm(256), Q: weight(512, 256), K: weight(128, 256), V: weight(128, 256), O: weight(256, 512), FFNGate: weight(512, 256), FFNUp: weight(512, 256), FFNDown: weight(256, 512)})
		state.AdaScale[i] = make([]float32, 256)
		for j := range state.AdaScale[i] {
			state.AdaScale[i][j] = float32(rng.NormFloat64()) * .1
		}
	}
	return cfg, w, state, rng
}

// Exercise Q6-only initialization, unequal residual/attention widths, Ada
// folding, sliding attention and cache compaction against the scalar decoder.
func TestVoxtralMetalStreamingParityAndCompaction(t *testing.T) {
	cfg, w, state, rng := voxtralMetalDecoderFixture(t)
	fast := newVoxtralFastDecoder(cfg, w, state)
	if fast == nil {
		t.Fatal("Metal decoder was not created")
	}
	defer fast.Close()
	gpu := fast.(*voxtralMetalDecoder)
	for pos := 0; pos < gpu.capacity+9; pos++ {
		input := make([]float32, 256)
		for i := range input {
			input[i] = float32(rng.NormFloat64()) * .2
		}
		want, err := ForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, pos)
		if err != nil {
			t.Fatal(err)
		}
		token, err := gpu.Step(input, pos, true)
		if err != nil {
			t.Fatal(err)
		}
		best := 0
		for i, v := range want {
			if v > want[best] {
				best = i
			}
		}
		if token != best {
			t.Fatalf("position %d: token %d, want %d", pos, token, best)
		}
		for i, v := range state.scratch.finalNormed {
			if math.Abs(float64(gpu.normed[i]-v)) > 2e-4 {
				t.Fatalf("position %d dimension %d: GPU %g CPU %g", pos, i, gpu.normed[i], v)
			}
		}
	}
	if gpu.physical >= gpu.capacity {
		t.Fatal("cache was not compacted")
	}
}

// The offline path owns the fast decoder for one transcription; the model's
// prepared weights remain borrowed. Closing must also work before the first
// step or after a failed step, when an early return still releases the cache.
func TestVoxtralMetalDecoderCloseOwnership(t *testing.T) {
	for _, stage := range []string{"before_step", "after_step", "step_error"} {
		t.Run(stage, func(t *testing.T) {
			cfg, w, state, _ := voxtralMetalDecoderFixture(t)
			output := w.Decoder.TokenEmbd
			borrowed := metalbackend.PrepareQ6K(output.Raw, output.Rows, output.Cols, false)
			if borrowed == nil {
				t.Fatal("cannot prepare borrowed output weight:", MetalError())
			}
			t.Cleanup(func() { metalbackend.Release(borrowed) })
			w.Decoder.TokenEmbd.Metal = &MetalWeight{q6: borrowed, typ: output.Type, rows: output.Rows, cols: output.Cols}
			fast := newVoxtralFastDecoder(cfg, w, state)
			if fast == nil {
				t.Fatal("Metal decoder was not created")
			}
			defer fast.Close()
			gpu := fast.(*voxtralMetalDecoder)
			if len(gpu.owned) != 7*len(w.Decoder.Layers) {
				t.Fatal("fixture did not mix owned layer weights with borrowed output weight")
			}
			owned := gpu.owned[0]
			input := make([]float32, cfg.Decoder.HiddenSize)
			projection := make([]float32, cfg.Decoder.NHeads*cfg.Decoder.HeadDim)
			if !metalbackend.MatvecQ6K(owned, input, projection) {
				t.Fatal("owned projection was not live")
			}
			if !gpu.decoder.Cache(0, 0, nil, nil, true) {
				t.Fatal("decoder cache was not live")
			}
			switch stage {
			case "after_step":
				if _, err := fast.Step(input, 0, true); err != nil {
					t.Fatal(err)
				}
			case "step_error":
				if _, err := fast.Step(input[:1], 0, true); err == nil {
					t.Fatal("invalid input accepted")
				}
			}
			fast.Close()
			fast.Close() // A later owner cleanup remains harmless.
			if gpu.decoder.Cache(0, 0, nil, nil, true) {
				t.Fatal("closed decoder retained its cache")
			}
			if metalbackend.MatvecQ6K(owned, input, projection) {
				t.Fatal("closed decoder retained an owned projection")
			}
			if len(gpu.owned) != 0 || len(gpu.refs) != 0 {
				t.Fatal("closed decoder retained weight references")
			}
			if !metalbackend.MatvecQ6K(borrowed, input, make([]float32, output.Rows)) {
				t.Fatal("closing decoder released model-owned weights")
			}
		})
	}
	var absent *voxtralMetalDecoder
	absent.Close()
}
