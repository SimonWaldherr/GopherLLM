//go:build darwin && cgo && metal

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// Exercise Q6-only initialization, unequal residual/attention widths, Ada
// folding, sliding attention and cache compaction against the scalar decoder.
func TestVoxtralMetalStreamingParityAndCompaction(t *testing.T) {
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
