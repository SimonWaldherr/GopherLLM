package engine

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestVoxtralDecodeProjectionAndScratchParity(t *testing.T) {
	for _, window := range []int{1, 3, 16} {
		for _, outputs := range []string{"prefix", "alternating", "all"} {
			t.Run(fmt.Sprintf("window=%d/outputs=%s", window, outputs), func(t *testing.T) {
				testVoxtralDecodeProjectionAndScratchParity(t, window, outputs)
			})
		}
	}
}

func testVoxtralDecodeProjectionAndScratchParity(t *testing.T, window int, outputs string) {
	t.Helper()
	cfg, w := buildTinyVoxtralRealtimeWeights()
	cfg.Decoder.SlidingWindow = window
	a, err := NewVoxtralRealtimeDecoderState(cfg, w, 2)
	if err != nil {
		t.Fatal(err)
	}
	b, err := referenceNewVoxtralRealtimeDecoderState(cfg, w, 2)
	if err != nil {
		t.Fatal(err)
	}
	for pos := 0; pos < 40; pos++ {
		input := randomExpertInput(cfg.Decoder.HiddenSize, int64(pos+100))
		want, err := referenceForwardVoxtralRealtimeDecoderStep(cfg, w, b, input, pos)
		if err != nil {
			t.Fatal(err)
		}
		generate := outputs == "all" || outputs == "prefix" && pos >= 12 || outputs == "alternating" && pos%2 == 0
		got, err := forwardVoxtralRealtimeDecoderStep(cfg, w, a, input, pos, generate)
		if err != nil {
			t.Fatal(err)
		}
		if generate && !reflect.DeepEqual(got, want) {
			t.Fatalf("logits at %d differ", pos)
		}
		if !generate && got != nil {
			t.Fatal("teacher-forced step materialized logits")
		}
		for layer := range a.KHistory {
			if !reflect.DeepEqual(a.KHistory[layer], b.KHistory[layer]) || !reflect.DeepEqual(a.VHistory[layer], b.VHistory[layer]) {
				t.Fatalf("K/V history at position %d, layer %d differs", pos, layer)
			}
			if got, want := len(a.KHistory[layer]), min(pos+1, window); got != want {
				t.Fatalf("K/V history at position %d, layer %d: %d rows, want %d", pos, layer, got, want)
			}
		}
		if a.nextPos != b.nextPos || !reflect.DeepEqual(a.scratch.x, b.scratch.x) || !reflect.DeepEqual(a.scratch.finalNormed, b.scratch.finalNormed) {
			t.Fatalf("hidden state or next position at %d differs", pos)
		}
	}
	if len(a.scratch.ffnGate) != len(b.scratch.ffnAct) {
		t.Fatal("activation width changed")
	}
}

var decodeMemorySink *DecodeBuffer

func BenchmarkMinistralOptionalScratch(b *testing.B) {
	type shape struct {
		name   string
		config Config
	}
	shapes := []shape{
		{"3b", Config{Dim: 3072, HiddenDim: 9216, NHeads: 32, NKVHeads: 8, HeadDim: 128, ValueDim: 128, VocabSize: 131072}},
		{"14b", Config{Dim: 5120, HiddenDim: 16384, NHeads: 32, NKVHeads: 8, HeadDim: 128, ValueDim: 128, VocabSize: 131072}},
	}
	if paths := os.Getenv("GOPHERLLM_MINISTRAL_MEMORY_MODELS"); paths != "" {
		shapes = nil
		for _, path := range filepath.SplitList(paths) {
			mapping, err := OpenMmap(path)
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() { _ = mapping.Close() })
			gguf, err := ParseGGUF(mapping.Bytes())
			if err != nil {
				b.Fatal(err)
			}
			c := ConfigFromGGUF(gguf)
			// Match LoadModel: projection tensors can refine dimensions inferred
			// from metadata, notably residual width / heads in Ministral 14B.
			inferAttentionShape(&c, indexTensors(gguf))
			b.Logf("model=%s dim=%d hidden=%d heads=%d kv_heads=%d head_dim=%d value_dim=%d vocab=%d", filepath.Base(path), c.Dim, c.HiddenDim, c.NHeads, c.NKVHeads, c.HeadDim, c.ValueDim, c.VocabSize)
			shapes = append(shapes, shape{filepath.Base(path), c})
		}
	}
	for _, shape := range shapes {
		c := shape.config
		for _, old := range []bool{true, false} {
			b.Run(fmt.Sprintf("%s/eager=%t", shape.name, old), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					x := NewDecodeBuffer(c, c.HeadDim, c.NKVHeads, c.ValueDim)
					if old {
						x.GateUp = make([]float32, 2*c.HiddenDim)
						x.MTPInput = make([]float32, 2*c.Dim)
					}
					decodeMemorySink = x
				}
			})
		}
	}
}

// BenchmarkVoxtralTeacherForcedDecode uses a real vocabulary width with a tiny
// synthetic transformer. It isolates discarded projections, not model throughput.
func BenchmarkVoxtralTeacherForcedDecode(b *testing.B) {
	setPerformanceThreads(b, 8)
	cfg, w := buildTinyVoxtralRealtimeWeights()
	cfg.Decoder.VocabSize = 131072
	w.Decoder.TokenEmbd = Weight{F32: make([]float32, cfg.Decoder.VocabSize*cfg.Decoder.HiddenSize), Rows: cfg.Decoder.VocabSize, Cols: cfg.Decoder.HiddenSize}
	for i := range w.Decoder.TokenEmbd.F32 {
		w.Decoder.TokenEmbd.F32[i] = float32(math.Sin(float64(i))) * .01
	}
	input := randomExpertInput(cfg.Decoder.HiddenSize, 123)
	for _, ref := range []bool{true, false} {
		b.Run(fmt.Sprintf("reference=%t", ref), func(b *testing.B) {
			var step func() error
			if ref {
				state, err := referenceNewVoxtralRealtimeDecoderState(cfg, w, 2)
				if err != nil {
					b.Fatal(err)
				}
				step = func() error {
					_, err := referenceForwardVoxtralRealtimeDecoderStep(cfg, w, state, input, state.nextPos)
					return err
				}
			} else {
				state, err := NewVoxtralRealtimeDecoderState(cfg, w, 2)
				if err != nil {
					b.Fatal(err)
				}
				step = func() error {
					_, err := forwardVoxtralRealtimeDecoderStep(cfg, w, state, input, state.nextPos, false)
					return err
				}
			}
			// Fill the window before timing so every measured step has the same
			// attention length. Construction and warmup allocations are excluded.
			for range cfg.Decoder.SlidingWindow {
				if err := step(); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			for b.Loop() {
				for range 32 {
					if err := step(); err != nil {
						b.Fatal(err)
					}
				}
			}
			b.ReportMetric(float64(b.N)*32/b.Elapsed().Seconds(), "steps/s")
		})
	}
}
