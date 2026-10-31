package engine

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestVoxtralDecodeProjectionAndScratchParity(t *testing.T) {
	cfg, w := buildTinyVoxtralRealtimeWeights()
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
		generate := pos >= 12
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
		if !reflect.DeepEqual(a.KHistory, b.KHistory) || !reflect.DeepEqual(a.VHistory, b.VHistory) || !reflect.DeepEqual(a.scratch.x, b.scratch.x) || !reflect.DeepEqual(a.scratch.finalNormed, b.scratch.finalNormed) {
			t.Fatalf("state at %d differs", pos)
		}
	}
	if len(a.scratch.ffnGate) != len(b.scratch.ffnAct) {
		t.Fatal("activation width changed")
	}
}

var decodeMemorySink *DecodeBuffer

func BenchmarkMinistralOptionalScratch(b *testing.B) {
	shapes := []struct {
		name   string
		config Config
	}{}
	for _, shape := range []struct {
		name              string
		dim, hidden, head int
	}{{"3b", 3072, 9216, 128}, {"14b", 5120, 16384, 160}} {
		shapes = append(shapes, struct {
			name   string
			config Config
		}{shape.name, Config{Dim: shape.dim, HiddenDim: shape.hidden, NHeads: 32, NKVHeads: 8, HeadDim: shape.head, ValueDim: 128, VocabSize: 131072}})
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
			b.Logf("model=%s dim=%d hidden=%d heads=%d kv_heads=%d head_dim=%d value_dim=%d vocab=%d", filepath.Base(path), c.Dim, c.HiddenDim, c.NHeads, c.NKVHeads, c.HeadDim, c.ValueDim, c.VocabSize)
			shapes = append(shapes, struct {
				name   string
				config Config
			}{filepath.Base(path), c})
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

func BenchmarkVoxtralTeacherForcedDecode(b *testing.B) {
	setPerformanceThreads(b, 8)
	cfg, w := buildTinyVoxtralRealtimeWeights()
	cfg.Decoder.VocabSize = 131072
	w.Decoder.TokenEmbd = Weight{F32: make([]float32, 131072*cfg.Decoder.HiddenSize), Rows: 131072, Cols: cfg.Decoder.HiddenSize}
	for i := range w.Decoder.TokenEmbd.F32 {
		w.Decoder.TokenEmbd.F32[i] = float32(math.Sin(float64(i))) * .01
	}
	input := randomExpertInput(cfg.Decoder.HiddenSize, 123)
	a, _ := NewVoxtralRealtimeDecoderState(cfg, w, 2)
	old, _ := referenceNewVoxtralRealtimeDecoderState(cfg, w, 2)
	// Keep the real vocabulary width but deliberately tiny transformer geometry;
	// this isolates the cost of discarded output projections, not full-model speed.
	run := func(ref bool) time.Duration {
		start := time.Now()
		for j := 0; j < 32; j++ {
			if ref {
				if _, err := referenceForwardVoxtralRealtimeDecoderStep(cfg, w, old, input, old.nextPos); err != nil {
					b.Fatal(err)
				}
			} else {
				if _, err := forwardVoxtralRealtimeDecoderStep(cfg, w, a, input, a.nextPos, false); err != nil {
					b.Fatal(err)
				}
			}
		}
		return time.Since(start)
	}
	run(true)
	run(false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var x, y time.Duration
		if i%2 == 0 {
			x = run(true)
			y = run(false)
		} else {
			y = run(false)
			x = run(true)
		}
		b.Logf("pair=%d reference_ns=%d changed_ns=%d", i, x, y)
	}
}

func BenchmarkVoxtralTeacherForcedAllocations(b *testing.B) {
	setPerformanceThreads(b, 8)
	cfg, w := buildTinyVoxtralRealtimeWeights()
	cfg.Decoder.VocabSize = 131072
	w.Decoder.TokenEmbd = Weight{F32: make([]float32, 131072*cfg.Decoder.HiddenSize), Rows: 131072, Cols: cfg.Decoder.HiddenSize}
	input := make([]float32, cfg.Decoder.HiddenSize)
	for _, ref := range []bool{true, false} {
		b.Run(fmt.Sprintf("reference=%t", ref), func(b *testing.B) {
			a, _ := NewVoxtralRealtimeDecoderState(cfg, w, 2)
			old, _ := referenceNewVoxtralRealtimeDecoderState(cfg, w, 2)
			run := func() {
				for j := 0; j < 32; j++ {
					if ref {
						if _, err := referenceForwardVoxtralRealtimeDecoderStep(cfg, w, old, input, old.nextPos); err != nil {
							b.Fatal(err)
						}
					} else {
						if _, err := forwardVoxtralRealtimeDecoderStep(cfg, w, a, input, a.nextPos, false); err != nil {
							b.Fatal(err)
						}
					}
				}
			}
			run()
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				run()
			}
		})
	}
}
