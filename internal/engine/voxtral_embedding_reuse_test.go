package engine

import (
	"context"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestVoxtralSessionEmbeddingReuseMatchesReference(t *testing.T) {
	cfg, weights := buildTinyVoxtralRealtimeWeights()
	state, err := NewVoxtralRealtimeDecoderState(cfg, weights, cfg.Time.DefaultNumDelayTokens)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := referenceNewVoxtralRealtimeDecoderState(cfg, weights, cfg.Time.DefaultNumDelayTokens)
	if err != nil {
		t.Fatal(err)
	}
	session := &VoxtralRealtimeSession{cfg: cfg, weights: weights, decoder: state, disableFast: true}
	t.Cleanup(func() { _ = session.Close() })
	originalWeights := append([]float32(nil), weights.Decoder.TokenEmbd.F32...)
	owned, err := EmbedVoxtralRealtimeToken(cfg, weights, 0)
	if err != nil {
		t.Fatal(err)
	}
	originalOwned := append([]float32(nil), owned...)
	var storage *float32
	for pos := range 64 {
		token := (pos*7 + 3) % cfg.Decoder.VocabSize
		audio := randomExpertInput(cfg.Decoder.HiddenSize, int64(pos+100))
		originalAudio := append([]float32(nil), audio...)
		fused, err := EmbedVoxtralRealtimeToken(cfg, weights, token)
		if err != nil {
			t.Fatal(err)
		}
		for i := range fused {
			fused[i] += audio[i]
		}
		want, err := referenceForwardVoxtralRealtimeDecoderStep(cfg, weights, reference, fused, pos)
		if err != nil {
			t.Fatal(err)
		}
		generate := pos >= 12
		got, err := session.step(context.Background(), audio, token, pos, generate)
		if err != nil {
			t.Fatal(err)
		}
		if generate && got != voxtralArgmax(want) {
			t.Fatalf("position %d: token %d, want %d", pos, got, voxtralArgmax(want))
		}
		if !generate && got != 0 {
			t.Fatalf("position %d: teacher-forced step returned token %d", pos, got)
		}
		if !reflect.DeepEqual(state.KHistory, reference.KHistory) || !reflect.DeepEqual(state.VHistory, reference.VHistory) || !reflect.DeepEqual(state.scratch.finalNormed, reference.scratch.finalNormed) {
			t.Fatalf("position %d: reused input changed decoder state", pos)
		}
		if !reflect.DeepEqual(session.embedding, fused) || !reflect.DeepEqual(audio, originalAudio) {
			t.Fatalf("position %d: fused input differs or audio was overwritten", pos)
		}
		if pos == 0 {
			storage = &session.embedding[0]
		} else if storage != &session.embedding[0] {
			t.Fatalf("position %d: input scratch was replaced", pos)
		}
	}
	if !reflect.DeepEqual(weights.Decoder.TokenEmbd.F32, originalWeights) || !reflect.DeepEqual(owned, originalOwned) {
		t.Fatal("embedding reuse overwrote model weights or a public embedding result")
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	if session.embedding != nil {
		t.Fatal("closed session retained input scratch")
	}
}

func TestVoxtralTranscribeRejectsInvalidRecordingBeforeWarmup(t *testing.T) {
	for _, invalid := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		t.Run(fmt.Sprint(invalid), func(t *testing.T) {
			model := tinyVoxtralHost()
			defer model.Close()
			// Warmup would reject this frontend. Invalid PCM must be rejected
			// before touching it, including a bad sample beyond the first chunk.
			model.source.cfg.Mel.NFFT = 0
			pcm := make([]float32, 6401)
			pcm[len(pcm)-1] = invalid
			if _, err := model.Transcribe(context.Background(), pcm); err == nil || !strings.Contains(err.Error(), "non-finite") {
				t.Fatalf("invalid recording reached warmup: %v", err)
			}
			if model.active {
				t.Fatal("invalid recording reserved model")
			}
		})
	}
}

var voxtralEmbeddingSink []float32

// This measures embedding lookup and audio fusion only, without decoder work.
func BenchmarkVoxtralFusedEmbedding(b *testing.B) {
	for _, width := range []int{12, 3072} {
		cfg := VoxtralRealtimeConfig{Decoder: VoxtralRealtimeDecoderConfig{HiddenSize: width, VocabSize: 32}}
		weights := VoxtralRealtimeWeights{Decoder: VoxtralRealtimeDecoderWeights{TokenEmbd: Weight{F32: make([]float32, width*32), Rows: 32, Cols: width}}}
		for i := range weights.Decoder.TokenEmbd.F32 {
			weights.Decoder.TokenEmbd.F32[i] = float32(i%17-8) * .01
		}
		audio := randomExpertInput(width, 123)
		for _, reuse := range []bool{false, true} {
			b.Run(fmt.Sprintf("width=%d/reuse=%t", width, reuse), func(b *testing.B) {
				var fused []float32
				if reuse {
					if err := embedVoxtralRealtimeTokenInto(cfg, weights, 3, &fused); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportAllocs()
				for b.Loop() {
					if reuse {
						if err := embedVoxtralRealtimeTokenInto(cfg, weights, 3, &fused); err != nil {
							b.Fatal(err)
						}
						for i := range fused {
							fused[i] += audio[i]
						}
					} else {
						embedding, err := EmbedVoxtralRealtimeToken(cfg, weights, 3)
						if err != nil {
							b.Fatal(err)
						}
						fused = make([]float32, width)
						for i := range fused {
							fused[i] = embedding[i] + audio[i]
						}
					}
					voxtralEmbeddingSink = fused
				}
			})
		}
	}
}
