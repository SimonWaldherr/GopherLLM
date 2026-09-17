package gopherllm

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
)

func TestVoxtralTranscriptionRejectsInvalidInputBeforeModelLoad(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := TranscribeVoxtralRealtime(ctx, "missing.gguf", []float32{0}, 0, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	for _, samples := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := TranscribeVoxtralRealtime(context.Background(), "missing.gguf", samples, 0, nil); err == nil || strings.Contains(err.Error(), "opening model") {
			t.Fatalf("invalid samples reached loader: %v", err)
		}
	}
	cfg, w := buildTinyVoxtralRealtimeWeights()
	if _, err := EncodeAudioVoxtralRealtimeContext(ctx, cfg, w, make([]float32, 1600)); !errors.Is(err, context.Canceled) {
		t.Fatalf("encoder cancellation: %v", err)
	}
}

func TestVoxtralTranscriptOmitsControlTokensAndPreservesUTF8(t *testing.T) {
	tok := &Tokenizer{Vocab: []string{"[STREAMING_PAD]", "[STREAMING_WORD]", "<other_control>", " Hello", "<0xC3>", "<0xBC>"}}
	types := []uint32{3, 3, 3, 1, 6, 6}
	var b strings.Builder
	for i := range tok.Vocab {
		b.WriteString(voxtralTranscriptPiece(tok, types, i))
	}
	if got := b.String(); got != " Helloü" {
		t.Fatalf("got %q", got)
	}
	if voxtralTranscriptPiece(tok, nil, 1) != "" {
		t.Fatal("marker fallback failed")
	}
}

func TestVoxtralOfflinePaddingAlignment(t *testing.T) {
	for _, n := range []int{1, 1279, 1280, 1281} {
		samples := make([]float32, n)
		samples[0] = .5
		samples[n-1] = .25
		padded := padVoxtralAudioOffline(samples, 1280)
		if len(padded)%1280 != 0 || len(padded) < n+(32+17)*1280 {
			t.Fatalf("invalid padding for %d", n)
		}
		for i, v := range samples {
			if padded[32*1280+i] != v {
				t.Fatal("audio shifted")
			}
		}
	}
}
