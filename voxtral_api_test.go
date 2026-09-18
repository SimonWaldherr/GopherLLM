package gopherllm

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
)

func tinyVoxtralHost() *VoxtralModel {
	cfg, w := buildTinyVoxtralRealtimeWeights()
	cfg.StreamingPadTokenID = 3
	vocab := make([]string, cfg.Decoder.VocabSize)
	for i := range vocab {
		vocab[i] = "a"
	}
	return &VoxtralModel{source: &VoxtralRealtimeSession{cfg: cfg, weights: w, tokenizer: &Tokenizer{Vocab: vocab}, bosID: 1, eosID: 2, disableFast: true}}
}

func TestVoxtralModelReusesWeightsWithFreshSessions(t *testing.T) {
	m := tinyVoxtralHost()
	defer m.Close()
	ctx := context.Background()
	pcm := make([]float32, 1927)
	for i := range pcm {
		pcm[i] = float32(math.Sin(float64(i) * .1))
	}
	first, err := m.NewSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.NewSession(ctx); !errors.Is(err, ErrVoxtralBusy) {
		t.Fatalf("busy: %v", err)
	}
	weights := first.weights.Decoder.TokenEmbd.F32
	want, err := first.Push(ctx, pcm, true)
	if err != nil {
		t.Fatal(err)
	}
	oldState := first.decoder
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	second, err := m.NewSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if len(weights) > 0 && &weights[0] != &second.weights.Decoder.TokenEmbd.F32[0] {
		t.Fatal("weights copied")
	}
	if second.decoder == oldState || second.text.Len() != 0 {
		t.Fatal("conversation state reused")
	}
	got, err := second.Push(ctx, pcm, true)
	if err != nil || got != want {
		t.Fatalf("got %q want %q err %v", got, want, err)
	}
}

func TestVoxtralModelCloseDefersReleaseUntilSessionClose(t *testing.T) {
	m := tinyVoxtralHost()
	s, err := m.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if m.source == nil {
		t.Fatal("released active weights")
	}
	if _, err := m.NewSession(context.Background()); !errors.Is(err, ErrVoxtralModelClosed) {
		t.Fatal(err)
	}
	if _, err := s.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if m.source != nil {
		t.Fatal("weights retained after final close")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Push(context.Background(), nil, false); !errors.Is(err, ErrVoxtralSessionClosed) {
		t.Fatal(err)
	}
}

func TestVoxtralModelCancellationAndPCM16(t *testing.T) {
	m := tinyVoxtralHost()
	defer m.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.NewSession(ctx); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if m.active {
		t.Fatal("cancelled request reserved model")
	}
	s, err := m.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if _, err := s.PushPCM16(context.Background(), []byte{0}, false); err == nil {
		t.Fatal("accepted partial sample")
	}
	if _, err := s.PushPCM16(ctx, make([]byte, 64), false); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := s.PushPCM16(context.Background(), make([]byte, 6400), true); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Flush(context.Background()); !errors.Is(err, ErrVoxtralSessionClosed) {
		t.Fatal(err)
	}
}

func TestVoxtralTranscribeReleasesSession(t *testing.T) {
	m := tinyVoxtralHost()
	defer m.Close()
	ctx := context.Background()
	if _, err := m.Transcribe(ctx, []float32{float32(math.NaN())}); err == nil {
		t.Fatal("accepted NaN")
	}
	if m.active {
		t.Fatal("failed transcription retained lease")
	}
	if _, err := m.Transcribe(ctx, make([]float32, 1927)); err != nil {
		t.Fatal(err)
	}
	if m.active {
		t.Fatal("transcription retained lease")
	}
}

func TestOpenVoxtralCancelledAndZeroValue(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := OpenVoxtral(ctx, "missing.gguf"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	var m VoxtralModel
	if _, err := m.NewSession(context.Background()); !errors.Is(err, ErrVoxtralModelClosed) {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestVoxtralConcurrentCloseAndPush(t *testing.T) {
	m := tinyVoxtralHost()
	s, err := m.NewSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		if _, err := s.Push(context.Background(), make([]float32, 1927), true); err != nil {
			t.Error(err)
		}
	}()
	go func() {
		defer wg.Done()
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	}()
	wg.Wait()
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if m.source != nil {
		t.Fatal("retained weights")
	}
}
