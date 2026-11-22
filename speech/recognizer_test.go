package speech

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"sync"
	"testing"
	"time"
)

type decoderFunc func(context.Context, []float32) (string, error)

func (f decoderFunc) transcribe(ctx context.Context, samples []float32) (string, error) {
	return f(ctx, samples)
}

func (decoderFunc) Close() error { return nil }

func TestPCM16ConversionReusesAndClearsScratch(t *testing.T) {
	pcm := []byte{0x00, 0x80, 0x00, 0x00, 0x00, 0x40, 0xff, 0x7f}
	want := []float32{-1, 0, 0.5, 32767.0 / 32768}
	r := &Recognizer{gate: make(chan struct{}, 1), decoder: decoderFunc(func(_ context.Context, samples []float32) (string, error) {
		if !slices.Equal(samples, want) {
			t.Fatalf("converted samples = %v, want %v", samples, want)
		}
		return " Computer, Licht an. ", nil
	})}
	var scratch *float32
	for range 2 {
		text, err := r.TranscribePCM16(t.Context(), pcm)
		if err != nil || text != "Computer, Licht an." {
			t.Fatalf("transcription = %q, %v", text, err)
		}
		if !slices.Equal(r.pcm, make([]float32, len(want))) {
			t.Fatal("scratch retained audio")
		}
		if scratch != nil && scratch != &r.pcm[0] {
			t.Fatal("scratch was reallocated")
		}
		scratch = &r.pcm[0]
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.TranscribePCM16(t.Context(), pcm); !errors.Is(err, ErrClosed) {
		t.Fatalf("transcription after close = %v", err)
	}
}

func TestInvalidFloatSamplesNeverReachDecoder(t *testing.T) {
	r := &Recognizer{gate: make(chan struct{}, 1), decoder: decoderFunc(func(context.Context, []float32) (string, error) {
		t.Error("invalid input reached the decoder")
		return "", nil
	})}
	for _, input := range [][]float32{nil, {float32(math.NaN())}, {1.1}, make([]float32, MaxSamples+1)} {
		if _, err := r.Transcribe(t.Context(), input); err == nil {
			t.Error("invalid input passed")
		}
	}
}

func TestQueuedCancellationDoesNotCancelActiveTranscription(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	defer unblock()
	r := &Recognizer{gate: make(chan struct{}, 1), decoder: decoderFunc(func(ctx context.Context, pcm []float32) (string, error) {
		if pcm[0] != 0.5 {
			t.Error("canceled request started decoding")
			return "", nil
		}
		close(started)
		<-release
		return "first", ctx.Err()
	})}
	active := make(chan error, 1)
	go func() {
		_, err := r.Transcribe(t.Context(), []float32{0.5})
		active <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first transcription did not start")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := r.Transcribe(ctx, []float32{0}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled request returned %v", err)
	}
	select {
	case err := <-active:
		t.Fatalf("active request completed before release: %v", err)
	default:
	}
	unblock()
	if err := <-active; err != nil {
		t.Fatalf("active request was canceled: %v", err)
	}
}

func TestWakePhraseMatchesOnlyWholeLeadingWords(t *testing.T) {
	phrase, err := ParseWakePhrase("Hällo-Computer 2")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		text, want string
		match      bool
	}{
		{"„HÄLLO, Computer 2“: Licht an.", "Licht an.", true},
		{"Hällo Computer 2?", "", true},
		{"Hällo Computer 21, Licht aus.", "", false},
		{"Sag Hällo Computer 2. Licht an.", "", false},
	} {
		if command, match := phrase.Match(c.text); command != c.want || match != c.match {
			t.Errorf("%q: got %q/%v, want %q/%v", c.text, command, match, c.want, c.match)
		}
	}
	if _, err := ParseWakePhrase("hello\nworld"); err == nil {
		t.Fatal("invalid phrase accepted")
	}
}

func BenchmarkPCM16Conversion(b *testing.B) {
	pcm := make([]byte, 4*SampleRate*2)
	for i := range len(pcm) / 2 {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(i))
	}
	decoder := decoderFunc(func(_ context.Context, samples []float32) (string, error) {
		if samples[len(samples)-1] >= 0 {
			return "unexpected", nil
		}
		return "light on", nil
	})
	b.Run("freshBuffer", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			samples := make([]float32, len(pcm)/2)
			for i := range samples {
				samples[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768
			}
			if _, err := decoder.transcribe(b.Context(), samples); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("reuseAndClear", func(b *testing.B) {
		r := &Recognizer{gate: make(chan struct{}, 1), decoder: decoder}
		b.ReportAllocs()
		for b.Loop() {
			if _, err := r.TranscribePCM16(b.Context(), pcm); err != nil {
				b.Fatal(err)
			}
		}
	})
}
