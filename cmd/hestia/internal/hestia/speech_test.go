package hestia

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestSpeechHostRetainsOneRecordingAndDecodesOnce(t *testing.T) {
	var calls, closed int
	ctx := t.Context()
	clip := []byte{0, 128, 255, 127}
	host := &SpeechHost{
		transcribe: func(_ context.Context, pcm []byte) (string, error) {
			calls++
			if !bytes.Equal(pcm, clip) {
				t.Fatalf("PCM = %v, want %v", pcm, clip)
			}
			return " test command ", nil
		},
		closeModel: func() error { closed++; return nil },
	}
	recording, err := host.StartVoiceSession(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := host.TranscribePCM16(ctx, clip); !errors.Is(err, ErrVoiceSessionBusy) {
		t.Fatalf("single-shot transcription during a recording: %v", err)
	}
	if _, err := recording.PushPCM16(ctx, clip[:1]); err == nil {
		t.Fatal("accepted odd-length PCM")
	}
	for _, part := range [][]byte{clip[:2], clip[2:]} {
		if _, err := recording.PushPCM16(ctx, part); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := recording.PushPCM16(ctx, make([]byte, MaxVoicePCMBytes)); err == nil {
		t.Fatal("accepted a recording exceeding the limit")
	}
	if calls != 0 {
		t.Fatal("decoded a partial recording")
	}
	if text, err := recording.Finish(ctx); text != "test command" || err != nil || calls != 1 || recording.pcm != nil {
		t.Fatalf("finished = %q, %v, calls = %d, retained bytes = %d", text, err, calls, len(recording.pcm))
	}
	if _, err := host.TranscribePCM16(ctx, clip); err != nil || calls != 2 {
		t.Fatalf("next transcription failed after finish: %v", err)
	}
	if err := host.Close(); err != nil {
		t.Fatal(err)
	}
	if err := host.Close(); err != nil || closed != 1 {
		t.Fatalf("host closed %d times, err = %v", closed, err)
	}
	if _, err := host.StartVoiceSession(ctx); !errors.Is(err, ErrSpeechNotConfigured) {
		t.Fatalf("start on closed host: %v", err)
	}
}

func TestSpeechHostReleasesAbortedAndCancelledRecordings(t *testing.T) {
	host := &SpeechHost{transcribe: func(context.Context, []byte) (string, error) {
		t.Fatal("cancelled audio must not be decoded")
		return "", nil
	}}
	first, err := host.StartVoiceSession(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.PushPCM16(t.Context(), []byte{0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := first.Abort(); err != nil || first.pcm != nil {
		t.Fatalf("abort = %v, retained bytes = %d", err, len(first.pcm))
	}
	second, err := host.StartVoiceSession(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.PushPCM16(t.Context(), []byte{0, 0}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := second.Finish(ctx); !errors.Is(err, context.Canceled) || second.pcm != nil {
		t.Fatalf("cancelled finish = %v, retained bytes = %d", err, len(second.pcm))
	}
	if _, err := host.StartVoiceSession(t.Context()); err != nil {
		t.Fatal(err)
	}
}

func TestSpeechConfigValidatesBackendInputs(t *testing.T) {
	for _, cfg := range []SpeechConfig{
		{},
		{Backend: SpeechVoxtral, ModelPath: "/tmp/voxtral.gguf"},
		{Backend: SpeechPhonon, ModelPath: "/tmp/model.fermion"},
		{Backend: SpeechWhisper, ModelPath: "/tmp/ggml-base.bin"},
	} {
		if err := cfg.Validate(); err != nil {
			t.Errorf("valid speech config %+v rejected: %v", cfg, err)
		}
	}
	for _, cfg := range []SpeechConfig{
		{ModelPath: "/tmp/model.fermion"},
		{Backend: SpeechVoxtral},
		{Backend: SpeechWhisper, Language: "de"},
		{Backend: SpeechWhisper, WhisperURL: "http://127.0.0.1:8093/inference"},
		{Backend: SpeechVoxtral, ModelPath: "/tmp/voxtral.gguf", Language: "de"},
		{Backend: SpeechPhonon, ModelPath: "/tmp/model.fsq"},
	} {
		if err := cfg.Validate(); err == nil {
			t.Errorf("invalid speech config %+v accepted", cfg)
		}
	}
}

func TestSpeechNativeBackendsRejectUnavailableModel(t *testing.T) {
	for _, name := range []string{SpeechVoxtral, SpeechPhonon} {
		ext := ".gguf"
		if name == SpeechPhonon {
			ext = ".fermion"
		}
		path := filepath.Join(t.TempDir(), "model"+ext)
		h, err := OpenConfiguredSpeechHost(t.Context(), SpeechConfig{Backend: name, ModelPath: path}, io.Discard)
		if h != nil || !errors.Is(err, os.ErrNotExist) {
			t.Errorf("open missing %s = %v, %v", name, h, err)
		}
	}
}
