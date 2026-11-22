package main

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func pcmWAV(t *testing.T) []byte {
	t.Helper()
	var wav bytes.Buffer
	if err := (gopherllm.SpeechAudio{Samples: []float32{0, 0.25, -0.25, 0}, SampleRate: 16000}).WriteWAV(&wav); err != nil {
		t.Fatal(err)
	}
	return wav.Bytes()
}

func TestVoiceAssistantTurn(t *testing.T) {
	wav := pcmWAV(t)
	a := &assistant{
		busy: make(chan struct{}, 1),
		transcribe: func(_ context.Context, recording []byte) (string, error) {
			if !bytes.Equal(wav, recording) {
				t.Fatal("audio changed in the assistant")
			}
			return "Wie heiße ich?", nil
		},
		reply: func(_ context.Context, messages []gopherllm.ChatMessage) (string, error) {
			var text []string
			for _, m := range messages {
				text = append(text, m.Content)
			}
			if !slices.Equal(text, []string{"Ich heiße Ada.", "Hallo Ada.", "Wie heiße ich?"}) {
				t.Fatalf("unexpected conversation: %v", text)
			}
			return "Du heißt Ada.", nil
		},
		speak: func(_ context.Context, _ string) (gopherllm.SpeechAudio, error) {
			return gopherllm.SpeechAudio{Samples: make([]float32, 4), SampleRate: 16000}, nil
		},
	}
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, err := form.CreateFormFile("audio", "test.wav")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write(wav)
	_ = form.WriteField("history", `[{"role":"user","text":"Ich heiße Ada."},{"role":"assistant","text":"Hallo Ada."}]`)
	_ = form.Close()
	req := httptest.NewRequest("POST", "http://127.0.0.1:8092/turn", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	rec := httptest.NewRecorder()
	a.routes().ServeHTTP(rec, req)
	var got answer
	if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &got) != nil || got.Input != "Wie heiße ich?" || got.Text != "Du heißt Ada." {
		t.Fatalf("unexpected response %d %s", rec.Code, rec.Body)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest("POST", "http://127.0.0.1:8092/speak", bytes.NewBufferString(got.Text))
	a.routes().ServeHTTP(rec, req)
	if rec.Code != 200 || rec.Header().Get("Content-Type") != "audio/wav" || !bytes.HasPrefix(rec.Body.Bytes(), []byte("RIFF")) {
		t.Fatalf("expected a spoken WAV reply, status %d", rec.Code)
	}
	req.Header.Set("Origin", "https://example.com")
	rec = httptest.NewRecorder()
	a.routes().ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("foreign origin got status %d", rec.Code)
	}
}
