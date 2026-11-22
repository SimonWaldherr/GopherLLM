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
	"time"

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
		reply: func(_ context.Context, messages []gopherllm.ChatMessage, _ func(string) bool) (string, error) {
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

func TestVoiceAssistantStreamsBeforeGenerationFinishes(t *testing.T) {
	release := make(chan struct{})
	const reply = "Guten Morgen!"
	a := &assistant{
		busy: make(chan struct{}, 1),
		reply: func(ctx context.Context, _ []gopherllm.ChatMessage, emit func(string) bool) (string, error) {
			if !emit("Guten") {
				return "", context.Canceled
			}
			select {
			case <-release:
				emit(" Morgen!")
				return reply, nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		},
	}
	srv := httptest.NewServer(a.routes())
	defer srv.Close()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("text", "Hallo")
	_ = form.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, srv.URL+"/turn", &body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", form.FormDataContentType())
	req.Header.Set("Accept", turnStreamType)
	res, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK || res.Header.Get("Content-Type") != turnStreamType {
		t.Fatalf("unexpected streaming response: %v", res)
	}
	decoder := json.NewDecoder(res.Body)
	next := func(kind string) turnEvent {
		t.Helper()
		var event turnEvent
		if err := decoder.Decode(&event); err != nil || event.Type != kind {
			t.Fatalf("event = %+v, error = %v, want %s", event, err, kind)
		}
		return event
	}
	if input, delta := next("input"), next("delta"); input.Text != "Hallo" || delta.Text != "Guten" {
		t.Fatalf("initial events = %+v %+v", input, delta)
	}
	close(release)
	if delta, final := next("delta"), next("done"); delta.Text != " Morgen!" || final.Result == nil ||
		final.Result.Input != "Hallo" || final.Result.Text != reply {
		t.Fatalf("remaining events = %+v %+v", delta, final)
	}
}
