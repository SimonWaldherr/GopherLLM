package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func transcriptionWAV(samples int, floating bool) []byte {
	width := 2
	if floating {
		width = 4
	}
	raw := make([]byte, 44+samples*width)
	copy(raw, "RIFF")
	binary.LittleEndian.PutUint32(raw[4:], uint32(len(raw)-8))
	copy(raw[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(raw[16:], 16)
	binary.LittleEndian.PutUint16(raw[20:], 1)
	if floating {
		binary.LittleEndian.PutUint16(raw[20:], 3)
	}
	binary.LittleEndian.PutUint16(raw[22:], 1)
	binary.LittleEndian.PutUint32(raw[24:], 16000)
	binary.LittleEndian.PutUint32(raw[28:], uint32(16000*width))
	binary.LittleEndian.PutUint16(raw[32:], uint16(width))
	binary.LittleEndian.PutUint16(raw[34:], uint16(width*8))
	copy(raw[36:], "data")
	binary.LittleEndian.PutUint32(raw[40:], uint32(samples*width))
	return raw
}

func TestDecodeTranscriptionWAV(t *testing.T) {
	for _, floating := range []bool{false, true} {
		raw := transcriptionWAV(2, floating)
		if floating {
			binary.LittleEndian.PutUint32(raw[44:], math.Float32bits(-.5))
		} else {
			binary.LittleEndian.PutUint16(raw[44:], 49152)
		}
		samples, err := decodeTranscriptionWAV(raw)
		if err != nil || len(samples) != 2 || samples[0] != -.5 {
			t.Fatalf("floating=%v got %v %v", floating, samples, err)
		}
	}
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"empty", nil}, {"no samples", transcriptionWAV(0, false)}, {"truncated", transcriptionWAV(2, false)[:45]},
		{"long", transcriptionWAV(16000*31, false)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := decodeTranscriptionWAV(tc.data); err == nil {
				t.Fatal("invalid WAV accepted")
			}
		})
	}
	for _, offset := range []int{22, 24, 32, 34, 40} {
		raw := transcriptionWAV(2, false)
		binary.LittleEndian.PutUint16(raw[offset:], 65535)
		if _, err := decodeTranscriptionWAV(raw); err == nil {
			t.Fatalf("invalid field at %d accepted", offset)
		}
	}
	raw := transcriptionWAV(1, true)
	binary.LittleEndian.PutUint32(raw[44:], math.Float32bits(float32(math.NaN())))
	if _, err := decodeTranscriptionWAV(raw); err == nil {
		t.Fatal("NaN accepted")
	}
}

func audioCatalogFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	data := buildGGUF(3, []ggufKV{{"general.architecture", ggufStr, "voxtral_realtime"}, {"general.name", ggufStr, "Test Voxtral"}}, nil)
	if err := os.WriteFile(filepath.Join(dir, "voxtral.gguf"), data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "chat.gguf"), buildTinyLlamaGGUF(), 0600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func transcriptionRequest(t *testing.T, model, format string, audio []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	if err := form.WriteField("model", model); err != nil {
		t.Fatal(err)
	}
	if err := form.WriteField("response_format", format); err != nil {
		t.Fatal(err)
	}
	file, err := form.CreateFormFile("file", "recording.wav")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.Write(audio); err != nil {
		t.Fatal(err)
	}
	if err := form.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	return req
}

func TestAudioRoutesCatalogAndTranscription(t *testing.T) {
	dir := audioCatalogFixture(t)
	mux := http.NewServeMux()
	calls := 0
	type contextKey struct{}
	registerAudioRoutes(mux, make(chan struct{}, 1), HandlerOptions{ModelDir: dir, Features: Features{ModelCatalog: true}}, func(ctx context.Context, path string, samples []float32, extra int, _ io.Writer) (string, error) {
		calls++
		if path != filepath.Join(dir, "voxtral.gguf") || len(samples) != 1600 || extra != 0 || ctx.Value(contextKey{}) != "request" {
			t.Fatal("wrong transcription inputs")
		}
		return "Hallo Welt", nil
	}, nil)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("GET", "/models/audio", nil))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"id":"voxtral"`) || strings.Contains(rec.Body.String(), "chat") || strings.Contains(rec.Body.String(), dir) {
		t.Fatalf("catalog: %d %s", rec.Code, rec.Body)
	}
	for _, format := range []string{"json", "text"} {
		req := transcriptionRequest(t, "voxtral", format, transcriptionWAV(1600, false))
		req = req.WithContext(context.WithValue(req.Context(), contextKey{}, "request"))
		rec = httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("transcription: %d %s", rec.Code, rec.Body)
		}
		if format == "json" {
			var result map[string]string
			if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil || result["text"] != "Hallo Welt" {
				t.Fatalf("invalid JSON %s", rec.Body)
			}
		} else if rec.Body.String() != "Hallo Welt" {
			t.Fatalf("invalid text %s", rec.Body)
		}
	}
	for _, tc := range []struct {
		model, format string
		data          []byte
		status        int
	}{
		{"", "json", transcriptionWAV(1, false), 400}, {"../voxtral", "json", transcriptionWAV(1, false), 400},
		{"chat", "json", transcriptionWAV(1, false), 400}, {"voxtral", "srt", transcriptionWAV(1, false), 400},
		{"voxtral", "json", []byte("invalid audio"), 400}, {"voxtral", "json", make([]byte, transcriptionMaxBytes), 413},
	} {
		rec = httptest.NewRecorder()
		mux.ServeHTTP(rec, transcriptionRequest(t, tc.model, tc.format, tc.data))
		if rec.Code != tc.status {
			t.Fatalf("invalid request: got %d want %d: %s", rec.Code, tc.status, rec.Body)
		}
	}
	if calls != 2 {
		t.Fatalf("invalid requests reached inference: %d", calls)
	}
}

func TestAudioRoutesDeploymentAndCancellation(t *testing.T) {
	for _, opts := range []HandlerOptions{{}, {Features: AllFeatures(), DeploymentMode: DeploymentBrowser}} {
		h := NewHandler(nil, opts)
		defer h.Close()
		for _, path := range []string{"/models/audio", "/v1/audio/transcriptions"} {
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, httptest.NewRequest("POST", path, nil))
			if rec.Code == 200 || rec.Code == 500 {
				t.Fatalf("disabled audio route %s: %d", path, rec.Code)
			}
		}
	}
	mux := http.NewServeMux()
	sem := make(chan struct{}, 1)
	sem <- struct{}{}
	registerAudioRoutes(mux, sem, HandlerOptions{Features: Features{ModelCatalog: true}}, func(context.Context, string, []float32, int, io.Writer) (string, error) {
		t.Fatal("cancelled request reached inference")
		return "", nil
	}, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	mux.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/audio/transcriptions", nil).WithContext(ctx))
}

type realtimeAudioSpy struct {
	chunks [][]float32
	final  bool
	closed bool
}

func (s *realtimeAudioSpy) Push(_ context.Context, pcm []float32, final bool) (string, error) {
	s.chunks = append(s.chunks, append([]float32(nil), pcm...))
	s.final = final
	return "live text", nil
}
func (s *realtimeAudioSpy) Close() error { s.closed = true; return nil }

type blockingRealtimeAudioSpy struct {
	started chan struct{}
	closed  bool
}

func (s *blockingRealtimeAudioSpy) Push(ctx context.Context, _ []float32, _ bool) (string, error) {
	close(s.started)
	<-ctx.Done()
	return "", ctx.Err()
}
func (s *blockingRealtimeAudioSpy) Close() error { s.closed = true; return nil }

func TestRealtimeAudioDeleteCancelsInference(t *testing.T) {
	mux := http.NewServeMux()
	spy := &blockingRealtimeAudioSpy{started: make(chan struct{})}
	closeRoutes := registerRealtimeAudioRoutes(mux, make(chan struct{}, 1), HandlerOptions{ModelDir: audioCatalogFixture(t), Features: Features{ModelCatalog: true}}, func(context.Context, string) (realtimeTranscriber, error) { return spy, nil })
	defer closeRoutes()
	created := httptest.NewRecorder()
	mux.ServeHTTP(created, httptest.NewRequest("POST", "/v1/audio/realtime/sessions", strings.NewReader(`{"model":"voxtral"}`)))
	var session struct{ ID string }
	if err := json.Unmarshal(created.Body.Bytes(), &session); err != nil || session.ID == "" {
		t.Fatalf("create: %s", created.Body)
	}
	path := "/v1/audio/realtime/sessions/" + session.ID
	done := make(chan struct{})
	go func() {
		defer close(done)
		mux.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", path, bytes.NewReader([]byte{0, 0})))
	}()
	select {
	case <-spy.started:
	case <-time.After(3 * time.Second):
		t.Fatal("inference did not start")
	}
	deleted := make(chan int, 1)
	go func() {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest("DELETE", path, nil))
		deleted <- rec.Code
	}()
	select {
	case code := <-deleted:
		if code != 204 || !spy.closed {
			t.Fatalf("delete %d closed %v", code, spy.closed)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("delete did not cancel inference")
	}
	<-done
}

func TestRealtimeAudioEmptyFinalAndValidation(t *testing.T) {
	mux := http.NewServeMux()
	spy := &realtimeAudioSpy{}
	closeRoutes := registerRealtimeAudioRoutes(mux, make(chan struct{}, 1), HandlerOptions{ModelDir: audioCatalogFixture(t), Features: Features{ModelCatalog: true}}, func(context.Context, string) (realtimeTranscriber, error) { return spy, nil })
	defer closeRoutes()
	created := httptest.NewRecorder()
	mux.ServeHTTP(created, httptest.NewRequest("POST", "/v1/audio/realtime/sessions", strings.NewReader(`{"model":"voxtral"}`)))
	var session struct{ ID string }
	if err := json.Unmarshal(created.Body.Bytes(), &session); err != nil || session.ID == "" {
		t.Fatalf("create: %s", created.Body)
	}
	path := "/v1/audio/realtime/sessions/" + session.ID
	for _, tc := range []struct {
		raw    []byte
		status int
	}{{nil, 400}, {[]byte{0}, 400}, {make([]byte, realtimeMaxChunkBytes+2), 413}, {[]byte{0, 0}, 200}} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest("POST", path, bytes.NewReader(tc.raw)))
		if rec.Code != tc.status {
			t.Fatalf("body %d got %d want %d", len(tc.raw), rec.Code, tc.status)
		}
	}
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", path+"?final=1", nil))
	if rec.Code != 200 || !spy.final || !spy.closed || len(spy.chunks) != 2 || len(spy.chunks[1]) != 0 {
		t.Fatalf("empty final: %d %#v", rec.Code, spy)
	}
	// Finalization releases capacity synchronously, allowing the next session.
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/audio/realtime/sessions", strings.NewReader(`{"model":"voxtral"}`)))
	if rec.Code != 200 {
		t.Fatalf("next session: %d %s", rec.Code, rec.Body)
	}
}

func TestRealtimeAudioSessionRoutes(t *testing.T) {
	dir := audioCatalogFixture(t)
	mux := http.NewServeMux()
	var spy *realtimeAudioSpy
	closeRoutes := registerAudioRoutesWithRealtime(mux, make(chan struct{}, 1), HandlerOptions{ModelDir: dir, Features: Features{ModelCatalog: true}}, func(context.Context, string, []float32, int, io.Writer) (string, error) { return "", nil }, nil, func(_ context.Context, path string) (realtimeTranscriber, error) {
		if path != filepath.Join(dir, "voxtral.gguf") {
			t.Fatalf("unexpected model path %q", path)
		}
		spy = &realtimeAudioSpy{}
		return spy, nil
	})
	t.Cleanup(closeRoutes)
	start := httptest.NewRequest(http.MethodPost, "/v1/audio/realtime/sessions", strings.NewReader(`{"model":"voxtral"}`))
	start.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, start)
	if rec.Code != http.StatusOK {
		t.Fatalf("start: %d %s", rec.Code, rec.Body)
	}
	var created struct {
		ID         string `json:"id"`
		SampleRate int    `json:"sample_rate"`
		ChunkMS    int    `json:"chunk_ms"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &created); err != nil || len(created.ID) != 36 || created.SampleRate != 16000 || created.ChunkMS != 200 {
		t.Fatalf("bad start %#v: %v", created, err)
	}
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/audio/realtime/sessions", strings.NewReader(`{"model":"voxtral"}`)))
	if rec.Code != http.StatusConflict {
		t.Fatalf("second live session: got %d", rec.Code)
	}
	pcm := make([]byte, 4)
	neg, pos := int16(-16384), int16(16384)
	binary.LittleEndian.PutUint16(pcm[0:], uint16(neg))
	binary.LittleEndian.PutUint16(pcm[2:], uint16(pos))
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/audio/realtime/sessions/"+created.ID+"?final=1", bytes.NewReader(pcm)))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"text":"live text"`) || spy == nil || len(spy.chunks) != 1 || !spy.final || spy.chunks[0][0] != -.5 || spy.chunks[0][1] != .5 {
		t.Fatalf("chunk: %d %s %#v", rec.Code, rec.Body, spy)
	}
	if !spy.closed {
		t.Fatal("final response returned before releasing model")
	}
	// DELETE remains idempotent after finalization.
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/v1/audio/realtime/sessions/"+created.ID, nil))
	if rec.Code != http.StatusNoContent || !spy.closed {
		t.Fatalf("delete: %d closed=%v", rec.Code, spy.closed)
	}
	for _, tc := range []struct {
		method, path string
		body         io.Reader
		want         int
	}{
		{http.MethodPost, "/v1/audio/realtime/sessions", strings.NewReader(`{}`), http.StatusBadRequest},
		{http.MethodPost, "/v1/audio/realtime/sessions/not-a-session", bytes.NewReader(pcm), http.StatusNotFound},
		{http.MethodPost, "/v1/audio/realtime/sessions/", nil, http.StatusNotFound},
	} {
		rec = httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(tc.method, tc.path, tc.body))
		if rec.Code != tc.want {
			t.Fatalf("%s %s: got %d want %d", tc.method, tc.path, rec.Code, tc.want)
		}
	}
}
