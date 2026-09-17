package server

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
)

const transcriptionMaxSeconds = 30
const transcriptionMaxBytes = 2 << 20

type transcriptionFunc func(context.Context, string, []float32, int, io.Writer) (string, error)

const realtimeMaxChunkBytes = 16000 * 2 * 2 // two seconds of PCM16
const realtimeSessionTTL = 2 * time.Minute

// realtimeTranscriber owns one audio-model instance for a browser session.
// Push receives little-endian signed 16-bit mono PCM at 16 kHz and returns the
// best transcript currently available. Implementations must be serial.
type realtimeTranscriber interface {
	Push(context.Context, []float32, bool) (string, error)
	Close() error
}

type realtimeFactory func(context.Context, string) (realtimeTranscriber, error)

// Audio uses a separate catalog selection and never replaces the chat runner.
// One transcription at a time bounds the offline encoder's working memory.
// The request owns its model mapping; cancellation/return releases it.
func registerAudioRoutes(mux *http.ServeMux, sem chan struct{}, opts HandlerOptions, transcribe transcriptionFunc) {
	_ = registerAudioRoutesWithRealtime(mux, sem, opts, transcribe, func(context.Context, string) (realtimeTranscriber, error) {
		return nil, fmt.Errorf("realtime transcriber is not configured")
	})
}

func registerAudioRoutesWithRealtime(mux *http.ServeMux, sem chan struct{}, opts HandlerOptions, transcribe transcriptionFunc, newRealtime realtimeFactory) func() {
	if !opts.Features.ModelCatalog {
		return func() {}
	}
	mux.HandleFunc("/models/audio", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		entries, err := discoverCatalog(opts.ModelDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		models := []map[string]string{}
		for _, entry := range entries {
			if entry.Architecture != "voxtral_realtime" {
				continue
			}
			name := entry.ModelName
			if name == "" {
				name = entry.FileName
			}
			models = append(models, map[string]string{"id": entry.ID, "name": name})
		}
		writeJSON(w, map[string]any{"models": models, "max_seconds": transcriptionMaxSeconds, "max_bytes": transcriptionMaxBytes})
	})
	audioSem := make(chan struct{}, 1)
	mux.HandleFunc("/v1/audio/transcriptions", withLimit(audioSem, withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		req.Body = http.MaxBytesReader(w, req.Body, transcriptionMaxBytes)
		defer req.Body.Close()
		err := req.ParseMultipartForm(transcriptionMaxBytes)
		if req.MultipartForm != nil {
			defer req.MultipartForm.RemoveAll()
		}
		if err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				http.Error(w, "audio upload exceeds 2 MiB", http.StatusRequestEntityTooLarge)
			} else {
				http.Error(w, "expected multipart form with model and file fields", http.StatusBadRequest)
			}
			return
		}
		selector := strings.TrimSpace(req.FormValue("model"))
		if selector == "" {
			http.Error(w, "choose a Voxtral audio model", http.StatusBadRequest)
			return
		}
		format := req.FormValue("response_format")
		if format != "" && format != "json" && format != "text" {
			http.Error(w, "response_format must be json or text", http.StatusBadRequest)
			return
		}
		entries, err := discoverCatalog(opts.ModelDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Only exact catalog IDs are accepted. Never treat a supplied selector as
		// a filesystem path or a chat-model load request.
		path := ""
		for _, entry := range entries {
			if entry.ID == selector && entry.Architecture == "voxtral_realtime" {
				path = entry.Path
				break
			}
		}
		if path == "" {
			http.Error(w, "Voxtral model not found in the configured model directory", http.StatusBadRequest)
			return
		}
		file, _, err := req.FormFile("file")
		if err != nil {
			http.Error(w, "missing audio file", http.StatusBadRequest)
			return
		}
		defer file.Close()
		raw, err := io.ReadAll(file)
		if err != nil {
			http.Error(w, "cannot read audio file", http.StatusBadRequest)
			return
		}
		samples, err := decodeTranscriptionWAV(raw)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		text, err := transcribe(req.Context(), path, samples, 0, io.Discard)
		if req.Context().Err() != nil {
			return
		}
		if err != nil {
			http.Error(w, "Voxtral transcription failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if format == "text" {
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			_, _ = io.WriteString(w, text)
			return
		}
		writeJSON(w, map[string]string{"text": text})
	})))
	return registerRealtimeAudioRoutes(mux, sem, opts, newRealtime)
}

type realtimeAudioSession struct {
	mu          sync.Mutex
	transcriber realtimeTranscriber
	timer       *time.Timer
	lastUsed    time.Time
	busy        bool
	closed      bool
	ctx         context.Context
	cancel      context.CancelFunc
}

func registerRealtimeAudioRoutes(mux *http.ServeMux, sem chan struct{}, opts HandlerOptions, newRealtime realtimeFactory) func() {
	if !opts.Features.ModelCatalog {
		return func() {}
	}
	var mu sync.Mutex
	sessions := map[string]*realtimeAudioSession{}
	active := make(chan struct{}, 1) // a live Voxtral model is large
	closed := false
	// Caller holds mu: detach atomically before waiting for in-flight inference.
	detach := func(id string) *realtimeAudioSession {
		session := sessions[id]
		delete(sessions, id)
		if session != nil {
			session.timer.Stop()
			session.cancel()
		}
		return session
	}
	closeSession := func(session *realtimeAudioSession) {
		if session != nil {
			session.mu.Lock()
			defer session.mu.Unlock()
			session.closed = true
			_ = session.transcriber.Close()
			<-active
		}
	}
	remove := func(id string) {
		mu.Lock()
		session := detach(id)
		mu.Unlock()
		closeSession(session)
	}
	var expire func(string)
	expire = func(id string) {
		mu.Lock()
		s := sessions[id]
		if s == nil {
			mu.Unlock()
			return
		}
		remaining := realtimeSessionTTL - time.Since(s.lastUsed)
		if s.busy || remaining > 0 {
			if s.busy {
				remaining = realtimeSessionTTL
			}
			s.timer.Reset(remaining)
			mu.Unlock()
			return
		}
		s = detach(id)
		mu.Unlock()
		closeSession(s)
	}
	lookupModel := func(selector string) (string, error) {
		entries, err := discoverCatalog(opts.ModelDir)
		if err != nil {
			return "", err
		}
		for _, entry := range entries {
			if entry.ID == selector && entry.Architecture == "voxtral_realtime" {
				return entry.Path, nil
			}
		}
		return "", errors.New("Voxtral model not found in the configured model directory")
	}
	mux.HandleFunc("/v1/audio/realtime/sessions", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var input struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 4096)).Decode(&input); err != nil {
			http.Error(w, "expected JSON with model", http.StatusBadRequest)
			return
		}
		path, err := lookupModel(strings.TrimSpace(input.Model))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		select {
		case active <- struct{}{}:
		case <-req.Context().Done():
			return
		default:
			http.Error(w, "another live transcription is active", http.StatusConflict)
			return
		}
		keepActive := false
		defer func() {
			if !keepActive {
				<-active
			}
		}()
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		case <-req.Context().Done():
			return
		}
		transcriber, err := newRealtime(req.Context(), path)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if req.Context().Err() != nil {
			_ = transcriber.Close()
			return
		}
		var token [18]byte
		if _, err := rand.Read(token[:]); err != nil {
			_ = transcriber.Close()
			http.Error(w, "cannot create audio session", http.StatusInternalServerError)
			return
		}
		id := hex.EncodeToString(token[:])
		mu.Lock()
		if closed {
			mu.Unlock()
			_ = transcriber.Close()
			http.Error(w, "audio service closed", http.StatusServiceUnavailable)
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		sessions[id] = &realtimeAudioSession{transcriber: transcriber, lastUsed: time.Now(), ctx: ctx, cancel: cancel, timer: time.AfterFunc(realtimeSessionTTL, func() { expire(id) })}
		mu.Unlock()
		keepActive = true
		writeJSON(w, map[string]any{"id": id, "sample_rate": 16000, "chunk_ms": 200})
	})
	mux.HandleFunc("/v1/audio/realtime/sessions/", func(w http.ResponseWriter, req *http.Request) {
		id := strings.TrimPrefix(req.URL.Path, "/v1/audio/realtime/sessions/")
		if id == "" || strings.Contains(id, "/") {
			http.NotFound(w, req)
			return
		}
		if req.Method == http.MethodDelete {
			remove(id)
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		req.Body = http.MaxBytesReader(w, req.Body, realtimeMaxChunkBytes)
		defer req.Body.Close()
		raw, err := io.ReadAll(req.Body)
		if err != nil {
			http.Error(w, "audio chunk exceeds 2 seconds", http.StatusRequestEntityTooLarge)
			return
		}
		final := req.URL.Query().Get("final") == "1"
		if (len(raw) == 0 && !final) || len(raw)%2 != 0 {
			http.Error(w, "expected 16-bit PCM audio", http.StatusBadRequest)
			return
		}
		pcm := make([]float32, len(raw)/2)
		for i := range pcm {
			pcm[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:]))) / 32768
		}
		mu.Lock()
		session := sessions[id]
		mu.Unlock()
		if session == nil {
			http.Error(w, "audio session not found or expired", http.StatusNotFound)
			return
		}
		select {
		case sem <- struct{}{}:
			defer func() { <-sem }()
		case <-req.Context().Done():
			return
		}
		session.mu.Lock()
		if session.closed {
			session.mu.Unlock()
			http.Error(w, "audio session closed", http.StatusGone)
			return
		}
		mu.Lock()
		session.busy = true
		session.lastUsed = time.Now()
		mu.Unlock()
		ctx, cancel := context.WithCancel(req.Context())
		stop := context.AfterFunc(session.ctx, cancel)
		text, err := session.transcriber.Push(ctx, pcm, final)
		stop()
		cancel()
		if final || err != nil {
			session.closed = true
		}
		mu.Lock()
		session.busy = false
		session.lastUsed = time.Now()
		mu.Unlock()
		session.mu.Unlock()
		if final || err != nil {
			remove(id)
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		writeJSON(w, map[string]any{"text": text, "received_seconds": float64(len(pcm)) / 16000})
	})
	return func() {
		mu.Lock()
		closed = true
		remaining := sessions
		sessions = map[string]*realtimeAudioSession{}
		for _, session := range remaining {
			session.timer.Stop()
			session.cancel()
		}
		mu.Unlock()
		for _, session := range remaining {
			session.mu.Lock()
			session.closed = true
			_ = session.transcriber.Close()
			session.mu.Unlock()
			<-active
		}
	}
}

// The browser resamples its supported input formats using Web Audio and sends
// WAV. API clients send mono 16 kHz PCM16 or IEEE float32 WAV directly. Keep
// the server parser strict so truncated chunks and NaNs never reach inference.
func decodeTranscriptionWAV(raw []byte) ([]float32, error) {
	if len(raw) < 12 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, fmt.Errorf("send a mono 16 kHz PCM16 or float32 WAV file")
	}
	if uint64(binary.LittleEndian.Uint32(raw[4:8]))+8 != uint64(len(raw)) {
		return nil, fmt.Errorf("invalid WAV container length")
	}
	var format, channels, bits, blockAlign uint16
	var rate uint32
	var data []byte
	haveFmt := false
	for offset := 12; offset < len(raw); {
		if len(raw)-offset < 8 {
			return nil, fmt.Errorf("truncated WAV chunk header")
		}
		size := uint64(binary.LittleEndian.Uint32(raw[offset+4 : offset+8]))
		start := offset + 8
		if size > uint64(len(raw)-start) {
			return nil, fmt.Errorf("truncated WAV chunk")
		}
		end := start + int(size)
		switch string(raw[offset : offset+4]) {
		case "fmt ":
			if haveFmt || size < 16 {
				return nil, fmt.Errorf("invalid WAV fmt chunk")
			}
			haveFmt = true
			format = binary.LittleEndian.Uint16(raw[start:])
			channels = binary.LittleEndian.Uint16(raw[start+2:])
			rate = binary.LittleEndian.Uint32(raw[start+4:])
			blockAlign = binary.LittleEndian.Uint16(raw[start+12:])
			bits = binary.LittleEndian.Uint16(raw[start+14:])
		case "data":
			if data != nil {
				return nil, fmt.Errorf("multiple WAV data chunks are unsupported")
			}
			data = raw[start:end]
		}
		offset = end + int(size%2)
		if offset > len(raw) {
			return nil, fmt.Errorf("missing WAV chunk padding")
		}
	}
	if !haveFmt || channels != 1 || rate != 16000 || !((format == 1 && bits == 16) || (format == 3 && bits == 32)) || blockAlign != bits/8 {
		return nil, fmt.Errorf("audio must be mono 16 kHz PCM16 or float32 WAV")
	}
	width := int(bits / 8)
	if len(data) == 0 || len(data)%width != 0 {
		return nil, fmt.Errorf("empty or incomplete WAV samples")
	}
	count := len(data) / width
	if count > 16000*transcriptionMaxSeconds {
		return nil, fmt.Errorf("audio exceeds the %d-second limit", transcriptionMaxSeconds)
	}
	samples := make([]float32, count)
	for i := range samples {
		if format == 1 {
			samples[i] = float32(int16(binary.LittleEndian.Uint16(data[i*width:]))) / 32768
		} else {
			samples[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*width:]))
			if math.IsNaN(float64(samples[i])) || math.IsInf(float64(samples[i]), 0) {
				return nil, fmt.Errorf("WAV contains non-finite samples")
			}
		}
	}
	return samples, nil
}
