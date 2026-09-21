package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
)

var (
	// ErrVoxtralModelClosed indicates that Close has stopped new sessions.
	ErrVoxtralModelClosed = errors.New("Voxtral model is closed")
	// ErrVoxtralBusy indicates that the model already has an active session.
	// Close that session before starting another, or use another model instance.
	ErrVoxtralBusy = errors.New("Voxtral model already has an active session")
	// ErrVoxtralSessionClosed indicates a closed, finalized or failed session.
	ErrVoxtralSessionClosed = errors.New("Voxtral realtime session is closed")
)

// VoxtralOption configures the local speech runtime without global settings.
type VoxtralOption func(*voxtralLoadSettings)
type voxtralLoadSettings struct {
	logw  io.Writer
	metal bool
}

// WithVoxtralLogWriter enables diagnostics. The default is io.Discard.
func WithVoxtralLogWriter(w io.Writer) VoxtralOption {
	return func(s *voxtralLoadSettings) {
		if w != nil {
			s.logw = w
		}
	}
}

// WithVoxtralMetal selects Metal when supported by the build. The default
// enables it when available; false disables both loading and fast decoding.
func WithVoxtralMetal(enabled bool) VoxtralOption {
	return func(s *voxtralLoadSettings) { s.metal = enabled }
}

// VoxtralModel owns one loaded Voxtral Realtime GGUF. Open once and create a
// fresh session for each recording. Exactly one session may be active per
// model; separate instances can serve independent concurrent recordings.
// All lifecycle methods are safe to call concurrently. Do not copy a model.
// The zero value is closed; use OpenVoxtral.
type VoxtralModel struct {
	mu     sync.Mutex
	source *VoxtralRealtimeSession
	active bool
	closed bool
}

// OpenVoxtral maps a local GGUF, loads its tokenizer, and prepares weights
// once. No network access or audio capture is performed. NewSession warms
// the silence prefix for each recording before accepting live audio.
func OpenVoxtral(ctx context.Context, path string, opts ...VoxtralOption) (*VoxtralModel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, fmt.Errorf("opening Voxtral model: %w", err)
	}
	m, err := openVoxtralBytes(ctx, mmap.Bytes(), mmap, opts...)
	if err != nil {
		_ = mmap.Close()
	}
	return m, err
}

// OpenVoxtralFromGGUFBytes loads a self-contained Voxtral Realtime GGUF
// already held in memory. It is intended for browser/WASM callers, where a
// user-selected File cannot be memory-mapped. The caller must keep data alive
// until Close returns when the supplied bytes are backed by external storage.
func OpenVoxtralFromGGUFBytes(ctx context.Context, data []byte, opts ...VoxtralOption) (*VoxtralModel, error) {
	return openVoxtralBytes(ctx, data, nil, opts...)
}

func openVoxtralBytes(ctx context.Context, data []byte, mmap *MmapFile, opts ...VoxtralOption) (*VoxtralModel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	settings := voxtralLoadSettings{logw: io.Discard, metal: MetalAvailable()}
	for _, opt := range opts {
		if opt != nil {
			opt(&settings)
		}
	}
	source := &VoxtralRealtimeSession{mmap: mmap, disableFast: !settings.metal}
	success := false
	defer func() {
		if !success {
			_ = source.Close()
		}
	}()
	gguf, err := ParseGGUF(data)
	if err != nil {
		return nil, fmt.Errorf("parsing Voxtral GGUF: %w", err)
	}
	source.cfg, source.weights, err = LoadVoxtralRealtimeModel(data, gguf, settings.metal, settings.logw)
	if err != nil {
		return nil, fmt.Errorf("loading Voxtral weights: %w", err)
	}
	source.tokenizer, err = TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("building Voxtral tokenizer: %w", err)
	}
	if source.cfg.Mel.SampleRate != VoxtralSampleRate {
		return nil, fmt.Errorf("Voxtral requires a %d Hz frontend, got %d", VoxtralSampleRate, source.cfg.Mel.SampleRate)
	}
	source.tokenTypes, _ = gguf.Metadata["tokenizer.ggml.token_type"].AsU32Array()
	source.eosID = int(gguf.GetU32("tokenizer.ggml.eos_token_id", 2))
	source.bosID = int(gguf.GetU32("tokenizer.ggml.bos_token_id", 1))
	if err := prepareVoxtralStreamWeights(ctx, &source.weights); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	success = true
	return &VoxtralModel{source: source}, nil
}

// NewSession reserves the model and warms a fresh frontend and decoder.
// Close the returned session even after Flush or an inference failure.
// Busy is returned immediately rather than blocking a caller indefinitely.
func (m *VoxtralModel) NewSession(ctx context.Context) (*VoxtralRealtimeSession, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	if m.closed || m.source == nil {
		m.mu.Unlock()
		return nil, ErrVoxtralModelClosed
	}
	if m.active {
		m.mu.Unlock()
		return nil, ErrVoxtralBusy
	}
	m.active = true
	source := m.source
	s := &VoxtralRealtimeSession{owner: m, cfg: source.cfg, weights: source.weights, tokenizer: source.tokenizer, tokenTypes: source.tokenTypes, eosID: source.eosID, bosID: source.bosID, disableFast: source.disableFast}
	m.mu.Unlock()
	if _, err := s.Push(ctx, nil, false); err != nil {
		_ = s.Close()
		return nil, err
	}
	return s, nil
}

// Close prevents new sessions. An already active session remains usable and
// keeps weights mapped until its Close. Close is idempotent; it does not
// interrupt inference. Cancel the Push context to stop ongoing work.
func (m *VoxtralModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	if !m.active {
		return m.releaseLocked()
	}
	return nil
}

func (m *VoxtralModel) releaseSession() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.active = false
	if m.closed {
		return m.releaseLocked()
	}
	return nil
}

func (m *VoxtralModel) releaseLocked() error {
	if m.source == nil {
		return nil
	}
	source := m.source
	m.source = nil
	return source.Close()
}

// VoxtralSampleRate is the required mono PCM sample rate, in Hz.
const VoxtralSampleRate = 16000

// Transcribe reuses loaded weights for a recording of normalized float32
// mono PCM at VoxtralSampleRate. It reserves and closes its own session and
// checks cancellation between 200 ms chunks. It uses the live decoder's
// finalization schedule; the legacy offline decoder remains separately available.
func (m *VoxtralModel) Transcribe(ctx context.Context, pcm []float32) (string, error) {
	if len(pcm) == 0 {
		return "", errors.New("audio is empty")
	}
	s, err := m.NewSession(ctx)
	if err != nil {
		return "", err
	}
	defer s.Close()
	for off := 0; off < len(pcm); off += 3200 {
		if _, err := s.Push(ctx, pcm[off:min(off+3200, len(pcm))], false); err != nil {
			return "", err
		}
	}
	return s.Flush(ctx)
}

// TranscribeOffline runs the same well-tested offline/batch encode-then-decode
// algorithm as the standalone TranscribeVoxtralRealtime, reusing this model's
// already-loaded weights instead of reopening the GGUF. Prefer this over
// Transcribe: the incremental live-session decoder Transcribe uses can return
// an empty transcript for some real speech inputs (see cmd/hestia's
// SpeechHost, which avoids it for the same reason), while this path shares
// the offline decoder's more thoroughly exercised behavior and, when the
// model was opened with Metal enabled, still gets the fast decoder step.
// Reserves and releases the model's single session slot, like Transcribe.
func (m *VoxtralModel) TranscribeOffline(ctx context.Context, samples []float32, maxExtraSteps int, logw io.Writer) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	m.mu.Lock()
	if m.closed || m.source == nil {
		m.mu.Unlock()
		return "", ErrVoxtralModelClosed
	}
	if m.active {
		m.mu.Unlock()
		return "", ErrVoxtralBusy
	}
	m.active = true
	source := m.source
	m.mu.Unlock()
	defer func() { _ = m.releaseSession() }()
	return decodeVoxtralRealtimeOffline(ctx, source.cfg, source.weights, source.tokenizer, source.tokenTypes, source.eosID, source.bosID, samples, maxExtraSteps, source.disableFast, logw)
}

// Flush finalizes the transcript without adding audio. Close is still required.
func (s *VoxtralRealtimeSession) Flush(ctx context.Context) (string, error) {
	return s.Push(ctx, nil, true)
}

// PushPCM16 accepts signed little-endian 16-bit mono PCM at 16 kHz. Each
// call must contain whole samples. WAV headers and resampling are not handled.
// The input is consumed synchronously and may be reused after return.
func (s *VoxtralRealtimeSession) PushPCM16(ctx context.Context, pcm []byte, final bool) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if len(pcm)%2 != 0 {
		return "", errors.New("PCM16 requires an even byte count")
	}
	samples := make([]float32, len(pcm)/2)
	for i := range samples {
		samples[i] = float32(int16(uint16(pcm[2*i])|uint16(pcm[2*i+1])<<8)) / 32768
	}
	return s.Push(ctx, samples, final)
}
