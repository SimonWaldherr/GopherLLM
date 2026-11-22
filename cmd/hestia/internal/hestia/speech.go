package hestia

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/SimonWaldherr/GopherLLM/speech"
)

const (
	SpeechVoxtral = string(speech.Voxtral)
	SpeechPhonon  = string(speech.Phonon)
	SpeechWhisper = string(speech.Whisper)

	voicePCMBytesPerSecond = speech.SampleRate * 2
	maxVoiceSeconds        = speech.MaxSamples / speech.SampleRate
	// MaxVoicePCMBytes bounds one recording of 16 kHz mono PCM16.
	MaxVoicePCMBytes = speech.MaxSamples * 2
)

var (
	ErrSpeechNotConfigured = errors.New("hestia: no speech model configured")
	ErrVoiceSessionBusy    = errors.New("hestia: a voice recording is already in progress")
	errVoiceSessionDone    = errors.New("hestia: voice session already finished")
)

// SpeechConfig selects local speech recognition. An empty configuration disables voice.
type SpeechConfig struct {
	Backend    string `json:"backend,omitempty"`
	ModelPath  string `json:"model_path,omitempty"`
	WhisperURL string `json:"whisper_url,omitempty"`
	Language   string `json:"language,omitempty"`
}

func (c SpeechConfig) Validate() error {
	if c == (SpeechConfig{}) {
		return nil
	}
	if c.WhisperURL != "" {
		return fmt.Errorf("speech: configure a local Whisper model_path instead of whisper_url")
	}
	switch c.Backend {
	case SpeechVoxtral, SpeechPhonon:
		if strings.TrimSpace(c.ModelPath) == "" || c.Language != "" {
			return fmt.Errorf("speech: %s requires model_path and detects the language automatically", c.Backend)
		}
		if c.Backend == SpeechPhonon && !strings.EqualFold(filepath.Ext(c.ModelPath), ".fermion") {
			return fmt.Errorf("speech: Phonon requires an extracted .fermion model beside config.json")
		}
	case SpeechWhisper:
		if strings.TrimSpace(c.ModelPath) == "" {
			return fmt.Errorf("speech: Whisper requires a local model_path")
		}
	default:
		return fmt.Errorf("speech: backend must be voxtral, phonon or whisper")
	}
	return nil
}

// SpeechHost owns one resident speech backend and one active recording slot.
type SpeechHost struct {
	backend string
	busy    atomic.Bool

	mu         sync.RWMutex
	transcribe func(context.Context, []byte) (string, error)
	closeModel func() error
}

// OpenSpeechHost opens a resident Voxtral model for callers using the legacy API.
func OpenSpeechHost(ctx context.Context, modelPath string, logw io.Writer) (*SpeechHost, error) {
	return OpenConfiguredSpeechHost(ctx, SpeechConfig{Backend: SpeechVoxtral, ModelPath: modelPath}, logw)
}

// OpenConfiguredSpeechHost initializes speech recognition without recording or
// sending any audio. Local models stay loaded until the host closes.
func OpenConfiguredSpeechHost(ctx context.Context, cfg SpeechConfig, logw io.Writer) (*SpeechHost, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.Backend == "" {
		return nil, nil
	}
	h := &SpeechHost{backend: cfg.Backend}
	model, err := speech.Open(ctx, cfg.ModelPath, speech.Options{Backend: speech.Backend(cfg.Backend), Language: cfg.Language}, logw)
	if err != nil {
		return nil, err
	}
	h.transcribe = model.TranscribePCM16
	h.closeModel = model.Close
	return h, nil
}

// Backend returns the configured engine name without exposing its address or path.
func (h *SpeechHost) Backend() string {
	if h == nil {
		return ""
	}
	return h.backend
}

func (h *SpeechHost) Close() error {
	if h == nil {
		return nil
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	h.transcribe = nil
	if h.closeModel != nil {
		closeModel := h.closeModel
		h.closeModel = nil
		return closeModel()
	}
	return nil
}

func (h *SpeechHost) reserve(ctx context.Context) error {
	if h == nil {
		return ErrSpeechNotConfigured
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if !h.busy.CompareAndSwap(false, true) {
		return ErrVoiceSessionBusy
	}
	h.mu.RLock()
	ready := h.transcribe != nil
	h.mu.RUnlock()
	if !ready {
		h.busy.Store(false)
		return ErrSpeechNotConfigured
	}
	return nil
}

// TranscribePCM16 transcribes a complete clip and releases the recording slot
// on completion or cancellation.
func (h *SpeechHost) TranscribePCM16(ctx context.Context, pcm []byte) (string, error) {
	if err := h.reserve(ctx); err != nil {
		return "", err
	}
	defer h.busy.Store(false)
	return h.transcribeClip(ctx, pcm)
}

func (h *SpeechHost) transcribeClip(ctx context.Context, pcm []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if len(pcm) == 0 {
		return "", fmt.Errorf("hestia: no audio was recorded")
	}
	if err := checkVoicePCM(len(pcm), 0); err != nil {
		return "", err
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.transcribe == nil {
		return "", ErrSpeechNotConfigured
	}
	text, err := h.transcribe(ctx, pcm)
	if err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	return strings.TrimSpace(text), nil
}

// VoiceSession keeps at most one turn of raw audio in memory. PushPCM16
// reports recording progress; Finish decodes the recording once.
type VoiceSession struct {
	host *SpeechHost
	mu   sync.Mutex
	pcm  []byte
	done bool
}

func (h *SpeechHost) StartVoiceSession(ctx context.Context) (*VoiceSession, error) {
	if err := h.reserve(ctx); err != nil {
		return nil, err
	}
	return &VoiceSession{host: h}, nil
}

func (vs *VoiceSession) PushPCM16(ctx context.Context, pcm []byte) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	vs.mu.Lock()
	defer vs.mu.Unlock()
	if vs.done {
		return "", errVoiceSessionDone
	}
	if err := checkVoicePCM(len(pcm), len(vs.pcm)); err != nil {
		return "", err
	}
	vs.pcm = append(vs.pcm, pcm...)
	seconds := float64(len(vs.pcm)) / voicePCMBytesPerSecond
	return fmt.Sprintf("… Aufnahme läuft (%.1fs) …", seconds), nil
}

// Finish transcribes the buffered recording and releases the recording slot.
func (vs *VoiceSession) Finish(ctx context.Context) (string, error) {
	vs.mu.Lock()
	if vs.done {
		vs.mu.Unlock()
		return "", errVoiceSessionDone
	}
	vs.done = true
	pcm := vs.pcm
	vs.pcm = nil
	vs.mu.Unlock()
	defer vs.host.busy.Store(false)
	return vs.host.transcribeClip(ctx, pcm)
}

// Abort releases the recording slot and discards captured audio.
func (vs *VoiceSession) Abort() error {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	if !vs.done {
		vs.done = true
		vs.pcm = nil
		vs.host.busy.Store(false)
	}
	return nil
}

func checkVoicePCM(n, recorded int) error {
	if n%2 != 0 {
		return fmt.Errorf("hestia: PCM16 chunk has an odd byte count")
	}
	if n > MaxVoicePCMBytes-recorded {
		return fmt.Errorf("hestia: audio exceeds the %d-second limit", maxVoiceSeconds)
	}
	return nil
}
