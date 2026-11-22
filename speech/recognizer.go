// Package speech provides resident, local speech recognition for native hosts.
package speech

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"strings"
	"unicode/utf8"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/engine"
)

type Backend string

const (
	Voxtral Backend = "voxtral"
	Phonon  Backend = "phonon"
	Whisper Backend = "whisper"

	SampleRate = 16000
	MaxSamples = 30 * SampleRate
)

var ErrClosed = errors.New("speech recognizer is closed")

type Options struct {
	Backend  Backend `json:"backend"`
	Language string  `json:"language,omitempty"` // Whisper language code, defaults to automatic detection.
	CPUOnly  bool    `json:"cpu_only,omitempty"` // Disable Metal for Voxtral.
}

// Available reports whether the build includes the requested backend.
func Available(backend Backend) bool {
	switch backend {
	case Voxtral, Phonon, Whisper:
		return true
	default:
		return false
	}
}

type decoder interface {
	transcribe(context.Context, []float32) (string, error)
	Close() error
}

// Recognizer owns one loaded model. Calls on one recognizer are serialized.
// An operation canceled while waiting never starts inference.
type Recognizer struct {
	gate    chan struct{}
	decoder decoder
	pcm     []float32
}

// Open loads a local model once. There are no downloads or microphone calls.
// Phonon takes model.fermion with its sibling config.json.
func Open(ctx context.Context, path string, opts Options, logw io.Writer) (*Recognizer, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(path) == "" {
		return nil, errors.New("speech model path is empty")
	}
	var model decoder
	switch opts.Backend {
	case Voxtral:
		if opts.Language != "" {
			return nil, errors.New("Voxtral detects the transcription language automatically")
		}
		m, err := gopherllm.OpenVoxtral(ctx, path, gopherllm.WithVoxtralLogWriter(logw), gopherllm.WithVoxtralMetal(!opts.CPUOnly && gopherllm.MetalAvailable()))
		if err != nil {
			return nil, err
		}
		model = voxtralDecoder{m}
	case Phonon:
		if opts.Language != "" || opts.CPUOnly {
			return nil, errors.New("Phonon detects the language automatically and uses the CPU")
		}
		if !strings.EqualFold(filepath.Ext(path), ".fermion") {
			return nil, errors.New("Phonon requires extracted model.fermion beside config.json")
		}
		m, err := gopherllm.OpenParakeet(path, logw)
		if err != nil {
			return nil, err
		}
		model = phononDecoder{m}
	case Whisper:
		m, err := engine.OpenWhisper(ctx, path, opts.Language)
		if err != nil {
			return nil, err
		}
		model = whisperDecoder{m, opts.Language}
	default:
		return nil, fmt.Errorf("unknown speech backend %q", opts.Backend)
	}
	if err := ctx.Err(); err != nil {
		_ = model.Close()
		return nil, err
	}
	return &Recognizer{gate: make(chan struct{}, 1), decoder: model}, nil
}

type voxtralDecoder struct{ *gopherllm.VoxtralModel }

func (d voxtralDecoder) transcribe(ctx context.Context, pcm []float32) (string, error) {
	return d.TranscribeOffline(ctx, pcm, 0, io.Discard)
}

type phononDecoder struct{ *gopherllm.ParakeetModel }

func (d phononDecoder) transcribe(ctx context.Context, pcm []float32) (string, error) {
	return d.TranscribeOffline(ctx, pcm, nil)
}

type whisperDecoder struct {
	*engine.WhisperModel
	language string
}

func (d whisperDecoder) transcribe(ctx context.Context, pcm []float32) (string, error) {
	return d.Transcribe(ctx, pcm, d.language)
}

// Transcribe accepts normalized 16 kHz mono samples. The input is borrowed
// for the duration of the call and may be reused once it returns.
func (r *Recognizer) Transcribe(ctx context.Context, samples []float32) (string, error) {
	if err := checkSamples(len(samples)); err != nil {
		return "", err
	}
	if err := r.acquire(ctx); err != nil {
		return "", err
	}
	defer r.release()
	for _, sample := range samples {
		if math.IsNaN(float64(sample)) || sample < -1 || sample > 1 {
			return "", errors.New("speech samples must be normalized finite float32 values")
		}
	}
	return r.run(ctx, samples)
}

// TranscribePCM16 accepts little-endian 16 kHz mono PCM and reuses conversion
// scratch. Scratch is cleared after inference, including on failure.
func (r *Recognizer) TranscribePCM16(ctx context.Context, pcm []byte) (string, error) {
	if len(pcm)%2 != 0 {
		return "", errors.New("PCM16 requires an even byte count")
	}
	n := len(pcm) / 2
	if err := checkSamples(n); err != nil {
		return "", err
	}
	if err := r.acquire(ctx); err != nil {
		return "", err
	}
	defer r.release()
	if cap(r.pcm) < n {
		r.pcm = make([]float32, n)
	}
	samples := r.pcm[:n]
	defer clear(samples)
	for i := range samples {
		samples[i] = float32(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768
	}
	return r.run(ctx, samples)
}

func checkSamples(n int) error {
	if n < 1 || n > MaxSamples {
		return fmt.Errorf("speech recording must contain 1 to %d samples at %d Hz", MaxSamples, SampleRate)
	}
	return nil
}

func (r *Recognizer) acquire(ctx context.Context) error {
	if r == nil || r.gate == nil {
		return ErrClosed
	}
	select {
	case r.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			r.release()
			return err
		}
		if r.decoder == nil {
			r.release()
			return ErrClosed
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *Recognizer) release() { <-r.gate }

func (r *Recognizer) run(ctx context.Context, samples []float32) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	text, err := r.decoder.transcribe(ctx, samples)
	if ctxErr := ctx.Err(); ctxErr != nil {
		return "", ctxErr
	}
	if err != nil {
		return "", err
	}
	if !utf8.ValidString(text) {
		text = strings.ToValidUTF8(text, "\uFFFD")
	}
	return strings.TrimSpace(text), nil
}

// Close waits for inference using the model to return and releases it.
// Cancel the transcription's context first to stop an active request.
func (r *Recognizer) Close() error {
	if r == nil || r.gate == nil {
		return nil
	}
	r.gate <- struct{}{}
	defer r.release()
	if r.decoder == nil {
		return nil
	}
	model := r.decoder
	r.decoder = nil
	r.pcm = nil
	return model.Close()
}
