package hestia

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// ErrSpeechNotConfigured is returned by voice endpoints when no Voxtral
// model path was configured (CONCEPT.md section 3: voice is optional, the
// text path keeps working without it).
var ErrSpeechNotConfigured = errors.New("hestia: no speech model configured")

// ErrVoiceSessionBusy is returned when a recording is already in progress
// (CONCEPT.md section 3's capacity limit, surfaced to the UI as the
// "Belegt-Anzeige").
var ErrVoiceSessionBusy = errors.New("hestia: a voice recording is already in progress")

// SpeechHost configures Hestia's speech transcription and enforces
// CONCEPT.md section 3's "genau eine aktive Sprachaufnahme" capacity limit.
//
// The model's incremental decoder can return an empty transcript for real
// speech. To keep live transcription reliable, VoiceSession periodically
// re-decodes the recording accumulated so far through the verified batch
// path. Partial text is advisory only and is never used to trigger an action.
type SpeechHost struct {
	ModelPath string

	mu   sync.Mutex
	busy bool
}

// OpenSpeechHost validates that a Voxtral Realtime GGUF exists and is
// loadable, without keeping it mapped (TranscribeVoxtralRealtime opens and
// releases its own mapping per call -- see this file's doc comment).
func OpenSpeechHost(ctx context.Context, modelPath string, logw io.Writer) (*SpeechHost, error) {
	if logw == nil {
		logw = io.Discard
	}
	// A short, throwaway clip validates the model loads and decodes without
	// keeping any state around afterward; "audio is empty" would be the
	// only expected error for one all-zero sample.
	probe := make([]float32, 1600)
	if _, err := gopherllm.TranscribeVoxtralRealtime(ctx, modelPath, probe, 0, logw); err != nil {
		return nil, fmt.Errorf("validating speech model: %w", err)
	}
	return &SpeechHost{ModelPath: modelPath}, nil
}

func (h *SpeechHost) Close() error { return nil }

// VoiceSession accumulates one turn's raw PCM16 audio in memory and refreshes
// a reliable partial transcript after each additional second of recording.
type VoiceSession struct {
	host     *SpeechHost
	mu       sync.Mutex
	pcm      []byte
	partial  string
	decoded  int
	decoding bool
	done     bool
}

// StartVoiceSession reserves the host's one recording slot.
func (h *SpeechHost) StartVoiceSession(ctx context.Context) (*VoiceSession, error) {
	if h == nil || h.ModelPath == "" {
		return nil, ErrSpeechNotConfigured
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.busy {
		return nil, ErrVoiceSessionBusy
	}
	h.busy = true
	return &VoiceSession{host: h}, nil
}

// PushPCM16 appends one chunk of little-endian 16-bit mono PCM at 16kHz and
// periodically refreshes its partial transcript using the known-good batch
// decoder. Decoding happens outside the session lock so recording can keep
// accepting audio while a previous partial is being rendered.
func (vs *VoiceSession) PushPCM16(ctx context.Context, pcm []byte) (string, error) {
	if len(pcm)%2 != 0 {
		return "", fmt.Errorf("hestia: PCM16 chunk has an odd byte count")
	}
	vs.mu.Lock()
	if vs.done {
		vs.mu.Unlock()
		return "", fmt.Errorf("hestia: voice session already finished")
	}
	vs.pcm = append(vs.pcm, pcm...)
	seconds := float64(len(vs.pcm)) / 2 / 16000
	vs.partial = fmt.Sprintf("… Aufnahme läuft (%.1fs) …", seconds)
	if vs.decoding || len(vs.pcm)-vs.decoded < 16000*2 {
		partial := vs.partial
		vs.mu.Unlock()
		return partial, nil
	}
	vs.decoding = true
	snapshot := append([]byte(nil), vs.pcm...)
	vs.mu.Unlock()
	text, err := gopherllm.TranscribeVoxtralRealtime(ctx, vs.host.ModelPath, pcm16ToFloat32(snapshot), 0, io.Discard)
	vs.mu.Lock()
	vs.decoding = false
	if err != nil {
		// Keep capturing and preserve the last visible partial. A transient
		// decode failure must not turn a microphone request into lost audio.
		partial := vs.partial
		vs.mu.Unlock()
		return partial, nil
	}
	if !vs.done {
		vs.decoded = len(snapshot)
		if text != "" {
			vs.partial = text
		}
	}
	partial := vs.partial
	vs.mu.Unlock()
	return partial, nil
}

// Finish transcribes the whole buffered recording in one batch call and
// releases the host's recording slot.
func (vs *VoiceSession) Finish(ctx context.Context) (string, error) {
	vs.mu.Lock()
	pcm := vs.pcm
	alreadyDone := vs.done
	vs.done = true
	vs.mu.Unlock()
	if alreadyDone {
		return "", fmt.Errorf("hestia: voice session already finished")
	}
	defer vs.release()

	if len(pcm) == 0 {
		return "", fmt.Errorf("hestia: no audio was recorded")
	}
	samples := pcm16ToFloat32(pcm)
	return gopherllm.TranscribeVoxtralRealtime(ctx, vs.host.ModelPath, samples, 0, io.Discard)
}

// Abort releases the recording slot without transcribing.
func (vs *VoiceSession) Abort() error {
	vs.mu.Lock()
	alreadyDone := vs.done
	vs.done = true
	vs.mu.Unlock()
	if !alreadyDone {
		vs.release()
	}
	return nil
}

func (vs *VoiceSession) release() {
	vs.host.mu.Lock()
	vs.host.busy = false
	vs.host.mu.Unlock()
}

func pcm16ToFloat32(pcm []byte) []float32 {
	samples := make([]float32, len(pcm)/2)
	for i := range samples {
		v := int16(binary.LittleEndian.Uint16(pcm[2*i:]))
		samples[i] = float32(v) / 32768
	}
	return samples
}
