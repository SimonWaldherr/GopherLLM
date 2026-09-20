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
// It deliberately calls gopherllm.TranscribeVoxtralRealtime -- the
// verified, correctness-checked offline/batch decode path (see
// voxtral_realtime.go and voxtral_transcribe.go in the module root) --
// rather than gopherllm.VoxtralModel's incremental Push/PushPCM16 API. The
// two use different decode schedules ("live decoder" vs. "legacy offline
// decoder", per Transcribe's own doc comment); as of this writing the
// incremental one returns empty transcripts on real speech that the batch
// one transcribes correctly (verified against samples/jfk.wav from
// antirez/voxtral.c). Buffering one turn's audio and transcribing it whole
// at Finish trades away live partial transcripts for a decoder actually
// known to work; CONCEPT.md section 4 already lists "kein geteilter
// Modellhost" as a known gap this accepts for now. Re-evaluate this
// tradeoff once the incremental path is fixed upstream.
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

// VoiceSession accumulates one turn's raw PCM16 audio in memory.
// Transcription happens once, at Finish -- see SpeechHost's doc comment.
type VoiceSession struct {
	host    *SpeechHost
	mu      sync.Mutex
	pcm     []byte
	partial string
	done    bool
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

// PushPCM16 appends one chunk of little-endian 16-bit mono PCM at 16kHz.
// It does not transcribe; the returned string is the same best-effort
// placeholder ("… Aufnahme läuft …") every chunk returns, since there is
// no incremental decoder in play -- callers should not treat this as a
// live transcript (CONCEPT.md section 6: partial revisions update the UI
// but never authorize an action; here there simply isn't one yet).
func (vs *VoiceSession) PushPCM16(ctx context.Context, pcm []byte) (string, error) {
	if len(pcm)%2 != 0 {
		return "", fmt.Errorf("hestia: PCM16 chunk has an odd byte count")
	}
	vs.mu.Lock()
	defer vs.mu.Unlock()
	if vs.done {
		return "", fmt.Errorf("hestia: voice session already finished")
	}
	vs.pcm = append(vs.pcm, pcm...)
	seconds := float64(len(vs.pcm)) / 2 / 16000
	vs.partial = fmt.Sprintf("… Aufnahme läuft (%.1fs) …", seconds)
	return vs.partial, nil
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
