package gopherllm

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"unicode/utf8"
)

// VoxtralRealtimeSession keeps one Voxtral GGUF mapped while a microphone is
// active. Audio may be appended in small chunks; model weights are never
// reloaded between chunks. Frontend, encoder and decoder state persist so
// only new frames and tokens are evaluated on each Push.
type VoxtralRealtimeSession struct {
	mu           sync.Mutex
	mmap         *MmapFile
	cfg          VoxtralRealtimeConfig
	weights      VoxtralRealtimeWeights
	tokenizer    *Tokenizer
	tokenTypes   []uint32
	eosID, bosID int
	audio        *voxtralStreamAudio
	decoder      *VoxtralRealtimeDecoderState
	fast         voxtralFastDecoder
	prev, pos    int
	text         strings.Builder
	pendingUTF8  string
	finished     bool
	closed       bool
}

type voxtralFastDecoder interface {
	Step(input []float32, pos int, generate bool) (int, error)
	Close()
}

// NewVoxtralRealtimeSession loads a local Voxtral Realtime GGUF once for a
// live transcription session. Call Close when microphone capture ends.
func NewVoxtralRealtimeSession(modelPath string, logw io.Writer) (*VoxtralRealtimeSession, error) {
	return NewVoxtralRealtimeSessionContext(context.Background(), modelPath, logw)
}

// NewVoxtralRealtimeSessionContext prepares the fixed silence prefix before
// microphone capture begins. Loading, warmup and subsequent pushes can cancel.
func NewVoxtralRealtimeSessionContext(ctx context.Context, modelPath string, logw io.Writer) (*VoxtralRealtimeSession, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if logw == nil {
		logw = io.Discard
	}
	mmap, err := OpenMmap(modelPath)
	if err != nil {
		return nil, fmt.Errorf("opening model: %w", err)
	}
	data := mmap.Bytes()
	gguf, err := ParseGGUF(data)
	if err != nil {
		mmap.Close()
		return nil, fmt.Errorf("parsing GGUF: %w", err)
	}
	cfg, weights, err := LoadVoxtralRealtimeModel(data, gguf, MetalAvailable(), logw)
	if err != nil {
		releaseVoxtralRealtimeWeights(&weights)
		mmap.Close()
		return nil, fmt.Errorf("loading Voxtral Realtime model: %w", err)
	}
	tok, err := TokenizerFromMetadata(gguf.Metadata)
	if err != nil {
		releaseVoxtralRealtimeWeights(&weights)
		mmap.Close()
		return nil, fmt.Errorf("building tokenizer: %w", err)
	}
	if cfg.Mel.SampleRate != 16000 {
		releaseVoxtralRealtimeWeights(&weights)
		mmap.Close()
		return nil, fmt.Errorf("Voxtral requires a 16 kHz frontend, got %d", cfg.Mel.SampleRate)
	}
	tokenTypes, _ := gguf.Metadata["tokenizer.ggml.token_type"].AsU32Array()
	s := &VoxtralRealtimeSession{mmap: mmap, cfg: cfg, weights: weights, tokenizer: tok, tokenTypes: tokenTypes, eosID: int(gguf.GetU32("tokenizer.ggml.eos_token_id", 2)), bosID: int(gguf.GetU32("tokenizer.ggml.bos_token_id", 1))}
	if err := prepareVoxtralStreamWeights(ctx, &s.weights); err != nil {
		s.Close()
		return nil, err
	}
	if _, err := s.Push(ctx, nil, false); err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

// Push adds 16 kHz mono PCM and returns the accumulated transcript. final
// flushes the delayed final words and may be called with no new samples.
// Calls are serial. After an inference error the session must be closed.
func (s *VoxtralRealtimeSession) Push(ctx context.Context, pcm []float32, final bool) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.finished {
		return "", fmt.Errorf("Voxtral realtime session is closed")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	for _, sample := range pcm {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return "", fmt.Errorf("audio contains non-finite samples")
		}
	}
	if s.audio == nil {
		var err error
		s.audio, err = newVoxtralStreamAudio(s.cfg)
		if err != nil {
			return "", err
		}
		s.decoder, err = NewVoxtralRealtimeDecoderState(s.cfg, s.weights, s.cfg.Time.DefaultNumDelayTokens)
		if err != nil {
			return "", err
		}
		s.fast = newVoxtralFastDecoder(s.cfg, s.weights, s.decoder)
	}
	embeds, err := s.audio.push(ctx, s.weights, pcm, final)
	if err != nil {
		s.finished = true
		return "", err
	}
	promptLen := 1 + voxtralLeftPadTokens + s.cfg.Time.DefaultNumDelayTokens
	for _, audio := range embeds {
		token := s.prev
		if s.pos == 0 {
			token = s.bosID
		} else if s.pos < promptLen {
			token = s.cfg.StreamingPadTokenID
		}
		next, err := s.step(ctx, audio, token, s.pos, s.pos >= promptLen-1)
		if err != nil {
			s.finished = true
			return "", err
		}
		s.pos++
		if s.pos < promptLen {
			continue
		}
		s.prev = next
		if s.prev == s.eosID {
			// A live stream has no file boundary. Keep advancing alignment with
			// silence/pad rather than freezing forever at the first EOS.
			s.prev = s.cfg.StreamingPadTokenID
			continue
		}
		s.pendingUTF8 += voxtralTranscriptPiece(s.tokenizer, s.tokenTypes, s.prev)
		for len(s.pendingUTF8) > 0 && utf8.FullRuneInString(s.pendingUTF8) {
			r, n := utf8.DecodeRuneInString(s.pendingUTF8)
			s.text.WriteRune(r)
			s.pendingUTF8 = s.pendingUTF8[n:]
		}
	}
	s.finished = final
	return strings.TrimSpace(s.text.String()), nil
}

func (s *VoxtralRealtimeSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if s.fast != nil {
		s.fast.Close()
		s.fast = nil
	}
	releaseVoxtralRealtimeWeights(&s.weights)
	s.weights = VoxtralRealtimeWeights{}
	s.audio = nil
	s.decoder = nil
	if s.mmap != nil {
		return s.mmap.Close()
	}
	return nil
}

func (s *VoxtralRealtimeSession) step(ctx context.Context, audio []float32, token, pos int, generate bool) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	te, err := EmbedVoxtralRealtimeToken(s.cfg, s.weights, token)
	if err != nil {
		return 0, err
	}
	for i := range te {
		te[i] += audio[i]
	}
	if s.fast != nil {
		return s.fast.Step(te, pos, generate)
	}
	logits, err := ForwardVoxtralRealtimeDecoderStep(s.cfg, s.weights, s.decoder, te, pos)
	if err != nil {
		return 0, err
	}
	return voxtralArgmax(logits), nil
}
