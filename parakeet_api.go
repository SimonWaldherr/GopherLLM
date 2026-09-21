package gopherllm

import (
	"context"
	"fmt"
	"io"
	"sync"
)

// ParakeetModel owns one loaded Parakeet-TDT GGUF's weights, mapped once and
// reused across calls -- the same "load once, reuse weights" shape as
// VoxtralModel, simplified: Parakeet has no live/streaming session support
// yet (see parakeet.go's doc comment), only offline transcription of a
// complete clip, so there is no session lifecycle to manage here. Safe for
// concurrent use; TranscribeOffline calls do not overlap each other, since
// ParakeetGreedyDecodeTDT's LSTM state is call-local, but callers wanting
// real concurrent transcription throughput should use separate ParakeetModel
// instances (each maps its own weights) rather than share one.
type ParakeetModel struct {
	mu      sync.Mutex
	mmap    *MmapFile
	cfg     ParakeetConfig
	weights ParakeetWeights
	closed  bool
}

// OpenParakeet maps a local GGUF and loads its config/weights once. No
// network access or audio capture is performed.
func OpenParakeet(path string, logw io.Writer) (*ParakeetModel, error) {
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, fmt.Errorf("opening Parakeet model: %w", err)
	}
	gguf, err := ParseGGUF(mmap.Bytes())
	if err != nil {
		_ = mmap.Close()
		return nil, fmt.Errorf("parsing Parakeet GGUF: %w", err)
	}
	cfg, weights, err := LoadParakeetModel(mmap.Bytes(), gguf, logw)
	if err != nil {
		_ = mmap.Close()
		return nil, fmt.Errorf("loading Parakeet model: %w", err)
	}
	return &ParakeetModel{mmap: mmap, cfg: cfg, weights: weights}, nil
}

// TranscribeOffline transcribes one complete 16kHz mono clip.
func (m *ParakeetModel) TranscribeOffline(ctx context.Context, samples []float32, logw func(string, ...any)) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return "", fmt.Errorf("Parakeet model is closed")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	frames, err := ParakeetEncode(m.cfg, m.weights, samples)
	if err != nil {
		return "", fmt.Errorf("encoding audio: %w", err)
	}
	if logw != nil {
		logw("encoded %d frames", len(frames))
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	tokens := ParakeetGreedyDecodeTDT(m.cfg, m.weights, frames)
	return ParakeetDecodeTokens(m.weights.Vocab, tokens)
}

// Close releases the mapped GGUF. Safe to call more than once.
func (m *ParakeetModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	if m.mmap != nil {
		return m.mmap.Close()
	}
	return nil
}
