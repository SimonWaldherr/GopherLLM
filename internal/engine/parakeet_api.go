package engine

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// ParakeetModel owns one loaded Parakeet-family speech model's weights.
// The weights are reused across calls, like those of
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

// OpenParakeet loads a local Parakeet-TDT GGUF or an unpacked Phonon
// model.fermion with its sibling config.json. It does not access the network.
func OpenParakeet(path string, logw io.Writer) (*ParakeetModel, error) {
	lower := strings.ToLower(path)
	if strings.HasSuffix(lower, ".fsq") || strings.HasSuffix(lower, ".bps.tar.zst") {
		return nil, fmt.Errorf("unpack the Phonon archive first, then open model.fermion beside its config.json")
	}
	mmap, err := OpenMmap(path)
	if err != nil {
		return nil, fmt.Errorf("opening Parakeet model: %w", err)
	}
	cfg, weights, err := loadParakeetFile(path, mmap.Bytes(), logw)
	if err != nil {
		_ = mmap.Close()
		return nil, err
	}
	return &ParakeetModel{mmap: mmap, cfg: cfg, weights: weights}, nil
}

func loadParakeetFile(path string, data []byte, logw io.Writer) (ParakeetConfig, ParakeetWeights, error) {
	if strings.EqualFold(filepath.Ext(path), ".fermion") {
		config, err := readPhononConfig(path)
		if err != nil {
			return ParakeetConfig{}, ParakeetWeights{}, err
		}
		return loadPhononModel(data, config, logw, true)
	}
	gguf, err := ParseGGUF(data)
	if err != nil {
		return ParakeetConfig{}, ParakeetWeights{}, fmt.Errorf("parsing Parakeet GGUF: %w", err)
	}
	cfg, weights, err := LoadParakeetModel(data, gguf, logw)
	if err != nil {
		return ParakeetConfig{}, ParakeetWeights{}, fmt.Errorf("loading Parakeet model: %w", err)
	}
	return cfg, weights, nil
}

func readPhononConfig(modelPath string) ([]byte, error) {
	path := filepath.Join(filepath.Dir(modelPath), "config.json")
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening Phonon config %s: %w", path, err)
	}
	defer f.Close()
	config, err := io.ReadAll(io.LimitReader(f, maxPhononConfigBytes+1))
	if err != nil {
		return nil, fmt.Errorf("reading Phonon config %s: %w", path, err)
	}
	if len(config) > maxPhononConfigBytes {
		return nil, fmt.Errorf("Phonon config %s exceeds %d bytes", path, maxPhononConfigBytes)
	}
	return config, nil
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
	if len(samples) == 0 {
		return "", fmt.Errorf("audio is empty")
	}
	if err := validateAudioSamples(samples); err != nil {
		return "", err
	}
	frames, err := parakeetEncodeContext(ctx, m.cfg, m.weights, samples)
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

// Close releases the model and its mapped weights. Safe to call more than once.
func (m *ParakeetModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	m.weights = ParakeetWeights{}
	if m.mmap != nil {
		return m.mmap.Close()
	}
	return nil
}
