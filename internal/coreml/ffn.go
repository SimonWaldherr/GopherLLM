package coreml

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// FFNInfo describes a compiled, stateless hidden-state projection. Inputs and
// outputs are fixed float32 [Tokens, Dimension] arrays; computation inside the
// exported graph may use FP16. This does not describe observed device usage.
type FFNInfo struct {
	Tokens, Dimension int
	// Manifest is the exporter's small JSON source binding stored in the
	// model's gopherllm.ffn creator-defined metadata. Empty is allowed for
	// low-level boundary probes; users attaching a GGUF must validate it.
	Manifest string
}

const (
	maxFFNTokens    = 128
	maxFFNDimension = 16 * 1024
)

// ErrFFNNonFiniteOutput lets a caller with the original source weights retry
// a stateless FFN on its ordinary backend after FP16 range overflow.
var ErrFFNNonFiniteOutput = errors.New("coreml: FFN returned a non-finite value")

type ffnBackend interface {
	predict(input, output []float32) error
	close()
}

// FFN owns one compiled feed-forward graph and its reusable native I/O.
// Predict and Close serialize access. The caller owns both slices; no Go
// storage is retained after Predict returns. The graph's input must be named
// hidden and its output output. Opening a model does not validate its weights
// against a GGUF; that binding belongs to the caller.
type FFN struct {
	mu      sync.Mutex
	info    FFNInfo
	backend ffnBackend
}

func (m *FFN) Info() FFNInfo {
	if m == nil {
		return FFNInfo{}
	}
	return m.info
}

func validFFNShape(info FFNInfo) bool {
	return info.Tokens >= 1 && info.Tokens <= maxFFNTokens &&
		info.Dimension >= 1 && info.Dimension <= maxFFNDimension &&
		info.Dimension <= maxBufferElements/info.Tokens
}

// Predict executes one complete fixed-size graph. Both buffers must contain
// exactly Tokens*Dimension finite float32 values. Input and output may alias:
// the native runtime copies the complete input before producing any output.
func (m *FFN) Predict(input, output []float32) error {
	if m == nil {
		return errors.New("coreml: FFN is closed")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.backend == nil {
		return errors.New("coreml: FFN is closed")
	}
	if !validFFNShape(m.info) {
		return errors.New("coreml: invalid or oversized fixed FFN shape")
	}
	count := m.info.Tokens * m.info.Dimension
	if len(input) != count || len(output) != count {
		return fmt.Errorf("coreml: FFN input and output must each contain %d values", count)
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.New("coreml: FFN input contains a non-finite value")
		}
	}
	if err := m.backend.predict(input, output); err != nil {
		return err
	}
	for _, value := range output {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return ErrFFNNonFiniteOutput
		}
	}
	return nil
}

func (m *FFN) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.backend != nil {
		m.backend.close()
		m.backend = nil
	}
	return nil
}
