// Package coreml executes compiled, stateless embedding models through Core ML.
// The native backend is opt-in: build with CGO_ENABLED=1 and -tags coreml.
package coreml

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"
)

// ComputeUnits restricts the devices Core ML may use. It does not guarantee
// that an allowed device executes any particular model operation.
type ComputeUnits uint8

const (
	CPUAndNeuralEngine ComputeUnits = iota
	CPUOnly
	CPUAndGPU
	All
)

// Config names the complete input feature set and the pooled embedding output.
// TokenTypeIDs is optional; all other feature names are required.
type Config struct {
	ComputeUnits                                  ComputeUnits
	InputIDs, AttentionMask, TokenTypeIDs, Output string
}

// Info describes the fixed model interface and permitted compute units.
// Neural Engine availability is reported by Core ML on macOS 14/iOS 17 and
// newer; false on older systems means that availability could not be queried.
// These fields describe hardware availability, not observed execution.
type Info struct {
	BatchSize, SequenceLength, Dimension int
	ComputeUnits                         ComputeUnits
	NeuralEngineAvailable                bool
	NeuralEngineCores                    int
}

// Keep these limits aligned with runtime.h. Bound reusable input/output scratch
// independently of model weights; larger batches need an explicit new export.
const (
	maxBatchSize      = 8
	maxBufferElements = 1 << 20
)

// Plan counts static operations by their preferred compute device. Functions,
// both sides of conditionals, and nested blocks are included once each. This
// describes Core ML's anticipated placement, not observed runtime utilization
// or the fraction of prediction time spent on each device.
type Plan struct {
	Operations, CPUOperations, GPUOperations, NeuralEngineOperations, UnassignedOperations int
}

var ErrUnavailable = errors.New("coreml: requires an Apple platform with cgo, the coreml build tag, and macOS 13/iOS 16 or newer")

type backend interface {
	predict(ids, mask, types []int32, out []float32) error
	close()
}

// Model owns a loaded Core ML model and reusable input and output buffers.
// Predict and Close are serialized, including calls made directly to this type.
type Model struct {
	mu      sync.Mutex
	info    Info
	backend backend
}

func (m *Model) Info() Info {
	if m == nil {
		return Info{}
	}
	return m.info
}

func (m *Model) Predict(ids, mask, types []int32, out []float32) error {
	if m == nil {
		return errors.New("coreml: model is closed")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.backend == nil {
		return errors.New("coreml: model is closed")
	}
	batch, sequence, dimension := m.info.BatchSize, m.info.SequenceLength, m.info.Dimension
	if batch < 1 || batch > maxBatchSize || sequence < 1 || dimension < 1 ||
		sequence > maxBufferElements/batch || dimension > maxBufferElements/batch {
		return errors.New("coreml: invalid or oversized fixed model shape")
	}
	inputCount, outputCount := batch*sequence, batch*dimension
	if len(ids) != inputCount || len(mask) != inputCount {
		return fmt.Errorf("coreml: input IDs and mask must each contain %d values", inputCount)
	}
	if len(types) != 0 && len(types) != inputCount {
		return fmt.Errorf("coreml: token types must contain %d values", inputCount)
	}
	if len(out) != outputCount {
		return fmt.Errorf("coreml: output must contain %d values", outputCount)
	}
	if err := m.backend.predict(ids, mask, types, out); err != nil {
		return err
	}
	for _, value := range out {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.New("coreml: model returned a non-finite embedding")
		}
	}
	return nil
}

func (m *Model) Close() error {
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

func validateConfig(c Config) error {
	if c.ComputeUnits > All {
		return fmt.Errorf("coreml: unknown compute units %d", c.ComputeUnits)
	}
	for _, field := range []struct{ label, name string }{{"input IDs", c.InputIDs}, {"attention mask", c.AttentionMask}, {"output", c.Output}} {
		if field.name == "" {
			return fmt.Errorf("coreml: %s feature name is required", field.label)
		}
	}
	for _, name := range []string{c.InputIDs, c.AttentionMask, c.TokenTypeIDs, c.Output} {
		if !utf8.ValidString(name) || strings.IndexByte(name, 0) >= 0 {
			return errors.New("coreml: feature names must be valid UTF-8 without NUL bytes")
		}
	}
	if c.InputIDs == c.AttentionMask || c.TokenTypeIDs != "" && (c.TokenTypeIDs == c.InputIDs || c.TokenTypeIDs == c.AttentionMask) {
		return errors.New("coreml: input feature names must be distinct")
	}
	return nil
}

func compiledModelPath(path string) (string, error) {
	if !utf8.ValidString(path) || strings.IndexByte(path, 0) >= 0 {
		return "", errors.New("coreml: model path must be valid UTF-8 without NUL bytes")
	}
	path = filepath.Clean(path)
	if filepath.Ext(path) != ".mlmodelc" {
		return "", errors.New("coreml: expected a compiled .mlmodelc directory at a persistent path")
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("coreml: open compiled model: %w", err)
	}
	if !info.IsDir() {
		return "", errors.New("coreml: compiled model must be a directory")
	}
	// Canonicalize aliases so repeatedly loading the same compiled model uses
	// the same specialization-cache identity. Never compile into a new temp path.
	path, err = filepath.EvalSymlinks(path)
	if err != nil {
		return "", fmt.Errorf("coreml: resolve compiled model: %w", err)
	}
	return filepath.Abs(path)
}
