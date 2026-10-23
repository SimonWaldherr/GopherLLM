// Package coreml runs separately exported embedding encoders through Core ML.
// The default permits CPU and Apple Neural Engine execution, leaving the GPU
// available to a GopherLLM chat model. Core ML chooses the actual placement;
// permitting the Neural Engine does not prove that it executes every operator.
//
// Models must export pooled vectors for fixed batch and sequence lengths. The
// supplied tokenizer, boundary tokens, padding ID and exported model must match.
// This package neither converts GGUF weights nor creates a chat KV cache.
package coreml

import (
	"context"
	"errors"
	"fmt"
	"math"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	backend "github.com/SimonWaldherr/GopherLLM/internal/coreml"
)

type ComputeUnits = backend.ComputeUnits

const (
	CPUAndNeuralEngine = backend.CPUAndNeuralEngine
	CPUOnly            = backend.CPUOnly
	CPUAndGPU          = backend.CPUAndGPU
	All                = backend.All
)

// Info describes the model's fixed input/output shapes and permitted compute
// units. NeuralEngineAvailable describes hardware availability, not placement.
type Info = backend.Info

var (
	ErrUnavailable = backend.ErrUnavailable
	ErrClosed      = errors.New("coreml embedder closed")
)

// Tokenizer includes every boundary token the encoder expects in its output.
// A matching *gopherllm.Tokenizer satisfies this interface.
type Tokenizer interface {
	Encode(string) []uint32
}

// Options names the exported model's features. Empty InputIDs, AttentionMask
// and Output use input_ids, attention_mask and embedding, respectively.
// An empty TokenTypeIDs disables that optional, all-zero input. The zero value
// permits CPU and Neural Engine execution and pads with token zero.
type Options struct {
	ComputeUnits                          ComputeUnits
	InputIDs, AttentionMask, TokenTypeIDs string
	Output                                string
	PadToken                              uint32
}

type predictor interface {
	Info() backend.Info
	Predict(ids, mask, types []int32, out []float32) error
	Close() error
}

// Embedder owns one loaded Core ML model and its reusable input/output arrays.
// EmbedBatch and Close are safe to call concurrently; predictions serialize.
// It satisfies rag.Embedder without loading a second copy of chat weights.
// Do not copy an Embedder. Its zero value is closed.
type Embedder struct {
	model     predictor
	info      Info
	tokenizer Tokenizer
	gate      chan struct{}
	closed    bool
	pad       int32
	ids       []int32
	mask      []int32
	types     []int32
	output    []float32
	counts    []int
}

// OpenEmbedder loads a compiled .mlmodelc encoder once. It does not download
// or convert weights or infer a tokenizer. Keep the compiled model at a stable
// path so Core ML can reuse its specialization cache. Close the embedder after
// all of its consumers.
func OpenEmbedder(path string, tokenizer Tokenizer, opts Options) (*Embedder, error) {
	if tokenizer == nil {
		return nil, errors.New("coreml: tokenizer is required")
	}
	if opts.PadToken > math.MaxInt32 {
		return nil, errors.New("coreml: padding token exceeds int32")
	}
	if opts.InputIDs == "" {
		opts.InputIDs = "input_ids"
	}
	if opts.AttentionMask == "" {
		opts.AttentionMask = "attention_mask"
	}
	if opts.Output == "" {
		opts.Output = "embedding"
	}
	m, err := backend.Open(path, backend.Config{
		ComputeUnits: opts.ComputeUnits, InputIDs: opts.InputIDs,
		AttentionMask: opts.AttentionMask, TokenTypeIDs: opts.TokenTypeIDs, Output: opts.Output,
	})
	if err != nil {
		return nil, err
	}
	e, err := newEmbedder(m, tokenizer, opts)
	if err != nil {
		_ = m.Close()
	}
	return e, err
}

func newEmbedder(m predictor, tokenizer Tokenizer, opts Options) (*Embedder, error) {
	info := m.Info()
	const maxValues = 1 << 20
	if info.BatchSize < 1 || info.BatchSize > 8 || info.SequenceLength <= 0 || info.Dimension <= 0 ||
		info.SequenceLength > maxValues/info.BatchSize || info.Dimension > maxValues/info.BatchSize {
		return nil, errors.New("coreml: model has an invalid input or output shape")
	}
	inputs, outputs := info.BatchSize*info.SequenceLength, info.BatchSize*info.Dimension
	e := &Embedder{
		model: m, info: info, tokenizer: tokenizer, gate: make(chan struct{}, 1),
		pad: int32(opts.PadToken), ids: make([]int32, inputs),
		mask: make([]int32, inputs), output: make([]float32, outputs), counts: make([]int, info.BatchSize),
	}
	if opts.TokenTypeIDs != "" {
		e.types = make([]int32, inputs)
	}
	e.gate <- struct{}{}
	return e, nil
}

// Info is immutable and remains available after Close.
func (e *Embedder) Info() Info {
	if e == nil {
		return Info{}
	}
	return e.info
}

func (e *Embedder) acquire(ctx context.Context) error {
	if e == nil || e.gate == nil {
		return ErrClosed
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-e.gate:
	}
	if err := ctx.Err(); err != nil {
		e.release()
		return err
	}
	if e.closed {
		e.release()
		return ErrClosed
	}
	return nil
}

func (e *Embedder) release() { e.gate <- struct{}{} }

// EmbedBatch returns L2-normalized pooled vectors in input order. Inputs that
// exceed the exported sequence length are rejected rather than truncated.
// Scratch is bounded by the model's fixed batch size, independent of len(texts).
// A partial final batch repeats its first text in unused slots and discards
// those output vectors, avoiding undefined all-masked attention in some models.
// Cancellation works while queued and between predictions;
// a synchronous Core ML prediction completes before its cancellation is returned.
// A failed or canceled batch never returns partial results.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([]gopherllm.EmbeddingResult, error) {
	if err := e.acquire(ctx); err != nil {
		return nil, err
	}
	defer e.release()
	results := make([]gopherllm.EmbeddingResult, 0, len(texts))
	// Prediction-sized, request-owned slabs reduce per-vector allocations.
	// A retained row keeps at most its physical batch alive, rather than the
	// whole request. No result storage is retained or reused by the embedder.
	var vectors []float32
	batch, sequence, dimension := e.info.BatchSize, e.info.SequenceLength, e.info.Dimension
	for start := 0; start < len(texts); start += batch {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for j := range e.ids {
			e.ids[j], e.mask[j] = e.pad, 0
		}
		count := min(batch, len(texts)-start)
		for row := 0; row < count; row++ {
			i := start + row
			tokens := e.tokenizer.Encode(texts[i])
			if len(tokens) == 0 || len(tokens) > sequence {
				return nil, fmt.Errorf("coreml: item %d has %d tokens; expected 1..%d", i, len(tokens), sequence)
			}
			e.counts[row] = len(tokens)
			for j, token := range tokens {
				if token > math.MaxInt32 {
					return nil, fmt.Errorf("coreml: item %d token %d exceeds int32", i, j)
				}
				e.ids[row*sequence+j], e.mask[row*sequence+j] = int32(token), 1
			}
		}
		for row := count; row < batch; row++ {
			copy(e.ids[row*sequence:(row+1)*sequence], e.ids[:sequence])
			copy(e.mask[row*sequence:(row+1)*sequence], e.mask[:sequence])
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := e.model.Predict(e.ids, e.mask, e.types, e.output); err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return nil, fmt.Errorf("coreml: batch starting at item %d: %w", start, err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for row := 0; row < count; row++ {
			output := e.output[row*dimension : (row+1)*dimension]
			var sum float64
			for _, value := range output {
				v := float64(value)
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return nil, fmt.Errorf("coreml: item %d produced a non-finite embedding", start+row)
				}
				sum += v * v
			}
			if sum == 0 {
				return nil, fmt.Errorf("coreml: item %d produced a zero-norm embedding", start+row)
			}
			scale := 1 / math.Sqrt(sum)
			if len(vectors) < dimension {
				vectors = make([]float32, (count-row)*dimension)
			}
			vector := vectors[:dimension:dimension]
			vectors = vectors[dimension:]
			for j, value := range output {
				vector[j] = float32(float64(value) * scale)
			}
			results = append(results, gopherllm.EmbeddingResult{Embedding: vector, TokenCount: e.counts[row]})
		}
	}
	return results, nil
}

// Close waits for active predictions and releases the model and retained
// scratch. It is idempotent. Subsequent embedding calls return ErrClosed.
func (e *Embedder) Close() error {
	if e == nil || e.gate == nil {
		return nil
	}
	<-e.gate
	defer e.release()
	if e.closed {
		return nil
	}
	e.closed = true
	err := e.model.Close()
	e.model, e.tokenizer, e.ids, e.mask, e.types, e.output = nil, nil, nil, nil, nil, nil
	e.counts = nil
	return err
}
