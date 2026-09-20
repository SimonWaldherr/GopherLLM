package mobile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

var (
	ErrClosed     = errors.New("engine is closed")
	ErrNoModel    = errors.New("model is not loaded")
	ErrLoaded     = errors.New("model is already loaded")
	ErrGenerating = errors.New("generation is already running")
)

// Engine owns one Model. Instances are independent; calls which use model
// memory are serialized so Unload cannot unmap weights under inference.
type Engine struct {
	opMu   sync.Mutex
	mu     sync.Mutex
	model  *gopherllm.Model
	cancel context.CancelFunc
	closed bool
}

func NewEngine() *Engine { return &Engine{} }

func (e *Engine) Load(path string, optionsJSON string) (err error) {
	if path == "" {
		return fmt.Errorf("model file path is empty")
	}
	if _, statErr := os.Stat(path); statErr != nil {
		return fmt.Errorf("model file does not exist: %w", statErr)
	}
	opts, err := parseLoadOptions(optionsJSON)
	if err != nil {
		return err
	}
	coreOpts, err := opts.coreOptions()
	if err != nil {
		return err
	}
	e.opMu.Lock()
	defer e.opMu.Unlock()
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return ErrClosed
	}
	if e.model != nil {
		e.mu.Unlock()
		return ErrLoaded
	}
	e.mu.Unlock()
	if opts.Metal && !gopherllm.MetalAvailable() {
		return fmt.Errorf("Metal is not available in this iOS build: %s", gopherllm.MetalError())
	}
	defer recoverError("failed to load GGUF", &err)
	m, err := gopherllm.Open(context.Background(), path, coreOpts...)
	if err != nil {
		return fmt.Errorf("failed to load GGUF: %w", err)
	}
	e.mu.Lock()
	e.model = m
	e.mu.Unlock()
	return nil
}

func (e *Engine) Unload() error {
	e.Cancel()
	e.opMu.Lock()
	defer e.opMu.Unlock()
	e.mu.Lock()
	m := e.model
	e.model = nil
	e.mu.Unlock()
	if m == nil {
		return nil
	}
	return m.Close()
}

func (e *Engine) Close() error {
	e.Cancel()
	e.opMu.Lock()
	defer e.opMu.Unlock()
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil
	}
	e.closed = true
	m := e.model
	e.model = nil
	e.mu.Unlock()
	if m == nil {
		return nil
	}
	return m.Close()
}

func (e *Engine) IsLoaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.model != nil && !e.closed
}
func (e *Engine) ModelName() string {
	e.opMu.Lock()
	defer e.opMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.model == nil {
		return ""
	}
	return e.model.Name()
}

// Cancel is safe at any time and only affects this Engine's active request.
func (e *Engine) Cancel() {
	e.mu.Lock()
	cancel := e.cancel
	e.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (e *Engine) Generate(prompt, optionsJSON string) (text string, err error) {
	o, err := parseGenerationOptions(optionsJSON)
	if err != nil {
		return "", err
	}
	e.opMu.Lock()
	defer e.opMu.Unlock()
	m, ctx, done, err := e.startGeneration()
	if err != nil {
		return "", err
	}
	defer done()
	defer recoverError("generation failed", &err)
	result, err := m.Generate(ctx, prompt, o.coreOptions()...)
	if err != nil {
		return result.Text, fmt.Errorf("generation failed: %w", err)
	}
	return result.Text, nil
}

func (e *Engine) startGeneration() (*gopherllm.Model, context.Context, func(), error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, nil, nil, ErrClosed
	}
	if e.model == nil {
		return nil, nil, nil, ErrNoModel
	}
	if e.cancel != nil {
		return nil, nil, nil, ErrGenerating
	}
	ctx, cancel := context.WithCancel(context.Background())
	e.cancel = cancel
	return e.model, ctx, func() { cancel(); e.mu.Lock(); e.cancel = nil; e.mu.Unlock() }, nil
}

func recoverError(prefix string, target *error) {
	if r := recover(); r != nil {
		*target = fmt.Errorf("%s: %v", prefix, r)
	}
}

// InfoJSON contains stable, inexpensive model metadata for mobile UIs.
func (e *Engine) InfoJSON() string {
	e.opMu.Lock()
	defer e.opMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.model == nil {
		return "{}"
	}
	c := e.model.Config()
	info := e.model.Info()
	v := struct {
		Name           string `json:"name"`
		FileSizeBytes  int    `json:"file_size_bytes"`
		Architecture   string `json:"architecture"`
		ContextLength  int    `json:"context_length"`
		VocabSize      int    `json:"vocab_size"`
		Mapped         bool   `json:"mapped"`
		MetalAvailable bool   `json:"metal_available"`
	}{e.model.Name(), info.FileSizeBytes, c.Arch, c.MaxSeqLen, c.VocabSize, e.model.IsMapped(), gopherllm.MetalAvailable()}
	b, _ := json.Marshal(v)
	return string(b)
}
