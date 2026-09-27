package mobile

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

var (
	ErrClosed  = errors.New("engine is closed")
	ErrNoModel = errors.New("model is not loaded")
	ErrLoaded  = errors.New("model is already loaded")
	// Deprecated: generation requests on one Engine queue behind each other,
	// so this error is never returned.
	ErrGenerating = errors.New("generation is already running")
)

// Engine owns at most one loaded model. Engines are independent of each other
// and every method is safe to call from any thread.
//
// Load, Unload and Close wait for in-flight work on the model to finish
// (Unload and Close cancel it first), so weights are never unmapped under
// inference. Generation requests on one Engine run one at a time in arrival
// order. IsLoaded, ModelName, InfoJSON and Cancel never wait for a running
// generation, and CountTokens runs alongside one.
type Engine struct {
	life sync.RWMutex // held exclusively to swap the model, shared to use it
	gen  sync.Mutex   // serializes generation requests

	mu     sync.Mutex // guards the fields below; never held across model work
	model  *gopherllm.Model
	info   string // InfoJSON, captured at load: static for the model's lifetime
	cancel context.CancelFunc
	closed bool
}

func NewEngine() *Engine { return &Engine{} }

// Load loads the GGUF at path. optionsJSON is empty for the defaults or an
// object with any of threads (0 = automatic), metal, prefault
// ("none"|"core"|"all"), prepare_quantized and out_of_core.
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
	e.life.Lock()
	defer e.life.Unlock()
	e.mu.Lock()
	closed, loaded := e.closed, e.model != nil
	e.mu.Unlock()
	switch {
	case closed:
		return ErrClosed
	case loaded:
		return ErrLoaded
	}
	if opts.Metal && !gopherllm.MetalAvailable() {
		return fmt.Errorf("Metal is not available in this build: %s", gopherllm.MetalError())
	}
	var m *gopherllm.Model
	defer func() {
		if r := recover(); r != nil {
			if m != nil {
				m.Close()
			}
			err = fmt.Errorf("failed to load GGUF: %v", r)
		}
	}()
	m, err = gopherllm.Open(context.Background(), path, opts.coreOptions()...)
	if err != nil {
		return fmt.Errorf("failed to load GGUF: %w", err)
	}
	info := loadedModelInfo(m)
	e.mu.Lock()
	e.model, e.info = m, info
	e.mu.Unlock()
	return nil
}

// Unload cancels any running generation and releases the model. The Engine
// stays usable: a later Load may load another model.
func (e *Engine) Unload() error {
	return e.release(false)
}

// Close is Unload plus retiring the Engine: every later call fails with
// ErrClosed. Closing twice is harmless.
func (e *Engine) Close() error {
	return e.release(true)
}

func (e *Engine) release(closing bool) error {
	e.Cancel()
	e.life.Lock()
	defer e.life.Unlock()
	e.mu.Lock()
	m := e.model
	e.model, e.info = nil, ""
	if closing {
		e.closed = true
	}
	e.mu.Unlock()
	if m == nil {
		return nil
	}
	return m.Close()
}

func (e *Engine) IsLoaded() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.model != nil
}

// ModelName is the loaded model's self-declared name, or "".
func (e *Engine) ModelName() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.model == nil {
		return ""
	}
	return e.model.Name()
}

// InfoJSON describes the loaded model (see InspectModel for the shared
// fields, plus file_size_bytes, mapped, out_of_core, load_time_ms and
// metal_available), or is "{}" when nothing is loaded.
func (e *Engine) InfoJSON() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.info == "" {
		return "{}"
	}
	return e.info
}

// Cancel stops this Engine's running generation, if any. It is safe to call
// at any time and only ever affects this Engine.
func (e *Engine) Cancel() {
	e.mu.Lock()
	cancel := e.cancel
	e.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// CountTokens reports how many tokens text encodes to with the loaded
// model's tokenizer (including BOS when the model adds one), e.g. to keep a
// chat history inside the model's context length.
func (e *Engine) CountTokens(text string) (int, error) {
	e.life.RLock()
	defer e.life.RUnlock()
	m, err := e.loadedModel()
	if err != nil {
		return 0, err
	}
	return len(m.Tokenize(text)), nil
}

// Generate answers a single user prompt and returns the generated text.
func (e *Engine) Generate(prompt, optionsJSON string) (string, error) {
	result, err := e.run(singlePrompt(prompt), optionsJSON, nil)
	return result.Text, err
}

// Chat continues a conversation and returns the result as JSON (see
// resultJSON). messagesJSON is an array of {"role","content"} objects with
// roles system, user and assistant; a leading system message takes the
// place of the system_prompt option.
func (e *Engine) Chat(messagesJSON, optionsJSON string) (string, error) {
	messages, err := parseMessages(messagesJSON)
	if err != nil {
		return "", err
	}
	result, err := e.run(messages, optionsJSON, nil)
	if err != nil {
		return "", err
	}
	return resultJSON(result), nil
}

// run executes one generation request: it waits for earlier requests on this
// Engine, then streams to onDelta (nil for none) until done or canceled.
func (e *Engine) run(messages []gopherllm.ChatMessage, optionsJSON string, onDelta func(string) error) (result gopherllm.GenerationResult, err error) {
	opts, err := parseGenerationOptions(optionsJSON)
	if err != nil {
		return result, err
	}
	e.gen.Lock()
	defer e.gen.Unlock()
	e.life.RLock()
	defer e.life.RUnlock()
	m, err := e.loadedModel()
	if err != nil {
		return result, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	e.mu.Lock()
	e.cancel = cancel
	e.mu.Unlock()
	defer func() {
		cancel()
		e.mu.Lock()
		e.cancel = nil
		e.mu.Unlock()
	}()
	defer recoverError("generation failed", &err)
	result, err = m.Stream(ctx, messages, onDelta, gopherllm.WithGenerationOptions(opts))
	if err != nil {
		return result, fmt.Errorf("generation failed: %w", err)
	}
	return result, nil
}

// loadedModel returns the current model; the caller holds e.life (shared),
// which keeps it loaded until released.
func (e *Engine) loadedModel() (*gopherllm.Model, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	switch {
	case e.closed:
		return nil, ErrClosed
	case e.model == nil:
		return nil, ErrNoModel
	}
	return e.model, nil
}

func singlePrompt(prompt string) []gopherllm.ChatMessage {
	return []gopherllm.ChatMessage{gopherllm.UserMessage(prompt)}
}

func recoverError(prefix string, target *error) {
	if r := recover(); r != nil {
		*target = fmt.Errorf("%s: %v", prefix, r)
	}
}
