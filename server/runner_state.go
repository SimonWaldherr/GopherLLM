package server

import (
	"errors"
	"sync"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type runnerState struct {
	mu   sync.RWMutex
	r    *gopherllm.Runner
	path string
	// baseline is captured before any server-side auto tuning. A model switch
	// restores it when the replacement has not been explicitly tuned, so
	// process-wide knobs measured for the previous model do not leak across.
	baseline gopherllm.RuntimeTuning
	// autoTune is the tuning result actually applied to r during this process
	// (nil if none has been). It is distinct from gopherllm.Runner.LoadAutoTune, which
	// only reports what is cached on disk and may not reflect what is
	// currently active if the server started without --auto.
	autoTune *gopherllm.AutoTuneResult
}

// embeddingState holds the optional, separately loaded model used by the
// browser RAG mode. Enabling RAG must never replace the chat generation model.
type embeddingState struct {
	mu sync.RWMutex
	r  *gopherllm.Runner
}

func (s *embeddingState) withRunner(fn func(*gopherllm.Runner)) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	fn(s.r)
}

// withEmbeddingRunner selects the dedicated embedding runner when one has
// been loaded, falling back to the chat runner for the useful single-model
// setup. Keeping the read lock for the entire callback also prevents a model
// swap from closing weights while an embedding pass is in progress.
func withEmbeddingRunner(chat *runnerState, embedder *embeddingState, fn func(*gopherllm.Runner)) {
	embedder.mu.RLock()
	if embedder.r != nil {
		defer embedder.mu.RUnlock()
		fn(embedder.r)
		return
	}
	embedder.mu.RUnlock()
	chat.withRunner(fn)
}

func (s *embeddingState) swap(r *gopherllm.Runner) {
	s.mu.Lock()
	old := s.r
	s.r = r
	s.mu.Unlock()
	if old != nil {
		_ = old.Close()
	}
}

func (s *embeddingState) close() error {
	s.mu.Lock()
	old := s.r
	s.r = nil
	s.mu.Unlock()
	if old == nil {
		return nil
	}
	return old.Close()
}

func (s *runnerState) get() *gopherllm.Runner {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.r
}

func (s *runnerState) withRunner(fn func(*gopherllm.Runner)) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	fn(s.r)
}

func (s *runnerState) swap(r *gopherllm.Runner, path string) {
	var old *gopherllm.Runner
	s.mu.Lock()
	old = s.r
	s.r = r
	s.path = path
	// A hot-swapped model has not had a model-specific calibration applied.
	// Restore the pre-auto baseline while holding the writer lock: all inference
	// paths hold its read lock for their whole run, so no generation observes a
	// mixture of old and restored process-global knobs.
	s.baseline.Apply()
	s.autoTune = nil
	s.mu.Unlock()
	if old != nil {
		_ = old.Close()
	}
}

func (s *runnerState) close() error {
	s.mu.Lock()
	old := s.r
	s.r = nil
	s.path = ""
	s.autoTune = nil
	s.mu.Unlock()
	if old == nil {
		return nil
	}
	return old.Close()
}

func (s *runnerState) getPath() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.path
}

// runAutoTune serializes calibration with model hot-swaps and all inference.
// gopherllm.AutoTuneResult.Apply changes process-wide settings, so holding the writer
// lock is intentional: inference paths hold a read lock for the entire run.
func (s *runnerState) runAutoTune(opts gopherllm.AutoTuneOptions, refresh bool) (gopherllm.AutoTuneResult, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.r == nil {
		return gopherllm.AutoTuneResult{}, false, errors.New("no model is loaded; load a model first")
	}
	res, cached, err := s.r.AutoTuneOrCached(opts, refresh)
	if err == nil {
		s.autoTune = &res
	}
	return res, cached, err
}

// autoTuneStatus reports what GET /autotune needs to render the web UI's
// panel: whether a tuning is active this session, whether one is persisted on
// disk for the current model+host, and the most relevant result to show.
func (s *runnerState) autoTuneStatus() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	status := map[string]any{"active": false, "cached": false}
	if s.r == nil {
		status["metal_available"] = gopherllm.MetalAvailable()
		if !gopherllm.MetalAvailable() {
			status["metal_hint"] = gopherllm.MetalError()
		}
		return status
	}
	if s.autoTune != nil {
		_, cached := s.r.LoadAutoTune()
		status = map[string]any{"active": true, "cached": cached, "result": s.autoTune}
	} else if cached, ok := s.r.LoadAutoTune(); ok {
		status = map[string]any{"active": false, "cached": true, "result": cached}
	}
	// Metal is a build-time choice, not something auto-tuning can switch on,
	// so a build without it silently leaves a large speedup on the table
	// (measured ~1.8x on a 3B Q4_K_M). Report it so the UI can say so rather
	// than letting it stay invisible.
	status["metal_available"] = gopherllm.MetalAvailable()
	if !gopherllm.MetalAvailable() {
		status["metal_hint"] = gopherllm.MetalError()
	}
	return status
}
