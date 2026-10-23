package gopherllm

import "sync"

// Only mapped, unprepared draft weights use deferred Metal registration.
// Copies of the execution weights share ownership through this object.
type qwen35MTPLazyMetal struct {
	bodyOnce, outputOnce sync.Once
	mapping              []byte
	q, o, gate, up, down *MetalWeight
	output               *MetalWeight
}

func (lazy *qwen35MTPLazyMetal) prepare(w Weight) *MetalWeight {
	if w.Metal != nil {
		return w.Metal
	}
	if w.F32 != nil || w.Prepared != nil || len(w.Raw) == 0 || len(w.Raw) > len(lazy.mapping) {
		return nil
	}
	if !weightBytesInMapping(w.Raw, lazy.mapping) {
		// The loader may pad a truncated tensor in Go-owned storage. Such a
		// buffer must never escape to Metal through bytesNoCopy.
		return nil
	}
	return prepareMetalWeight(w.Raw, w.Type, w.Rows, w.Cols, true)
}

func (w Qwen35MTPWeights) withLazyMetal() Qwen35MTPWeights {
	lazy := w.lazyMetal
	if lazy == nil {
		return w
	}
	if mtpMetalBodyEnabled {
		lazy.bodyOnce.Do(func() {
			lazy.q, lazy.o = lazy.prepare(w.Attention.Q), lazy.prepare(w.Attention.O)
			lazy.gate, lazy.up, lazy.down = lazy.prepare(w.FFN.Gate), lazy.prepare(w.FFN.Up), lazy.prepare(w.FFN.Down)
		})
		w.Attention.Q.Metal, w.Attention.O.Metal = lazy.q, lazy.o
		w.FFN.Gate.Metal, w.FFN.Up.Metal, w.FFN.Down.Metal = lazy.gate, lazy.up, lazy.down
	}
	lazy.outputOnce.Do(func() { lazy.output = lazy.prepare(w.Output) })
	w.Output.Metal = lazy.output
	w.lazyMetal = nil
	return w
}

func (w *Qwen35MTPWeights) releaseLazyMetal(releaser *weightResourceReleaser) {
	if w.lazyMetal == nil {
		return
	}
	lazy := w.lazyMetal
	for _, handle := range []*MetalWeight{lazy.q, lazy.o, lazy.gate, lazy.up, lazy.down, lazy.output} {
		weight := Weight{Metal: handle}
		releaser.release(&weight)
	}
	w.lazyMetal = nil
}
