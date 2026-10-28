package engine

import "context"

// A rejected hybrid block needs its accepted recurrent prefix, not another
// set of large matrix projections. Retain the small, independent inputs to
// each DeltaNet layer while verifying, then replay only its state updates.
// Attention K/V and the target hidden rows already have the correct causal
// prefix. This stores no per-position recurrent-state matrices.
type qwen35RecurrentReplay struct {
	layers []qwen35ReplayLayer
	valid  bool
}

type qwen35ReplayLayer struct {
	qkv, gate, alpha, beta []float32
}

const qwen35MaxReplayBytes = 32 << 20

func (r *qwen35RecurrentReplay) begin(cfg Config, layers, positions int, enabled bool) {
	r.valid = false
	width := cfg.SSMInner*2 + 2*cfg.SSMGroups*cfg.SSMState + 2*cfg.SSMHeads
	if !enabled || layers <= 0 || positions <= 0 || positions > speculativeMaxPositions || width <= 0 ||
		int64(layers)*int64(positions)*int64(width)*4 > qwen35MaxReplayBytes {
		return
	}
	ensureLenNoClear(&r.layers, layers)
	r.valid = true
}

func (r *qwen35RecurrentReplay) capture(layer int, qkv, gate, alpha, beta []float32) {
	if !r.valid {
		return
	}
	l := &r.layers[layer]
	ensureLenNoClear(&l.qkv, len(qkv))
	ensureLenNoClear(&l.gate, len(gate))
	ensureLenNoClear(&l.alpha, len(alpha))
	ensureLenNoClear(&l.beta, len(beta))
	copy(l.qkv, qkv)
	copy(l.gate, gate)
	copy(l.alpha, alpha)
	copy(l.beta, beta)
}

func (r *qwen35RecurrentReplay) replay(ctx context.Context, cfg Config, weights Qwen35Weights, state *Qwen35Cache, buf *DecodeBuffer, keep int) error {
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	for _, layer := range weights.Layers {
		if layer.Kind != qwen35DeltaNet {
			continue
		}
		l := &r.layers[layer.RecurrentCacheSlot]
		for i := range keep {
			if err := ctx.Err(); err != nil {
				return err
			}
			qwen35DeltaNetStep(cfg, layer.DeltaNet, state, l.qkv[i*channels:(i+1)*channels],
				l.gate[i*cfg.SSMInner:(i+1)*cfg.SSMInner], l.alpha[i*cfg.SSMHeads:(i+1)*cfg.SSMHeads],
				l.beta[i*cfg.SSMHeads:(i+1)*cfg.SSMHeads], layer.RecurrentCacheSlot, buf)
		}
	}
	return ctx.Err()
}
