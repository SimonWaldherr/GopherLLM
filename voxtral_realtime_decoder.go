package gopherllm

import (
	"fmt"
	"math"
)

// This file implements the Voxtral Realtime decoder: a Ministral-style GQA
// transformer (split-half RoPE, sliding-window causal attention, SwiGLU FFN,
// with GGUF Q/K rows permuted from the interleaved safetensors layout;
// tied token embedding / output projection) whose one addition is an
// Ada-RMSNorm-style scale gate on each block's FFN branch, conditioned on a
// FIXED "time" signal representing the checkpoint's built-in streaming
// delay -- not a per-token varying value. See voxtral_realtime.go's doc
// comment for how this was verified, and voxtral_realtime_audio.go for the
// audio encoder that feeds this decoder's input embeddings.
//
// This is single-step (one token position per call) autoregressive decode
// with a bounded sliding-window K/V history and reusable scratch buffers, not yet
// the engine's quantized/ring-buffer KV cache types (kv_f16.go/kv_i8.go),
// which are Runner-integration work landing separately.

// VoxtralRealtimeDecoderState is one generation's mutable state: per-layer
// K/V history (each entry NKVHeads*HeadDim wide) and the fixed per-layer Ada
// scale gates, computed once from the checkpoint's built-in delay and reused
// for every position.
type VoxtralRealtimeDecoderState struct {
	KHistory [][][]float32 // [layer][retained position][NKVHeads*HeadDim], oldest first
	VHistory [][][]float32
	AdaScale [][]float32 // [layer][HiddenSize], fixed for the whole generation

	// State belongs to one sequential generation and is not safe for concurrent use.
	nextPos int
	scratch voxtralRealtimeDecoderScratch
}

type voxtralRealtimeDecoderScratch struct {
	invFreq                                              []float32
	ropeHeadDim                                          int
	ropeTheta                                            float32
	x, normed, q, k, v, attnOut, outProj                 []float32
	ffnGate, ffnUp, ffnAct, ffnDown, finalNormed, scores []float32
	sin, cos                                             []float32
}

// NewVoxtralRealtimeDecoderState allocates decode state and precomputes the
// fixed per-layer Ada scale gates. numDelayTokens selects the streaming
// delay to condition on; pass cfg.Time.DefaultNumDelayTokens for the
// checkpoint's own default (see this file's doc comment -- MODEL.md verified
// this is a scalar "how many tokens do we lag behind the audio", not a
// position-dependent signal).
func NewVoxtralRealtimeDecoderState(cfg VoxtralRealtimeConfig, weights VoxtralRealtimeWeights, numDelayTokens int) (*VoxtralRealtimeDecoderState, error) {
	nLayers := len(weights.Decoder.Layers)
	if nLayers == 0 {
		return nil, fmt.Errorf("voxtral realtime decoder state: no decoder layers loaded")
	}
	tCond, err := voxtralRealtimeTimeCond(cfg.Time, weights.Decoder.TimeEmbedInvFreq, numDelayTokens)
	if err != nil {
		return nil, fmt.Errorf("voxtral realtime decoder state: %w", err)
	}
	adaScale := make([][]float32, nLayers)
	for l := range weights.Decoder.Layers {
		layer := &weights.Decoder.Layers[l]
		var bottleneck []float32
		layer.AdaLinear1.MatvecInto(tCond, &bottleneck)
		for i, v := range bottleneck {
			bottleneck[i] = geluTanh(v)
		}
		var scale []float32
		layer.AdaLinear2.MatvecInto(bottleneck, &scale)
		if len(scale) != cfg.Decoder.HiddenSize {
			return nil, fmt.Errorf("voxtral realtime decoder state: layer %d ada scale has %d elements, want HiddenSize=%d", l, len(scale), cfg.Decoder.HiddenSize)
		}
		adaScale[l] = scale
	}
	return &VoxtralRealtimeDecoderState{
		KHistory: make([][][]float32, nLayers),
		VHistory: make([][][]float32, nLayers),
		AdaScale: adaScale,
	}, nil
}

// voxtralRealtimeTimeCond builds the fixed time-conditioning vector:
// t_cond = concat(cos(t*inv_freq), sin(t*inv_freq)), t = float(numDelayTokens),
// using the checkpoint's own dec.time_embed.inv_freq tensor (length
// HiddenSize/2) rather than recomputing it from Time.EmbedTheta, for
// byte-exact fidelity to the reference.
func voxtralRealtimeTimeCond(timeCfg VoxtralRealtimeTimeConfig, invFreq []float32, numDelayTokens int) ([]float32, error) {
	half := timeCfg.EmbedDim / 2
	if len(invFreq) != half {
		return nil, fmt.Errorf("time embedding: inv_freq has %d elements, want EmbedDim/2=%d", len(invFreq), half)
	}
	t := float64(numDelayTokens)
	out := make([]float32, timeCfg.EmbedDim)
	for i, f := range invFreq {
		angle := t * float64(f)
		s, c := math.Sincos(angle)
		out[i] = float32(c)
		out[half+i] = float32(s)
	}
	return out, nil
}

// EmbedVoxtralRealtimeToken looks up one token's embedding row (also usable
// as the tied output projection's input width check).
func EmbedVoxtralRealtimeToken(cfg VoxtralRealtimeConfig, weights VoxtralRealtimeWeights, tokenID int) ([]float32, error) {
	if tokenID < 0 || tokenID >= cfg.Decoder.VocabSize {
		return nil, fmt.Errorf("embedding token: id %d out of range [0,%d)", tokenID, cfg.Decoder.VocabSize)
	}
	var out []float32
	weights.Decoder.TokenEmbd.RowInto(tokenID, cfg.Decoder.HiddenSize, &out)
	return out, nil
}

// ForwardVoxtralRealtimeDecoderStep runs one autoregressive decode step.
// input is the already-fused audio+token embedding for this position (see
// voxtral_realtime.go's doc comment point 4: audio embeddings are summed
// elementwise into the text token embedding, not interleaved or placed at
// special tokens). pos is this position's index in the sequence, used for
// RoPE and the sliding-window causal mask; state accumulates K/V history
// across calls and must be reused across a single generation's steps in
// order.
func ForwardVoxtralRealtimeDecoderStep(cfg VoxtralRealtimeConfig, weights VoxtralRealtimeWeights, state *VoxtralRealtimeDecoderState, input []float32, pos int) (logits []float32, err error) {
	dc := cfg.Decoder
	if state == nil {
		return nil, fmt.Errorf("decoder step: nil state")
	}
	if pos != state.nextPos {
		return nil, fmt.Errorf("decoder step: position %d, want next position %d", pos, state.nextPos)
	}
	if dc.HiddenSize <= 0 || dc.HeadDim <= 0 || dc.HeadDim%2 != 0 || dc.NHeads <= 0 {
		return nil, fmt.Errorf("decoder step: invalid decoder dimensions")
	}
	if len(state.KHistory) != len(weights.Decoder.Layers) || len(state.VHistory) != len(weights.Decoder.Layers) {
		return nil, fmt.Errorf("decoder step: state K/V layer count does not match weights")
	}
	for li := range weights.Decoder.Layers {
		if len(state.KHistory[li]) != len(state.VHistory[li]) {
			return nil, fmt.Errorf("decoder step: layer %d has inconsistent K/V history", li)
		}
	}
	if len(input) != dc.HiddenSize {
		return nil, fmt.Errorf("decoder step: input has %d elements, want HiddenSize=%d", len(input), dc.HiddenSize)
	}
	if len(weights.Decoder.Layers) != len(state.AdaScale) {
		return nil, fmt.Errorf("decoder step: %d layers loaded but state has %d ada-scale entries", len(weights.Decoder.Layers), len(state.AdaScale))
	}
	for li, scale := range state.AdaScale {
		if len(scale) != dc.HiddenSize {
			return nil, fmt.Errorf("decoder step: layer %d has invalid ada scale width", li)
		}
	}
	qDim := dc.NHeads * dc.HeadDim
	kvDim := dc.NKVHeads * dc.HeadDim
	if dc.NKVHeads <= 0 || dc.NHeads%dc.NKVHeads != 0 {
		return nil, fmt.Errorf("decoder step: NHeads=%d not a multiple of NKVHeads=%d", dc.NHeads, dc.NKVHeads)
	}
	kvMul := dc.NHeads / dc.NKVHeads

	b := &state.scratch
	ensureLenNoClear(&b.x, dc.HiddenSize)
	x := b.x
	copy(x, input)

	if b.ropeHeadDim != dc.HeadDim || b.ropeTheta != dc.RopeTheta {
		b.invFreq = standardRopeInvFreqSlice(dc.HeadDim, dc.RopeTheta)
		b.ropeHeadDim, b.ropeTheta = dc.HeadDim, dc.RopeTheta
	}
	scale := float32(1 / math.Sqrt(float64(dc.HeadDim)))
	half, nCache := prepareRopeScratch(pos, dc.HeadDim, dc.HeadDim, b.invFreq, 1, &b.sin, &b.cos)

	for li := range weights.Decoder.Layers {
		layer := &weights.Decoder.Layers[li]

		rmsNormInto(x, layer.AttnNorm, dc.Epsilon, &b.normed)
		layer.Q.MatvecInto(b.normed, &b.q)
		layer.K.MatvecInto(b.normed, &b.k)
		layer.V.MatvecInto(b.normed, &b.v)
		if len(b.q) != qDim || len(b.k) != kvDim || len(b.v) != kvDim {
			return nil, fmt.Errorf("decoder step: layer %d projected widths q=%d k=%d v=%d, want q=%d kv=%d", li, len(b.q), len(b.k), len(b.v), qDim, kvDim)
		}
		if nCache > 0 {
			applyPreparedRope(b.q, dc.HeadDim, dc.NHeads, half, nCache, b.sin, b.cos, false)
			applyPreparedRope(b.k, dc.HeadDim, dc.NKVHeads, half, nCache, b.sin, b.cos, false)
		}

		// Recycle the oldest row once the attention window is full. Keep
		// public histories chronological, with storage bounded by the window.
		appendVoxtralHistory(&state.KHistory[li], b.k, dc.SlidingWindow)
		appendVoxtralHistory(&state.VHistory[li], b.v, dc.SlidingWindow)
		hist := len(state.KHistory[li])
		ensureLenNoClear(&b.scores, hist)
		ensureLenNoClear(&b.attnOut, qDim)
		for h := range dc.NHeads {
			off := h * dc.HeadDim
			kvHead := h / kvMul
			kvOff := kvHead * dc.HeadDim
			query := b.q[off : off+dc.HeadDim]
			maxScore := negMaxF32
			for j := 0; j < hist; j++ {
				s := DotF32(query, state.KHistory[li][j][kvOff:kvOff+dc.HeadDim]) * scale
				b.scores[j] = s
				maxScore = max(maxScore, s)
			}
			denom := float32(0)
			for j := 0; j < hist; j++ {
				b.scores[j] = fastExpF32(b.scores[j] - maxScore)
				denom += b.scores[j]
			}
			out := b.attnOut[off : off+dc.HeadDim]
			clear(out)
			if denom > 0 {
				inv := 1 / denom
				for j := 0; j < hist; j++ {
					AxpyF32(out, b.scores[j]*inv, state.VHistory[li][j][kvOff:kvOff+dc.HeadDim])
				}
			}
		}

		layer.O.MatvecInto(b.attnOut, &b.outProj)
		if len(b.outProj) != dc.HiddenSize {
			return nil, fmt.Errorf("decoder step: layer %d output projection has %d elements, want HiddenSize=%d", li, len(b.outProj), dc.HiddenSize)
		}
		addInPlace(x, b.outProj)

		rmsNormInto(x, layer.FFNNorm, dc.Epsilon, &b.normed)
		// The one architectural addition over a stock Mistral/Ministral
		// block: scale the FFN input by (1 + fixed per-layer ada gate)
		// before the SwiGLU FFN. Never applied to the attention branch --
		// see this file's doc comment and voxtral_realtime.go point 4.
		adaScale := state.AdaScale[li]
		for i := range b.normed {
			b.normed[i] *= 1 + adaScale[i]
		}
		layer.FFNGate.MatvecInto(b.normed, &b.ffnGate)
		layer.FFNUp.MatvecInto(b.normed, &b.ffnUp)
		ensureLenNoClear(&b.ffnAct, len(b.ffnGate))
		siluMulF32(b.ffnGate, b.ffnUp, b.ffnAct)
		layer.FFNDown.MatvecInto(b.ffnAct, &b.ffnDown)
		if len(b.ffnDown) != dc.HiddenSize {
			return nil, fmt.Errorf("decoder step: layer %d ffn_down has %d elements, want HiddenSize=%d", li, len(b.ffnDown), dc.HiddenSize)
		}
		addInPlace(x, b.ffnDown)
	}

	rmsNormInto(x, weights.Decoder.OutputNorm, dc.Epsilon, &b.finalNormed)
	weights.Decoder.TokenEmbd.MatvecInto(b.finalNormed, &logits)
	state.nextPos++
	return logits, nil
}

// appendVoxtralHistory copies a projection into owned storage so that scratch
// reuse cannot change cached keys or values. A nonpositive window keeps all rows.
func appendVoxtralHistory(history *[][]float32, row []float32, window int) {
	if window > 0 && len(*history) >= window {
		recycled := (*history)[0]
		copy(*history, (*history)[1:])
		copy(recycled, row)
		(*history)[len(*history)-1] = recycled
		return
	}
	*history = append(*history, append([]float32(nil), row...))
}
