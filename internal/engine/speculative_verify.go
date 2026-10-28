package engine

import (
	"context"
	"fmt"
	"os"
)

const speculativeMaxPositions = 8
const speculativeMaxCheckpointBytes = 256 << 20

// One workspace holds a speculative window, not a recurrent snapshot for
// every candidate. Rejection restores the checkpoint and replays only the
// accepted input prefix. Attention rows beyond that prefix are never read.
type speculativeVerifyScratch struct {
	inputs                         [speculativeMaxPositions]uint32
	history                        [speculativeMaxPositions][repeatPenaltyWindow]uint32
	recent                         [speculativeMaxPositions][]uint32
	hidden                         [][]float32
	predictions                    []uint32
	outputInput                    []float32
	outputLogits                   []float32
	outputRows                     [][]float32
	boundaryX, boundaryXN, pending []float32
	checkpoint                     *Qwen35Cache
	qwenReplay                     qwen35RecurrentReplay
	lookup                         []uint32
	start, count                   int
	active                         bool
}

func qwen35HasQuantizedProjections(w Qwen35Weights) bool {
	quantized := func(w Weight) bool { return w.F32 == nil && len(w.Raw) > 0 && !rawScalarWeight(w) }
	if quantized(w.Output) {
		return true
	}
	for _, l := range w.Layers {
		if quantized(l.FFN.Gate) || quantized(l.FFN.Up) || quantized(l.FFN.Down) {
			return true
		}
		if l.Kind == qwen35Attention {
			w := l.Attention
			if quantized(w.Q) || quantized(w.K) || quantized(w.V) || quantized(w.O) {
				return true
			}
		} else {
			w := l.DeltaNet
			if quantized(w.QKVConv) || quantized(w.Gate) || quantized(w.AlphaProj) || quantized(w.BetaProj) || quantized(w.Out) {
				return true
			}
		}
	}
	return false
}

func (r *Runner) canVerifyQwen35() bool {
	// Quantized float-only row kernels and batched float reductions have
	// different operation trees. Keep that backend's established MTP schedule
	// until multi-position verification has its own numerical parity coverage.
	return r.canBatchQwen35() && (useQ8Activations.Load() || !qwen35HasQuantizedProjections(r.qwen35))
}

func (r *Runner) canVerifyBlock(cache *KVCache) bool {
	if os.Getenv("GOPHERLLM_NO_BLOCK_VERIFY") != "" || cache == nil {
		return false
	}
	if r.kind == loadedQwen35 {
		return r.config.Arch == "qwen35" && (r.canVerifyQwen35() || r.canBatchQwen35Metal(cache)) && cache.Qwen35 != nil && cache.Qwen35.bytes() <= speculativeMaxCheckpointBytes
	}
	if !useQ8Activations.Load() && (cache.F16 || cache.I8) {
		quantized := func(w Weight) bool { return w.F32 == nil && len(w.Raw) > 0 && !rawScalarWeight(w) }
		weights, ok := r.batchPrefillWeights()
		if !ok || quantized(weights.Output) {
			return false
		}
		for _, l := range weights.Layers {
			if quantized(l.WQ) || quantized(l.WK) || quantized(l.WV) || quantized(l.WO) || quantized(l.W1) || quantized(l.W2) || quantized(l.W3) || quantized(l.WQKV) || quantized(l.WGateUp) {
				return false
			}
		}
	}
	return r.canBatchPrefill()
}

func (r *Runner) forwardQwen35BlockStates(ctx context.Context, cache *KVCache, buf *DecodeBuffer, inputs []uint32, start int) ([][]float32, error) {
	if r.canBatchQwen35Metal(cache) {
		return forwardQwen35MetalBatchStatesInto(ctx, r.config, r.qwen35, cache, buf, inputs, start)
	}
	return forwardQwen35BatchStatesInto(ctx, r.config, r.qwen35, cache, buf, inputs, start)
}

// recent already includes inputs[0], the seed just emitted by the caller.
// Each later output row sees the preceding candidate in its repetition window.
func (s *speculativeVerifyScratch) histories(recent []uint32) {
	for i := 0; i < s.count; i++ {
		row := s.history[i][:0]
		if i == 0 {
			row = append(row, recent[max(0, len(recent)-repeatPenaltyWindow):]...)
		} else {
			prev := s.recent[i-1]
			row = append(row, prev[max(0, len(prev)+1-repeatPenaltyWindow):]...)
			row = append(row, s.inputs[i])
		}
		s.recent[i] = row
	}
}

func (r *Runner) prepareSpeculativeBlock(ctx context.Context, cache *KVCache, buf *DecodeBuffer, inputs []uint32, start int, recent []uint32, penalty float32) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if !r.canVerifyBlock(cache) || len(inputs) < 2 || len(inputs) > speculativeMaxPositions {
		return false, nil
	}
	if start < 0 || start > cache.MaxLen || len(inputs) > cache.MaxLen-start {
		return false, fmt.Errorf("gopherllm: speculative block exceeds the cache range")
	}
	s := &buf.speculative
	if s.active {
		return false, fmt.Errorf("gopherllm: speculative block already active")
	}
	s.start, s.count = start, len(inputs)
	copy(s.inputs[:], inputs)
	ensureLenNoClear(&s.boundaryX, len(buf.X))
	ensureLenNoClear(&s.boundaryXN, len(buf.XN))
	copy(s.boundaryX, buf.X)
	copy(s.boundaryXN, buf.XN)
	if r.kind == loadedQwen35 {
		cache.Qwen35.snapshotInto(&s.checkpoint)
		if cache.Qwen35MTP != nil {
			ensureLenNoClear(&s.pending, len(cache.Qwen35MTP.PendingHidden))
			copy(s.pending, cache.Qwen35MTP.PendingHidden)
		}
	}
	s.active = true
	s.histories(recent)
	var err error
	fusedOutput := false
	if r.kind == loadedQwen35 {
		s.hidden, err = r.forwardQwen35BlockStates(ctx, cache, buf, s.inputs[:s.count], start)
	} else {
		weights, _ := r.batchPrefillWeights()
		s.hidden, fusedOutput, err = tryMetalDenseVerify(ctx, r.config, weights, cache, buf, s.inputs[:s.count], start, s.recent[:s.count], penalty, &s.predictions)
		if err == nil && !fusedOutput {
			s.hidden, err = forwardBatchStatesIntoContext(ctx, r.config, weights, cache, buf, s.inputs[:s.count], start)
		}
	}
	if err != nil {
		r.abortSpeculativeBlock(cache, buf)
		return false, err
	}
	weights := r.standard
	if r.kind == loadedQwen35 {
		weights = ModelWeights{Output: r.qwen35.Output}
	} else {
		weights, _ = r.batchPrefillWeights()
	}
	if !fusedOutput {
		greedyBatchOutput(r.config, weights, buf, s.hidden, s.recent[:s.count], penalty, &s.predictions, &s.outputInput)
	}
	if err := ctx.Err(); err != nil {
		r.abortSpeculativeBlock(cache, buf)
		return false, err
	}
	return true, nil
}

// The output operation retains per-position penalty histories and the normal
// bias/scaling/softcap order. Only eligible Q6_K heads use the fused GPU path.
func greedyBatchOutput(cfg Config, weights ModelWeights, buf *DecodeBuffer, hidden [][]float32, recent [][]uint32, penalty float32, tokens *[]uint32, flat *[]float32) {
	ensureLenNoClear(tokens, len(hidden))
	forceLogits := os.Getenv("GOPHERLLM_NO_GREEDY_ARGMAX") == "1"
	if !forceLogits && greedyCPUQuantizedBatchOutput(cfg, weights, buf, hidden, recent, penalty, *tokens) {
		return
	}
	ordering := !forceLogits && len(weights.OutputBias) == 0 && finite32(cfg.LogitScale) && cfg.LogitScale > 0 && finite32(cfg.FinalLogitSoftcap) && cfg.FinalLogitSoftcap >= 0 && (penalty == 1 || cfg.FinalLogitSoftcap == 0)
	if ordering && penalty == 1 && weights.Output.ArgmaxMatvecBatch(hidden, tokens) {
		return
	}
	if ordering && weights.Output.Type == GGMLTypeQ6_K && weights.Output.Metal != nil {
		ensureLenNoClear(flat, len(hidden)*cfg.Dim)
		for i, row := range hidden {
			copy((*flat)[i*cfg.Dim:(i+1)*cfg.Dim], row)
		}
		if argmaxMetalQ6KBatchPenalized(weights.Output.Metal, *flat, *tokens, recent, penalty) {
			return
		}
	}
	output := DecodeBuffer{Logits: buf.Logits}
	for i, row := range hidden {
		output.XN = row
		var token uint32
		ok := false
		if !forceLogits {
			token, ok = argmaxOutputTokenPenalizedInto(cfg, weights, &output, recent[i], penalty, &output.Logits)
		}
		if !ok {
			ProjectLogitsInto(cfg, weights, &output, &output.Logits)
			applyRepeatPenalty(output.Logits, recent[i], penalty)
			token = argmaxFiniteToken(output.Logits)
		}
		(*tokens)[i] = token
	}
	buf.Logits = output.Logits
}

func (r *Runner) abortSpeculativeBlock(cache *KVCache, buf *DecodeBuffer) {
	s := &buf.speculative
	if !s.active {
		return
	}
	if r.kind == loadedQwen35 {
		cache.Qwen35.restore(s.checkpoint)
		if cache.Qwen35MTP != nil {
			copy(cache.Qwen35MTP.PendingHidden, s.pending)
		}
	}
	copy(buf.X, s.boundaryX)
	copy(buf.XN, s.boundaryXN)
	s.active = false
}

// keep counts target inputs, including the already-emitted seed. On rejection
// only that chronological prefix becomes resident. Cancellation restores the
// original boundary, leaving previously resident metadata usable.
func (r *Runner) finishSpeculativeBlock(ctx context.Context, cache *KVCache, buf *DecodeBuffer, keep int, mtp, seedPrepared bool) error {
	s := &buf.speculative
	if !s.active || keep < 0 || keep > s.count {
		return fmt.Errorf("gopherllm: invalid speculative commit boundary")
	}
	if err := ctx.Err(); err != nil {
		r.abortSpeculativeBlock(cache, buf)
		return err
	}
	if keep == 0 {
		r.abortSpeculativeBlock(cache, buf)
		return nil
	}
	if r.kind == loadedQwen35 && keep < s.count {
		cache.Qwen35.restore(s.checkpoint)
		if s.qwenReplay.valid {
			if err := s.qwenReplay.replay(ctx, r.config, r.qwen35, cache.Qwen35, buf, keep); err != nil {
				r.abortSpeculativeBlock(cache, buf)
				return err
			}
		} else {
			for i, token := range s.inputs[:keep] {
				if err := ctx.Err(); err != nil {
					r.abortSpeculativeBlock(cache, buf)
					return err
				}
				r.forwardHiddenToken(cache, buf, token, s.start+i)
			}
		}
	}
	if mtp {
		first := 0
		if seedPrepared {
			copy(cache.Qwen35MTP.PendingHidden, s.hidden[0])
			first = 1
		}
		if err := qwen35MTPProcessBatch(ctx, r.config, r.qwen35.MTP, cache.Qwen35MTP, s.inputs[first:keep], s.hidden[first:keep], s.start+first); err != nil {
			r.abortSpeculativeBlock(cache, buf)
			return err
		}
	}
	copy(buf.XN, s.hidden[keep-1])
	if r.kind == loadedQwen35 {
		copy(buf.X, buf.qwenBatch.X[keep-1])
	} else {
		copy(buf.X, buf.batch.X[keep-1])
	}
	if err := ctx.Err(); err != nil {
		r.abortSpeculativeBlock(cache, buf)
		return err
	}
	s.active = false
	return nil
}

func (r *Runner) prefillQwen35MTPAt(ctx context.Context, cache *KVCache, buf *DecodeBuffer, tokens []uint32, start int, logits *[]float32) error {
	chunk := min(r.prefillChunkSize(), 32)
	for first := 0; first < len(tokens); first += chunk {
		end := min(first+chunk, len(tokens))
		rows, err := r.forwardQwen35BlockStates(ctx, cache, buf, tokens[first:end], start+first)
		if err != nil {
			return err
		}
		if err := qwen35MTPProcessBatch(ctx, r.config, r.qwen35.MTP, cache.Qwen35MTP, tokens[first:end], rows, start+first); err != nil {
			return err
		}
	}
	if logits != nil {
		r.qwen35.Output.MatvecInto(buf.XN, logits)
	}
	return ctx.Err()
}
