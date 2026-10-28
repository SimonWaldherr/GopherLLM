package engine

// Reuse each CPU vocabulary-weight row across the verifier's positions, just
// as the body projections do. Calling ArgmaxMatvec once per position streams
// the often larger-than-cache vocabulary matrix repeatedly. The existing Q8
// batch kernels retain the serial dot-product arithmetic, including ARM's
// four-position kernels, so penalty, bias and softcap can follow unchanged.
func greedyCPUQuantizedBatchOutput(cfg Config, weights ModelWeights, buf *DecodeBuffer, hidden [][]float32, recent [][]uint32, penalty float32, tokens []uint32) bool {
	w := weights.Output
	const maxLogitBytes = 64 << 20
	if !useQ8Activations.Load() || w.F32 != nil || w.Metal != nil || w.GPU != nil ||
		len(hidden) < 4 || len(hidden) > speculativeMaxPositions || w.Rows <= 0 ||
		w.Rows > maxLogitBytes/4/len(hidden) {
		return false
	}
	switch w.Type {
	case GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0:
	default:
		return false
	}
	s := &buf.speculative
	rows := reuseBatchViews(&s.outputLogits, &s.outputRows, len(hidden), w.Rows)
	if !matvecBatchQ8(w, hidden, rows) {
		return false
	}
	rawArgmax := penalty == 1 && len(weights.OutputBias) == 0 && finite32(cfg.LogitScale) && cfg.LogitScale > 0 &&
		finite32(cfg.FinalLogitSoftcap) && cfg.FinalLogitSoftcap >= 0
	for i, logits := range rows {
		// Match the serial greedy shortcut's raw argmax when applicable;
		// rounded softcap values can otherwise collapse distinct logits to ties.
		if !rawArgmax {
			finishLogits(cfg, weights, &logits)
			applyRepeatPenalty(logits, recent[i], penalty)
		}
		tokens[i] = argmaxFiniteToken(logits)
	}
	return true
}
