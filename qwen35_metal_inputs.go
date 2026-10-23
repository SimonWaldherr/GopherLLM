package gopherllm

// Keep CPU batching on the same Q8 row kernels as token decode. Prepared
// projections and float matrices can select a different reduction tree.
func qwen35MetalBatchCPUInput(w Weight) bool {
	if w.Prepared != nil || !qwen35MetalBatchCPUOut(w) {
		return false
	}
	rowBytes, ok := w.Type.DataSize(w.Cols)
	return ok && rowBytes > 0 && w.Rows <= len(w.Raw)/rowBytes
}

// Independent input projections can run before the chronological recurrence.
// Treat CPU pairs together: token decode may fuse/demote a pair, so moving
// just one member could change its arithmetic. A direct QKV explicitly skips
// that pair in qwen35DeltaNetMix and permits independent gate batching.
func qwen35MetalBatchDeltaInputs(w Qwen35DeltaNetWeights, xs [][]float32, b *qwen35BatchScratch) (qkv, gate, alphaBeta bool) {
	if len(xs) <= 1 {
		return
	}
	if qwen35MetalBatchCPUInput(w.QKVConv) && qwen35MetalBatchCPUInput(w.Gate) {
		matvecBatch2Slabs(w.QKVConv, w.Gate, xs, b.QKV, b.Gate, b.XNFlat, b.QKVFlat, b.GateFlat)
		qkv, gate = true, true
	} else if w.QKVConv.GPU == nil && metalWeightUsesDirect(w.QKVConv.Metal) {
		qkv = matvecMetalQ6KDecodeBatchInto(w.QKVConv.Metal, b.XNFlat, len(xs), &b.QKVFlat)
		if qwen35MetalBatchCPUInput(w.Gate) {
			matvecBatchSlabs(w.Gate, xs, b.Gate, b.XNFlat, b.GateFlat)
			gate = true
		}
	}
	if qwen35MetalBatchCPUInput(w.AlphaProj) && qwen35MetalBatchCPUInput(w.BetaProj) {
		matvecBatch2Slabs(w.AlphaProj, w.BetaProj, xs, b.Alpha, b.Beta, b.XNFlat, b.AlphaFlat, b.BetaFlat)
		alphaBeta = true
	}
	return
}

// Compute only projections not prepared by the batch. Pair fallback retains
// the original fusion decisions, including the CPU/GPU split of mixed types.
func qwen35DeltaNetBatchInputs(w Qwen35DeltaNetWeights, x []float32, buf *DecodeBuffer,
	qkv, gate, alpha, beta []float32, haveQKV, haveGate, haveAlphaBeta bool,
) ([]float32, []float32, []float32, []float32) {
	if !haveQKV && !haveGate {
		if metalWeightUsesDirect(w.QKVConv.Metal) ||
			!tryMatvec2Into(w.QKVConv, w.Gate, x, &buf.Q4KXSums, &buf.MambaIn, &buf.MambaZ) {
			w.QKVConv.MatvecInto(x, &buf.MambaIn)
			w.Gate.MatvecInto(x, &buf.MambaZ)
		}
		qkv, gate = buf.MambaIn, buf.MambaZ
	} else {
		if !haveQKV {
			w.QKVConv.MatvecInto(x, &buf.MambaIn)
			qkv = buf.MambaIn
		}
		if !haveGate {
			w.Gate.MatvecInto(x, &buf.MambaZ)
			gate = buf.MambaZ
		}
	}
	if !haveAlphaBeta {
		if !tryMatvec2Into(w.AlphaProj, w.BetaProj, x, &buf.Q4KXSums, &buf.MambaDT, &buf.MambaBeta) {
			w.AlphaProj.MatvecInto(x, &buf.MambaDT)
			w.BetaProj.MatvecInto(x, &buf.MambaBeta)
		}
		alpha, beta = buf.MambaDT, buf.MambaBeta
	}
	return qkv, gate, alpha, beta
}
