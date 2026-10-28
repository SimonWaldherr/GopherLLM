package engine

// copyConversationKVRange restores [start, end) while keeping a separately
// token-verified resident prefix intact. It copies quantized I8 rows verbatim,
// including their scales; recurrent state is outside this attention-only path.
// A successful copy returns end, matching copyKVPrefix's position convention.
func copyConversationKVRange(dst, src *KVCache, start, end int) int {
	if start < 0 || start > end || !kvPrefixSnapshotCompatible(dst, src, end) {
		return 0
	}
	if start == end {
		return end
	}
	if dst.kvFormat() == kvI8 {
		kRow, vRow := q8RowBytes(dst.PerPosKDim), q8RowBytes(dst.PerPosVDim)
		for layer := range dst.layerCount() {
			copy(dst.K8[layer][start*kRow:end*kRow], src.K8[layer][start*kRow:end*kRow])
			copy(dst.V8[layer][start*vRow:end*vRow], src.V8[layer][start*vRow:end*vRow])
		}
		return end
	}
	kStart, kEnd := start*dst.PerPosKDim, end*dst.PerPosKDim
	vStart, vEnd := start*dst.PerPosVDim, end*dst.PerPosVDim
	for layer := range dst.layerCount() {
		if dst.F16 {
			copy(dst.K16[layer][kStart:kEnd], src.K16[layer][kStart:kEnd])
			copy(dst.V16[layer][vStart:vEnd], src.V16[layer][vStart:vEnd])
		} else {
			copy(dst.K[layer][kStart:kEnd], src.K[layer][kStart:kEnd])
			copy(dst.V[layer][vStart:vEnd], src.V[layer][vStart:vEnd])
		}
	}
	return end
}
