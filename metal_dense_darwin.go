//go:build darwin && cgo && metal

package gopherllm

import (
	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
	"math"
	"os"
	"runtime"
	"unsafe"
)

var metalDenseDecodeEnabled = os.Getenv("GOPHERLLM_METAL_DENSE_DECODE") != "0"

type metalDenseState struct {
	pin                                         runtime.Pinner
	sharedK, sharedV                            [][]float32
	output                                      *metalbackend.Weight
	decoder                                     *metalbackend.Decoder
	owner                                       *LayerWeights
	cache                                       *KVCache
	dim, hidden, heads, kvheads, layers, window int
	eps, scale                                  float32
	refs, owned                                 []*metalbackend.Weight
	// batchKVSynced[layer] is how far (position count) that layer's persistent
	// Metal K/V buffer has been uploaded by metalDenseBatchAttention, letting
	// each successive prefill chunk append only the new suffix instead of
	// re-copying the whole prefix. It can only lag behind reality (e.g. after
	// single-token decode's own full resync), never overstate it, so a stale
	// value causes redundant-but-correct re-upload rather than stale data.
	batchKVSynced []int
}

func (s *metalDenseState) close() {
	if s == nil {
		return
	}
	s.decoder.Close()
	s.decoder = nil
	s.pin.Unpin()
	s.sharedK = nil
	s.sharedV = nil
	for _, w := range s.owned {
		metalbackend.Release(w)
	}
	s.owned = nil
	s.refs = nil
	s.output = nil
	runtime.SetFinalizer(s, nil)
}

func metalDenseEligible(c Config, w ModelWeights, k *KVCache) bool {
	// k.MaxLen must not exceed metalbackend.NewDecoder's own buffer-sizing
	// ceiling (its partial-attention and per-layer K/V buffers are allocated
	// once, sized proportional to maxlen, up to that limit) -- keeping this
	// check here (rather than only inside newMetalDenseState) avoids retrying
	// a doomed, non-trivial device allocation on every single decoded token
	// once a too-long context is configured.
	if !metalDenseDecodeEnabled || k == nil || k.F16 || k.I8 || k.MaxLen > 16384 || k.MaxLen <= 0 ||
		(c.Arch != "mistral3" && c.Arch != "ministral" && c.Arch != "qwen3" && c.Arch != "llama") ||
		c.HeadDim != 128 || c.ValueDim != 128 || c.RopeDimensionCount != 128 || c.RopeThetaSWA != 0 ||
		c.UseLayerNorm || c.ParallelResidual || c.UseGELU || c.UsesMLA || c.AttnLogitSoftcap != 0 ||
		c.ResidualScale != 1 || c.usesAbsolutePositionEmbd() || c.usesALiBi() ||
		c.NLayers <= 0 || len(w.Layers) != c.NLayers || len(w.OutputNorm) != c.Dim || !metalDenseZero(w.OutputNormBias) ||
		c.NHeads <= 0 || c.NKVHeads <= 0 || c.NHeads%c.NKVHeads != 0 || max(1, c.KVMul) != c.NHeads/c.NKVHeads || len(c.LayerHeads)+len(c.LayerKVHeads)+len(c.LayerFFNDim) != 0 ||
		k.PerPosKDim != c.NKVHeads*128 || k.PerPosVDim != c.NKVHeads*128 || len(k.K) != c.NLayers || len(k.V) != c.NLayers {
		return false
	}
	for i := range w.Layers {
		l := &w.Layers[i]
		if l.W1.Metal == nil || l.HasQKV || l.HasGateUp || l.MoE != nil || l.MLA != nil ||
			len(l.AttnNorm) != c.Dim || len(l.FFNNorm) != c.Dim ||
			!metalDenseZero(l.AttnNormBias, l.FFNNormBias, l.BQ, l.BK, l.BV, l.BO, l.FFNUpBias, l.FFNDownBias) || len(l.PostAttnNorm)+len(l.PostFFNNorm)+len(l.AttnSinks) != 0 ||
			(len(l.AttnQNorm) != 0 && len(l.AttnQNorm) != 128) || (len(l.AttnKNorm) != 0 && len(l.AttnKNorm) != 128) ||
			len(k.K[i]) < k.MaxLen*k.PerPosKDim || len(k.V[i]) < k.MaxLen*k.PerPosVDim {
			return false
		}
	}
	return true
}

func newMetalDenseState(c Config, w ModelWeights, k *KVCache) *metalDenseState {
	scale := c.AttentionScale
	if scale == 0 {
		scale = float32(1 / math.Sqrt(128))
	}
	s := &metalDenseState{sharedK: make([][]float32, c.NLayers), sharedV: make([][]float32, c.NLayers), owner: &w.Layers[0], cache: k, dim: c.Dim, hidden: c.HiddenDim, heads: c.NHeads, kvheads: c.NKVHeads, layers: c.NLayers, window: c.SlidingWindow, eps: c.RMSNormEps, scale: c.AttentionScale, batchKVSynced: make([]int, c.NLayers)}
	s.decoder = metalbackend.NewDecoder(c.Dim, c.HiddenDim, c.NHeads, c.NKVHeads, c.NLayers, k.MaxLen, c.RMSNormEps, scale, w.OutputNorm)
	if s.decoder == nil {
		return nil
	}
	fail := func() *metalDenseState { s.close(); return nil }
	for i := range w.Layers {
		l := &w.Layers[i]
		matrices := [7]Weight{l.WQ, l.WK, l.WV, l.WO, l.W1, l.W3, l.W2}
		rows := [7]int{c.NHeads * 128, c.NKVHeads * 128, c.NKVHeads * 128, c.Dim, c.HiddenDim, c.HiddenDim, c.Dim}
		cols := [7]int{c.Dim, c.Dim, c.Dim, c.NHeads * 128, c.Dim, c.Dim, c.HiddenDim}
		var refs [7]*metalbackend.Weight
		var quant [7]uint32
		for j, m := range matrices {
			if m.Rows != rows[j] || m.Cols != cols[j] || m.F32 != nil || m.Cols%256 != 0 {
				return fail()
			}
			var p *metalbackend.Weight
			switch m.Type {
			case GGMLTypeQ4_K:
				quant[j] = 4
				if m.Metal != nil {
					p = m.Metal.q4
				}
			case GGMLTypeQ6_K:
				quant[j] = 6
				if m.Metal != nil {
					p = m.Metal.q6
				}
			case GGMLTypeQ8_0:
				quant[j] = 8
				if m.Metal != nil {
					p = m.Metal.q8
				}
			default:
				return fail()
			}
			if p == nil {
				switch quant[j] {
				case 4:
					p = metalbackend.PrepareQ4K(m.Raw, m.Rows, m.Cols, false)
				case 6:
					p = metalbackend.PrepareQ6K(m.Raw, m.Rows, m.Cols, false)
				case 8:
					p = metalbackend.PrepareQ8_0(m.Raw, m.Rows, m.Cols, false)
				}
				if p == nil {
					return fail()
				}
				s.owned = append(s.owned, p)
			}
			refs[j] = p
			s.refs = append(s.refs, p)
		}
		// layerUsesSWA consults c.SWAPattern per layer when one is present
		// (Ministral 8B's interleaved local/global attention), so this
		// already binds the correct per-layer window regardless of whether
		// the model uses a uniform window on every layer or an interleaved
		// pattern -- metalDenseEligible no longer needs to reject the latter.
		window := 0
		if c.layerUsesSWA(i) {
			window = c.SlidingWindow
		}
		if !s.decoder.BindLayer(i, refs, quant, l.AttnNorm, l.FFNNorm, l.AttnQNorm, l.AttnKNorm, window) {
			return fail()
		}
		if os.Getenv("GOPHERLLM_METAL_DENSE_SHARED_KV") != "0" && metalKVShareable(k.K[i]) && metalKVShareable(k.V[i]) {
			s.pin.Pin(&k.K[i][0])
			s.pin.Pin(&k.V[i][0])
			if s.decoder.BindCache(i, k.K[i], k.V[i]) {
				s.sharedK[i] = k.K[i]
				s.sharedV[i] = k.V[i]
			}
		}
	}
	runtime.SetFinalizer(s, (*metalDenseState).close)
	return s
}

func tryMetalDenseDecode(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer, token uint32, pos int) bool {
	return tryMetalDenseDecodeOutput(c, w, k, b, token, pos, nil)
}

func tryMetalDenseDecodeOutput(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer, token uint32, pos int, logits *[]float32) bool {
	return tryMetalDenseStep(c, w, k, b, token, pos, logits, nil, 1, nil)
}

func tryMetalDenseGreedy(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer, token uint32, pos int, recent []uint32, penalty float32) (uint32, bool) {
	if len(w.OutputBias) > 0 || !finite32(c.LogitScale) || c.LogitScale <= 0 || !finite32(c.FinalLogitSoftcap) || c.FinalLogitSoftcap < 0 || (penalty != 1 && c.FinalLogitSoftcap > 0) || !finite32(penalty) || penalty <= 0 || len(recent) > 64 {
		return 0, false
	}
	var next uint32
	ok := tryMetalDenseStep(c, w, k, b, token, pos, nil, recent, penalty, &next)
	return next, ok
}

func tryMetalDenseStep(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer, token uint32, pos int, logits *[]float32, recent []uint32, penalty float32, next *uint32) bool {
	s := metalDenseFor(c, w, k, b)
	if s == nil || pos < 0 || pos >= k.MaxLen {
		return false
	}

	if logits != nil || next != nil {
		if w.Output.F32 != nil || w.Output.Metal == nil || w.Output.Cols != c.Dim || w.Output.Rows != c.VocabSize {
			return false
		}
		var output *metalbackend.Weight
		var quant uint32
		switch w.Output.Type {
		case GGMLTypeQ4_K:
			output = w.Output.Metal.q4
			quant = 4
		case GGMLTypeQ6_K:
			output = w.Output.Metal.q6
			quant = 6
		case GGMLTypeQ8_0:
			output = w.Output.Metal.q8
			quant = 8
		default:
			return false
		}
		if output == nil {
			return false
		}
		if s.output != output {
			if !s.decoder.BindOutput(output, quant) {
				return false
			}
			s.output = output
		}
		if logits != nil {
			ensureLenNoClear(logits, c.VocabSize)
		}
	}
	var projected []float32
	if logits != nil {
		projected = *logits
	}
	_, pairs := prepareRopeScratch(pos, 128, c.RopeDimensionCount, b.RopeInvFreq, b.RopeMscale, &b.RopeSin, &b.RopeCos)
	if pairs != 64 {
		return false
	}
	w.TokenEmbd.RowInto(int(token), c.Dim, &b.X)
	if image, ok := b.ImageEmbeds[pos]; ok {
		copy(b.X, image)
	} else if c.EmbeddingScale != 1 {
		ScaleF32(b.X, c.EmbeddingScale)
	}
	ensureLenNoClear(&b.XN, c.Dim)
	// CPU cache remains authoritative. Page-aligned pinned caches are shared
	// directly; external allocations copy the prefix and read back the new row.
	// Both preserve cache edits, rewinds, restoration and CPU fallback.
	for i := range w.Layers {
		if !s.decoder.Cache(i, pos, k.K[i], k.V[i], true) {
			return false
		}
	}
	if !s.decoder.Step(b.X, b.RopeSin, b.RopeCos, pairs, pos, ropeInterleaved(c.Arch), attentionTemperatureAt(c, pos), b.X, b.XN, projected, recent, penalty, next) {
		s.close()
		b.metalDense = nil
		return false
	}
	for i := range w.Layers {
		s.decoder.Cache(i, pos, k.K[i][pos*k.PerPosKDim:], k.V[i][pos*k.PerPosVDim:], false)
	}
	runtime.KeepAlive(s)
	return true
}

func metalDenseZero(vectors ...[]float32) bool {
	for _, v := range vectors {
		for _, x := range v {
			if x != 0 {
				return false
			}
		}
	}
	return true
}

// Metal's no-copy buffers require page-aligned starts and lengths. Retain
// enough capacity after the visible KV rows to round the buffer length up.
func makeKVF32(n int) []float32 {
	page := os.Getpagesize() / 4
	if n <= 0 || n > int(^uint(0)>>1)-2*page {
		return make([]float32, n)
	}
	storage := make([]float32, n+2*page)
	offset := int((uintptr(os.Getpagesize())-uintptr(unsafe.Pointer(&storage[0]))%uintptr(os.Getpagesize()))%uintptr(os.Getpagesize())) / 4
	return storage[offset : offset+n]
}
func metalKVShareable(v []float32) bool {
	page := os.Getpagesize()
	return len(v) > 0 && uintptr(unsafe.Pointer(&v[0]))%uintptr(page) == 0 && cap(v) >= (len(v)*4+page-1)/page*(page/4)
}
func (s *metalDenseState) matchesCache(k *KVCache) bool {
	if s.cache != k {
		return false
	}
	for i, v := range s.sharedK {
		if v != nil && (&v[0] != &k.K[i][0] || &s.sharedV[i][0] != &k.V[i][0]) {
			return false
		}
	}
	return true
}

// valid reports whether s is still the correct bound decoder for c/w/k: the
// same loaded model (owner is a pointer into that model's own layer slice),
// the same (unresized, unreplaced) KV cache, and every config field the
// decoder was actually built from. Once true, none of metalDenseEligible's
// checks can have started failing, since nothing they inspect changes
// without also changing one of these.
func (s *metalDenseState) valid(c Config, w ModelWeights, k *KVCache) bool {
	return s.matchesCache(k) && s.owner == &w.Layers[0] && s.cache == k && s.dim == c.Dim &&
		s.hidden == c.HiddenDim && s.heads == c.NHeads && s.kvheads == c.NKVHeads &&
		s.layers == c.NLayers && s.window == c.SlidingWindow && s.eps == c.RMSNormEps && s.scale == c.AttentionScale
}

func metalDenseFor(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer) *metalDenseState {
	if b == nil {
		return nil
	}
	// Fast path: metalDenseEligible is a long per-layer scan (dozens of field
	// comparisons plus a zero-check over every bias slice in every layer),
	// re-run from scratch on every call by the code below -- profiling a real
	// decode loop showed this costing a measurable slice of per-token CPU
	// time despite deciding the same "yes" every single token. Once a decoder
	// is already correctly bound for this exact model/cache/config, nothing
	// eligibility would check can have changed without also changing one of
	// the fields valid() compares, so re-deriving the same answer the
	// expensive way is pure waste.
	if s := b.metalDense; s != nil && s.valid(c, w, k) {
		return s
	}
	if !metalDenseEligible(c, w, k) {
		if b.metalDense != nil {
			b.metalDense.close()
			b.metalDense = nil
		}
		return nil
	}
	s := b.metalDense
	if s != nil && !s.valid(c, w, k) {
		s.close()
		b.metalDense = nil
		s = nil
	}
	if s == nil {
		s = newMetalDenseState(c, w, k)
		if s == nil {
			return nil
		}
		b.metalDense = s
	}
	return s
}

func prepareMetalDenseBatch(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer, batch int) bool {
	return batch >= 16 && batch <= metalBatchFFNMaxTokens && os.Getenv("GOPHERLLM_METAL_DENSE_PREFILL") != "0" && metalDenseFor(c, w, k, b) != nil
}
func metalDenseBatchProjection(b *DecodeBuffer, layer, matrix int, x, out []float32, batch int) bool {
	if b == nil || b.metalDense == nil {
		return false
	}
	s := b.metalDense
	ok := s.decoder.Project(layer, matrix, x, out, batch)
	runtime.KeepAlive(s) // The state owns/pins the decoder's borrowed weights and KV.
	return ok
}

// metalDenseBatchProjectionQKV is metalDenseBatchProjection for Q, K, and V
// fused into one command buffer instead of three -- see
// gllm_decode_project_qkv's doc comment for why that matters at realistic
// prefill chunk sizes.
func metalDenseBatchProjectionQKV(b *DecodeBuffer, layer int, x, qOut, kOut, vOut []float32, batch int) bool {
	if b == nil || b.metalDense == nil {
		return false
	}
	s := b.metalDense
	ok := s.decoder.ProjectQKV(layer, x, qOut, kOut, vOut, batch)
	runtime.KeepAlive(s) // The state owns/pins the decoder's borrowed weights and KV.
	return ok
}

var metalBatchAttentionEnabled = os.Getenv("GOPHERLLM_METAL_BATCH_ATTENTION") != "0"

// metalDenseBatchAttention runs causal batched multi-head attention for one
// layer's prompt chunk on Metal: gllm_decode_batch_attention's doc comment
// has the full contract. q is [batch][heads*128], already RoPE-rotated and
// temperature-scaled on the CPU exactly as the CPU attention path expects;
// out is the same shape. It first syncs the decoder's persistent per-layer
// K/V buffers (a plain memcpy, or a no-op when they already alias cache
// directly) through the end of this chunk, since attendHeadGroup's CPU
// counterpart reads the equivalent Go-side cache.K/cache.V slices, which
// forward_batch.go's cache.storeKV calls have already updated by the time
// attention runs.
//
// A profiled real prefill (long prompt, real Ministral-3B checkpoint) found
// this CPU attention step -- not the projections -- was 93% of prefill wall
// time: an O(n^2) computation the CPU path parallelizes only across
// goroutines, where Metal parallelizes across thousands of independent
// (token, head) GPU threads instead.
func metalDenseBatchAttention(b *DecodeBuffer, layer int, cache *KVCache, q []float32, startPos, batch int, out []float32) bool {
	if !metalBatchAttentionEnabled || b == nil || b.metalDense == nil || cache == nil {
		return false
	}
	s := b.metalDense
	pos := startPos + batch
	from := 0
	if layer < len(s.batchKVSynced) {
		from = s.batchKVSynced[layer]
		if from > pos {
			from = 0
		}
	}
	if !s.decoder.CacheAppend(layer, from, pos, cache.K[layer], cache.V[layer]) {
		return false
	}
	if layer < len(s.batchKVSynced) {
		s.batchKVSynced[layer] = pos
	}
	ok := s.decoder.BatchAttention(layer, q, startPos, batch, out)
	runtime.KeepAlive(s) // The state owns/pins the decoder's borrowed weights and KV.
	return ok
}
