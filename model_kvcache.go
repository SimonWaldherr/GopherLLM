package gopherllm

// KVCache stores the attention keys and values of every processed position,
// one flat slice per layer laid out position-major: position p's keys occupy
// K[layer][p*PerPosKDim : (p+1)*PerPosKDim] (all KV heads concatenated), and
// likewise for V. Sized to prompt+max_tokens (capped at the model context
// length) and reused by Runner when capacity permits; there is no ring/eviction
// and generation stops at its request-local cache length.
type KVCache struct {
	K          [][]float32
	V          [][]float32
	PerPosKDim int
	PerPosVDim int
	MaxLen     int
	// F16 selects half-precision row storage: K16/V16 replace K/V entirely
	// and attention converts rows in-register (see kv_f16.go). Halves the
	// cache's memory footprint and the bytes attention streams per token.
	F16 bool
	// I8 selects Q8_0-block row storage: K8/V8 replace K/V entirely (see
	// kv_i8.go). ~3.76x smaller than f32 and ~1.88x smaller than f16, at the
	// cost of a coarser, data-dependent quantization the fixed f16 conversion
	// doesn't have. Only ever set when kvI8Eligible has already verified every
	// relevant dimension is a multiple of 32 — the format assumes block
	// (not element) addressability wherever it slices K8/V8 by KV head.
	// Kept as a second bool alongside F16, not merged into one field, so the
	// exported F16 field and NewKVCache/NewKVCacheF16 stay unchanged for any
	// external caller; kvFormat below is the single place that resolves both
	// bools into one tri-state so no call site re-derives it independently.
	I8 bool
	// Nemotron is the shared recurrent cache for Mamba-2 graphs. Hybrid
	// Nemotron-H also uses K/V rows; pure Mamba2 leaves those dimensions empty.
	Nemotron *NemotronHCache
	// Qwen35 is the Gated DeltaNet recurrent state for the qwen35/qwen35moe
	// hybrid graph. Its periodic full-attention layers use the K/V rows above,
	// same dual-cache pattern as Nemotron-H.
	Qwen35 *Qwen35Cache
	// Qwen35MTP is the optional one-layer NextN draft head's independent KV
	// cache and staging scratch. It is allocated only for a generation request
	// that explicitly enables MTP speculation.
	Qwen35MTP *Qwen35MTPState
	K16       [][]uint16
	V16       [][]uint16
	K8        [][]byte
	V8        [][]byte
}

// kvFormat is the tri-state storage format a KVCache actually holds. Every
// internal dispatch site should call (*KVCache).kvFormat() rather than
// re-checking F16/I8 independently — that guarantees a single, consistent
// precedence rule everywhere instead of two branches silently disagreeing on
// a malformed cache.
type kvFormat uint8

const (
	kvF32 kvFormat = iota
	kvF16
	kvI8
)

func (c *KVCache) kvFormat() kvFormat {
	switch {
	case c.I8:
		return kvI8
	case c.F16:
		return kvF16
	default:
		return kvF32
	}
}

// NewKVCache allocates an f32 cache for `layers` layers of maxLen positions
// with the given per-position K and V widths (see KVCache).
func NewKVCache(layers, kDim, vDim, maxLen int) *KVCache {
	k := make([][]float32, layers)
	v := make([][]float32, layers)
	for i := range layers {
		k[i] = make([]float32, maxLen*kDim)
		v[i] = make([]float32, maxLen*vDim)
	}
	return &KVCache{K: k, V: v, PerPosKDim: kDim, PerPosVDim: vDim, MaxLen: maxLen}
}

// NewKVCacheF16 is NewKVCache with half-precision row storage.
func NewKVCacheF16(layers, kDim, vDim, maxLen int) *KVCache {
	k := make([][]uint16, layers)
	v := make([][]uint16, layers)
	for i := range layers {
		k[i] = make([]uint16, maxLen*kDim)
		v[i] = make([]uint16, maxLen*vDim)
	}
	return &KVCache{K16: k, V16: v, F16: true, PerPosKDim: kDim, PerPosVDim: vDim, MaxLen: maxLen}
}

// NewKVCacheI8 is NewKVCache with Q8_0-block row storage. Callers must have
// already verified kvI8Eligible(kDim, vDim, headDim, valueDim) — this
// constructor does not re-check alignment itself, matching NewKVCacheF16's
// contract of trusting its caller's format decision.
func NewKVCacheI8(layers, kDim, vDim, maxLen int) *KVCache {
	k := make([][]byte, layers)
	v := make([][]byte, layers)
	kRowBytes := q8RowBytes(kDim)
	vRowBytes := q8RowBytes(vDim)
	for i := range layers {
		k[i] = make([]byte, maxLen*kRowBytes)
		v[i] = make([]byte, maxLen*vRowBytes)
	}
	return &KVCache{K8: k, V8: v, I8: true, PerPosKDim: kDim, PerPosVDim: vDim, MaxLen: maxLen}
}

// desiredKVFormat resolves the global toggles into the format a new cache
// should be built as, applying the same I8-then-F16-then-F32 precedence
// kvFormat uses for an existing cache. allowI8 gates I8 on this request's
// dimensions already having passed kvI8Eligible — see generationWorkspace.
func desiredKVFormat(allowI8 bool) kvFormat {
	switch {
	case allowI8 && useI8KVCache.Load():
		return kvI8
	case useF16KVCache.Load():
		return kvF16
	default:
		return kvF32
	}
}

// newKVCacheAuto picks the fastest storage format the platform/model shape
// supports: int8 if explicitly enabled and allowI8 (dimension-eligible), else
// f16 where the platform supports it (see useF16KVCache), else exact f32.
func newKVCacheAuto(layers, kDim, vDim, maxLen int, allowI8 bool) *KVCache {
	switch desiredKVFormat(allowI8) {
	case kvI8:
		return NewKVCacheI8(layers, kDim, vDim, maxLen)
	case kvF16:
		return NewKVCacheF16(layers, kDim, vDim, maxLen)
	default:
		return NewKVCache(layers, kDim, vDim, maxLen)
	}
}

// layerCount reports the number of layers the cache was allocated for,
// regardless of element format.
func (c *KVCache) layerCount() int {
	switch c.kvFormat() {
	case kvI8:
		return len(c.K8)
	case kvF16:
		return len(c.K16)
	default:
		return len(c.K)
	}
}

// storeKV writes one position's K and V rows into the cache in its native
// element format.
func (c *KVCache) storeKV(l, pos int, k, v []float32) {
	switch c.kvFormat() {
	case kvI8:
		kRowBytes := q8RowBytes(c.PerPosKDim)
		vRowBytes := q8RowBytes(c.PerPosVDim)
		kStart8 := pos * kRowBytes
		vStart8 := pos * vRowBytes
		f32ToQ8Row(c.K8[l][kStart8:kStart8+kRowBytes], k)
		f32ToQ8Row(c.V8[l][vStart8:vStart8+vRowBytes], v)
		return
	case kvF16:
		kStart := pos * c.PerPosKDim
		vStart := pos * c.PerPosVDim
		f32ToF16Row(c.K16[l][kStart:kStart+min(len(k), c.PerPosKDim)], k)
		f32ToF16Row(c.V16[l][vStart:vStart+min(len(v), c.PerPosVDim)], v)
		return
	}
	kStart := pos * c.PerPosKDim
	vStart := pos * c.PerPosVDim
	copy(c.K[l][kStart:kStart+min(len(k), c.PerPosKDim)], k)
	copy(c.V[l][vStart:vStart+min(len(v), c.PerPosVDim)], v)
}

// attendHead runs online attention for one query head against this cache's
// rows, dispatching to the storage format's kernel set.
func (c *KVCache) attendHead(l, kvH int, query []float32, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	c.attendHeadWithSink(l, kvH, query, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, 0, 0, false, out)
}

// attendHeadWithSink is attendHead with an optional learned no-value sink
// logit. GPT-OSS appends that logit to each head's softmax denominator without
// adding a value row, which dampens attention when the learned sink wins.
func (c *KVCache) attendHeadWithSink(l, kvH int, query []float32, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap, alibiSlope, sink float32, hasSink bool, out []float32) {
	switch c.kvFormat() {
	case kvI8:
		onlineAttentionI8WithSink(query, c.K8[l][q8RowBytes(kvH*keyHeadDim):], c.V8[l][q8RowBytes(kvH*valueHeadDim):],
			c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, alibiSlope, sink, hasSink, out)
		return
	case kvF16:
		onlineAttentionF16WithSink(query, c.K16[l][kvH*keyHeadDim:], c.V16[l][kvH*valueHeadDim:],
			c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, alibiSlope, sink, hasSink, out)
		return
	}
	onlineAttentionWithSink(query, c.K[l][kvH*keyHeadDim:], c.V[l][kvH*valueHeadDim:],
		c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim, startT, endT, scale, softcap, alibiSlope, sink, hasSink, out)
}

// attendHeadGroup evaluates all query heads that share one GQA/MQA KV head.
// Keeping their score and value passes adjacent lets the CPU reuse each K/V
// cacheline instead of streaming the same cache once per query head.
func (c *KVCache) attendHeadGroup(l, kvH int, queries []float32, queryHeads, keyHeadDim, valueHeadDim, startT, endT int, scale, softcap float32, out []float32) {
	switch c.kvFormat() {
	case kvI8:
		onlineAttentionGroupI8(queries, c.K8[l][q8RowBytes(kvH*keyHeadDim):], c.V8[l][q8RowBytes(kvH*valueHeadDim):],
			queryHeads, c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim,
			startT, endT, scale, softcap, out)
		return
	case kvF16:
		onlineAttentionGroupF16(queries, c.K16[l][kvH*keyHeadDim:], c.V16[l][kvH*valueHeadDim:],
			queryHeads, c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim,
			startT, endT, scale, softcap, out)
		return
	}
	onlineAttentionGroup(queries, c.K[l][kvH*keyHeadDim:], c.V[l][kvH*valueHeadDim:],
		queryHeads, c.PerPosKDim, c.PerPosVDim, keyHeadDim, valueHeadDim,
		startT, endT, scale, softcap, out)
}
