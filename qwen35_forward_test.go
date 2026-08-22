package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func qwen35TestWeight(rows, cols int, values []float32) Weight {
	return Weight{F32: append([]float32(nil), values...), Rows: rows, Cols: cols}
}

func requireQwen35Close(t *testing.T, got, want float32) {
	t.Helper()
	if math.Abs(float64(got-want)) > 3e-5 {
		t.Fatalf("got %.8f, want %.8f", got, want)
	}
}

func TestQwen35AttentionUsesSigmoidGate(t *testing.T) {
	cfg := Config{
		Dim: 2, NHeads: 1, NKVHeads: 1, HeadDim: 2, ValueDim: 2, KVMul: 1,
		RMSNormEps: 1e-6, AttentionScale: 1,
	}
	identity := []float32{1, 0, 0, 1}
	w := Qwen35AttentionWeights{
		// Q has [query | gate] rows. A zero gate must retain half of the
		// attention value (sigmoid(0)), rather than erase it as SiLU(0) would.
		Q:     qwen35TestWeight(4, 2, []float32{1, 0, 0, 1, 0, 0, 0, 0}),
		K:     qwen35TestWeight(2, 2, identity),
		V:     qwen35TestWeight(2, 2, identity),
		O:     qwen35TestWeight(2, 2, identity),
		QNorm: []float32{1, 1}, KNorm: []float32{1, 1},
	}
	cache := NewKVCache(1, 2, 2, 1)
	buf := NewDecodeBuffer(cfg, 2, 1, 2)
	qwen35AttentionForward(cfg, w, cache, []float32{2, 1}, 0, 0, 0, 0, buf)
	requireQwen35Close(t, buf.Proj[0], 1)
	requireQwen35Close(t, buf.Proj[1], .5)
}

// TestQwen35LongContextGroupedGQAMatchesScalar locks the specialized Qwen
// dispatch to the former per-head result. Qwen3.8-27B uses this exact 24:4
// GQA ratio (six queries share each KV head); F16 is the normal compact cache
// on Apple Silicon and exercises the grouped F16 kernel used in production.
func TestQwen35LongContextGroupedGQAMatchesScalar(t *testing.T) {
	const (
		nHeads   = 24
		nKVHeads = 4
		headDim  = 2
		dim      = 4
		ctx      = groupedGQADecodeMinContext + 1
	)
	cfg := Config{
		Dim: dim, NHeads: nHeads, NKVHeads: nKVHeads, HeadDim: headDim, ValueDim: headDim,
		KVMul: nHeads / nKVHeads, RMSNormEps: 1e-5, RopeDimensionCount: headDim,
		RopeTheta: 10000,
	}
	rng := rand.New(rand.NewSource(38))
	w := Qwen35AttentionWeights{
		Q:     qwen35TestWeight(nHeads*2*headDim, dim, randomVec(rng, nHeads*2*headDim*dim)),
		K:     qwen35TestWeight(nKVHeads*headDim, dim, randomVec(rng, nKVHeads*headDim*dim)),
		V:     qwen35TestWeight(nKVHeads*headDim, dim, randomVec(rng, nKVHeads*headDim*dim)),
		O:     qwen35TestWeight(dim, nHeads*headDim, randomVec(rng, dim*nHeads*headDim)),
		QNorm: []float32{1, 1},
		KNorm: []float32{1, 1},
	}
	input := randomVec(rng, dim)
	newCache := func() *KVCache {
		return NewKVCacheF16(1, nKVHeads*headDim, nKVHeads*headDim, ctx)
	}
	scalarCache, groupedCache := newCache(), newCache()
	for pos := 0; pos < ctx-1; pos++ {
		k := randomVec(rng, nKVHeads*headDim)
		v := randomVec(rng, nKVHeads*headDim)
		scalarCache.storeKV(0, pos, k, v)
		groupedCache.storeKV(0, pos, k, v)
	}

	run := func(cache *KVCache) (attnOut, proj []float32) {
		buf := NewDecodeBuffer(cfg, headDim, nKVHeads, headDim)
		ropeHalf, ropePairs := prepareRopeScratch(ctx-1, headDim, cfg.RopeDimensionCount, buf.RopeInvFreq, buf.RopeMscale, &buf.RopeSin, &buf.RopeCos)
		qwen35AttentionForward(cfg, w, cache, input, 0, ctx-1, ropeHalf, ropePairs, buf)
		return append([]float32(nil), buf.AttnOut...), append([]float32(nil), buf.Proj...)
	}

	oldGrouped := useGroupedGQAAttention
	defer func() { useGroupedGQAAttention = oldGrouped }()
	useGroupedGQAAttention = false
	wantAttn, wantProj := run(scalarCache)
	useGroupedGQAAttention = true
	gotAttn, gotProj := run(groupedCache)
	for i := range gotAttn {
		requireQwen35Close(t, gotAttn[i], wantAttn[i])
	}
	for i := range gotProj {
		requireQwen35Close(t, gotProj[i], wantProj[i])
	}
}

// qwen35DeltaReferenceStep is a deliberately small, scalar reference for
// one-token Gated DeltaNet decode. It is independent from the production
// helpers so it locks the public Qwen graph semantics: [Q|K|V] layout, L2
// Q/K, tiled key-group mapping, delta update, and RMSNorm before SiLU(z).
func qwen35DeltaReferenceStep(cfg Config, qkv, gate, alphaW, betaW, a, dt, norm []float32, input float32, state [][][]float32) []float32 {
	dInner, dState, heads, groups := cfg.SSMInner, cfg.SSMState, cfg.SSMHeads, cfg.SSMGroups
	headDim := dInner / heads
	conv := make([]float32, len(qkv))
	for i, v := range qkv {
		y := v * input
		conv[i] = y / (1 + float32(math.Exp(float64(-y))))
	}
	keyDim := groups * dState
	q := append([]float32(nil), conv[:keyDim]...)
	k := append([]float32(nil), conv[keyDim:2*keyDim]...)
	v := conv[2*keyDim:]
	for group := 0; group < groups; group++ {
		qPart := q[group*dState : (group+1)*dState]
		kPart := k[group*dState : (group+1)*dState]
		qSum, kSum := float32(0), float32(0)
		for d := 0; d < dState; d++ {
			qSum += qPart[d] * qPart[d]
			kSum += kPart[d] * kPart[d]
		}
		qScale := float32(1/math.Sqrt(float64(dState))) / float32(math.Sqrt(float64(qSum+cfg.RMSNormEps)))
		kScale := float32(1 / math.Sqrt(float64(kSum+cfg.RMSNormEps)))
		for d := 0; d < dState; d++ {
			qPart[d] *= qScale
			kPart[d] *= kScale
		}
	}

	y := make([]float32, dInner)
	for h := 0; h < heads; h++ {
		alpha := alphaW[h]*input + dt[h]
		softplus := float32(math.Log1p(math.Exp(float64(alpha))))
		decay := float32(math.Exp(float64(softplus * a[h])))
		betaRaw := betaW[h] * input
		beta := float32(1 / (1 + math.Exp(float64(-betaRaw))))
		group := h % groups
		qPart := q[group*dState : (group+1)*dState]
		kPart := k[group*dState : (group+1)*dState]
		for d := 0; d < headDim; d++ {
			row := state[h][d]
			recall := float32(0)
			for j := 0; j < headDim; j++ {
				row[j] *= decay
				recall += row[j] * kPart[j]
			}
			delta := beta * (v[h*headDim+d] - recall)
			out := float32(0)
			for j := 0; j < headDim; j++ {
				row[j] += delta * kPart[j]
				out += row[j] * qPart[j]
			}
			y[h*headDim+d] = out
		}
	}
	for h := 0; h < heads; h++ {
		part := y[h*headDim : (h+1)*headDim]
		ss := float32(0)
		for _, value := range part {
			ss += value * value
		}
		scale := float32(1 / math.Sqrt(float64(ss/float32(headDim)+cfg.RMSNormEps)))
		for d := range part {
			part[d] *= norm[d] * scale
		}
	}
	for i, z := range gate {
		z *= input
		y[i] *= z / (1 + float32(math.Exp(float64(-z))))
	}
	return y
}

func TestQwen35DeltaNetMatchesScalarReferenceAcrossSteps(t *testing.T) {
	cfg := Config{Dim: 1, SSMConv: 1, SSMInner: 8, SSMState: 2, SSMHeads: 4, SSMGroups: 2, RMSNormEps: 1e-5}
	qkv := []float32{1, -.5, .25, 1.5, .8, .4, 1.1, -.6, .2, .8, -.3, 1.4, .9, -.7, .6, .1}
	gate := []float32{.2, -.4, .6, .8, -.3, .5, .7, -.2}
	alpha := []float32{.1, .2, -.1, 0}
	beta := []float32{.3, -.2, .4, -.5}
	a := []float32{-.4, -.7, -.6, -.9}
	dt := []float32{.05, -.1, .02, .12}
	norm := []float32{1.2, .8}
	identity := make([]float32, 64)
	for i := 0; i < 8; i++ {
		identity[i*8+i] = 1
	}
	w := Qwen35DeltaNetWeights{
		QKVConv:    qwen35TestWeight(16, 1, qkv),
		ConvKernel: qwen35TestWeight(16, 1, make([]float32, 16)),
		Gate:       qwen35TestWeight(8, 1, gate),
		AlphaProj:  qwen35TestWeight(4, 1, alpha),
		BetaProj:   qwen35TestWeight(4, 1, beta),
		A:          a, DTBias: dt, Norm: norm,
		Out: qwen35TestWeight(8, 8, identity),
	}
	for i := range w.ConvKernel.F32 {
		w.ConvKernel.F32[i] = 1
	}

	refState := make([][][]float32, cfg.SSMHeads)
	for h := range refState {
		refState[h] = make([][]float32, cfg.SSMState)
		for d := range refState[h] {
			refState[h][d] = make([]float32, cfg.SSMState)
		}
	}
	cache := newQwen35Cache(cfg, 1)
	buf := NewDecodeBuffer(cfg, 2, 1, 2)
	for _, input := range []float32{1, .6} {
		want := qwen35DeltaReferenceStep(cfg, qkv, gate, alpha, beta, a, dt, norm, input, refState)
		qwen35DeltaNetForward(cfg, w, cache, []float32{input}, 0, buf)
		for i, value := range want {
			requireQwen35Close(t, buf.Proj[i], value)
		}
	}
}

// The production Qwen3.8 geometry updates large independent DeltaNet state
// matrices per head concurrently. Keep that path locked to the scalar graph,
// not only the small serial fixture above.
func TestQwen35DeltaNetParallelHeadsMatchesScalarReference(t *testing.T) {
	const (
		heads   = 8
		headDim = 64
	)
	cfg := Config{
		Dim: 1, SSMConv: 1, SSMInner: heads * headDim, SSMState: headDim,
		SSMHeads: heads, SSMGroups: 1, RMSNormEps: 1e-5,
	}
	rng := rand.New(rand.NewSource(81))
	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	qkv := randomVec(rng, channels)
	gate := randomVec(rng, cfg.SSMInner)
	alpha := randomVec(rng, heads)
	beta := randomVec(rng, heads)
	a := make([]float32, heads)
	dt := randomVec(rng, heads)
	for h := range heads {
		a[h] = -.2 - float32(h)*.03
		dt[h] *= .1
	}
	norm := make([]float32, headDim)
	for i := range norm {
		norm[i] = .8 + float32(i%7)*.04
	}
	identity := make([]float32, cfg.SSMInner*cfg.SSMInner)
	for i := range cfg.SSMInner {
		identity[i*cfg.SSMInner+i] = 1
	}
	w := Qwen35DeltaNetWeights{
		QKVConv:    qwen35TestWeight(channels, 1, qkv),
		ConvKernel: qwen35TestWeight(channels, 1, make([]float32, channels)),
		Gate:       qwen35TestWeight(cfg.SSMInner, 1, gate),
		AlphaProj:  qwen35TestWeight(heads, 1, alpha),
		BetaProj:   qwen35TestWeight(heads, 1, beta),
		A:          a,
		DTBias:     dt,
		Norm:       norm,
		Out:        qwen35TestWeight(cfg.SSMInner, cfg.SSMInner, identity),
	}
	for i := range w.ConvKernel.F32 {
		w.ConvKernel.F32[i] = 1
	}

	refState := make([][][]float32, heads)
	for h := range refState {
		refState[h] = make([][]float32, headDim)
		for d := range refState[h] {
			refState[h][d] = make([]float32, headDim)
		}
	}
	cache := newQwen35Cache(cfg, 1)
	buf := NewDecodeBuffer(cfg, headDim, 1, headDim)
	for _, input := range []float32{.35, -.2} {
		want := qwen35DeltaReferenceStep(cfg, qkv, gate, alpha, beta, a, dt, norm, input, refState)
		qwen35DeltaNetForward(cfg, w, cache, []float32{input}, 0, buf)
		for i, value := range want {
			requireQwen35Close(t, buf.Proj[i], value)
		}
	}
}
