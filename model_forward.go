package gopherllm

import (
	"math"
	"os"
	"sync"
)

// Forward runs one token through the transformer and returns its
// next-token logits; ForwardInto is the allocation-free form. Both append the
// token's K/V to the cache at position pos as a side effect.
func Forward(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	logits := make([]float32, 0)
	ForwardInto(config, weights, cache, buf, token, pos, &logits)
	return logits
}

func ForwardInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardBodyInto(config, weights, cache, buf, token, pos)
	ProjectLogitsInto(config, weights, buf, logits)
}

func ProjectLogitsInto(config Config, weights ModelWeights, buf *DecodeBuffer, logits *[]float32) {
	weights.Output.MatvecInto(buf.XN, logits)
	addInPlace(*logits, weights.OutputBias)
	if config.LogitScale != 1 {
		ScaleF32(*logits, 1/config.LogitScale)
	}
	if config.FinalLogitSoftcap > 0 {
		softcapF32(*logits, config.FinalLogitSoftcap)
	}
}

func ArgmaxOutputToken(config Config, weights ModelWeights, buf *DecodeBuffer) (uint32, bool) {
	return argmaxOutputTokenInto(config, weights, buf, nil)
}

func argmaxOutputTokenInto(config Config, weights ModelWeights, buf *DecodeBuffer, logits *[]float32) (uint32, bool) {
	return argmaxOutputTokenPenalizedInto(config, weights, buf, nil, 1, logits)
}

func argmaxOutputTokenPenalizedInto(config Config, weights ModelWeights, buf *DecodeBuffer, recent []uint32, repeatPenalty float32, logits *[]float32) (uint32, bool) {
	if !finite32(config.LogitScale) || config.LogitScale <= 0 || config.FinalLogitSoftcap < 0 ||
		!finite32(repeatPenalty) || repeatPenalty <= 0 {
		return 0, false
	}
	// Positive linear logit scaling commutes with repeat penalty, but adding a
	// bias or applying the nonlinear softcap does not. Materialize those rare
	// combinations so penalty ordering stays exactly the sampler's ordering.
	if len(weights.OutputBias) > 0 || (repeatPenalty != 1 && config.FinalLogitSoftcap > 0) {
		ProjectLogitsInto(config, weights, buf, &buf.Logits)
		applyRepeatPenalty(buf.Logits, recent, repeatPenalty)
		if logits != nil {
			ensureLenNoClear(logits, len(buf.Logits))
			copy(*logits, buf.Logits)
		}
		return argmaxFiniteToken(buf.Logits), true
	}
	// Greedy decode only needs the winning token. Positive logit scaling and the
	// optional positive softcap preserve argmax ordering, so Metal can reduce its
	// Q6_K output buffer on-device and avoid a 131k-logit readback plus CPU scan.
	// Sampling and unsupported output types retain the materialized fallback.
	if token, ok := argmaxMetalQ6KPenalized(weights.Output.Metal, buf.XN, recent, repeatPenalty); ok {
		return token, true
	}
	if weights.Output.Metal != nil && logits != nil {
		ProjectLogitsInto(config, weights, buf, logits)
		if len(*logits) == 0 {
			return 0, false
		}
		applyRepeatPenalty(*logits, recent, repeatPenalty)
		return argmaxFiniteToken(*logits), true
	}
	if repeatPenalty != 1 {
		return 0, false
	}
	return weights.Output.ArgmaxMatvec(buf.XN)
}

// ForwardBodyInto is the transformer body shared by logits, prefill, and
// embedding paths: embed the token, then per layer run pre-norm attention
// (RoPE'd Q/K, K/V appended to the cache, online-softmax attention over all
// cached positions, output projection, residual add) followed by a pre-norm
// SwiGLU FFN, and finally apply the output norm, leaving the normed hidden
// state in buf.XN for the caller to project (or pool, for embeddings).
// Attention heads are spread across the worker pool once the attended span is
// long enough to amortize dispatch (see the comment at the call site); the
// Q/K/V and gate/up matvecs go through the fused multi-matrix kernels when
// the quant types allow (tryMatvec3Into/tryMatvec2Into).
func ForwardBodyInto(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	if config.UsesMLA {
		ForwardDeepSeek2BodyInto(config, weights, cache, buf, token, pos)
		return
	}
	dim := config.Dim
	headDim := config.HeadDim
	kvMul := max(1, config.KVMul)
	ropeInvFreq, ropeMscale := buf.RopeInvFreq, buf.RopeMscale
	if config.Arch == "gpt-oss" && len(buf.RopeGptOssInvFreq) > 0 {
		// GPT-OSS uses its own YaRN concentration rule, precomputed alongside
		// the regular table in NewDecodeBuffer.
		ropeInvFreq, ropeMscale = buf.RopeGptOssInvFreq, buf.RopeGptOssConcentration
	}
	ropeHalf, ropePairs := prepareRopeScratch(pos, headDim, config.RopeDimensionCount, ropeInvFreq, ropeMscale, &buf.RopeSin, &buf.RopeCos)
	swaRopeHalf, swaRopePairs := 0, 0
	if len(buf.RopeSWAInvFreq) > 0 {
		swaRopeHalf, swaRopePairs = prepareRopeScratch(pos, headDim, config.RopeDimensionCount, buf.RopeSWAInvFreq, buf.RopeSWAMscale, &buf.RopeSWASin, &buf.RopeSWACos)
	}
	ropeIsInterleaved := ropeInterleaved(config.Arch)
	weights.TokenEmbd.RowInto(int(token), dim, &buf.X)
	if emb, ok := buf.ImageEmbeds[pos]; ok {
		copy(buf.X[:dim], emb)
	} else if config.EmbeddingScale != 1 {
		ScaleF32(buf.X[:dim], config.EmbeddingScale)
	}
	if config.usesAbsolutePositionEmbd() {
		weights.PositionEmbd.RowInto(pos, dim, &buf.PosEmbd)
		addInPlace(buf.X[:dim], buf.PosEmbd[:dim])
	}
	for l := range config.NLayers {
		layer := weights.Layers[l]
		if config.usesPostNormOnly() {
			ensureLenNoClear(&buf.XN, dim)
			copy(buf.XN, buf.X[:dim])
		} else {
			normalizeDecoderInto(config, buf.X, layer.AttnNorm, layer.AttnNormBias, &buf.XN)
		}
		if layer.HasQKV {
			layer.WQKV.MatvecInto(buf.XN, &buf.QKV)
			qLen := config.NHeads * headDim
			kLen := config.NKVHeads * headDim
			vLen := config.NKVHeads * config.ValueDim
			ensureLenNoClear(&buf.Q, qLen)
			ensureLenNoClear(&buf.K, kLen)
			ensureLenNoClear(&buf.V, vLen)
			copy(buf.Q, buf.QKV[:qLen])
			copy(buf.K, buf.QKV[qLen:qLen+kLen])
			copy(buf.V, buf.QKV[qLen+kLen:qLen+kLen+vLen])
		} else {
			tryMatvecAttentionInto(layer.WQ, layer.WK, layer.WV, buf.XN, &buf.Q4KXSums, &buf.Q, &buf.K, &buf.V)
			addInPlace(buf.Q, layer.BQ)
			addInPlace(buf.K, layer.BK)
			addInPlace(buf.V, layer.BV)
		}
		// Normalize projected Q/K before RoPE. Qwen3/EXAONE 4 use one RMSNorm
		// per head; OLMo 2/3 normalize each complete projection.
		normalizeProjectedQKInPlace(config, layer, buf.Q, buf.K)
		if config.layerUsesRoPE(l) {
			activeHalf, activePairs := ropeHalf, ropePairs
			activeSin, activeCos := buf.RopeSin, buf.RopeCos
			if config.layerUsesSWA(l) && swaRopePairs > 0 {
				activeHalf, activePairs = swaRopeHalf, swaRopePairs
				activeSin, activeCos = buf.RopeSWASin, buf.RopeSWACos
			}
			applyPreparedRope(buf.Q, headDim, config.NHeads, activeHalf, activePairs, activeSin, activeCos, ropeIsInterleaved)
			applyPreparedRope(buf.K, headDim, config.NKVHeads, activeHalf, activePairs, activeSin, activeCos, ropeIsInterleaved)
		}
		if temperature := attentionTemperatureAt(config, pos); temperature != 1 {
			ScaleF32(buf.Q, temperature)
		}

		cache.storeKV(l, pos, buf.K, buf.V)

		clear(buf.AttnOut)
		scale := config.AttentionScale
		if scale == 0 {
			scale = float32(1 / math.Sqrt(float64(headDim)))
		}
		if config.Arch == "phi2" {
			// Phi-2 scales Q before the dot product and uses a unit attention
			// scale to keep intermediate scores in the trained precision range.
			ScaleF32(buf.Q, scale)
			scale = 1
		}
		attnStart := 0
		if config.layerUsesSWA(l) {
			attnStart = max(0, pos-config.SlidingWindow)
		}
		// GQA/MQA query heads share K/V rows. Process each complete group together
		// so the shared cacheline is fetched once, then parallelize across KV
		// groups at contexts long enough to amortize dispatch.
		groupedGQA := useGroupedGQAAttention && kvMul > 1 && config.NKVHeads > 0 && len(layer.AttnSinks) == 0 && !config.usesALiBi()
		if attnLen := pos - attnStart + 1; groupedGQA && shouldParallelGroupedGQAAttention(cache, kvMul, config.NKVHeads, attnLen) {
			parallelAttendHeadGroups(config, cache, buf, l, pos, attnStart, scale, kvMul)
		} else if groupedGQA && attnLen < 128 {
			attendHeadGroupsRange(&config, cache, buf, l, pos, attnStart, scale, kvMul, 0, config.NKVHeads)
		} else if attnLen >= 128 && config.NHeads > 1 {
			parallelAttendHeads(config, layer, cache, buf, l, pos, attnStart, scale, kvMul)
		} else {
			attendHeadsRange(&config, &layer, cache, buf, l, pos, attnStart, scale, kvMul, 0, config.NHeads)
		}
		layer.WO.MatvecInto(buf.AttnOut, &buf.Proj)
		addInPlace(buf.Proj, layer.BO)
		if layer.PostAttnNorm != nil {
			rmsNormInto(buf.Proj, layer.PostAttnNorm, config.RMSNormEps, &buf.Proj)
		}
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		if config.ParallelResidual {
			copy(buf.AttnProj[:dim], buf.Proj[:dim])
		} else {
			addInPlace(buf.X[:dim], buf.Proj)
		}

		if config.sharesParallelBranchNorm() {
			ensureLenNoClear(&buf.XN2, dim)
			copy(buf.XN2, buf.XN[:dim])
		} else if config.usesPostNormOnly() {
			ensureLenNoClear(&buf.XN2, dim)
			copy(buf.XN2, buf.X[:dim])
		} else {
			normalizeDecoderInto(config, buf.X, layer.FFNNorm, layer.FFNNormBias, &buf.XN2)
		}
		if layer.MoE != nil {
			sparseMoEForward(layer.MoE, buf.XN2, buf)
		} else {
			// Decode bottleneck: the selective Metal path previously synchronized and
			// copied Gate/Up to the CPU for SiLU, then copied Hidden back for Down.
			// Keep all three stages in one command buffer when the measured
			// Q4_K/Q4_K/Q6_K shape matches. Any unsupported shape or GPU failure falls
			// through to the unchanged CPU/GPU path; removing this branch is rollback.
			fusedMetalFFN := !config.usesPlainMLP() && !layer.HasGateUp && !config.UseGELU &&
				matvecMetalSwiGLUInto(layer.W1.Metal, layer.W3.Metal, layer.W2.Metal, buf.XN2, &buf.Proj)
			if !fusedMetalFFN {
				if config.usesPlainMLP() {
					layer.W3.MatvecInto(buf.XN2, &buf.Up)
					addInPlace(buf.Up, layer.FFNUpBias)
					ensureLenNoClear(&buf.Hidden, config.HiddenDim)
					if config.UseExactGELU {
						for i := range config.HiddenDim {
							buf.Hidden[i] = geluExact(buf.Up[i])
						}
					} else {
						for i := range config.HiddenDim {
							buf.Hidden[i] = geluTanhScalar(buf.Up[i])
						}
					}
					layer.W2.MatvecInto(buf.Hidden, &buf.Proj)
					addInPlace(buf.Proj, layer.FFNDownBias)
				} else {
					if layer.HasGateUp {
						layer.WGateUp.MatvecInto(buf.XN2, &buf.GateUp)
						ensureLenNoClear(&buf.Gate, config.HiddenDim)
						ensureLenNoClear(&buf.Up, config.HiddenDim)
						copy(buf.Gate, buf.GateUp[:config.HiddenDim])
						copy(buf.Up, buf.GateUp[config.HiddenDim:2*config.HiddenDim])
					} else {
						if !tryMatvec2Into(layer.W1, layer.W3, buf.XN2, &buf.Q4KXSums, &buf.Gate, &buf.Up) {
							layer.W1.MatvecInto(buf.XN2, &buf.Gate)
							layer.W3.MatvecInto(buf.XN2, &buf.Up)
						}
					}
					hDim := config.HiddenDim
					ensureLenNoClear(&buf.Hidden, hDim)
					if hDim > 0 {
						gate := buf.Gate
						up := buf.Up
						hidden := buf.Hidden
						_ = gate[hDim-1]
						_ = up[hDim-1]
						_ = hidden[hDim-1]
						if config.UseGELU {
							geluMulF32(gate[:hDim], up[:hDim], hidden[:hDim])
						} else {
							siluMulF32(gate[:hDim], up[:hDim], hidden[:hDim])
						}
					}
					layer.W2.MatvecInto(buf.Hidden, &buf.Proj)
					addInPlace(buf.Proj, layer.FFNDownBias)
				}
			}
		}
		if layer.PostFFNNorm != nil {
			rmsNormInto(buf.Proj, layer.PostFFNNorm, config.RMSNormEps, &buf.Proj)
		}
		if config.ResidualScale != 1 {
			ScaleF32(buf.Proj, config.ResidualScale)
		}
		addInPlace(buf.X[:dim], buf.Proj)
		if config.ParallelResidual {
			addInPlace(buf.X[:dim], buf.AttnProj[:dim])
		}
	}
	normalizeDecoderInto(config, buf.X, weights.OutputNorm, weights.OutputNormBias, &buf.XN)
}

// attendHeadsRange lives outside ForwardBodyInto so the common short-context
// serial path does not construct an escaping closure for every layer. The
// parallel wrapper owns the copies captured by its worker closure.
func attendHeadsRange(config *Config, layer *LayerWeights, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul, hStart, hEnd int) {
	headDim := config.HeadDim
	valueDim := config.ValueDim
	alibi := config.usesALiBi()
	for h := hStart; h < hEnd; h++ {
		kvH := h / kvMul
		qOff := h * headDim
		outOff := h * valueDim
		sink, hasSink := float32(0), h < len(layer.AttnSinks)
		if hasSink {
			sink = layer.AttnSinks[h]
		}
		var alibiSlope float32
		if alibi {
			alibiSlope = aLiBiSlope(h, config.NHeads, config.ALiBiMaxBias)
		}
		cache.attendHeadWithSink(l, kvH, buf.Q[qOff:qOff+headDim], headDim, valueDim,
			attnStart, pos, scale, config.AttnLogitSoftcap, alibiSlope, sink, hasSink,
			buf.AttnOut[outOff:outOff+valueDim])
	}
}

type attendHeadsTask struct {
	config            Config
	layer             LayerWeights
	cache             *KVCache
	buf               *DecodeBuffer
	l, pos, attnStart int
	scale             float32
	kvMul             int
}

func (t *attendHeadsTask) runRows(start, end int) {
	attendHeadsRange(&t.config, &t.layer, t.cache, t.buf, t.l, t.pos, t.attnStart, t.scale, t.kvMul, start, end)
}

var attendHeadsTaskPool = sync.Pool{New: func() any { return new(attendHeadsTask) }}

func parallelAttendHeads(config Config, layer LayerWeights, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul int) {
	task := attendHeadsTaskPool.Get().(*attendHeadsTask)
	task.config, task.layer, task.cache, task.buf = config, layer, cache, buf
	task.l, task.pos, task.attnStart, task.scale, task.kvMul = l, pos, attnStart, scale, kvMul
	parallelChunksTask(config.NHeads, task)
	*task = attendHeadsTask{}
	attendHeadsTaskPool.Put(task)
}

func attendHeadGroupsRange(config *Config, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul, kvStart, kvEnd int) {
	headDim := config.HeadDim
	valueDim := config.ValueDim
	for kvH := kvStart; kvH < kvEnd; kvH++ {
		hStart := kvH * kvMul
		hEnd := min(hStart+kvMul, config.NHeads)
		if hStart >= hEnd {
			break
		}
		cache.attendHeadGroup(l, kvH,
			buf.Q[hStart*headDim:hEnd*headDim], hEnd-hStart, headDim, valueDim,
			attnStart, pos, scale, config.AttnLogitSoftcap,
			buf.AttnOut[hStart*valueDim:hEnd*valueDim])
	}
}

type attendHeadGroupsTask struct {
	config            Config
	cache             *KVCache
	buf               *DecodeBuffer
	l, pos, attnStart int
	scale             float32
	kvMul, nKVHeads   int
}

func (t *attendHeadGroupsTask) runRows(start, end int) {
	if start >= t.nKVHeads {
		return
	}
	attendHeadGroupsRange(&t.config, t.cache, t.buf, t.l, t.pos, t.attnStart, t.scale, t.kvMul, start, min(end, t.nKVHeads))
}

var attendHeadGroupsTaskPool = sync.Pool{New: func() any { return new(attendHeadGroupsTask) }}

func parallelAttendHeadGroups(config Config, cache *KVCache, buf *DecodeBuffer, l, pos, attnStart int, scale float32, kvMul int) {
	// Keep the configured worker set awake for the projection matvec that
	// immediately follows attention. GQA often exposes only eight groups on a
	// 12-core Apple SoC; dispatching exactly eight jobs made four workers sleep
	// and added a repeated wake-up penalty at every layer boundary.
	workItems := max(config.NKVHeads, min(numThreads(), config.NHeads))
	task := attendHeadGroupsTaskPool.Get().(*attendHeadGroupsTask)
	task.config, task.cache, task.buf = config, cache, buf
	task.l, task.pos, task.attnStart, task.scale = l, pos, attnStart, scale
	task.kvMul, task.nKVHeads = kvMul, config.NKVHeads
	parallelChunksTask(workItems, task)
	*task = attendHeadGroupsTask{}
	attendHeadGroupsTaskPool.Put(task)
}

func ForwardHidden(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	ForwardBodyInto(config, weights, cache, buf, token, pos)
	out := make([]float32, len(buf.XN))
	copy(out, buf.XN)
	return out
}

func ForwardPrefill(config Config, weights ModelWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) {
	ForwardBodyInto(config, weights, cache, buf, token, pos)
}

func ForwardGptOssInto(config Config, weights GptOssWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardInto(config, weights.Standard, cache, buf, token, pos, logits)
}

func ForwardHiddenGptOss(config Config, weights GptOssWeights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	return ForwardHidden(config, weights.Standard, cache, buf, token, pos)
}

func ForwardGemma4Into(config Config, weights Gemma4Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	if weights.Native {
		forwardNativeGemma4BodyInto(config, weights, cache, buf, token, pos)
		projectNativeGemma4Logits(config, weights, buf, logits)
		return
	}
	ForwardInto(config, weights.Standard, cache, buf, token, pos, logits)
}

func ForwardHiddenGemma4(config Config, weights Gemma4Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int) []float32 {
	if weights.Native {
		forwardNativeGemma4BodyInto(config, weights, cache, buf, token, pos)
		out := make([]float32, len(buf.XN))
		copy(out, buf.XN)
		return out
	}
	ForwardBodyInto(config, weights.Standard, cache, buf, token, pos)
	out := make([]float32, len(buf.XN))
	copy(out, buf.XN)
	return out
}

// Kept as a package variable so correctness and end-to-end performance tests
// can A/B the legacy path in one process without changing model state.
var useGroupedGQAAttention = os.Getenv("GOPHERLLM_NO_GROUPED_GQA") == ""

// Below this point, 32-way head scheduling beats the lower data movement of
// eight grouped jobs on the M2 Max. At long context the KV bandwidth saved by
// the NEON x4 kernels dominates. Prefill has independent token-level
// parallelism and therefore uses grouping without this decode-only threshold.
const groupedGQADecodeMinContext = 4096

// shouldParallelGroupedGQAAttention selects the long-context grouped GQA
// schedule. The f32 crossover stays conservative: on an M2 Max, its eight
// Ministral KV groups leave enough cores idle that independent heads win below
// 4k context. With f16 KV rows, though, the arm64 four-way kernels load and
// convert each shared K/V row once. The reduced group cost moves that
// crossover down to the existing 128-token parallel-attention threshold. At
// shorter contexts, ForwardBodyInto keeps the serial grouped path to avoid
// worker dispatch overhead entirely.
func shouldParallelGroupedGQAAttention(cache *KVCache, kvMul, nKVHeads, attnLen int) bool {
	if nKVHeads <= 1 || attnLen < 128 {
		return false
	}
	if attnLen >= groupedGQADecodeMinContext {
		return true
	}
	return hasFastF16GQA4 && kvMul == 4 && cache != nil && cache.kvFormat() == kvF16
}

func addInPlace(dst, src []float32) {
	AxpyF32(dst, 1.0, src)
}
