package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	workerthreads "github.com/SimonWaldherr/GopherLLM/internal/threads"
)

// Cover the production projection formats (Q4_K Q/K/gate/up and Q6_K V/O/
// down), partial RoPE, nontrivial convolution and parallel DeltaNet heads.
func qwen35BatchFixture(quantized bool) (Config, Qwen35Weights) {
	dim, hidden, head, heads, kvHeads, inner, ssmHeads, groups := 8, 16, 4, 2, 1, 8, 2, 2
	if quantized {
		dim, hidden, head, heads, kvHeads, inner, ssmHeads, groups = 256, 512, 64, 4, 2, 512, 8, 2
	}
	cfg := Config{Arch: "qwen35", Dim: dim, HiddenDim: hidden, NLayers: 4, VocabSize: 32, MaxSeqLen: 128,
		NHeads: heads, NKVHeads: kvHeads, HeadDim: head, ValueDim: head, KVDim: kvHeads * head, KVMul: heads / kvHeads,
		SSMInner: inner, SSMHeads: ssmHeads, SSMGroups: groups, SSMState: inner / ssmHeads, SSMConv: 3,
		RMSNormEps: 1e-5, RopeTheta: 10000, RopeDimensionCount: head / 2, FullAttentionInterval: 2}
	rng := rand.New(rand.NewSource(3505))
	values := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64()) * .035
		}
		return v
	}
	norm := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = .8 + rng.Float32()*.3
		}
		return v
	}
	weight := func(rows, cols int, typ GGMLType) Weight {
		v := values(rows * cols)
		w := Weight{F32: v, Rows: rows, Cols: cols}
		if quantized && typ != GGMLTypeF32 {
			w.F32, w.Type = nil, typ
			for row := range rows {
				if typ == GGMLTypeQ4_K {
					w.Raw = append(w.Raw, QuantizeRowQ4K(v[row*cols:(row+1)*cols], cols)...)
				} else {
					w.Raw = append(w.Raw, QuantizeRowQ6K(v[row*cols:(row+1)*cols], cols)...)
				}
			}
		}
		return w
	}
	w := Qwen35Weights{TokenEmbd: weight(cfg.VocabSize, dim, GGMLTypeF32), OutputNorm: norm(dim), Output: weight(cfg.VocabSize, dim, GGMLTypeQ6_K)}
	channels := inner + 2*groups*cfg.SSMState
	for layer := range cfg.NLayers {
		l := Qwen35LayerWeights{Norm: norm(dim), PostAttnNorm: norm(dim)}
		l.FFN = Qwen35FFNWeights{Gate: weight(hidden, dim, GGMLTypeQ4_K), Up: weight(hidden, dim, GGMLTypeQ4_K), Down: weight(dim, hidden, GGMLTypeQ6_K)}
		if layer%2 == 0 {
			l.Kind, l.RecurrentCacheSlot = qwen35DeltaNet, layer/2
			l.DeltaNet = Qwen35DeltaNetWeights{QKVConv: weight(channels, dim, GGMLTypeQ4_K), Gate: weight(inner, dim, GGMLTypeQ4_K),
				ConvKernel: weight(channels, cfg.SSMConv, GGMLTypeF32), AlphaProj: weight(ssmHeads, dim, GGMLTypeF32), BetaProj: weight(ssmHeads, dim, GGMLTypeF32),
				A: make([]float32, ssmHeads), DTBias: values(ssmHeads), Norm: norm(cfg.SSMState), Out: weight(dim, inner, GGMLTypeQ6_K)}
			for h := range l.DeltaNet.A {
				l.DeltaNet.A[h] = -.2 - rng.Float32()
			}
		} else {
			l.Kind, l.KVCacheSlot = qwen35Attention, layer/2
			l.Attention = Qwen35AttentionWeights{Q: weight(2*heads*head, dim, GGMLTypeQ4_K), K: weight(kvHeads*head, dim, GGMLTypeQ4_K),
				V: weight(kvHeads*head, dim, GGMLTypeQ6_K), O: weight(dim, heads*head, GGMLTypeQ6_K), QNorm: norm(head), KNorm: norm(head)}
		}
		w.Layers = append(w.Layers, l)
	}
	return cfg, w
}

func qwen35BatchCache(cfg Config, w Qwen35Weights, format kvFormat) *KVCache {
	layers, width := qwen35AttentionLayerCount(w), cfg.NKVHeads*cfg.HeadDim
	var cache *KVCache
	switch format {
	case kvF16:
		cache = NewKVCacheF16(layers, width, width, cfg.MaxSeqLen)
	case kvI8:
		cache = NewKVCacheI8(layers, width, width, cfg.MaxSeqLen)
	default:
		cache = NewKVCache(layers, width, width, cfg.MaxSeqLen)
	}
	cache.Qwen35 = newQwen35Cache(cfg, qwen35RecurrentLayerCount(w))
	return cache
}

func requireQwen35BatchValues(t *testing.T, name string, got, want []float32) {
	t.Helper()
	requireQwen35BatchValuesWithin(t, name, got, want, 5e-5)
}

func requireQwen35BatchValuesWithin(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d, want %d", name, len(got), len(want))
	}
	for i, value := range got {
		// Float batch dots/BLAS may reduce products differently. The shared
		// recurrence itself keeps its exact token order and operation tree.
		if !finite32(value) || !finite32(want[i]) || math.Abs(float64(value-want[i])) > tolerance*max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d]=%.9g, want %.9g", name, i, value, want[i])
		}
	}
}

func requireQwen35BatchF16(t *testing.T, name string, got, want []uint16) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d, want %d", name, len(got), len(want))
	}
	for i, bits := range got {
		value, ref := F16ToF32(bits), F16ToF32(want[i])
		// Without Q8 activations, decode folds quantized block scales/mins
		// into scalar sub-dots; batch dequantizes first and uses float dots.
		// Their small float errors can cross an f16 rounding boundary. Allow
		// at most one half ULP, retaining the existing 5e-5 absolute float
		// bound near zero where half ULPs are much smaller than that bound.
		ulp := int(max(bits, want[i])) - int(min(bits, want[i]))
		if !finite32(value) || !finite32(ref) || (math.Abs(float64(value-ref)) > 5e-5 && (bits>>15 != want[i]>>15 || ulp > 1)) {
			t.Fatalf("%s[%d]=%.9g (%04x), want %.9g (%04x), half ULPs=%d", name, i, value, bits, ref, want[i], ulp)
		}
	}
}

func requireQwen35BatchI8Continuation(t *testing.T, name string, got, want []byte) {
	t.Helper()
	if len(got) != len(want) || len(got)%34 != 0 {
		t.Fatalf("%s invalid Q8_0 lengths %d/%d", name, len(got), len(want))
	}
	for block := 0; block < len(got); block += 34 {
		gotBits, wantBits := binaryLE16(got[block:]), binaryLE16(want[block:])
		gotScale, wantScale := F16ToF32(gotBits), F16ToF32(wantBits)
		scaleULP := int(max(gotBits, wantBits)) - int(min(gotBits, wantBits))
		if !finite32(gotScale) || !finite32(wantScale) || gotBits>>15 != wantBits>>15 || scaleULP > 1 {
			t.Fatalf("%s Q8_0 block %d scales %04x/%04x differ by %d half ULPs", name, block/34, gotBits, wantBits, scaleULP)
		}
		for i := 2; i < 34; i++ {
			gotQ, wantQ := int(int8(got[block+i])), int(int8(want[block+i]))
			if delta := max(gotQ, wantQ) - min(gotQ, wantQ); delta > 1 {
				t.Fatalf("%s Q8_0 block %d quant %d=%d, want %d", name, block/34, i-2, gotQ, wantQ)
			}
			gotValue, wantValue := gotScale*float32(gotQ), wantScale*float32(wantQ)
			// One quantization step plus one f16 scale step bounds the
			// dequantized error by |dg*(qg-qw)| + |(dg-dw)*qw|. A half
			// scale times an int8 is exactly representable as float32.
			bound := math.Abs(float64(gotScale))*math.Abs(float64(gotQ-wantQ)) + math.Abs(float64(gotScale)-float64(wantScale))*math.Abs(float64(wantQ))
			if !finite32(gotValue) || !finite32(wantValue) || math.Abs(float64(gotValue)-float64(wantValue)) > bound {
				t.Fatalf("%s Q8_0 block %d value %d=%.9g, want %.9g, codec bound %.9g", name, block/34, i-2, gotValue, wantValue, bound)
			}
		}
	}
}

func requireQwen35BatchCache(t *testing.T, got, want *KVCache, tolerance float64, continuation bool) {
	t.Helper()
	requireQwen35BatchValuesWithin(t, "conv", got.Qwen35.Conv, want.Qwen35.Conv, tolerance)
	requireQwen35BatchValuesWithin(t, "recurrent state", got.Qwen35.State, want.Qwen35.State, tolerance)
	for layer := range got.layerCount() {
		switch got.kvFormat() {
		case kvI8:
			if continuation && tolerance > 5e-5 {
				// Prefill remains bit-exact, including Q8_0 scales. During
				// subsequent decode a float perturbation can cross either a
				// quant or scale boundary; bound those two codec steps separately.
				requireQwen35BatchI8Continuation(t, "K", got.K8[layer], want.K8[layer])
				requireQwen35BatchI8Continuation(t, "V", got.V8[layer], want.V8[layer])
				continue
			}
			if !slices.Equal(got.K8[layer], want.K8[layer]) || !slices.Equal(got.V8[layer], want.V8[layer]) {
				t.Fatalf("I8 K/V differ at layer %d", layer)
			}
		case kvF16:
			if tolerance > 5e-5 {
				requireQwen35BatchF16(t, "K", got.K16[layer], want.K16[layer])
				requireQwen35BatchF16(t, "V", got.V16[layer], want.V16[layer])
				continue
			}
			if !slices.Equal(got.K16[layer], want.K16[layer]) || !slices.Equal(got.V16[layer], want.V16[layer]) {
				t.Fatalf("F16 K/V differ at layer %d", layer)
			}
		default:
			requireQwen35BatchValues(t, "K", got.K[layer], want.K[layer])
			requireQwen35BatchValues(t, "V", got.V[layer], want.V[layer])
		}
	}
}

func TestQwen35ChunkedQuantizedProjectionRounding(t *testing.T) {
	if useQ8Activations.Load() {
		t.Skip("Q8 activation kernels share the same row arithmetic in decode and batch")
	}
	cfg, weights := qwen35BatchFixture(true)
	input := make([]float32, cfg.Dim)
	for i := range input {
		input[i] = float32(math.Sin(float64(i))) * .125
	}
	for _, weight := range []Weight{weights.Layers[1].Attention.Q, weights.Layers[1].Attention.V} {
		var serial []float32
		weight.MatvecInto(input, &serial)
		batch := make([]float32, weight.Rows)
		matvecBatch(weight, [][]float32{input}, [][]float32{batch})
		// With the SSE2 fallback these differ by <=4.5e-8 for this input:
		// scalar quantized-block algebra versus dequantize-first float dots.
		// Keep this isolated bound much tighter than the whole-graph bound.
		requireQwen35BatchValuesWithin(t, fmt.Sprintf("projection type %d", weight.Type), batch, serial, 1e-6)
	}
}

func TestQwen35ChunkedPrefillMatchesTokenGraphAndPrefixAppend(t *testing.T) {
	for _, quantized := range []bool{false, true} {
		cfg, w := qwen35BatchFixture(quantized)
		for _, format := range []kvFormat{kvF32, kvF16, kvI8} {
			if format == kvI8 && !quantized { // The compact fixture has sub-block head widths.
				continue
			}
			t.Run(fmt.Sprintf("quant=%t/kv=%d", quantized, format), func(t *testing.T) {
				tolerance, hiddenTolerance := 5e-5, 5e-5
				if quantized && format != kvF32 && !useQ8Activations.Load() {
					// Quantized-block float reductions differ before compressed
					// KV rounding on non-Q8 backends. The f16 boundary crossing
					// reaches 6.1e-5 in logits in the SSE2 fixture; allow 1e-4
					// only for its downstream outputs/state. Float KV and
					// unquantized weights retain the stricter 5e-5 bound.
					tolerance = 1e-4
					// Normalizing the perturbed final residual amplifies the
					// measured hidden error to 1.035e-4, while logits, Conv
					// and recurrent state remain below the 1e-4 budget.
					hiddenTolerance = 1.25e-4
				}
				r := &Runner{kind: loadedQwen35, config: cfg, qwen35: w}
				batchBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
				for _, n := range []int{1, 3, 8, 32, 35, 67} {
					gotCache, wantCache := qwen35BatchCache(cfg, w, format), qwen35BatchCache(cfg, w, format)
					serialBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
					tokens := make([]uint32, n)
					for i := range tokens {
						tokens[i] = uint32((i*7 + 3) % cfg.VocabSize)
					}
					var got, want []float32
					prefix := min(n, 7)
					for pos, token := range tokens {
						ForwardQwen35Into(cfg, w, wantCache, serialBuf, token, pos, &want)
					}
					if err := r.prefillQwen35At(context.Background(), gotCache, batchBuf, tokens[:prefix], 0, &got); err != nil {
						t.Fatal(err)
					}
					if prefix < n {
						if err := r.prefillQwen35At(context.Background(), gotCache, batchBuf, tokens[prefix:], prefix, &got); err != nil {
							t.Fatal(err)
						}
					}
					requireQwen35BatchValuesWithin(t, "logits", got, want, tolerance)
					requireQwen35BatchValuesWithin(t, "final hidden", batchBuf.XN, serialBuf.XN, hiddenTolerance)
					requireQwen35BatchCache(t, gotCache, wantCache, tolerance, false)
					if len(batchBuf.qwenBatch.X) > 32 {
						t.Fatal("hybrid activation workspace exceeded the 32-token chunk bound")
					}
					// A close final logit alone cannot establish that the whole
					// hybrid state is usable. Continue ordinary decode from both
					// caches and require the same greedy decisions at every step.
					for step := range 8 {
						gotToken, wantToken := argmaxFiniteToken(got), argmaxFiniteToken(want)
						if gotToken != wantToken {
							t.Fatalf("n=%d greedy token at decode step %d=%d, want %d", n, step, gotToken, wantToken)
						}
						ForwardQwen35Into(cfg, w, gotCache, batchBuf, gotToken, n+step, &got)
						ForwardQwen35Into(cfg, w, wantCache, serialBuf, wantToken, n+step, &want)
						requireQwen35BatchValuesWithin(t, "decode logits", got, want, tolerance)
						requireQwen35BatchValuesWithin(t, "decode hidden", batchBuf.XN, serialBuf.XN, hiddenTolerance)
						requireQwen35BatchCache(t, gotCache, wantCache, tolerance, true)
					}
				}
			})
		}
	}
}

func qwen35DenseRuntimeRunner(t *testing.T, mtp bool) *Runner {
	t.Helper()
	r, err := RunnerFromGGUFBytes(buildTinyQwen35MoEGGUF(mtp))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	r.config.Arch, r.arch, r.config.ExpertCount, r.config.ExpertUsedCount = "qwen35", "qwen35", 0, 0
	r.config.HiddenDim = 16
	for i := range r.qwen35.Layers {
		r.qwen35.Layers[i].FFN = Qwen35FFNWeights{
			Gate: qwen35TestWeight(16, 8, smallWeights(16*8, 720+i)), Up: qwen35TestWeight(16, 8, smallWeights(16*8, 740+i)), Down: qwen35TestWeight(8, 16, smallWeights(16*8, 760+i)),
		}
	}
	return r
}

func TestQwen35ChunkedPrefillRuntimePrefixAndMTPStates(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, true)
	if r.canBatchPrefill() || !r.canBatchQwen35() {
		t.Fatal("hybrid batching entered the standard transformer capability")
	}
	opts := tinyMistralKVPrefixOptions()
	opts.SystemPrompt, opts.MaxTokens = "", 2
	messages := []ChatMessage{UserMessage("abcdef")}
	first, err := r.GenerateChat(messages, opts)
	if err != nil || len(r.workspaceBuf.qwenBatch.X) == 0 {
		t.Fatalf("dense runtime did not enter hybrid batching: %+v %v", first, err)
	}
	next := append(append([]ChatMessage(nil), messages...), AssistantMessage(first.Text), UserMessage("ghij"))
	got, err := r.GenerateChat(next, opts)
	if err != nil {
		t.Fatal(err)
	}
	gotState, gotLogits := r.prefixCache.qwen35.snapshot(), slices.Clone(r.prefixCache.promptLogits)
	if got.PromptCache == nil || !got.PromptCache.Hit {
		t.Fatal("chunked hybrid prefix append did not reuse the complete state")
	}
	r.ClearConversationCache()
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "1")
	want, err := r.GenerateChat(next, opts)
	if err != nil || got.Text != want.Text {
		t.Fatalf("prefix append differs from serial cold generation: got=%+v want=%+v err=%v", got, want, err)
	}
	requireQwen35BatchValues(t, "prompt logits", gotLogits, r.prefixCache.promptLogits)
	requireQwen35BatchValues(t, "resident recurrence", gotState.State, r.prefixCache.qwen35.State)
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "")
	r.ClearConversationCache()
	r.workspaceBuf.qwenBatch = qwen35BatchScratch{}
	opts.MTPDraftTokens = 1
	mtpResult, err := r.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.workspaceBuf.qwenBatch.X) == 0 || r.prefixCache.qwen35MTP == nil {
		t.Fatal("active MTP missed batched prefill or lost its pending hidden snapshot")
	}
	batchPending, batchState := slices.Clone(r.prefixCache.qwen35MTP), r.prefixCache.qwen35.snapshot()
	r.ClearConversationCache()
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "1")
	serialMTP, err := r.GenerateChat(messages, opts)
	if err != nil || mtpResult.Text != serialMTP.Text || mtpResult.FinishReason != serialMTP.FinishReason {
		t.Fatalf("batched MTP prefill differs from serial MTP: got=%+v want=%+v err=%v", mtpResult, serialMTP, err)
	}
	requireQwen35BatchValues(t, "MTP pending target hidden", batchPending, r.prefixCache.qwen35MTP)
	requireQwen35BatchValues(t, "MTP prefill recurrence", batchState.State, r.prefixCache.qwen35.State)
}

func TestQwen35ChunkedPrefillCapabilityFallbacks(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, false)
	r.outOfCore = true
	if r.canBatchQwen35() {
		t.Fatal("out-of-core hybrid entered batching")
	}
	r.outOfCore = false
	r.config.Arch = "qwen35moe"
	if r.canBatchQwen35() {
		t.Fatal("MoE hybrid entered dense batching")
	}
	r.config.Arch = "qwen35"
	r.qwen35.Layers[0].DeltaNet.QKVConv.Metal = new(MetalWeight)
	if r.canBatchQwen35() {
		t.Fatal("selective Metal hybrid was demoted to CPU batching")
	}
	r.qwen35.Layers[0].DeltaNet.QKVConv.Metal = nil
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "1")
	if r.canBatchQwen35() {
		t.Fatal("disabled batching was ignored")
	}
}

type qwen35CancelAfterStateWrite struct {
	context.Context
	runner *Runner
	cancel context.CancelFunc
}

func (c *qwen35CancelAfterStateWrite) Err() error {
	if cache := c.runner.workspaceCache; cache != nil && cache.Qwen35 != nil {
		for _, v := range cache.Qwen35.State {
			if v != 0 {
				c.cancel()
				break
			}
		}
	}
	return c.Context.Err()
}

func TestQwen35ChunkedPrefillCancellationDoesNotRetainPartialState(t *testing.T) {
	r := qwen35DenseRuntimeRunner(t, false)
	opts := tinyMistralKVPrefixOptions()
	opts.SystemPrompt = ""
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ctx = &qwen35CancelAfterStateWrite{Context: ctx, runner: r, cancel: cancel}
	messages := []ChatMessage{UserMessage("abcdef")}
	if _, err := r.GenerateChat(messages, opts.WithContext(ctx)); !errors.Is(err, context.Canceled) {
		t.Fatalf("chunked hybrid ignored cancellation during recurrence: %v", err)
	}
	if r.prefixCache.cache != nil {
		t.Fatal("partial layer-wise recurrence became a reusable prefix")
	}
	got, err := r.GenerateChat(messages, opts)
	if err != nil {
		t.Fatal(err)
	}
	state := r.prefixCache.qwen35.snapshot()
	r.ClearConversationCache()
	t.Setenv("GOPHERLLM_NO_BATCH_PREFILL", "1")
	want, err := r.GenerateChat(messages, opts)
	if err != nil || got.Text != want.Text {
		t.Fatalf("request after canceled prefill differs from cold serial generation: %v", err)
	}
	requireQwen35BatchValues(t, "state after cancellation", state.State, r.prefixCache.qwen35.State)
}

func benchmarkQwen35Chunked(b *testing.B, r *Runner, lengths []int) {
	previous, procs := configuredThreads.Load(), runtime.GOMAXPROCS(8)
	SetNumThreads(8)
	b.Cleanup(func() { configuredThreads.Store(previous); workerthreads.Set(int(previous)); runtime.GOMAXPROCS(procs) })
	for _, n := range lengths {
		var serialLogits, serialState, serialConv []float32
		for _, batch := range []bool{false, true} {
			b.Run(fmt.Sprintf("tokens%d/batch=%t", n, batch), func(b *testing.B) {
				cache, buf := r.generationWorkspace(n)
				tokens := make([]uint32, n)
				for i := range tokens {
					tokens[i] = uint32((i*7 + 3) % r.config.VocabSize)
				}
				var logits []float32
				run := func() {
					cache.Qwen35.reset()
					if batch {
						if err := r.prefillQwen35At(context.Background(), cache, buf, tokens, 0, &logits); err != nil {
							b.Fatal(err)
						}
					} else {
						for pos, token := range tokens {
							if pos == n-1 {
								ForwardQwen35Into(r.config, r.qwen35, cache, buf, token, pos, &logits)
							} else {
								ForwardQwen35BodyInto(r.config, r.qwen35, cache, buf, token, pos)
							}
						}
					}
				}
				run()
				maxLogitDiff, maxStateDiff := float64(0), float64(0)
				if batch {
					check := func(name string, got, want []float32) float64 {
						if len(got) != len(want) {
							b.Fatalf("%s length=%d, want %d", name, len(got), len(want))
						}
						maxDiff := float64(0)
						for i, value := range got {
							diff := math.Abs(float64(value - want[i]))
							if !finite32(value) || !finite32(want[i]) || diff > 1e-3*max(1, math.Abs(float64(want[i]))) {
								b.Fatalf("%s[%d]=%.9g, want %.9g", name, i, value, want[i])
							}
							maxDiff = max(maxDiff, diff)
						}
						return maxDiff
					}
					maxLogitDiff = check("configured logits", logits, serialLogits)
					if argmaxFiniteToken(logits) != argmaxFiniteToken(serialLogits) {
						b.Fatal("chunked prefill changed the configured model's greedy next token")
					}
					maxStateDiff = check("configured complete recurrence", cache.Qwen35.State, serialState)
					check("configured complete convolution", cache.Qwen35.Conv, serialConv)
				} else {
					serialLogits = slices.Clone(logits)
					serialState, serialConv = slices.Clone(cache.Qwen35.State), slices.Clone(cache.Qwen35.Conv)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					run()
				}
				b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
				if batch {
					b.ReportMetric(maxLogitDiff, "logits_max_abs")
					b.ReportMetric(maxStateDiff, "state_max_abs")
				}
			})
		}
	}
}

func BenchmarkQwen35ChunkedPrefill(b *testing.B) {
	cfg, weights := qwen35BatchFixture(true)
	benchmarkQwen35Chunked(b, &Runner{kind: loadedQwen35, config: cfg, qwen35: weights}, []int{8, 32})
}

func BenchmarkQwen35ChunkedPrefillConfiguredModel(b *testing.B) {
	path := os.Getenv("GOPHERLLM_BENCH_MODEL")
	if path == "" {
		b.Skip("set GOPHERLLM_BENCH_MODEL to a local dense Qwen3.5/3.6/3.8 model")
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		b.Fatal(err)
	}
	defer r.Close()
	if !r.canBatchQwen35() {
		b.Skip("configured model is not a dense CPU Qwen hybrid")
	}
	lengths := []int{8, 32}
	if strings.TrimSpace(os.Getenv("GOPHERLLM_BENCH_QWEN_SHORT")) != "" {
		lengths = []int{8}
	}
	// This diagnostic deliberately changes the graph. It checks real loaded
	// projection/head geometries with a bounded resident weight set; its name
	// must never be presented as a throughput result for the complete model.
	if raw := strings.TrimSpace(os.Getenv("GOPHERLLM_BENCH_QWEN_LAYER_LIMIT")); raw != "" {
		limit, err := strconv.Atoi(raw)
		whole := len(r.qwen35.Layers)
		if err != nil || limit <= 0 || limit >= whole {
			b.Fatalf("GOPHERLLM_BENCH_QWEN_LAYER_LIMIT must be between 1 and %d", whole-1)
		}
		r.qwen35.Layers, r.config.NLayers = r.qwen35.Layers[:limit], limit
		if len(r.config.QwenRecurrentLayers) > limit {
			r.config.QwenRecurrentLayers = r.config.QwenRecurrentLayers[:limit]
		}
		if qwen35AttentionLayerCount(r.qwen35) == 0 || qwen35RecurrentLayerCount(r.qwen35) == 0 {
			b.Fatal("partial graph must contain both recurrent and full-attention layers")
		}
		b.Run(fmt.Sprintf("partial_layers%d_of%d", limit, whole), func(b *testing.B) { benchmarkQwen35Chunked(b, r, lengths) })
		return
	}
	benchmarkQwen35Chunked(b, r, lengths)
}
