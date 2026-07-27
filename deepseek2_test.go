package gopherllm

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// buildTinyDeepSeek2GGUF is deliberately two layers: the first uses the
// leading dense FFN and the second exercises Kimi/DeepSeek's sparse sigmoid
// router plus its always-on shared expert. The dimensions are tiny, but retain
// the distinct MLA query/key/value/cache dimensions of a real Kimi K2 GGUF.
func buildTinyDeepSeek2GGUF(expertGroups uint32) []byte {
	const (
		dim, heads, vocab = 8, 2, 16
		qRank, kvRank     = 4, 3
		rope, keyMLA      = 2, 4 // 2 no-position + 2 RoPE dimensions
		valueMLA          = 2
		hidden, expertFF  = 8, 4
		experts, used     = 3, 2
	)
	if expertGroups == 0 {
		expertGroups = 1
	}
	tokens := make([]any, vocab)
	scores := make([]any, vocab)
	for i := range tokens {
		switch i {
		case 0:
			tokens[i] = "<unk>"
		case 1:
			tokens[i] = "<s>"
		case 2:
			tokens[i] = "</s>"
		default:
			tokens[i] = string(rune('a' + i - 3))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "deepseek2"},
		{"general.name", ggufStr, "tiny-deepseek2"},
		{"deepseek2.embedding_length", ggufU32, uint32(dim)},
		{"deepseek2.block_count", ggufU32, uint32(2)},
		{"deepseek2.attention.head_count", ggufU32, uint32(heads)},
		{"deepseek2.attention.head_count_kv", ggufU32, uint32(1)},
		// key/value_length are the compact cache widths, not MLA's expanded
		// per-head Q/value widths.
		{"deepseek2.attention.key_length", ggufU32, uint32(kvRank + rope)},
		{"deepseek2.attention.value_length", ggufU32, uint32(kvRank)},
		{"deepseek2.attention.q_lora_rank", ggufU32, uint32(qRank)},
		{"deepseek2.attention.kv_lora_rank", ggufU32, uint32(kvRank)},
		{"deepseek2.attention.key_length_mla", ggufU32, uint32(keyMLA)},
		{"deepseek2.attention.value_length_mla", ggufU32, uint32(valueMLA)},
		{"deepseek2.feed_forward_length", ggufU32, uint32(hidden)},
		{"deepseek2.expert_feed_forward_length", ggufU32, uint32(expertFF)},
		{"deepseek2.expert_count", ggufU32, uint32(experts)},
		{"deepseek2.expert_used_count", ggufU32, uint32(used)},
		{"deepseek2.expert_shared_count", ggufU32, uint32(1)},
		{"deepseek2.expert_group_count", ggufU32, expertGroups},
		{"deepseek2.expert_group_used_count", ggufU32, uint32(1)},
		{"deepseek2.expert_gating_func", ggufU32, uint32(deepSeekExpertGatingSigmoid)},
		{"deepseek2.expert_weights_norm", ggufBool, true},
		{"deepseek2.expert_weights_scale", ggufF32, float32(2.827)},
		{"deepseek2.leading_dense_block_count", ggufU32, uint32(1)},
		{"deepseek2.context_length", ggufU32, uint32(32)},
		{"deepseek2.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{"deepseek2.rope.freq_base", ggufF32, float32(10000)},
		{"deepseek2.rope.dimension_count", ggufU32, uint32(rope)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	expert := func(name string, input, output, count, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(input), uint64(output), uint64(count)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(input*output*count, seed))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		f32t("output.weight", vocab, dim, 2),
	}
	for layer := 0; layer < 2; layer++ {
		prefix := "blk." + string(rune('0'+layer)) + "."
		tensors = append(tensors,
			vec(prefix+"attn_norm.weight", dim),
			f32t(prefix+"attn_q_a.weight", qRank, dim, 10+layer),
			vec(prefix+"attn_q_a_norm.weight", qRank),
			f32t(prefix+"attn_q_b.weight", heads*keyMLA, qRank, 20+layer),
			f32t(prefix+"attn_kv_a_mqa.weight", kvRank+rope, dim, 30+layer),
			vec(prefix+"attn_kv_a_norm.weight", kvRank),
			expert(prefix+"attn_k_b.weight", keyMLA-rope, kvRank, heads, 40+layer),
			expert(prefix+"attn_v_b.weight", kvRank, valueMLA, heads, 50+layer),
			f32t(prefix+"attn_output.weight", dim, heads*valueMLA, 60+layer),
			vec(prefix+"ffn_norm.weight", dim),
		)
		if layer == 0 {
			tensors = append(tensors,
				f32t(prefix+"ffn_gate.weight", hidden, dim, 70),
				f32t(prefix+"ffn_up.weight", hidden, dim, 71),
				f32t(prefix+"ffn_down.weight", dim, hidden, 72),
			)
			continue
		}
		tensors = append(tensors,
			f32t(prefix+"ffn_gate_inp.weight", experts, dim, 80),
			vec(prefix+"exp_probs_b.bias", experts),
			expert(prefix+"ffn_gate_exps.weight", dim, expertFF, experts, 81),
			expert(prefix+"ffn_up_exps.weight", dim, expertFF, experts, 82),
			expert(prefix+"ffn_down_exps.weight", expertFF, dim, experts, 83),
			f32t(prefix+"ffn_gate_shexp.weight", expertFF, dim, 84),
			f32t(prefix+"ffn_up_shexp.weight", expertFF, dim, 85),
			f32t(prefix+"ffn_down_shexp.weight", dim, expertFF, 86),
		)
	}
	return buildGGUF(3, kvs, tensors)
}

func TestDeepSeek2MLAReferenceThreePositions(t *testing.T) {
	// This compact hand-built MLA block has no RoPE, making its reference
	// computation transparent while still testing Wk_b absorption, latent KV
	// cache writes at the actual position, latent attention, V_b expansion, and
	// the output projection over three successively cached tokens.
	cfg := Config{
		Dim: 2, NHeads: 1, NKVHeads: 1,
		HeadDim: 2, ValueDim: 2, KVDim: 2,
		UsesMLA: true, MLAKVLoRARank: 2, MLAKeyDim: 1, MLAValueDim: 1,
		RMSNormEps: 1e-6, ResidualScale: 1,
	}
	layer := LayerWeights{
		MLA: &MLAAttentionWeights{
			Q:       Weight{F32: []float32{1, 0}},
			KVA:     Weight{F32: []float32{1, 0, 0, 1}},
			KVANorm: []float32{1, 1},
			KB:      ExpertWeight{Weight: Weight{F32: []float32{1, 0}}, Input: 1, Output: 2, Experts: 1},
			VB:      ExpertWeight{Weight: Weight{F32: []float32{0, 1}}, Input: 2, Output: 1, Experts: 1},
		},
		WO: Weight{F32: []float32{1, 2}},
		BO: []float32{0, 0},
	}
	inputs := [][]float32{{1, 1}, {2, 1}, {-1, 3}}
	cache := NewKVCache(1, 2, 2, len(inputs))
	buf := NewDecodeBuffer(cfg, 2, 1, 2)
	latents := make([][]float32, 0, len(inputs))
	for pos, x := range inputs {
		copy(buf.XN, x)
		forwardMLAAttentionInto(cfg, layer, 0, cache, buf, pos, 0, 0)

		ss := x[0]*x[0] + x[1]*x[1]
		rms := float32(1 / math.Sqrt(float64(ss/2+cfg.RMSNormEps)))
		latent := []float32{x[0] * rms, x[1] * rms}
		latents = append(latents, latent)
		if got := cache.K[0][pos*2 : pos*2+2]; math.Abs(float64(got[0]-latent[0])) > 1e-6 || math.Abs(float64(got[1]-latent[1])) > 1e-6 {
			t.Fatalf("cache K at pos %d = %v, want %v", pos, got, latent)
		}
		if got := cache.V[0][pos*2 : pos*2+2]; math.Abs(float64(got[0]-latent[0])) > 1e-6 || math.Abs(float64(got[1]-latent[1])) > 1e-6 {
			t.Fatalf("cache V at pos %d = %v, want %v", pos, got, latent)
		}

		var maxScore float32 = -float32(math.MaxFloat32)
		scores := make([]float32, pos+1)
		for i := range scores {
			scores[i] = x[0] * latents[i][0]
			if scores[i] > maxScore {
				maxScore = scores[i]
			}
		}
		var denom, value float32
		weights := make([]float32, len(scores))
		for i, score := range scores {
			weights[i] = float32(math.Exp(float64(score - maxScore)))
			denom += weights[i]
		}
		for i, weight := range weights {
			value += weight / denom * latents[i][1]
		}
		closeMoEFloat(t, "MLA output[0]", buf.Proj[0], value)
		closeMoEFloat(t, "MLA output[1]", buf.Proj[1], 2*value)
	}
}

func TestDeepSeek2GGUFLoadsRunsAndKeepsMoELayout(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyDeepSeek2GGUF(1))
	if err != nil {
		t.Fatal(err)
	}
	if r.kind != loadedStandard || !r.config.UsesMLA || r.config.MLAKVLoRARank != 3 || r.config.MLAKeyDim != 4 || r.config.MLAValueDim != 2 {
		t.Fatalf("unexpected DeepSeek2 runner config: kind=%d config=%+v", r.kind, r.config)
	}
	if r.canBatchPrefill() {
		t.Fatal("MLA must use the per-token prefill path")
	}
	if r.standard.Layers[0].MLA == nil || r.standard.Layers[0].MoE != nil || r.standard.Layers[1].MLA == nil || r.standard.Layers[1].MoE == nil {
		t.Fatal("expected leading dense MLA layer followed by sparse MLA MoE layer")
	}
	moe := r.standard.Layers[1].MoE
	if !moe.RoutingSigmoid || !moe.SharedAlways || moe.SharedGateIn != nil || len(moe.RouterCorrectionBias) != r.config.ExpertCount {
		t.Fatalf("Kimi/DeepSeek MoE layout was not loaded: %+v", moe)
	}
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	if kDim != 5 || vDim != 3 {
		t.Fatalf("MLA cache dimensions = %d/%d, want compact 5/3", kDim, vDim)
	}
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 3)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	for pos, token := range []uint32{3, 4, 5} {
		logits := Forward(r.config, r.standard, cache, buf, token, pos)
		if len(logits) != r.config.VocabSize {
			t.Fatalf("pos %d logits=%d, want %d", pos, len(logits), r.config.VocabSize)
		}
		for i, v := range logits {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("non-finite logit at position %d, index %d: %v", pos, i, v)
			}
		}
	}
}

// MLA keeps compact K/V latents rather than normal per-head rows. Exercise the
// f16 cache explicitly so its different storage layout cannot accidentally
// regress the DeepSeek/Kimi attention path (it is opt-in on non-amd64).
func TestDeepSeek2MLAForwardWithF16KVCache(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyDeepSeek2GGUF(1))
	if err != nil {
		t.Fatal(err)
	}
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCacheF16(r.config.NLayers, kDim, vDim, 3)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	if !cache.F16 {
		t.Fatal("expected f16 MLA cache")
	}
	for pos, token := range []uint32{3, 4, 5} {
		logits := Forward(r.config, r.standard, cache, buf, token, pos)
		if len(logits) != r.config.VocabSize {
			t.Fatalf("pos %d logits=%d, want %d", pos, len(logits), r.config.VocabSize)
		}
		for i, value := range logits {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("f16 MLA cache: non-finite logit at position %d, index %d: %v", pos, i, value)
			}
		}
	}
}

func TestKimiK2AliasReadsDeepSeek2MetadataNamespace(t *testing.T) {
	data := buildTinyDeepSeek2GGUF(1)
	gguf, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	// Some converters retain general.architecture=kimi_k2 but write the
	// llama.cpp-native deepseek2.* hparams. The alias must not turn all MLA
	// dimensions into zero merely because the metadata namespace differs.
	gguf.Metadata["general.architecture"] = MetaValue{Kind: "str", Value: "kimi_k2"}
	r, err := runnerFromParsedGGUF(data, gguf, false, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if r.Architecture() != "kimi_k2" || r.config.Arch != "kimi_k2" || !r.config.UsesMLA || r.config.Dim != 8 || r.config.MLAKVLoRARank != 3 {
		t.Fatalf("Kimi alias did not use deepseek2 metadata: runner=%q config=%+v", r.Architecture(), r.config)
	}
}

func TestDeepSeek2SigmoidCorrectionBiasAndAlwaysOnSharedExpert(t *testing.T) {
	// Correction bias promotes experts 1 and 2, while their mixture weights
	// must still come from the *uncorrected* sigmoid router scores. The shared
	// branch is added unconditionally (no Qwen-style gate).
	w := &SparseMoEWeights{
		Router:               Weight{F32: []float32{2, 0, -2}},
		RouterCorrectionBias: []float32{0, 10, 9},
		RoutingSigmoid:       true,
		NormalizeTopK:        true,
		Scale:                2,
		ExpertUsed:           2,
		Gate:                 ExpertWeight{Weight: Weight{F32: []float32{1, 1, 1}}, Input: 1, Output: 1, Experts: 3},
		Up:                   ExpertWeight{Weight: Weight{F32: []float32{1, 1, 1}}, Input: 1, Output: 1, Experts: 3},
		Down:                 ExpertWeight{Weight: Weight{F32: []float32{1, 3, 5}}, Input: 1, Output: 1, Experts: 3},
		SharedAlways:         true,
		SharedGate:           &Weight{F32: []float32{1}},
		SharedUp:             &Weight{F32: []float32{1}},
		SharedDown:           &Weight{F32: []float32{7}},
	}
	buf := &DecodeBuffer{}
	sparseMoEForward(w, []float32{1}, buf)
	if len(buf.TopExperts) != 2 || (buf.TopExperts[0].Index != 1 && buf.TopExperts[1].Index != 1) || (buf.TopExperts[0].Index != 2 && buf.TopExperts[1].Index != 2) {
		t.Fatalf("correction bias did not select experts 1 and 2: %#v", buf.TopExperts)
	}
	p1, p2 := nemotronSigmoid(0), nemotronSigmoid(-2)
	weight1 := p1 / (p1 + p2)
	weight2 := p2 / (p1 + p2)
	silu := nemotronSigmoid(1)
	want := 2*(weight1*3+weight2*5)*silu + 7*silu
	closeMoEFloat(t, "DeepSeek/Kimi sigmoid mixture + shared", buf.Proj[0], want)
}

func TestDeepSeek2RejectsGroupedRoutingAndMetal(t *testing.T) {
	if _, err := RunnerFromGGUFBytes(buildTinyDeepSeek2GGUF(2)); err == nil || !strings.Contains(err.Error(), "grouped MoE") {
		t.Fatalf("grouped DeepSeek2 routing error = %v, want clear unsupported-group diagnostic", err)
	}
	if _, err := RunnerFromGGUFBytesWithOptions(buildTinyDeepSeek2GGUF(1), LoadOptions{UseMetal: true}); err == nil || !strings.Contains(err.Error(), "does not support Metal") {
		t.Fatalf("DeepSeek2 Metal error = %v, want explicit MLA CPU diagnostic", err)
	}
}

func TestDeepSeek2OutOfCoreKeepsMLAAndExpertPlanesMapped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-deepseek2.gguf")
	if err := os.WriteFile(path, buildTinyDeepSeek2GGUF(1), 0o600); err != nil {
		t.Fatal(err)
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{OutOfCore: true})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.OutOfCore() || r.standard.Layers[0].MLA.KB.Weight.F32 != nil || !rawScalarWeight(r.standard.Layers[0].MLA.KB.Weight) {
		t.Fatal("MLA Wk_b was expanded instead of remaining mmap-backed")
	}
	moe := r.standard.Layers[1].MoE
	if moe.Gate.Weight.F32 != nil || !rawScalarWeight(moe.Gate.Weight) || moe.SharedDown.F32 != nil || !rawScalarWeight(*moe.SharedDown) {
		t.Fatal("DeepSeek/Kimi expert planes were expanded instead of remaining mmap-backed")
	}
	kDim, vDim, maxHead, maxKV, maxVal := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 2)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxVal)
	for pos, token := range []uint32{3, 4} {
		logits := Forward(r.config, r.standard, cache, buf, token, pos)
		if len(logits) != r.config.VocabSize || math.IsNaN(float64(logits[0])) || math.IsInf(float64(logits[0]), 0) {
			t.Fatalf("mmap-backed DeepSeek/Kimi forward at pos %d produced invalid logits", pos)
		}
	}
}
