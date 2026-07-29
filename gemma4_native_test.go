package gopherllm

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// buildTinyNativeGemma4GGUF models the key structural split of production
// Gemma 4: a 256-style local layer (smallened to headDim 2) followed by a
// global layer with a larger head dimension and K-as-V.  It intentionally
// uses F32 so native graph tests remain fast and deterministic.
func buildTinyNativeGemma4GGUF(withPLE bool) []byte {
	return buildTinyNativeGemma4GGUFWithMoE(withPLE, false, "")
}

// buildTinyNativeGemma4MoEGGUF adds the structural pieces unique to the 26B
// A4B graph: ordinary ffn_* tensors are the shared GEGLU branch while the
// fused expert gate/up bank, scaled router and three branch norms provide the
// sparse branch. omit lets loader tests prove that required special tensors
// are not silently ignored.
func buildTinyNativeGemma4MoEGGUF(omit string) []byte {
	return buildTinyNativeGemma4GGUFWithMoE(false, true, omit)
}

func buildTinyNativeGemma4GGUFWithMoE(withPLE, withMoE bool, omit string) []byte {
	const (
		dim          = 8
		heads        = 2
		hidden       = 16
		vocab        = 18
		experts      = 4
		expertHidden = 4
	)
	tokens := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{"<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "▁", "\n"}
	for i := range tokens {
		if i < len(special) {
			tokens[i] = special[i]
		} else {
			tokens[i] = string(rune('a' + i - len(special)))
		}
		scores[i] = float32(0)
	}
	perLayer := uint32(0)
	if withPLE {
		perLayer = 2
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "gemma4"},
		{"general.name", ggufStr, "tiny-native-gemma4"},
		{"gemma4.embedding_length", ggufU32, uint32(dim)},
		{"gemma4.block_count", ggufU32, uint32(2)},
		{"gemma4.attention.head_count", ggufU32, uint32(heads)},
		{"gemma4.attention.head_count_kv", ggufArr, ggufArray{ggufU32, []any{uint32(2), uint32(1)}}},
		{"gemma4.attention.key_length", ggufU32, uint32(4)},
		{"gemma4.attention.value_length", ggufU32, uint32(4)},
		{"gemma4.attention.key_length_swa", ggufU32, uint32(2)},
		{"gemma4.attention.value_length_swa", ggufU32, uint32(2)},
		{"gemma4.attention.sliding_window", ggufU32, uint32(3)},
		{"gemma4.attention.sliding_window_pattern", ggufArr, ggufArray{ggufBool, []any{true, false}}},
		{"gemma4.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-6)},
		{"gemma4.rope.freq_base", ggufF32, float32(1_000_000)},
		{"gemma4.rope.freq_base_swa", ggufF32, float32(10_000)},
		{"gemma4.rope.dimension_count", ggufU32, uint32(4)},
		{"gemma4.rope.dimension_count_swa", ggufU32, uint32(2)},
		{"gemma4.feed_forward_length", ggufU32, uint32(hidden)},
		{"gemma4.context_length", ggufU32, uint32(64)},
		{"gemma4.final_logit_softcapping", ggufF32, float32(30)},
		{"gemma4.embedding_length_per_layer_input", ggufU32, perLayer},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
		{"tokenizer.chat_template", ggufStr, "{{ '<|turn>user\\n' }}{{ '<turn|>' }}"},
	}
	if withMoE {
		kvs = append(kvs,
			ggufKV{"gemma4.expert_count", ggufU32, uint32(experts)},
			ggufKV{"gemma4.expert_used_count", ggufU32, uint32(2)},
			ggufKV{"gemma4.expert_feed_forward_length", ggufU32, uint32(expertHidden)},
		)
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	f32expert := func(name string, input, output, count, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(input), uint64(output), uint64(count)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(input*output*count, seed))}
	}
	ones := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		ones("output_norm.weight", dim),
		// Proportional global RoPE: second pair is intentionally frozen.
		{name: "rope_freqs.weight", dims: []uint64{2}, dtype: GGMLTypeF32, data: f32Bytes([]float32{1, 1e30})},
	}
	addLayer := func(layer, headDim, kvHeads int, includeV bool, scale float32) {
		p := "blk." + string(rune('0'+layer)) + "."
		qRows := heads * headDim
		kvRows := kvHeads * headDim
		tensors = append(tensors,
			ones(p+"attn_norm.weight", dim),
			f32t(p+"attn_q.weight", qRows, dim, 10+layer),
			f32t(p+"attn_k.weight", kvRows, dim, 20+layer),
			ones(p+"attn_q_norm.weight", headDim),
			ones(p+"attn_k_norm.weight", headDim),
			f32t(p+"attn_output.weight", dim, qRows, 30+layer),
			ones(p+"post_attention_norm.weight", dim),
			ones(p+"ffn_norm.weight", dim),
			f32t(p+"ffn_gate.weight", hidden, dim, 40+layer),
			f32t(p+"ffn_up.weight", hidden, dim, 50+layer),
			f32t(p+"ffn_down.weight", dim, hidden, 60+layer),
			ones(p+"post_ffw_norm.weight", dim),
			ggufTensor{name: p + "layer_output_scale.weight", dims: []uint64{1}, dtype: GGMLTypeF32, data: f32Bytes([]float32{scale})},
		)
		if includeV {
			tensors = append(tensors, f32t(p+"attn_v.weight", kvRows, dim, 70+layer))
		}
		if withMoE {
			moeTensors := []ggufTensor{
				f32t(p+"ffn_gate_inp.weight", experts, dim, 80+layer),
				ones(p+"ffn_gate_inp.scale", dim),
				ones(p+"pre_ffw_norm_2.weight", dim),
				ones(p+"post_ffw_norm_1.weight", dim),
				ones(p+"post_ffw_norm_2.weight", dim),
				f32expert(p+"ffn_gate_up_exps.weight", dim, 2*expertHidden, experts, 90+layer),
				f32expert(p+"ffn_down_exps.weight", expertHidden, dim, experts, 100+layer),
				{name: p + "ffn_down_exps.scale", dims: []uint64{experts}, dtype: GGMLTypeF32, data: f32Bytes([]float32{0.5, 1, 1.5, 2})},
			}
			for _, tensor := range moeTensors {
				if tensor.name != omit {
					tensors = append(tensors, tensor)
				}
			}
		}
	}
	addLayer(0, 2, 2, true, 0.5)
	addLayer(1, 4, 1, false, 1)
	return buildGGUF(3, kvs, tensors)
}

func TestNativeGemma4LoadsMixedAttentionAndRuns(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyNativeGemma4GGUF(false))
	if err != nil {
		t.Fatal(err)
	}
	if !r.gemma4.Native || len(r.gemma4.Layers) != 2 {
		t.Fatalf("native Gemma4=%v layers=%d", r.gemma4.Native, len(r.gemma4.Layers))
	}
	local, global := r.gemma4.Layers[0], r.gemma4.Layers[1]
	if !local.IsSWA || local.HeadDim != 2 || local.NKVHeads != 2 || local.UsesKAsV {
		t.Fatalf("local geometry = %+v", local)
	}
	if global.IsSWA || global.HeadDim != 4 || global.NKVHeads != 1 || !global.UsesKAsV || global.HasAttnV {
		t.Fatalf("global K-as-V geometry = %+v", global)
	}
	if global.OutputScale != 1 || local.OutputScale != 0.5 {
		t.Fatalf("output scales = %g/%g", local.OutputScale, global.OutputScale)
	}
	if r.config.AttentionScale != 1 {
		t.Fatalf("Gemma4 attention scale = %g, want 1", r.config.AttentionScale)
	}
	if got := len(global.RopeInvFreq); got != 2 || global.RopeInvFreq[1] >= 1e-20 {
		t.Fatalf("global proportional RoPE not applied: %v", global.RopeInvFreq)
	}

	kDim, vDim, maxHead, maxKV, maxValue := r.cacheDims()
	if kDim != 4 || vDim != 4 || maxHead != 4 || maxKV != 2 || maxValue != 4 {
		t.Fatalf("native cache dimensions = %d/%d head=%d kv=%d value=%d", kDim, vDim, maxHead, maxKV, maxValue)
	}
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 4)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxValue)
	logits := []float32{}
	ForwardGemma4Into(r.config, r.gemma4, cache, buf, 9, 0, &logits)
	if len(logits) != 18 {
		t.Fatalf("logit count = %d, want 18", len(logits))
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v <= -30 || v >= 30 {
			t.Fatalf("logit[%d]=%g, want finite and softcapped", i, v)
		}
	}
	// Exercise the Runner's native prefill/greedy dispatch, not only the
	// exported direct forward helper.
	opts := DefaultGenerationOptions()
	opts.SystemPrompt = ""
	opts.MaxTokens = 1
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	if _, err := r.Generate("a", opts); err != nil {
		t.Fatalf("native Gemma4 Generate: %v", err)
	}
}

func TestNativeGemma4PLERequiresPLETensors(t *testing.T) {
	_, err := RunnerFromGGUFBytes(buildTinyNativeGemma4GGUF(true))
	if err == nil || !strings.Contains(err.Error(), "per_layer_token_embd") {
		t.Fatalf("PLE layout error = %v", err)
	}
}

// buildTinyNativeGemma4E2BGGUF is a structural E2B fixture. It has the real
// 35/20 split (fifteen physical K/V layers and twenty query-only tail layers),
// scalar KV-head metadata, PLE tensors, and deliberately serialized tail K/V
// weights that must never be loaded or stored by the native path.
func buildTinyNativeGemma4E2BGGUF() []byte {
	const (
		dim         = 8
		heads       = 2
		localHead   = 2
		globalHead  = 4
		hidden      = 8
		perLayerDim = 2
		layers      = 35
		vocab       = 18
	)
	tokens := make([]any, vocab)
	scores := make([]any, vocab)
	for i := range tokens {
		tokens[i] = fmt.Sprintf("t%d", i)
		scores[i] = float32(0)
	}
	pattern := make([]any, layers)
	feedForward := make([]any, layers)
	for layer := range layers {
		pattern[layer] = layer%5 != 4 // four local blocks, then one global
		feedForward[layer] = uint32(hidden)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "gemma4"},
		{"general.name", ggufStr, "tiny-native-gemma4-e2b"},
		{"gemma4.embedding_length", ggufU32, uint32(dim)},
		{"gemma4.block_count", ggufU32, uint32(layers)},
		{"gemma4.attention.head_count", ggufU32, uint32(heads)},
		// E2B's actual file uses this scalar, not a per-layer array.
		{"gemma4.attention.head_count_kv", ggufU32, uint32(1)},
		{"gemma4.attention.key_length", ggufU32, uint32(globalHead)},
		{"gemma4.attention.value_length", ggufU32, uint32(globalHead)},
		{"gemma4.attention.key_length_swa", ggufU32, uint32(localHead)},
		{"gemma4.attention.value_length_swa", ggufU32, uint32(localHead)},
		{"gemma4.attention.sliding_window", ggufU32, uint32(3)},
		{"gemma4.attention.sliding_window_pattern", ggufArr, ggufArray{ggufBool, pattern}},
		{"gemma4.attention.shared_kv_layers", ggufU32, uint32(20)},
		{"gemma4.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-6)},
		{"gemma4.rope.freq_base", ggufF32, float32(1_000_000)},
		{"gemma4.rope.freq_base_swa", ggufF32, float32(10_000)},
		{"gemma4.rope.dimension_count", ggufU32, uint32(globalHead)},
		{"gemma4.rope.dimension_count_swa", ggufU32, uint32(localHead)},
		{"gemma4.feed_forward_length", ggufArr, ggufArray{ggufU32, feedForward}},
		{"gemma4.embedding_length_per_layer_input", ggufU32, uint32(perLayerDim)},
		{"gemma4.context_length", ggufU32, uint32(64)},
		{"gemma4.final_logit_softcapping", ggufF32, float32(30)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
	}
	f32t := func(name string, rows, cols int, values []float32) ggufTensor {
		if values == nil {
			values = make([]float32, rows*cols)
		}
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(values)}
	}
	ones := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	tokenValues := make([]float32, vocab*dim)
	// Keep every residual block deterministic: all ordinary projections are
	// zero, so the source K at layer 13 derives directly from this row.
	tokenValues[9*dim] = 1
	tokenValues[9*dim+1] = 2
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, tokenValues),
		ones("output_norm.weight", dim),
		{name: "rope_freqs.weight", dims: []uint64{2}, dtype: GGMLTypeF32, data: f32Bytes([]float32{1, 1})},
		f32t("per_layer_token_embd.weight", vocab, perLayerDim*layers, nil),
		f32t("per_layer_model_proj.weight", perLayerDim*layers, dim, nil),
		ones("per_layer_proj_norm.weight", perLayerDim),
	}
	for layer := range layers {
		isSWA := layer%5 != 4
		headDim := globalHead
		if isSWA {
			headDim = localHead
		}
		prefix := fmt.Sprintf("blk.%d.", layer)
		qRows := heads * headDim
		kvRows := headDim
		kValues := make([]float32, kvRows*dim)
		if layer == 13 { // the E2B local shared-KV source
			kValues[0] = 1
			kValues[dim+1] = 1
		}
		if layer == 15 || layer == 19 { // query-only tail K must be ignored
			for i := range kValues {
				kValues[i] = 99
			}
		}
		tensors = append(tensors,
			ones(prefix+"attn_norm.weight", dim),
			f32t(prefix+"attn_q.weight", qRows, dim, nil),
			f32t(prefix+"attn_k.weight", kvRows, dim, kValues),
			f32t(prefix+"attn_v.weight", kvRows, dim, nil),
			ones(prefix+"attn_q_norm.weight", headDim),
			ones(prefix+"attn_k_norm.weight", headDim),
			f32t(prefix+"attn_output.weight", dim, qRows, nil),
			ones(prefix+"post_attention_norm.weight", dim),
			ones(prefix+"ffn_norm.weight", dim),
			f32t(prefix+"ffn_gate.weight", hidden, dim, nil),
			f32t(prefix+"ffn_up.weight", hidden, dim, nil),
			f32t(prefix+"ffn_down.weight", dim, hidden, nil),
			ones(prefix+"post_ffw_norm.weight", dim),
			ggufTensor{name: prefix + "layer_output_scale.weight", dims: []uint64{1}, dtype: GGMLTypeF32, data: f32Bytes([]float32{1})},
			f32t(prefix+"inp_gate.weight", perLayerDim, dim, nil),
			f32t(prefix+"proj.weight", dim, perLayerDim, nil),
			ones(prefix+"post_norm.weight", dim),
		)
	}
	return buildGGUF(3, kvs, tensors)
}

func TestNativeGemma4E2BSharedKVSlotsAreExact(t *testing.T) {
	pattern := make([]bool, 35)
	for i := range pattern {
		pattern[i] = i%5 != 4
	}
	slots, count, err := gemma4KVCacheSlots(pattern, 20)
	if err != nil {
		t.Fatal(err)
	}
	if count != 15 || len(slots) != 35 {
		t.Fatalf("E2B KV slots count=%d mapping=%v", count, slots)
	}
	for layer := 0; layer < 15; layer++ {
		if slots[layer] != layer {
			t.Fatalf("owned E2B layer %d maps to slot %d, want itself", layer, slots[layer])
		}
	}
	for layer := 15; layer < 35; layer++ {
		want := 14
		if pattern[layer] {
			want = 13
		}
		if slots[layer] != want {
			t.Fatalf("tail E2B layer %d maps to slot %d, want %d", layer, slots[layer], want)
		}
	}
}

func TestNativeGemma4E2BLoadsPLEAndIgnoresTailKV(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyNativeGemma4E2BGGUF())
	if err != nil {
		t.Fatal(err)
	}
	if r.gemma4.PerLayer == nil || r.gemma4.PerLayer.Dim != 2 || r.kvCacheLayerCount() != 15 {
		t.Fatalf("E2B PLE/cache configuration = %+v slots=%d", r.gemma4.PerLayer, r.kvCacheLayerCount())
	}
	for layer := 0; layer < 15; layer++ {
		got := r.gemma4.Layers[layer]
		if !got.HasKV || got.KVCacheSlot != layer {
			t.Fatalf("E2B owned layer %d KV = has=%v slot=%d", layer, got.HasKV, got.KVCacheSlot)
		}
	}
	for _, layer := range []int{15, 16, 19, 34} {
		got := r.gemma4.Layers[layer]
		want := 14
		if got.IsSWA {
			want = 13
		}
		if got.HasKV || got.KVCacheSlot != want || got.AttnK.F32 != nil || got.AttnK.Raw != nil {
			t.Fatalf("E2B tail layer %d KV = has=%v slot=%d K=%+v, want query-only source %d", layer, got.HasKV, got.KVCacheSlot, got.AttnK, want)
		}
	}

	kDim, vDim, maxHead, maxKV, maxValue := r.cacheDims()
	cache := NewKVCache(r.kvCacheLayerCount(), kDim, vDim, 1)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxValue)
	logits := []float32{}
	ForwardGemma4Into(r.config, r.gemma4, cache, buf, 9, 0, &logits)
	if len(buf.Gemma4PLE) != 35*2 || len(buf.Gemma4PLEInput) != 35*2 || len(logits) != 18 {
		t.Fatalf("E2B PLE/logits sizes = %d/%d/%d", len(buf.Gemma4PLE), len(buf.Gemma4PLEInput), len(logits))
	}
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("E2B logit[%d]=%g, want finite", i, value)
		}
	}
	// Source layer 13's K is normalized [1,2] at position zero. Tail layer 15
	// serializes an all-99 K tensor, but it must use slot 13 without overwriting
	// it; layer 19 likewise must not overwrite global slot 14.
	wantK := []float32{1 / float32(math.Sqrt(2.5)), 2 / float32(math.Sqrt(2.5))}
	for i, want := range wantK {
		if got := cache.K[13][i]; math.Abs(float64(got-want)) > 1e-4 {
			t.Fatalf("E2B slot13 K[%d]=%g, want source-layer value %g (tail K must be ignored)", i, got, want)
		}
	}
	for i, value := range cache.K[14][:4] {
		if value != 0 {
			t.Fatalf("E2B global slot14 K[%d]=%g, want zero source K (tail K must be ignored)", i, value)
		}
	}
}

func TestNativeGemma4E2BAnalysisUsesPhysicalKVLayers(t *testing.T) {
	gguf, err := ParseGGUFQuiet(buildTinyNativeGemma4E2BGGUF())
	if err != nil {
		t.Fatal(err)
	}
	analysis := AnalyzeGGUF(gguf, nil)
	if analysis.HiddenDim != 8 {
		t.Fatalf("E2B per-layer hidden width=%d, want 8", analysis.HiddenDim)
	}
	// 15 physical slots * 1 KV head * (4 K + 4 V) f32 values * 64 context.
	const want = 15 * 1 * (4 + 4) * 4 * 64
	if analysis.KVCacheBytesAtFullContext != want {
		t.Fatalf("E2B KV estimate=%d, want %d (not decoder-depth-sized)", analysis.KVCacheBytesAtFullContext, want)
	}
}

func TestNativeGemma4MoELoadsAndRunsSharedDensePlusExperts(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyNativeGemma4MoEGGUF(""))
	if err != nil {
		t.Fatal(err)
	}
	if r.config.ExpertCount != 4 || r.config.ExpertUsedCount != 2 || len(r.gemma4.MoE) != 2 {
		t.Fatalf("MoE config/layers = experts=%d used=%d layers=%d", r.config.ExpertCount, r.config.ExpertUsedCount, len(r.gemma4.MoE))
	}
	for layer, moe := range r.gemma4.MoE {
		if moe == nil || moe.ExpertUsed != 2 || moe.Gate.Input != 8 || moe.Gate.Output != 4 || moe.Gate.Experts != 4 ||
			moe.Up.RowOffset != 4 || moe.Gate.StorageOutput != 8 || len(moe.DownScale) != 4 {
			t.Fatalf("layer %d special MoE state = %#v", layer, moe)
		}
	}
	kDim, vDim, maxHead, maxKV, maxValue := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 4)
	buf := NewDecodeBuffer(r.config, maxHead, maxKV, maxValue)
	logits := []float32{}
	ForwardGemma4Into(r.config, r.gemma4, cache, buf, 9, 0, &logits)
	if len(logits) != 18 {
		t.Fatalf("logit count = %d, want 18", len(logits))
	}
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logit[%d]=%g, want finite", i, value)
		}
	}
	if len(buf.TopExperts) != 2 || len(buf.ExpertProbs) != 2 {
		t.Fatalf("MoE routing scratch = selected=%v probs=%v", buf.TopExperts, buf.ExpertProbs)
	}
}

func TestNativeGemma4MoERequiresAllSpecialTensors(t *testing.T) {
	_, err := RunnerFromGGUFBytes(buildTinyNativeGemma4MoEGGUF("blk.0.ffn_down_exps.scale"))
	if err == nil || !strings.Contains(err.Error(), "ffn_down_exps.scale") {
		t.Fatalf("missing MoE scale error = %v", err)
	}
}

func TestNativeGemma4MoERouterInputUsesRMSScaleAndSqrtDim(t *testing.T) {
	const dim = 2
	config := Config{Dim: dim, HiddenDim: 1, ExpertCount: 2, ExpertUsedCount: 1, RMSNormEps: 0}
	zeroDense := Weight{F32: []float32{0, 0}, Rows: 1, Cols: dim}
	layer := Gemma4LayerWeights{
		FFNNorm:      []float32{1, 1},
		FFNGate:      zeroDense,
		FFNUp:        zeroDense,
		FFNDown:      Weight{F32: []float32{0, 0}, Rows: dim, Cols: 1},
		FFNHiddenDim: 1,
		PostFFNNorm:  []float32{1, 1},
	}
	zeroGateUp := Weight{F32: make([]float32, dim*1*2), Rows: 2, Cols: dim}
	zeroDown := Weight{F32: make([]float32, 1*dim*2), Rows: 2 * dim, Cols: 1}
	moe := &Gemma4MoEWeights{
		Router:      Weight{F32: []float32{1, 0, 0, 1}, Rows: 2, Cols: dim},
		RouterScale: []float32{2, 3},
		PreNorm2:    []float32{1, 1},
		PostNorm1:   []float32{1, 1},
		PostNorm2:   []float32{1, 1},
		Gate:        ExpertWeight{Weight: zeroGateUp, Input: dim, Output: 1, Experts: 2},
		Up:          ExpertWeight{Weight: zeroGateUp, Input: dim, Output: 1, Experts: 2},
		Down:        ExpertWeight{Weight: zeroDown, Input: 1, Output: dim, Experts: 2},
		DownScale:   []float32{1, 1},
		ExpertUsed:  1,
	}
	buf := NewDecodeBuffer(config, 1, 1, 1)
	buf.X = []float32{3, 4}
	forwardNativeGemma4MoE(config, layer, moe, buf)
	rms := float32(math.Sqrt((3*3 + 4*4) / 2.0))
	want := []float32{3 / rms * 2 / float32(math.Sqrt(dim)), 4 / rms * 3 / float32(math.Sqrt(dim))}
	for i := range want {
		if got := buf.RouterLogits[i]; math.Abs(float64(got-want[i])) > 1e-5 {
			t.Fatalf("router[%d]=%g, want %g", i, got, want[i])
		}
	}
}

func TestRenderGemma4MessagesAndStopToken(t *testing.T) {
	tok := newChatTokenizer("<|turn>", "<turn|>", "<|channel>", "<channel|>")
	r := &Runner{
		tok:  tok,
		arch: "gemma4",
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "str", Value: "<|turn>user\\n<turn|>"},
		}},
	}
	if kind := r.chatTemplateKind(); kind != "gemma4-chat" {
		t.Fatalf("chatTemplateKind=%q", kind)
	}
	tokens, ok := r.renderGemma4Messages([]ChatMessage{UserMessage("hi")}, "be precise")
	if !ok {
		t.Fatal("renderGemma4Messages ok=false")
	}
	if tokens[0] != tok.BOSID {
		t.Fatalf("first token=%d, want BOS", tokens[0])
	}
	if got := countToken(tokens, tok.TokenToID["<|turn>"]); got != 3 { // system, user, open model
		t.Fatalf("<|turn> count=%d, want 3", got)
	}
	if got := countToken(tokens, tok.TokenToID["<turn|>"]); got != 2 { // closed system + user
		t.Fatalf("<turn|> count=%d, want 2", got)
	}
	if !r.isStopToken(tok.TokenToID["<turn|>"]) {
		t.Fatal("Gemma4 must stop at <turn|>")
	}
}

// TestRenderGemma4MessagesOnlyClosesThoughtChannelWhenTheCheckpointWantsIt
// guards a real bug: E2B's own chat_template never emits
// '<|channel>thought\n<channel|>' in its add_generation_prompt branch, while
// other Gemma 4 checkpoints (12B/26B) do so by default. Injecting it
// unconditionally fed E2B an out-of-distribution prompt suffix and derailed
// its generation into gibberish; the fix reads each checkpoint's own
// chat_template to decide.
func TestRenderGemma4MessagesOnlyClosesThoughtChannelWhenTheCheckpointWantsIt(t *testing.T) {
	tok := newChatTokenizer("<|turn>", "<turn|>", "<|channel>", "<channel|>")

	wantsIt := &Runner{
		tok:  tok,
		arch: "gemma4",
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "str", Value: `{%- if not enable_thinking | default(false) -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}`},
		}},
	}
	tokens, ok := wantsIt.renderGemma4Messages([]ChatMessage{UserMessage("hi")}, "")
	if !ok {
		t.Fatal("renderGemma4Messages ok=false")
	}
	if !hasAll(tokens, tok.TokenToID["<|channel>"], tok.TokenToID["<channel|>"]) {
		t.Fatalf("expected the thought channel to be opened and closed: %v", tokens)
	}

	doesNotWantIt := &Runner{
		tok:  tok,
		arch: "gemma4",
		gguf: &GGUFFile{Metadata: map[string]MetaValue{
			"tokenizer.chat_template": {Kind: "str", Value: `{%- if add_generation_prompt -%}{{- '<|turn>model\n' -}}{%- endif -%}`},
		}},
	}
	tokens, ok = doesNotWantIt.renderGemma4Messages([]ChatMessage{UserMessage("hi")}, "")
	if !ok {
		t.Fatal("renderGemma4Messages ok=false")
	}
	if hasAll(tokens, tok.TokenToID["<|channel>"]) {
		t.Fatalf("E2B-style template must not get a fabricated thought channel: %v", tokens)
	}
}
