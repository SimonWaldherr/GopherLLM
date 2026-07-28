package gopherllm

import (
	"math"
	"testing"
)

// buildTinyQwen35MoEGGUF is deliberately small, but contains both kinds of
// Qwen3.5 layer: a Gated DeltaNet layer followed by its periodic attention
// layer. Each layer has the Ornith/Qwen35-MoE sparse expert layout, including
// the gated shared expert.
func buildTinyQwen35MoEGGUF(withMTP bool) []byte {
	return buildTinyQwen35MoEGGUFWithSchedule(withMTP, nil, 2)
}

func buildTinyQwen35MoEGGUFWithSchedule(withMTP bool, recurrentSchedule []bool, fullAttentionInterval int) []byte {
	const (
		arch                                          = "qwen35moe"
		dim, heads, kvHeads, headDim, vocab           = 8, 2, 1, 4, 16
		baseLayers, experts, used, expertHidden       = 2, 3, 2, 4
		ssmInner, ssmHeads, ssmGroups, ssmState, conv = 8, 2, 2, 4, 2
	)
	channels := ssmInner + 2*ssmGroups*ssmState
	layers := baseLayers
	if withMTP {
		layers++
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
		{"general.architecture", ggufStr, arch},
		{"general.name", ggufStr, "tiny-qwen35moe"},
		{arch + ".embedding_length", ggufU32, uint32(dim)},
		{arch + ".block_count", ggufU32, uint32(layers)},
		{arch + ".context_length", ggufU32, uint32(32)},
		{arch + ".attention.head_count", ggufU32, uint32(heads)},
		{arch + ".attention.head_count_kv", ggufU32, uint32(kvHeads)},
		{arch + ".attention.key_length", ggufU32, uint32(headDim)},
		{arch + ".attention.value_length", ggufU32, uint32(headDim)},
		{arch + ".attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{arch + ".rope.freq_base", ggufF32, float32(10000)},
		{arch + ".rope.dimension_count", ggufU32, uint32(headDim)},
		{arch + ".expert_count", ggufU32, uint32(experts)},
		{arch + ".expert_used_count", ggufU32, uint32(used)},
		{arch + ".expert_feed_forward_length", ggufU32, uint32(expertHidden)},
		{arch + ".expert_shared_feed_forward_length", ggufU32, uint32(expertHidden)},
		{arch + ".full_attention_interval", ggufU32, uint32(fullAttentionInterval)},
		{arch + ".ssm.conv_kernel", ggufU32, uint32(conv)},
		{arch + ".ssm.inner_size", ggufU32, uint32(ssmInner)},
		{arch + ".ssm.state_size", ggufU32, uint32(ssmState)},
		{arch + ".ssm.time_step_rank", ggufU32, uint32(ssmHeads)},
		{arch + ".ssm.group_count", ggufU32, uint32(ssmGroups)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
	}
	if len(recurrentSchedule) > 0 {
		items := make([]any, len(recurrentSchedule))
		for i, recurrent := range recurrentSchedule {
			items[i] = recurrent
		}
		kvs = append(kvs, ggufKV{arch + ".attention.recurrent_layers", ggufArr, ggufArray{ggufBool, items}})
	}
	if withMTP {
		kvs = append(kvs, ggufKV{arch + ".nextn_predict_layers", ggufU32, uint32(1)})
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(n, seed))}
	}
	expert := func(name string, input, output, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(input), uint64(output), uint64(experts)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(input*output*experts, seed))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim, 2),
		f32t("output.weight", vocab, dim, 3),
	}
	for layer := range baseLayers {
		prefix := "blk." + string(rune('0'+layer)) + "."
		tensors = append(tensors,
			vec(prefix+"attn_norm.weight", dim, 10+layer),
			vec(prefix+"post_attention_norm.weight", dim, 20+layer),
			f32t(prefix+"ffn_gate_inp.weight", experts, dim, 30+layer),
			expert(prefix+"ffn_gate_exps.weight", dim, expertHidden, 40+layer),
			expert(prefix+"ffn_up_exps.weight", dim, expertHidden, 50+layer),
			expert(prefix+"ffn_down_exps.weight", expertHidden, dim, 60+layer),
			vec(prefix+"ffn_gate_inp_shexp.weight", dim, 70+layer),
			f32t(prefix+"ffn_gate_shexp.weight", expertHidden, dim, 80+layer),
			f32t(prefix+"ffn_up_shexp.weight", expertHidden, dim, 90+layer),
			f32t(prefix+"ffn_down_shexp.weight", dim, expertHidden, 100+layer),
		)
		if layer == 0 {
			tensors = append(tensors,
				f32t(prefix+"attn_qkv.weight", channels, dim, 110),
				f32t(prefix+"ssm_conv1d.weight", channels, conv, 111),
				f32t(prefix+"attn_gate.weight", dim, dim, 112),
				f32t(prefix+"ssm_alpha.weight", ssmHeads, dim, 113),
				f32t(prefix+"ssm_beta.weight", ssmHeads, dim, 114),
				vec(prefix+"ssm_a", ssmHeads, 115),
				vec(prefix+"ssm_dt.bias", ssmHeads, 116),
				vec(prefix+"ssm_norm.weight", headDim, 117),
				f32t(prefix+"ssm_out.weight", dim, ssmInner, 118),
			)
		} else {
			tensors = append(tensors,
				f32t(prefix+"attn_q.weight", heads*2*headDim, dim, 120),
				f32t(prefix+"attn_k.weight", kvHeads*headDim, dim, 121),
				f32t(prefix+"attn_v.weight", kvHeads*headDim, dim, 122),
				f32t(prefix+"attn_output.weight", dim, heads*headDim, 123),
				vec(prefix+"attn_q_norm.weight", headDim, 124),
				vec(prefix+"attn_k_norm.weight", headDim, 125),
			)
		}
	}
	if withMTP {
		// A marker is sufficient: the loader must recognize the draft layer
		// before attempting to load its non-base transformer tensors.
		tensors = append(tensors, f32t("blk.2.nextn.eh_proj.weight", dim*2, dim, 130))
	}
	return buildGGUF(3, kvs, tensors)
}

func TestQwen35MoELoadsAndRunsHybridGraph(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyQwen35MoEGGUF(false))
	if err != nil {
		t.Fatal(err)
	}
	if r.kind != loadedQwen35 || r.config.Arch != "qwen35moe" {
		t.Fatalf("runner kind=%d arch=%q, want qwen35moe", r.kind, r.config.Arch)
	}
	if !ArchitectureSupported("qwen35moe") {
		t.Fatal("qwen35moe must be advertised as supported")
	}
	if r.canBatchPrefill() {
		t.Fatal("hybrid Qwen35 graph must not use the standard batched prefill")
	}
	for i, layer := range r.qwen35.Layers {
		if layer.FFN.MoE == nil {
			t.Fatalf("layer %d sparse MoE was not loaded", i)
		}
		moe := layer.FFN.MoE
		if moe.SharedGateIn == nil || moe.SharedGate == nil || moe.SharedUp == nil || moe.SharedDown == nil {
			t.Fatalf("layer %d gated shared expert was not loaded", i)
		}
	}
	if r.qwen35.Layers[0].Kind != qwen35DeltaNet || r.qwen35.Layers[1].Kind != qwen35Attention {
		t.Fatalf("layer schedule = %v, %v", r.qwen35.Layers[0].Kind, r.qwen35.Layers[1].Kind)
	}

	cache, buf := r.generationWorkspace(4)
	if cache.Qwen35 == nil {
		t.Fatal("Qwen35 recurrent cache was not initialized")
	}
	if cache.layerCount() != 1 {
		t.Fatalf("Qwen35 K/V cache layers = %d, want 1 attention layer", cache.layerCount())
	}
	if cache.Qwen35.Layers != 1 {
		t.Fatalf("Qwen35 recurrent cache layers = %d, want 1 DeltaNet layer", cache.Qwen35.Layers)
	}
	var logits []float32
	for pos, token := range []uint32{1, 3, 4} {
		r.forwardTokenInto(cache, buf, token, pos, &logits)
	}
	if len(logits) != r.config.VocabSize {
		t.Fatalf("logit length = %d, want %d", len(logits), r.config.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d is not finite: %v", i, v)
		}
	}
}

func TestQwen35SkipsTrailingMTPDraftLayer(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyQwen35MoEGGUF(true))
	if err != nil {
		t.Fatal(err)
	}
	if r.config.NLayers != 2 || len(r.qwen35.Layers) != 2 {
		t.Fatalf("MTP-adjusted layers = %d/%d, want 2/2", r.config.NLayers, len(r.qwen35.Layers))
	}
}

func TestQwen35UsesExplicitRecurrentScheduleWithoutInterval(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyQwen35MoEGGUFWithSchedule(false, []bool{true, false}, 0))
	if err != nil {
		t.Fatal(err)
	}
	if r.qwen35.Layers[0].Kind != qwen35DeltaNet || r.qwen35.Layers[1].Kind != qwen35Attention {
		t.Fatalf("explicit layer schedule = %v, %v", r.qwen35.Layers[0].Kind, r.qwen35.Layers[1].Kind)
	}
	if got := r.qwen35.Layers[0].RecurrentCacheSlot; got != 0 {
		t.Fatalf("DeltaNet cache slot = %d, want 0", got)
	}
	if got := r.qwen35.Layers[1].KVCacheSlot; got != 0 {
		t.Fatalf("attention K/V cache slot = %d, want 0", got)
	}
	cache, _ := r.generationWorkspace(3)
	if cache.layerCount() != 1 || cache.Qwen35.Layers != 1 {
		t.Fatalf("compact cache layers K/V=%d recurrent=%d, want 1/1", cache.layerCount(), cache.Qwen35.Layers)
	}
}
