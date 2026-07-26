package gopherllm

import (
	"math"
	"testing"
)

func TestNemotronHMoELoaderAndForward(t *testing.T) {
	const (
		dim, vocab                                       = 4, 8
		ssmInner, ssmHeads, ssmGroups, ssmState, ssmConv = 4, 2, 1, 1, 2
		experts, used, expertFF                          = 2, 1, 3
	)
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	expert := func(name string, input, output, count, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(input), uint64(output), uint64(count)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(input*output*count, seed))}
	}
	tokens := []any{"<unk>", "<s>", "</s>", "a", "b", "c", "d", "e"}
	scores := make([]any, len(tokens))
	for i := range scores {
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "nemotron_h_moe"},
		{"nemotron_h_moe.embedding_length", ggufU32, uint32(dim)},
		{"nemotron_h_moe.block_count", ggufU32, uint32(3)},
		{"nemotron_h_moe.context_length", ggufU32, uint32(64)},
		{"nemotron_h_moe.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{"nemotron_h_moe.attention.head_count", ggufArr, ggufArray{ggufU32, []any{uint32(0), uint32(2), uint32(0)}}},
		{"nemotron_h_moe.attention.head_count_kv", ggufArr, ggufArray{ggufU32, []any{uint32(0), uint32(1), uint32(0)}}},
		{"nemotron_h_moe.feed_forward_length", ggufArr, ggufArray{ggufU32, []any{uint32(0), uint32(0), uint32(expertFF)}}},
		{"nemotron_h_moe.ssm.conv_kernel", ggufU32, uint32(ssmConv)},
		{"nemotron_h_moe.ssm.inner_size", ggufU32, uint32(ssmInner)},
		{"nemotron_h_moe.ssm.state_size", ggufU32, uint32(ssmState)},
		{"nemotron_h_moe.ssm.time_step_rank", ggufU32, uint32(ssmHeads)},
		{"nemotron_h_moe.ssm.group_count", ggufU32, uint32(ssmGroups)},
		{"nemotron_h_moe.expert_count", ggufU32, uint32(experts)},
		{"nemotron_h_moe.expert_used_count", ggufU32, uint32(used)},
		{"nemotron_h_moe.expert_weights_scale", ggufF32, float32(1)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1), vec("output_norm.weight", dim), f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim), f32t("blk.0.ssm_in.weight", 2*ssmInner+2*ssmGroups*ssmState+ssmHeads, dim, 3),
		f32t("blk.0.ssm_conv1d.weight", ssmInner+2*ssmGroups*ssmState, ssmConv, 4), vec("blk.0.ssm_conv1d.bias", ssmInner+2*ssmGroups*ssmState),
		vec("blk.0.ssm_dt.bias", ssmHeads), vec("blk.0.ssm_a", ssmHeads), vec("blk.0.ssm_d", ssmHeads), vec("blk.0.ssm_norm.weight", ssmInner), f32t("blk.0.ssm_out.weight", dim, ssmInner, 5),
		vec("blk.1.attn_norm.weight", dim), f32t("blk.1.attn_q.weight", 4, dim, 6), f32t("blk.1.attn_k.weight", 2, dim, 7), f32t("blk.1.attn_v.weight", 2, dim, 8), f32t("blk.1.attn_output.weight", dim, 4, 9),
		vec("blk.2.attn_norm.weight", dim), f32t("blk.2.ffn_gate_inp.weight", experts, dim, 10), expert("blk.2.ffn_up_exps.weight", dim, expertFF, experts, 11), expert("blk.2.ffn_down_exps.weight", expertFF, dim, experts, 12),
	}
	gguf, err := ParseGGUFQuiet(buildGGUF(3, kvs, tensors))
	if err != nil {
		t.Fatal(err)
	}
	r, err := runnerFromParsedGGUF(buildGGUF(3, kvs, tensors), gguf, false, LoadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if r.kind != loadedNemotronH {
		t.Fatalf("kind = %d", r.kind)
	}
	cache, buf := r.generationWorkspace(4)
	var logits []float32
	for pos, token := range []uint32{1, 3, 4} {
		r.forwardTokenInto(cache, buf, token, pos, &logits)
	}
	if len(logits) != vocab {
		t.Fatalf("logits length = %d, want %d", len(logits), vocab)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d is not finite: %v", i, v)
		}
	}
}

func TestDenseNemotronHLoadsWithGlobalHeadCount(t *testing.T) {
	const (
		dim, vocab                                       = 4, 8
		ssmInner, ssmHeads, ssmGroups, ssmState, ssmConv = 4, 2, 1, 1, 2
		ffnDim                                           = 3
	)
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	tokens := []any{"<unk>", "<s>", "</s>", "a", "b", "c", "d", "e"}
	scores := make([]any, len(tokens))
	for i := range scores {
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "nemotron_h"},
		{"nemotron_h.embedding_length", ggufU32, uint32(dim)},
		{"nemotron_h.block_count", ggufU32, uint32(3)},
		{"nemotron_h.context_length", ggufU32, uint32(64)},
		{"nemotron_h.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		// Dense Nemotron-H uses a global Q-head count, while KV/FFN are
		// scheduled per layer. This matches NVIDIA's 4B GGUF layout.
		{"nemotron_h.attention.head_count", ggufU32, uint32(2)},
		{"nemotron_h.attention.head_count_kv", ggufArr, ggufArray{ggufU32, []any{uint32(0), uint32(0), uint32(1)}}},
		{"nemotron_h.feed_forward_length", ggufArr, ggufArray{ggufU32, []any{uint32(0), uint32(ffnDim), uint32(0)}}},
		{"nemotron_h.attention.key_length", ggufU32, uint32(2)},
		{"nemotron_h.attention.value_length", ggufU32, uint32(2)},
		{"nemotron_h.ssm.conv_kernel", ggufU32, uint32(ssmConv)},
		{"nemotron_h.ssm.inner_size", ggufU32, uint32(ssmInner)},
		{"nemotron_h.ssm.state_size", ggufU32, uint32(ssmState)},
		{"nemotron_h.ssm.time_step_rank", ggufU32, uint32(ssmHeads)},
		{"nemotron_h.ssm.group_count", ggufU32, uint32(ssmGroups)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1), vec("output_norm.weight", dim), f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim), f32t("blk.0.ssm_in.weight", 2*ssmInner+2*ssmGroups*ssmState+ssmHeads, dim, 3),
		f32t("blk.0.ssm_conv1d.weight", ssmInner+2*ssmGroups*ssmState, ssmConv, 4), vec("blk.0.ssm_conv1d.bias", ssmInner+2*ssmGroups*ssmState),
		vec("blk.0.ssm_dt.bias", ssmHeads), vec("blk.0.ssm_a", ssmHeads), vec("blk.0.ssm_d", ssmHeads), vec("blk.0.ssm_norm.weight", ssmInner), f32t("blk.0.ssm_out.weight", dim, ssmInner, 5),
		vec("blk.1.attn_norm.weight", dim), f32t("blk.1.ffn_up.weight", ffnDim, dim, 6), f32t("blk.1.ffn_down.weight", dim, ffnDim, 7), vec("blk.1.ffn_up.bias", ffnDim), vec("blk.1.ffn_down.bias", dim),
		vec("blk.2.attn_norm.weight", dim), f32t("blk.2.attn_q.weight", 4, dim, 8), f32t("blk.2.attn_k.weight", 2, dim, 9), f32t("blk.2.attn_v.weight", 2, dim, 10), f32t("blk.2.attn_output.weight", dim, 4, 11), vec("blk.2.attn_output.bias", dim),
	}
	r, err := RunnerFromGGUFBytes(buildGGUF(3, kvs, tensors))
	if err != nil {
		t.Fatal(err)
	}
	if r.kind != loadedNemotronH || r.config.NHeads != 2 || r.config.NKVHeads != 1 {
		t.Fatalf("runner config = kind %d %+v", r.kind, r.config)
	}
	if r.nemotronH.Layers[1].Kind != nemotronDenseFFN {
		t.Fatalf("layer 1 kind = %d, want dense FFN", r.nemotronH.Layers[1].Kind)
	}
	cache, buf := r.generationWorkspace(4)
	var logits []float32
	for pos, token := range []uint32{1, 3, 4} {
		r.forwardTokenInto(cache, buf, token, pos, &logits)
	}
	if len(logits) != vocab {
		t.Fatalf("logits length = %d, want %d", len(logits), vocab)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d is not finite: %v", i, v)
		}
	}
}

func TestNemotronDenseFFNUsesReLUSquaredAndBiases(t *testing.T) {
	w := NemotronDenseFFNWeights{
		// Up*x for x=[1,1] gives [-1,2]; the second bias changes it to 1.
		Up:       Weight{F32: []float32{-2, 1, 2, 0}},
		Down:     Weight{F32: []float32{1, 2, -1, 1}},
		UpBias:   []float32{0, -1},
		DownBias: []float32{0.5, -0.5},
	}
	buf := &DecodeBuffer{}
	nemotronDenseFFNForward(w, []float32{1, 1}, buf)
	// ReLU²([-1, 1]) = [0, 1], then Down + bias = [2.5, 0.5].
	if len(buf.Proj) != 2 || buf.Proj[0] != 2.5 || buf.Proj[1] != 0.5 {
		t.Fatalf("dense FFN result = %v, want [2.5 0.5]", buf.Proj)
	}
}

func TestNemotronMoELatentProjectionPreservesInputForAllExperts(t *testing.T) {
	// The latent path has a different expert input/output width than the model
	// width. With two selected experts this also verifies that one expert's
	// output cannot overwrite the latent input used by the next one.
	cfg := Config{
		Dim:                2,
		ExpertCount:        2,
		ExpertUsedCount:    2,
		ExpertWeightsNorm:  true,
		ExpertWeightsScale: 1,
	}
	w := NemotronMoEWeights{
		Router:    Weight{F32: []float32{0, 0, 0, 0}},
		LatentIn:  &Weight{F32: []float32{1, 1}}, // [1, 2] => latent [3]
		LatentOut: &Weight{F32: []float32{1, 2}}, // latent [22.5] => [22.5, 45]
		Up: ExpertWeight{
			Weight: Weight{F32: []float32{1, 2}}, // expert outputs [3], [6]
			Input:  1, Output: 1, Experts: 2,
		},
		Down: ExpertWeight{
			Weight: Weight{F32: []float32{1, 1}},
			Input:  1, Output: 1, Experts: 2,
		},
	}
	buf := &DecodeBuffer{}
	nemotronMoEForward(cfg, w, []float32{1, 2}, buf)
	// ReLU²([3]) and ReLU²([6]) are 9 and 36; equal sigmoid routing
	// probabilities normalize to 0.5, yielding (9+36)/2 = 22.5 before
	// the latent-up projection.
	if len(buf.Proj) != 2 || math.Abs(float64(buf.Proj[0]-22.5)) > 1e-5 || math.Abs(float64(buf.Proj[1]-45)) > 1e-5 {
		t.Fatalf("latent MoE output = %v, want [22.5 45]", buf.Proj)
	}
}
