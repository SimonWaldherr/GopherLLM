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
