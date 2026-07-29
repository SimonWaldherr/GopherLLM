package gopherllm

import (
	"math"
	"strings"
	"testing"
)

func buildTinyMamba2GGUF(ffnDim int) []byte {
	const (
		dim, vocab                                       = 4, 8
		ssmInner, ssmHeads, ssmGroups, ssmState, ssmConv = 4, 2, 1, 1, 2
	)
	channels := ssmInner + 2*ssmGroups*ssmState
	inRows := 2*ssmInner + 2*ssmGroups*ssmState + ssmHeads
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	negVec := func(name string, n int) ggufTensor {
		v := make([]float32, n)
		for i := range v {
			v[i] = -1
		}
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(v)}
	}
	tokens := []any{"<unk>", "<s>", "</s>", "a", "b", "c", "d", "e"}
	scores := make([]any, len(tokens))
	for i := range scores {
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "mamba2"},
		{"mamba2.embedding_length", ggufU32, uint32(dim)},
		{"mamba2.block_count", ggufU32, uint32(1)},
		{"mamba2.context_length", ggufU32, uint32(64)},
		{"mamba2.attention.head_count", ggufU32, uint32(0)},
		{"mamba2.feed_forward_length", ggufU32, uint32(ffnDim)},
		{"mamba2.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{"mamba2.ssm.conv_kernel", ggufU32, uint32(ssmConv)},
		{"mamba2.ssm.inner_size", ggufU32, uint32(ssmInner)},
		{"mamba2.ssm.state_size", ggufU32, uint32(ssmState)},
		{"mamba2.ssm.time_step_rank", ggufU32, uint32(ssmHeads)},
		{"mamba2.ssm.group_count", ggufU32, uint32(ssmGroups)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1), vec("output_norm.weight", dim), f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim), f32t("blk.0.ssm_in.weight", inRows, dim, 3),
		f32t("blk.0.ssm_conv1d.weight", channels, ssmConv, 4), vec("blk.0.ssm_conv1d.bias", channels),
		vec("blk.0.ssm_dt.bias", ssmHeads), negVec("blk.0.ssm_a", ssmHeads), vec("blk.0.ssm_d", ssmHeads), vec("blk.0.ssm_norm.weight", ssmInner), f32t("blk.0.ssm_out.weight", dim, ssmInner, 5),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestMamba2LoaderForwardAndCacheReset(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyMamba2GGUF(0))
	if err != nil {
		t.Fatal(err)
	}
	if r.kind != loadedMamba2 || r.config.SSMInner != 4 || r.config.SSMHeads != 2 {
		t.Fatalf("runner = kind %d config %+v", r.kind, r.config)
	}
	cache, buf := r.generationWorkspace(4)
	if cache.PerPosKDim != 0 || cache.PerPosVDim != 0 || cache.Nemotron == nil {
		t.Fatalf("pure Mamba2 cache = %+v", cache)
	}
	var logits []float32
	for pos, token := range []uint32{1, 3, 4} {
		r.forwardTokenInto(cache, buf, token, pos, &logits)
	}
	if len(logits) != 8 {
		t.Fatalf("logits length = %d, want 8", len(logits))
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d is not finite: %v", i, v)
		}
	}
	cache.Nemotron.State[0] = 123
	cache.Nemotron.Conv[0] = 456
	cache2, _ := r.generationWorkspace(4)
	if cache2 != cache || cache.Nemotron.State[0] != 0 || cache.Nemotron.Conv[0] != 0 {
		t.Fatalf("recurrent workspace was not reset: cache same=%v state=%v conv=%v", cache2 == cache, cache.Nemotron.State[0], cache.Nemotron.Conv[0])
	}
}

func TestMamba2RejectsMLPVariant(t *testing.T) {
	_, err := RunnerFromGGUFBytes(buildTinyMamba2GGUF(1))
	if err == nil || !strings.Contains(err.Error(), "pure Mamba-2") {
		t.Fatalf("Mamba2 MLP error = %v", err)
	}
}

func TestMamba2AndNemotronShareGateOrdering(t *testing.T) {
	y, z := float32(2), float32(-1)
	got := mambaGatedValue(y, z)
	want := y * (z / (1 + float32(math.Exp(float64(-z)))))
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("shared Mamba2/Nemotron gate=%v, want=%v", got, want)
	}
}
