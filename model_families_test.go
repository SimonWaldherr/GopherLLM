package gopherllm

import (
	"strings"
	"testing"
)

func buildTinyExaone4GGUF() []byte {
	return buildTinyExaone4GGUFWithNextN(false)
}

func buildTinyExaone4GGUFWithNextN(withNextN bool) []byte {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads
		hidden = 16
		vocab  = 24
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{
		"<unk>", "<s>", "</s>", "[|system|]", "[|user|]", "[|assistant|]",
		"[|tool|]", "[|endofturn|]", "▁", "\n",
	}
	for i := 0; i < vocab; i++ {
		if i < len(special) {
			toks[i] = special[i]
		} else {
			toks[i] = string(rune('a' + i - len(special)))
		}
		scores[i] = float32(0)
	}
	blockCount := uint32(1)
	if withNextN {
		blockCount = 2
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "exaone4"},
		{"general.name", ggufStr, "tiny-exaone4"},
		{"exaone4.embedding_length", ggufU32, uint32(dim)},
		{"exaone4.block_count", ggufU32, blockCount},
		{"exaone4.attention.head_count", ggufU32, uint32(heads)},
		{"exaone4.attention.head_count_kv", ggufU32, uint32(kv)},
		{"exaone4.attention.key_length", ggufU32, uint32(hdim)},
		{"exaone4.attention.value_length", ggufU32, uint32(hdim)},
		{"exaone4.feed_forward_length", ggufU32, uint32(hidden)},
		{"exaone4.context_length", ggufU32, uint32(1024)},
		{"exaone4.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-6)},
		{"exaone4.rope.freq_base", ggufF32, float32(1000000)},
		{"exaone4.rope.dimension_count", ggufU32, uint32(hdim)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.add_bos_token", ggufBool, false},
	}
	if withNextN {
		kvs = append(kvs, ggufKV{"exaone4.nextn_predict_layers", ggufU32, uint32(1)})
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		f32t("output.weight", vocab, dim, 2),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 3),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 4),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 5),
		vec("blk.0.attn_q_norm.weight", hdim),
		vec("blk.0.attn_k_norm.weight", hdim),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		vec("blk.0.post_attention_norm.weight", dim),
		f32t("blk.0.ffn_gate.weight", hidden, dim, 7),
		f32t("blk.0.ffn_up.weight", hidden, dim, 8),
		f32t("blk.0.ffn_down.weight", dim, hidden, 9),
		vec("blk.0.post_ffw_norm.weight", dim),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestSmolLM3LoadsAndUsesItsRopeSchedule(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyStandardGGUF("smollm3"))
	if err != nil {
		t.Fatal(err)
	}
	if !r.canBatchPrefill() {
		t.Fatal("SmolLM3 should support batched prefill")
	}
	if !ropeInterleaved("smollm3") {
		t.Fatal("SmolLM3 requires normal/interleaved RoPE")
	}
	cfg := Config{Arch: "smollm3"}
	for il, want := range []bool{true, true, true, false, true, true, true, false} {
		if got := cfg.layerUsesRoPE(il); got != want {
			t.Fatalf("layer %d uses RoPE = %v, want %v", il, got, want)
		}
	}
	assertStandardBatchParity(t, r, r.tok.Encode(strings.Repeat("abcdefghij", 3)))
}

func TestExaone4LoadsPostNormGraphAndBatchedPrefill(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyExaone4GGUF())
	if err != nil {
		t.Fatal(err)
	}
	layer := r.standard.Layers[0]
	if layer.AttnNorm != nil || layer.FFNNorm != nil {
		t.Fatal("EXAONE 4 must not fabricate pre-normalization tensors")
	}
	if len(layer.AttnQNorm) != r.config.HeadDim || len(layer.AttnKNorm) != r.config.HeadDim ||
		len(layer.PostAttnNorm) != r.config.Dim || len(layer.PostFFNNorm) != r.config.Dim {
		t.Fatal("EXAONE 4 QK/post norms were not loaded")
	}
	if !r.canBatchPrefill() {
		t.Fatal("dense EXAONE 4 should support batched prefill")
	}
	assertStandardBatchParity(t, r, r.tok.Encode(strings.Repeat("abcdefghij", 3)))
}

func TestExaone4SkipsTrailingNextNBlocks(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyExaone4GGUFWithNextN(true))
	if err != nil {
		t.Fatal(err)
	}
	if r.config.NextNPredictLayers != 1 || r.config.NLayers != 1 || len(r.standard.Layers) != 1 {
		t.Fatalf("unexpected EXAONE 4 trunk/NextN split: config=%+v loaded=%d", r.config, len(r.standard.Layers))
	}
}

func TestExaone4LocalGlobalAndRopeSchedule(t *testing.T) {
	cfg := Config{
		Arch:          "exaone4",
		SlidingWindow: 4096,
		SWAPattern:    []bool{true, true, true, false},
	}
	for il, want := range []bool{true, true, true, false} {
		if got := cfg.layerUsesSWA(il); got != want {
			t.Fatalf("layer %d SWA = %v, want %v", il, got, want)
		}
		if got := cfg.layerUsesRoPE(il); got != want {
			t.Fatalf("layer %d RoPE = %v, want %v", il, got, want)
		}
	}
	cfg.SlidingWindow = 0
	if !cfg.layerUsesRoPE(3) {
		t.Fatal("dense EXAONE 4 without SWA must use RoPE in every layer")
	}

	gguf := &GGUFFile{Metadata: map[string]MetaValue{
		"exaone4.attention.sliding_window_pattern": {Kind: "u32", Value: uint32(3)},
	}}
	got := swaPattern(gguf, "exaone4", 6)
	want := []bool{true, true, false, true, true, false}
	for il := range want {
		if got[il] != want[il] {
			t.Fatalf("explicit period layer %d = %v, want %v", il, got[il], want[il])
		}
	}
}

func TestRenderExaone4Messages(t *testing.T) {
	tok := newChatTokenizer("[|system|]", "[|user|]", "[|assistant|]", "[|tool|]", "[|endofturn|]")
	r := &Runner{tok: tok, arch: "exaone4"}
	tokens, ok := r.renderExaone4Messages([]ChatMessage{
		UserMessage("hello"),
		AssistantMessage("hi"),
		UserMessage("again"),
	}, "be useful")
	if !ok {
		t.Fatal("ok=false")
	}
	text := decodeAll(tok, tokens)
	want := "[|system|]be useful[|endofturn|]\n" +
		"[|user|]hello\n" +
		"[|assistant|]hi[|endofturn|]\n" +
		"[|user|]again\n[|assistant|]"
	// The synthetic SentencePiece tokenizer exposes its phantom word-boundary
	// spaces when decoded token-by-token; real special-token rendering does
	// not. Compare the protocol after removing that test-only artifact.
	if strings.ReplaceAll(text, " ", "") != strings.ReplaceAll(want, " ", "") {
		t.Fatalf("rendered = %q, want %q", text, want)
	}
	if kind := r.chatTemplateKind(); kind != "exaone4-chat" {
		t.Fatalf("template kind = %q", kind)
	}
}
