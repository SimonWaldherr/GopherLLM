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

func buildTinyOlmo2GGUF() []byte {
	return buildTinyOlmo2GGUFWithOutput(true)
}

func buildTinyOlmo2GGUFWithOutput(withOutput bool) []byte {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads
		hidden = 16
		vocab  = 18
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	for i := 0; i < vocab; i++ {
		if i == 0 {
			toks[i] = "<unk>"
		} else if i == 1 {
			toks[i] = "<|endoftext|>"
		} else {
			toks[i] = string(rune('a' + i - 2))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "olmo2"},
		{"general.name", ggufStr, "tiny-olmo3-layout"},
		{"olmo2.embedding_length", ggufU32, uint32(dim)},
		{"olmo2.block_count", ggufU32, uint32(1)},
		{"olmo2.attention.head_count", ggufU32, uint32(heads)},
		{"olmo2.attention.head_count_kv", ggufU32, uint32(kv)},
		{"olmo2.attention.key_length", ggufU32, uint32(hdim)},
		{"olmo2.attention.value_length", ggufU32, uint32(hdim)},
		{"olmo2.feed_forward_length", ggufU32, uint32(hidden)},
		{"olmo2.context_length", ggufU32, uint32(1024)},
		{"olmo2.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-6)},
		{"olmo2.attention.sliding_window", ggufU32, uint32(128)},
		{"olmo2.attention.sliding_window_pattern", ggufU32, uint32(4)},
		{"olmo2.rope.freq_base", ggufF32, float32(10000)},
		{"olmo2.rope.freq_base_swa", ggufF32, float32(500000)},
		{"olmo2.rope.dimension_count", ggufU32, uint32(hdim)},
		{"tokenizer.ggml.model", ggufStr, "gpt2"},
		{"tokenizer.ggml.pre", ggufStr, "olmo"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.add_bos_token", ggufBool, false},
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 21),
		vec("output_norm.weight", dim),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 22),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 23),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 24),
		vec("blk.0.attn_q_norm.weight", heads*hdim),
		vec("blk.0.attn_k_norm.weight", kv*hdim),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 25),
		vec("blk.0.post_attention_norm.weight", dim),
		f32t("blk.0.ffn_gate.weight", hidden, dim, 26),
		f32t("blk.0.ffn_up.weight", hidden, dim, 27),
		f32t("blk.0.ffn_down.weight", dim, hidden, 28),
		vec("blk.0.post_ffw_norm.weight", dim),
	}
	if withOutput {
		tensors = append(tensors, f32t("output.weight", vocab, dim, 29))
	}
	return buildGGUF(3, kvs, tensors)
}

func buildTinyPhi2GGUF() []byte {
	return buildTinyPhi2GGUFWithOutputBias(true)
}

func buildTinyPhi2GGUFWithOutputBias(withOutputBias bool) []byte {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads
		hidden = 16
		vocab  = 18
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	for i := 0; i < vocab; i++ {
		if i == 0 {
			toks[i] = "<unk>"
		} else if i == 1 {
			toks[i] = "<|endoftext|>"
		} else {
			toks[i] = string(rune('a' + i - 2))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "phi2"},
		{"general.name", ggufStr, "tiny-phi2"},
		{"phi2.embedding_length", ggufU32, uint32(dim)},
		{"phi2.block_count", ggufU32, uint32(1)},
		{"phi2.attention.head_count", ggufU32, uint32(heads)},
		{"phi2.attention.head_count_kv", ggufU32, uint32(kv)},
		{"phi2.attention.key_length", ggufU32, uint32(hdim)},
		{"phi2.attention.value_length", ggufU32, uint32(hdim)},
		{"phi2.feed_forward_length", ggufU32, uint32(hidden)},
		{"phi2.context_length", ggufU32, uint32(1024)},
		{"phi2.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		{"phi2.rope.freq_base", ggufF32, float32(10000)},
		{"phi2.rope.dimension_count", ggufU32, uint32(hdim)},
		{"tokenizer.ggml.model", ggufStr, "gpt2"},
		{"tokenizer.ggml.pre", ggufStr, "gpt2"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.add_bos_token", ggufBool, false},
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, values []float32) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(len(values))}, dtype: GGMLTypeF32, data: f32Bytes(values)}
	}
	zeros := func(n int) []float32 { return make([]float32, n) }
	outputBias := make([]float32, vocab)
	for i := range outputBias {
		outputBias[i] = float32(i-4) * 0.01
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 31),
		vec("output_norm.weight", onesF32(dim)),
		vec("output_norm.bias", zeros(dim)),
		f32t("output.weight", vocab, dim, 32),
		vec("blk.0.attn_norm.weight", onesF32(dim)),
		vec("blk.0.attn_norm.bias", zeros(dim)),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 33),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 34),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 35),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 36),
		vec("blk.0.attn_output.bias", zeros(dim)),
		f32t("blk.0.ffn_up.weight", hidden, dim, 37),
		vec("blk.0.ffn_up.bias", zeros(hidden)),
		f32t("blk.0.ffn_down.weight", dim, hidden, 38),
		vec("blk.0.ffn_down.bias", zeros(dim)),
	}
	if withOutputBias {
		tensors = append(tensors, vec("output.bias", outputBias))
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

func TestOlmo2AndOlmo3PostNormGraph(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyOlmo2GGUF())
	if err != nil {
		t.Fatal(err)
	}
	if r.Architecture() != "olmo2" || !r.config.usesFullProjectionQKNorm() {
		t.Fatalf("unexpected OLMo config: %+v", r.config)
	}
	if r.config.RopeThetaSWA != 500000 {
		t.Fatalf("SWA RoPE base = %v, want 500000", r.config.RopeThetaSWA)
	}
	layer := r.standard.Layers[0]
	if layer.AttnNorm != nil || layer.FFNNorm != nil {
		t.Fatal("OLMo 2/3 must not fabricate pre-normalization tensors")
	}
	if len(layer.AttnQNorm) != r.config.NHeads*r.config.HeadDim ||
		len(layer.AttnKNorm) != r.config.NKVHeads*r.config.HeadDim {
		t.Fatal("OLMo full-projection QK norms were not loaded")
	}
	if !r.canBatchPrefill() {
		t.Fatal("dense OLMo 2/3 should support batched prefill")
	}
	tokens := r.tok.Encode("abcdefghijklm")
	if len(tokens) < 2 {
		t.Fatalf("synthetic OLMo prompt too short: %v", tokens)
	}
	assertStandardBatchParity(t, r, tokens)
}

func TestOlmoFullProjectionQKNormIsNotPerHead(t *testing.T) {
	config := Config{Arch: "olmo2", HeadDim: 2, NHeads: 2, NKVHeads: 2, RMSNormEps: 1e-6}
	layer := LayerWeights{
		AttnQNorm: onesF32(4),
		AttnKNorm: onesF32(4),
	}
	q := []float32{1, 1, 10, 10}
	k := append([]float32(nil), q...)
	normalizeProjectedQKInPlace(config, layer, q, k)
	if q[0] >= 0.2 || q[2] <= 1 {
		t.Fatalf("OLMo Q norm looks per-head instead of full-projection: %v", q)
	}
	for i := range q {
		if q[i] != k[i] {
			t.Fatalf("Q/K norm mismatch at %d: %v vs %v", i, q[i], k[i])
		}
	}
}

func TestOlmo3UsesSeparateUnscaledSWARope(t *testing.T) {
	config := Config{
		Arch:                      "olmo2",
		Dim:                       8,
		HiddenDim:                 16,
		NHeads:                    2,
		NKVHeads:                  2,
		HeadDim:                   4,
		ValueDim:                  4,
		RopeDimensionCount:        4,
		RopeTheta:                 10000,
		RopeThetaSWA:              500000,
		RopeScalingType:           "yarn",
		RopeScalingFactor:         4,
		RopeOriginalContextLength: 128,
		MaxSeqLen:                 512,
	}
	buf := NewDecodeBuffer(config, 4, 2, 4)
	if len(buf.RopeInvFreq) != 2 || len(buf.RopeSWAInvFreq) != 2 {
		t.Fatalf("rope table sizes global=%d swa=%d", len(buf.RopeInvFreq), len(buf.RopeSWAInvFreq))
	}
	if buf.RopeSWAMscale != 1 {
		t.Fatalf("SWA OLMo 3 RoPE magnitude = %v, want 1", buf.RopeSWAMscale)
	}
	wantSWA := float32(1 / 707.1067811865476) // 1 / 500000^(2/4)
	if d := buf.RopeSWAInvFreq[1] - wantSWA; d < -1e-6 || d > 1e-6 {
		t.Fatalf("SWA inverse frequency = %v, want %v", buf.RopeSWAInvFreq[1], wantSWA)
	}
	if d := buf.RopeInvFreq[1] - buf.RopeSWAInvFreq[1]; d > -1e-7 && d < 1e-7 {
		t.Fatalf("global and local OLMo 3 RoPE tables unexpectedly match: %v", buf.RopeInvFreq)
	}
}

func TestOlmo2RequiresUntiedOutput(t *testing.T) {
	_, err := RunnerFromGGUFBytes(buildTinyOlmo2GGUFWithOutput(false))
	if err == nil || !strings.Contains(err.Error(), "olmo2 requires output.weight") {
		t.Fatalf("error = %v, want required output diagnostic", err)
	}
}

func TestPhi2LoadsNativeParallelBiasedGraph(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyPhi2GGUF())
	if err != nil {
		t.Fatal(err)
	}
	if !r.config.UseLayerNorm || !r.config.ParallelResidual || !r.config.UseGELU {
		t.Fatalf("unexpected Phi-2 graph flags: %+v", r.config)
	}
	layer := r.standard.Layers[0]
	if layer.FFNNorm != nil || layer.W1.F32 != nil || layer.W1.Raw != nil || layer.W3.F32 == nil {
		t.Fatal("Phi-2 must load one shared norm and an ungated up projection")
	}
	if len(layer.AttnNormBias) != r.config.Dim ||
		len(layer.FFNUpBias) != r.config.HiddenDim ||
		len(layer.FFNDownBias) != r.config.Dim ||
		len(r.standard.OutputBias) != r.config.VocabSize {
		t.Fatal("Phi-2 mandatory biases were not loaded")
	}
	if !r.canBatchPrefill() {
		t.Fatal("Phi-2 should support batched prefill")
	}
	tokens := r.tok.Encode("abcdefghijklm")
	assertStandardBatchParity(t, r, tokens)
}

func TestPhi2OutputBiasIsApplied(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyPhi2GGUF())
	if err != nil {
		t.Fatal(err)
	}
	kDim, vDim, mh, mk, mv := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 2)
	buf := NewDecodeBuffer(r.config, mh, mk, mv)
	ForwardBodyInto(r.config, r.standard, cache, buf, 2, 0)
	withBias := []float32{}
	ProjectLogitsInto(r.config, r.standard, buf, &withBias)
	withoutWeights := r.standard
	withoutWeights.OutputBias = nil
	withoutBias := []float32{}
	ProjectLogitsInto(r.config, withoutWeights, buf, &withoutBias)
	for i, bias := range r.standard.OutputBias {
		if d := (withBias[i] - withoutBias[i]) - bias; d < -1e-6 || d > 1e-6 {
			t.Fatalf("logit %d bias delta = %v, want %v", i, withBias[i]-withoutBias[i], bias)
		}
	}
}

func TestPhi2RejectsMissingOutputBias(t *testing.T) {
	_, err := RunnerFromGGUFBytes(buildTinyPhi2GGUFWithOutputBias(false))
	if err == nil || !strings.Contains(err.Error(), "phi2 requires 18-element output.bias") {
		t.Fatalf("error = %v, want output bias diagnostic", err)
	}
}

func TestGreedyArgmaxIncludesOutputBias(t *testing.T) {
	config := Config{Dim: 2, VocabSize: 2, LogitScale: 1}
	weights := ModelWeights{
		Output:     Weight{F32: make([]float32, 4), Rows: 2, Cols: 2},
		OutputBias: []float32{-1, 2},
	}
	buf := &DecodeBuffer{XN: []float32{3, 4}, Logits: make([]float32, 2)}
	got, ok := argmaxOutputTokenInto(config, weights, buf, nil)
	if !ok || got != 1 {
		t.Fatalf("argmax = %d, ok=%v; want biased token 1", got, ok)
	}
}
