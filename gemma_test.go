package main

import (
	"math"
	"testing"
)

// buildTinyGemmaGGUF builds a minimal 2-layer gemma2-flavored model carrying
// every Gemma-family mechanism the runtime implements: QK-norm and
// post-attention/post-FFN norm tensors, softcapping metadata, a sliding
// window, and the <start_of_turn>/<end_of_turn> chat vocabulary.
func buildTinyGemmaGGUF() []byte {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads // 4
		hidden = 16
		vocab  = 20
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{"<pad>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>", "▁", "\n"}
	for i := 0; i < vocab; i++ {
		if i < len(special) {
			toks[i] = special[i]
		} else {
			toks[i] = string(rune('a' + (i - len(special))))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "gemma2"},
		{"general.name", ggufStr, "tiny-gemma"},
		{"gemma2.embedding_length", ggufU32, uint32(dim)},
		{"gemma2.block_count", ggufU32, uint32(2)},
		{"gemma2.attention.head_count", ggufU32, uint32(heads)},
		{"gemma2.attention.head_count_kv", ggufU32, uint32(kv)},
		{"gemma2.attention.key_length", ggufU32, uint32(hdim)},
		{"gemma2.attention.value_length", ggufU32, uint32(hdim)},
		{"gemma2.feed_forward_length", ggufU32, uint32(hidden)},
		{"gemma2.context_length", ggufU32, uint32(1024)},
		{"gemma2.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-6)},
		{"gemma2.rope.freq_base", ggufF32, float32(10000)},
		{"gemma2.rope.dimension_count", ggufU32, uint32(hdim)},
		{"gemma2.attention.sliding_window", ggufU32, uint32(4)},
		{"gemma2.attn_logit_softcapping", ggufF32, float32(50)},
		{"gemma2.final_logit_softcapping", ggufF32, float32(30)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
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
		// output tied to embeddings (typical for Gemma) — no output.weight.
	}
	for l := 0; l < 2; l++ {
		p := "blk." + string(rune('0'+l)) + "."
		tensors = append(tensors,
			vec(p+"attn_norm.weight", dim),
			f32t(p+"attn_q.weight", heads*hdim, dim, 3+l),
			f32t(p+"attn_k.weight", kv*hdim, dim, 4+l),
			f32t(p+"attn_v.weight", kv*hdim, dim, 5+l),
			f32t(p+"attn_output.weight", dim, heads*hdim, 6+l),
			vec(p+"attn_q_norm.weight", hdim),
			vec(p+"attn_k_norm.weight", hdim),
			vec(p+"post_attention_norm.weight", dim),
			vec(p+"ffn_norm.weight", dim),
			f32t(p+"ffn_gate.weight", hidden, dim, 7+l),
			f32t(p+"ffn_up.weight", hidden, dim, 8+l),
			f32t(p+"ffn_down.weight", dim, hidden, 9+l),
			vec(p+"post_ffw_norm.weight", dim),
		)
	}
	return buildGGUF(3, kvs, tensors)
}

func TestGemmaConfigMechanics(t *testing.T) {
	g, err := ParseGGUFQuiet(buildTinyGemmaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	cfg := ConfigFromGGUF(g)
	if !cfg.UseGELU {
		t.Fatal("gemma2 should select GELU")
	}
	want := float32(math.Sqrt(8))
	if math.Abs(float64(cfg.EmbeddingScale-want)) > 1e-6 {
		t.Fatalf("EmbeddingScale = %v, want sqrt(dim) = %v", cfg.EmbeddingScale, want)
	}
	if cfg.AttnLogitSoftcap != 50 || cfg.FinalLogitSoftcap != 30 {
		t.Fatalf("softcaps = %v/%v, want 50/30", cfg.AttnLogitSoftcap, cfg.FinalLogitSoftcap)
	}
	// gemma2 default: alternate SWA/global (period 2, layer 0 windowed).
	if len(cfg.SWAPattern) != 2 || !cfg.SWAPattern[0] || cfg.SWAPattern[1] {
		t.Fatalf("SWAPattern = %v, want [true false]", cfg.SWAPattern)
	}
	if !cfg.layerUsesSWA(0) || cfg.layerUsesSWA(1) {
		t.Fatal("layerUsesSWA should be true for layer 0, false for layer 1")
	}
}

func TestGemmaLoadsOptionalNormsAndGenerates(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyGemmaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	for l, layer := range r.gemma4.Standard.Layers {
		if layer.AttnQNorm == nil || layer.AttnKNorm == nil || layer.PostAttnNorm == nil || layer.PostFFNNorm == nil {
			t.Fatalf("layer %d: optional norms not loaded: %+v", l, layer)
		}
	}
	// Llama-style models must keep nil optional norms (zero-filled would
	// destroy activations).
	lr, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	if lr.standard.Layers[0].AttnQNorm != nil || lr.standard.Layers[0].PostAttnNorm != nil {
		t.Fatal("llama layer should have nil optional norms")
	}
	if !lr.canBatchPrefill() {
		t.Fatal("plain llama model must still batch-prefill after the gemma guards")
	}

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 6
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	a, err := r.Generate("a b c", opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := r.Generate("a b c", opts)
	if err != nil {
		t.Fatal(err)
	}
	if a.Text != b.Text {
		t.Fatalf("gemma greedy output not deterministic: %q vs %q", a.Text, b.Text)
	}
	// Final softcap bounds every logit to (-30, 30): verify via a direct
	// forward call.
	kDim, vDim, mh, mk, mv := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, 4)
	buf := NewDecodeBuffer(r.config, mh, mk, mv)
	logits := []float32{}
	ForwardInto(r.config, r.gemma4.Standard, cache, buf, 3, 0, &logits)
	for i, v := range logits {
		if math.IsNaN(float64(v)) || v <= -30 || v >= 30 {
			t.Fatalf("logit[%d] = %v, want finite in (-30, 30) under final softcap", i, v)
		}
	}
}

func TestGemmaChatTemplate(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyGemmaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	if kind := r.chatTemplateKind(); kind != "gemma-chat" {
		t.Fatalf("chatTemplateKind = %q, want gemma-chat", kind)
	}
	start, _ := r.tok.SpecialID("<start_of_turn>")
	end, _ := r.tok.SpecialID("<end_of_turn>")
	tokens, ok := r.renderGemmaMessages([]ChatMessage{UserMessage("abc"), AssistantMessage("de"), UserMessage("f")}, "gg")
	if !ok {
		t.Fatal("renderGemmaMessages ok=false")
	}
	if tokens[0] != r.tok.BOSID {
		t.Fatalf("expected BOS first, got %d", tokens[0])
	}
	if got := countToken(tokens, start); got != 4 { // 3 turns + trailing model turn
		t.Fatalf("<start_of_turn> count = %d, want 4", got)
	}
	if got := countToken(tokens, end); got != 3 {
		t.Fatalf("<end_of_turn> count = %d, want 3", got)
	}
	// <end_of_turn> also terminates generation for gemma archs.
	if !r.isStopToken(end) {
		t.Fatal("<end_of_turn> should be a stop token for gemma")
	}
}

func TestGeluTanhValues(t *testing.T) {
	// Reference values of gelu_pytorch_tanh.
	cases := []struct{ x, want float64 }{
		{0, 0},
		{1, 0.8411919906082768},
		{-1, -0.15880800939172324},
		{3, 2.9963627039350904},
	}
	for _, c := range cases {
		got := float64(geluTanh(float32(c.x)))
		if math.Abs(got-c.want) > 1e-5 {
			t.Fatalf("geluTanh(%v) = %v, want %v", c.x, got, c.want)
		}
	}
}

func TestSoftcapF32Bounds(t *testing.T) {
	v := []float32{-1000, -30, 0, 30, 1000}
	softcapF32(v, 30)
	if v[2] != 0 {
		t.Fatalf("softcap(0) = %v, want 0", v[2])
	}
	for i, x := range v {
		if x <= -30.0001 || x >= 30.0001 {
			t.Fatalf("v[%d] = %v, want within (-30, 30)", i, x)
		}
	}
	if !(v[0] < -29.9 && v[4] > 29.9) {
		t.Fatalf("large inputs should saturate near the cap: %v", v)
	}
	if math.Abs(float64(v[1])+float64(v[3])) > 1e-5 {
		t.Fatalf("softcap should be odd-symmetric: %v vs %v", v[1], v[3])
	}
}

func TestSwaPatternExplicitArrayWins(t *testing.T) {
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "gemma4"},
		{"gemma4.block_count", ggufU32, uint32(4)},
		{"gemma4.attention.sliding_window_pattern", ggufArr, ggufArray{ggufBool, []any{true, true, false, true}}},
	}
	data := buildGGUF(3, kvs, nil)
	g, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	pattern := swaPattern(g, "gemma4", 4)
	want := []bool{true, true, false, true}
	if len(pattern) != len(want) {
		t.Fatalf("pattern = %v", pattern)
	}
	for i := range want {
		if pattern[i] != want[i] {
			t.Fatalf("pattern[%d] = %v, want %v", i, pattern[i], want[i])
		}
	}
	// Without the array, gemma4 synthesizes the 5:1 default.
	g2, _ := ParseGGUFQuiet(buildGGUF(3, []ggufKV{{"general.architecture", ggufStr, "gemma4"}}, nil))
	p2 := swaPattern(g2, "gemma4", 12)
	if len(p2) != 12 || p2[5] || p2[11] || !p2[0] || !p2[4] || !p2[6] {
		t.Fatalf("default gemma4 pattern = %v, want 5 local then 1 global", p2)
	}
}

func TestGemmaSystemPromptFoldedIntoFirstUserTurn(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyGemmaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	with, _ := r.renderGemmaMessages([]ChatMessage{UserMessage("abc")}, "dd")
	without, _ := r.renderGemmaMessages([]ChatMessage{UserMessage("abc")}, "")
	if len(with) <= len(without) {
		t.Fatalf("system prompt should lengthen the first user turn: %d vs %d", len(with), len(without))
	}
	start, _ := r.tok.SpecialID("<start_of_turn>")
	if countToken(with, start) != countToken(without, start) {
		t.Fatal("system prompt must not add a turn")
	}
}
