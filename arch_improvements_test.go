package gopherllm

import (
	"strings"
	"testing"
)

// buildTinyQwen3GGUF builds a minimal qwen3-flavored model: qwen2-style
// llama graph plus per-head QK-norm tensors and a ChatML vocabulary — the
// shape the DeepSeek-R1-0528 Qwen3 distills use.
func buildTinyQwen3GGUF() []byte {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads
		hidden = 16
		vocab  = 22
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{"<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "▁", "\n"}
	for i := 0; i < vocab; i++ {
		if i < len(special) {
			toks[i] = special[i]
		} else {
			toks[i] = string(rune('a' + (i - len(special))))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, "qwen3"},
		{"general.name", ggufStr, "tiny-qwen3"},
		{"qwen3.embedding_length", ggufU32, uint32(dim)},
		{"qwen3.block_count", ggufU32, uint32(1)},
		{"qwen3.attention.head_count", ggufU32, uint32(heads)},
		{"qwen3.attention.head_count_kv", ggufU32, uint32(kv)},
		{"qwen3.attention.key_length", ggufU32, uint32(hdim)},
		{"qwen3.attention.value_length", ggufU32, uint32(hdim)},
		{"qwen3.feed_forward_length", ggufU32, uint32(hidden)},
		{"qwen3.context_length", ggufU32, uint32(1024)},
		{"qwen3.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-6)},
		{"qwen3.rope.freq_base", ggufF32, float32(1000000)},
		{"qwen3.rope.dimension_count", ggufU32, uint32(hdim)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.add_bos_token", ggufBool, false},
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
		vec("blk.0.attn_norm.weight", dim),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 3),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 4),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 5),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		vec("blk.0.attn_q_norm.weight", hdim), // the qwen3 signature
		vec("blk.0.attn_k_norm.weight", hdim),
		vec("blk.0.ffn_norm.weight", dim),
		f32t("blk.0.ffn_gate.weight", hidden, dim, 7),
		f32t("blk.0.ffn_up.weight", hidden, dim, 8),
		f32t("blk.0.ffn_down.weight", dim, hidden, 9),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestQwen3LoadsWithQKNormAndChatML(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyQwen3GGUF())
	if err != nil {
		t.Fatal(err)
	}
	if r.Architecture() != "qwen3" {
		t.Fatalf("arch = %q", r.Architecture())
	}
	layer := r.standard.Layers[0]
	if layer.AttnQNorm == nil || layer.AttnKNorm == nil {
		t.Fatal("qwen3 QK-norm tensors not loaded")
	}
	if r.config.UseGELU {
		t.Fatal("qwen3 must keep SiLU (GELU is gemma-only)")
	}
	if ropeInterleaved("qwen3") {
		t.Fatal("qwen3 uses NeoX (non-interleaved) rope")
	}
	if kind := r.chatTemplateKind(); kind != "chatml" {
		t.Fatalf("template kind = %q, want chatml", kind)
	}
	// QK-norm forces the fully-featured per-token prefill path.
	if r.canBatchPrefill() {
		t.Fatal("QK-norm models must not take the batched prefill path")
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 4
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	a, err := r.Generate("a b", opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := r.Generate("a b", opts)
	if err != nil {
		t.Fatal(err)
	}
	if a.Text != b.Text {
		t.Fatalf("qwen3 greedy not deterministic: %q vs %q", a.Text, b.Text)
	}
}

func TestExtractThinkForcedOpenLeadingClose(t *testing.T) {
	// DeepSeek-R1-0528-style: the prompt already contains <think>, so output
	// starts mid-reasoning and the first tag we see is </think>.
	content, reasoning := extractThink("let me reason about this...</think>The answer is 4.")
	if content != "The answer is 4." {
		t.Fatalf("content = %q", content)
	}
	if reasoning != "let me reason about this..." {
		t.Fatalf("reasoning = %q", reasoning)
	}
	// A normal <think>...</think> AFTER the forced-open block still works.
	content, reasoning = extractThink("pre</think>mid<think>more</think>post")
	if content != "midpost" {
		t.Fatalf("content = %q", content)
	}
	if reasoning != "pre\n\nmore" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	// No think markers at all: unchanged passthrough.
	content, reasoning = extractThink("plain")
	if content != "plain" || reasoning != "" {
		t.Fatalf("content=%q reasoning=%q", content, reasoning)
	}
}

func TestMistralTrailingAssistantIsOpenContinuation(t *testing.T) {
	tok := newMistralToolTokenizer()
	r := &Runner{tok: tok, arch: "ministral"}
	// History assistant message (not last) still closes with EOS...
	closed, _ := r.renderMistralInstMessages([]ChatMessage{
		UserMessage("hi"), AssistantMessage("yo"), UserMessage("more"),
	}, "", nil)
	if countToken(closed, tok.EOSID) != 1 {
		t.Fatalf("history assistant should close with EOS: %v", closed)
	}
	// ...but a TRAILING assistant message stays open for continuation.
	open, _ := r.renderMistralInstMessages([]ChatMessage{
		UserMessage("hi"), AssistantMessage("Once upon a"),
	}, "", nil)
	if countToken(open, tok.EOSID) != 0 {
		t.Fatalf("trailing assistant must not emit EOS (prefill continuation): %v", open)
	}
	// The prefix text must be the last thing rendered.
	tail := decodeAll(tok, open[len(open)-2:])
	if !strings.Contains(decodeAll(tok, open), "Once upon a") || strings.Contains(tail, "[INST]") {
		t.Fatalf("continuation prefix not trailing: %q", decodeAll(tok, open))
	}
}