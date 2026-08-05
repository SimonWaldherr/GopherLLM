package gopherllm

import (
	"io"
	"strings"
	"testing"
)

// TestLoadModelRejectsZeroLayers guards against a GGUF whose block_count is
// missing or zero: without this check LoadModel would proceed to allocate a
// zero-length layer slice and fail confusingly (or silently produce a
// no-op model) deep inside the forward pass instead of at load time.
func TestLoadModelRejectsZeroLayers(t *testing.T) {
	const dim, heads, kv, hdim, hidden, vocab = 8, 2, 2, 4, 16, 16
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	for i := range toks {
		toks[i] = string(rune('a' + i))
		scores[i] = float32(0)
	}
	data := buildGGUF(3, []ggufKV{
		{"general.architecture", ggufStr, "llama"},
		{"general.name", ggufStr, "zero-layer"},
		{"llama.embedding_length", ggufU32, uint32(dim)},
		{"llama.block_count", ggufU32, uint32(0)},
		{"llama.attention.head_count", ggufU32, uint32(heads)},
		{"llama.attention.head_count_kv", ggufU32, uint32(kv)},
		{"llama.attention.key_length", ggufU32, uint32(hdim)},
		{"llama.attention.value_length", ggufU32, uint32(hdim)},
		{"llama.feed_forward_length", ggufU32, uint32(hidden)},
		{"llama.context_length", ggufU32, uint32(1024)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
	}, nil)

	gguf, err := ParseGGUF(data)
	if err != nil {
		t.Fatalf("ParseGGUF: %v", err)
	}
	_, _, err = LoadModel(data, gguf, false, false, false, io.Discard)
	if err == nil {
		t.Fatal("LoadModel: want error for block_count=0, got nil")
	}
	if !strings.Contains(err.Error(), "layers=0") {
		t.Fatalf("LoadModel error = %q, want it to mention layers=0", err.Error())
	}
}
