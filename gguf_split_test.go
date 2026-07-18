package gopherllm

import (
	"os"
	"path/filepath"
	"testing"
)

// splitTinyLlamaGGUF returns the two shard byte blobs that, merged, decode to
// the same model as buildTinyLlamaGGUF: shard 1 carries all the metadata plus
// the first half of tensors and split.no=0; shard 2 carries the rest of the
// tensors and split.no=1. Neither shard alone is a loadable model — only the
// merge is.
func splitTinyLlamaGGUF() (shard1, shard2 []byte) {
	const (
		dim    = 8
		heads  = 2
		kv     = 2
		hdim   = dim / heads
		hidden = 16
		vocab  = 16
	)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{"<unk>", "<s>", "</s>"}
	for i := 0; i < vocab; i++ {
		if i < len(special) {
			toks[i] = special[i]
		} else {
			toks[i] = string(rune('a' + (i - len(special))))
		}
		scores[i] = float32(0)
	}
	baseKVs := []ggufKV{
		{"general.architecture", ggufStr, "llama"},
		{"general.name", ggufStr, "tiny-split"},
		{"llama.embedding_length", ggufU32, uint32(dim)},
		{"llama.block_count", ggufU32, uint32(1)},
		{"llama.attention.head_count", ggufU32, uint32(heads)},
		{"llama.attention.head_count_kv", ggufU32, uint32(kv)},
		{"llama.attention.key_length", ggufU32, uint32(hdim)},
		{"llama.attention.value_length", ggufU32, uint32(hdim)},
		{"llama.feed_forward_length", ggufU32, uint32(hidden)},
		{"llama.context_length", ggufU32, uint32(1024)},
		{"llama.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{"llama.rope.freq_base", ggufF32, float32(10000)},
		{"llama.rope.dimension_count", ggufU32, uint32(hdim)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}

	part1Tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 3),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 4),
	}
	part2Tensors := []ggufTensor{
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 5),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		vec("blk.0.ffn_norm.weight", dim),
		f32t("blk.0.ffn_gate.weight", hidden, dim, 7),
		f32t("blk.0.ffn_up.weight", hidden, dim, 8),
		f32t("blk.0.ffn_down.weight", dim, hidden, 9),
	}

	shard1KVs := append(append([]ggufKV{}, baseKVs...),
		ggufKV{"split.no", ggufU16, uint16(0)},
		ggufKV{"split.count", ggufU16, uint16(2)},
	)
	shard2KVs := []ggufKV{
		{"split.no", ggufU16, uint16(1)},
		{"split.count", ggufU16, uint16(2)},
	}
	return buildGGUF(3, shard1KVs, part1Tensors), buildGGUF(3, shard2KVs, part2Tensors)
}

func TestLoadSplitGGUFMergesShards(t *testing.T) {
	dir := t.TempDir()
	shard1, shard2 := splitTinyLlamaGGUF()
	path1 := filepath.Join(dir, "tiny-split-00001-of-00002.gguf")
	path2 := filepath.Join(dir, "tiny-split-00002-of-00002.gguf")
	if err := os.WriteFile(path1, shard1, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path2, shard2, 0o644); err != nil {
		t.Fatal(err)
	}

	r, info, err := RunnerFromPath(path1)
	if err != nil {
		t.Fatalf("RunnerFromPath(shard 1): %v", err)
	}
	defer r.Close()

	if r.Architecture() != "llama" {
		t.Fatalf("architecture = %q", r.Architecture())
	}
	if name, _ := r.ModelName(); name != "tiny-split" {
		t.Fatalf("model name = %q, want merged metadata from shard 1", name)
	}
	if len(r.GGUF().Tensors) != 12 {
		t.Fatalf("merged tensor count = %d, want 12", len(r.GGUF().Tensors))
	}
	if info.FileSizeBytes <= 0 {
		t.Fatalf("merged FileSizeBytes = %d", info.FileSizeBytes)
	}

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 3
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	if _, err := r.Generate("a b c", opts); err != nil {
		t.Fatalf("generate on merged split model: %v", err)
	}
}

func TestLoadSplitGGUFMissingShardErrors(t *testing.T) {
	dir := t.TempDir()
	shard1, _ := splitTinyLlamaGGUF()
	path1 := filepath.Join(dir, "tiny-split-00001-of-00002.gguf")
	if err := os.WriteFile(path1, shard1, 0o644); err != nil {
		t.Fatal(err)
	}
	// Shard 2 deliberately not written.
	if _, _, err := RunnerFromPath(path1); err == nil {
		t.Fatal("expected an error when a sibling shard is missing")
	}
}

func TestLoadSplitGGUFRejectsNonConformingFilename(t *testing.T) {
	dir := t.TempDir()
	shard1, _ := splitTinyLlamaGGUF()
	// A file that declares split.count>1 but was renamed away from the
	// "-NNNNN-of-MMMMM.gguf" convention has no discoverable siblings.
	path := filepath.Join(dir, "not-a-split-name.gguf")
	if err := os.WriteFile(path, shard1, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := RunnerFromPath(path); err == nil {
		t.Fatal("expected an error for a non-conforming split filename")
	}
}
