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

// Out-of-core split loading is the case that matters for genuinely large
// models: every model big enough to want demand paging ships sharded, and the
// merging path would materialise all of it in anonymous memory first.
func TestOutOfCoreSplitGGUFLoadsWithoutMerging(t *testing.T) {
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

	r, info, err := RunnerFromPathWithOptions(path1, LoadOptions{OutOfCore: true})
	if err != nil {
		t.Fatalf("out-of-core split load: %v", err)
	}
	defer r.Close()

	if !info.OutOfCore {
		t.Fatal("LoadInfo.OutOfCore = false; the split path silently fell back to merging")
	}
	if len(r.GGUF().Tensors) != 12 {
		t.Fatalf("tensor count = %d, want 12 across both shards", len(r.GGUF().Tensors))
	}
	// Every tensor must resolve through its own shard view rather than a
	// merged buffer — that is the property that keeps the model off the heap.
	for _, tensor := range r.GGUF().Tensors {
		if tensor.Shard == nil {
			t.Fatalf("tensor %s has no shard view; it was merged into memory", tensor.Name)
		}
	}
	// Two distinct shard mappings must be retained, and both must still be
	// open — the weights alias them for the Runner's whole lifetime.
	if got := len(r.extraMappedFiles); got != 2 {
		t.Fatalf("retained shard mappings = %d, want 2", got)
	}

	// The model has to actually run: a wrong per-shard offset rebase would
	// still load and only show up as garbage here.
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 3
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	if _, err := r.Generate("a b c", opts); err != nil {
		t.Fatalf("generate on out-of-core split model: %v", err)
	}
}

// The out-of-core and merging paths must agree exactly. They resolve tensor
// bytes through different code (per-shard views vs one concatenated buffer),
// so a rebasing bug in either shows up as different logits for the same input.
func TestOutOfCoreSplitMatchesMergedSplit(t *testing.T) {
	write := func(dir string) string {
		shard1, shard2 := splitTinyLlamaGGUF()
		path1 := filepath.Join(dir, "tiny-split-00001-of-00002.gguf")
		path2 := filepath.Join(dir, "tiny-split-00002-of-00002.gguf")
		if err := os.WriteFile(path1, shard1, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path2, shard2, 0o644); err != nil {
			t.Fatal(err)
		}
		return path1
	}

	merged, _, err := RunnerFromPathWithOptions(write(t.TempDir()), LoadOptions{})
	if err != nil {
		t.Fatalf("merged load: %v", err)
	}
	defer merged.Close()

	ooc, _, err := RunnerFromPathWithOptions(write(t.TempDir()), LoadOptions{OutOfCore: true})
	if err != nil {
		t.Fatalf("out-of-core load: %v", err)
	}
	defer ooc.Close()

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 6
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1

	want, err := merged.Generate("a b c", opts)
	if err != nil {
		t.Fatalf("merged generate: %v", err)
	}
	got, err := ooc.Generate("a b c", opts)
	if err != nil {
		t.Fatalf("out-of-core generate: %v", err)
	}
	if got.Text != want.Text {
		t.Fatalf("out-of-core split output %q != merged split output %q", got.Text, want.Text)
	}
}
