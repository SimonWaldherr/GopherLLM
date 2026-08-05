package gopherllm

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// buildRequantizableLlamaGGUF is buildTinyStandardGGUF's shape (1-layer
// pre-norm RoPE/GQA/SwiGLU llama), but sized so every weight matrix's row
// length (256) divides evenly into every target block format this package
// supports (32 for Q8_0/Q4_0, 256 for Q4_K/Q6_K) — buildTinyLlamaGGUF's
// dim=8 is too small to exercise requantization at all, since nothing
// would pass planTensor's block-size check.
func buildRequantizableLlamaGGUF() []byte {
	const (
		arch   = "llama"
		dim    = 256
		heads  = 4
		kv     = 4
		hdim   = dim / heads // 64
		hidden = 256
		vocab  = 32
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
	kvs := []ggufKV{
		{"general.architecture", ggufStr, arch},
		{"general.name", ggufStr, "tiny-requant"},
		{arch + ".embedding_length", ggufU32, uint32(dim)},
		{arch + ".block_count", ggufU32, uint32(1)},
		{arch + ".attention.head_count", ggufU32, uint32(heads)},
		{arch + ".attention.head_count_kv", ggufU32, uint32(kv)},
		{arch + ".attention.key_length", ggufU32, uint32(hdim)},
		{arch + ".attention.value_length", ggufU32, uint32(hdim)},
		{arch + ".feed_forward_length", ggufU32, uint32(hidden)},
		{arch + ".context_length", ggufU32, uint32(1024)},
		{arch + ".attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{arch + ".rope.freq_base", ggufF32, float32(10000)},
		{arch + ".rope.dimension_count", ggufU32, uint32(hdim)},
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
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		f32t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 3),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 4),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 5),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		f32t("blk.0.ffn_gate.weight", hidden, dim, 7),
		f32t("blk.0.ffn_up.weight", hidden, dim, 8),
		f32t("blk.0.ffn_down.weight", dim, hidden, 9),
		vec("blk.0.ffn_norm.weight", dim),
	}
	return buildGGUF(3, kvs, tensors)
}

func writeTempGGUF(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write temp GGUF: %v", err)
	}
	return path
}

// TestCompressModelRejectsInPlace confirms compressing a file onto itself
// (or an equivalent path to it) is rejected before anything is touched,
// rather than corrupting the still-mmap'd source.
func TestCompressModelRejectsInPlace(t *testing.T) {
	srcPath := writeTempGGUF(t, buildRequantizableLlamaGGUF())
	before, err := os.ReadFile(srcPath)
	if err != nil {
		t.Fatal(err)
	}

	err = CompressModel(srcPath, srcPath, CompressOptions{TargetFormat: GGMLTypeQ4_K})
	if err == nil {
		t.Fatal("CompressModel(same path, same path): want error, got nil")
	}

	// A relative-vs-absolute-equivalent path must be caught too, not just
	// byte-for-byte identical strings.
	rel, err := filepath.Rel(".", srcPath)
	if err == nil {
		if err := CompressModel(srcPath, rel, CompressOptions{TargetFormat: GGMLTypeQ4_K}); err == nil {
			t.Fatal("CompressModel(abs, equivalent relative path): want error, got nil")
		}
	}

	after, err := os.ReadFile(srcPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("source file was modified despite the in-place compress being rejected")
	}
}

// TestCompressModelRejectsSplitShard confirms pointing --compress at one
// shard of a split GGUF fails clearly instead of silently writing a file
// containing only that shard's tensor subset.
func TestCompressModelRejectsSplitShard(t *testing.T) {
	shard1, _ := splitTinyLlamaGGUF()
	shardPath := writeTempGGUF(t, shard1)
	outPath := filepath.Join(t.TempDir(), "out.gguf")

	err := CompressModel(shardPath, outPath, CompressOptions{TargetFormat: GGMLTypeQ4_K})
	if err == nil {
		t.Fatal("CompressModel(split shard): want error, got nil")
	}
	if _, statErr := os.Stat(outPath); statErr == nil {
		t.Error("CompressModel left a partial output file behind after rejecting a split shard")
	}
}

// TestCompressModelOutputFloor confirms token_embd.weight/output.weight get
// bumped to the Q6_K quality floor when the main target is lower quality,
// and that Uniform disables the floor for callers who want strict uniform
// quantization.
func TestCompressModelOutputFloor(t *testing.T) {
	srcPath := writeTempGGUF(t, buildRequantizableLlamaGGUF())

	outPath := filepath.Join(t.TempDir(), "floored.gguf")
	if err := CompressModel(srcPath, outPath, CompressOptions{TargetFormat: GGMLTypeQ4_K}); err != nil {
		t.Fatalf("CompressModel: %v", err)
	}
	g := parseGGUFFile(t, outPath)
	if got := tensorType(g, "token_embd.weight"); got != GGMLTypeQ6_K {
		t.Errorf("token_embd.weight = %s, want Q6_K floor", got)
	}
	if got := tensorType(g, "output.weight"); got != GGMLTypeQ6_K {
		t.Errorf("output.weight = %s, want Q6_K floor", got)
	}
	if got := tensorType(g, "blk.0.attn_q.weight"); got != GGMLTypeQ4_K {
		t.Errorf("blk.0.attn_q.weight = %s, want the requested Q4_K (no floor applies to attention weights)", got)
	}

	uniformPath := filepath.Join(t.TempDir(), "uniform.gguf")
	if err := CompressModel(srcPath, uniformPath, CompressOptions{TargetFormat: GGMLTypeQ4_K, Uniform: true}); err != nil {
		t.Fatalf("CompressModel(Uniform): %v", err)
	}
	gu := parseGGUFFile(t, uniformPath)
	if got := tensorType(gu, "token_embd.weight"); got != GGMLTypeQ4_K {
		t.Errorf("Uniform=true: token_embd.weight = %s, want Q4_K (floor disabled)", got)
	}
}

func parseGGUFFile(t *testing.T, path string) *GGUFFile {
	t.Helper()
	mmap, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { mmap.Close() })
	g, err := ParseGGUF(mmap.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func tensorType(g *GGUFFile, name string) GGMLType {
	for _, t := range g.Tensors {
		if t.Name == name {
			return t.DType
		}
	}
	return GGMLTypeUnknown
}

// TestCompressModelEndToEnd requantizes a small but block-size-compatible
// synthetic F32 model to Q4_K, then loads the RESULT with the real,
// production RunnerFromGGUFBytes and runs an actual forward pass — the
// strongest available check, since it exercises the exact code path a real
// user's `gopherllm --compress` output would hit.
func TestCompressModelEndToEnd(t *testing.T) {
	srcPath := writeTempGGUF(t, buildRequantizableLlamaGGUF())
	outPath := filepath.Join(t.TempDir(), "out.gguf")

	var log bytes.Buffer
	if err := CompressModel(srcPath, outPath, CompressOptions{TargetFormat: GGMLTypeQ4_K, LogWriter: &log}); err != nil {
		t.Fatalf("CompressModel: %v", err)
	}
	if log.Len() == 0 {
		t.Error("expected a non-empty compression log")
	}

	srcInfo, err := os.Stat(srcPath)
	if err != nil {
		t.Fatalf("stat source: %v", err)
	}
	outInfo, err := os.Stat(outPath)
	if err != nil {
		t.Fatalf("stat output: %v", err)
	}
	if outInfo.Size() >= srcInfo.Size() {
		t.Errorf("compressed output (%d bytes) is not smaller than source (%d bytes)", outInfo.Size(), srcInfo.Size())
	}

	outBytes, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	runner, err := RunnerFromGGUFBytes(outBytes)
	if err != nil {
		t.Fatalf("RunnerFromGGUFBytes(compressed output): %v", err)
	}
	defer runner.Close()

	opts := DefaultGenerationOptions()
	opts.MaxTokens = 4
	// A tiny synthetic model has essentially random logits over a 32-token
	// vocabulary, so an immediate EOS (empty text) is plausible and not
	// itself a failure — matching this repo's own tiny-fixture generation
	// tests (runtime_test.go), which only assert Generate doesn't error.
	if _, err := runner.Generate("a b c", opts); err != nil {
		t.Fatalf("Generate on compressed model: %v", err)
	}
}

// TestCompressModelWeightsPlausible spot-checks that a requantized tensor's
// values are still a reasonable approximation of the source — catching a
// gross encoder bug (e.g. wrong row/col orientation) that a pure
// "does it load and run" test could miss if the model happens to still
// produce *some* finite-but-garbage output.
func TestCompressModelWeightsPlausible(t *testing.T) {
	srcPath := writeTempGGUF(t, buildRequantizableLlamaGGUF())
	outPath := filepath.Join(t.TempDir(), "out.gguf")
	if err := CompressModel(srcPath, outPath, CompressOptions{TargetFormat: GGMLTypeQ8_0}); err != nil {
		t.Fatalf("CompressModel: %v", err)
	}

	srcMmap, err := OpenMmap(srcPath)
	if err != nil {
		t.Fatalf("OpenMmap(source): %v", err)
	}
	defer srcMmap.Close()
	srcGGUF, err := ParseGGUF(srcMmap.Bytes())
	if err != nil {
		t.Fatalf("ParseGGUF(source): %v", err)
	}
	outMmap, err := OpenMmap(outPath)
	if err != nil {
		t.Fatalf("OpenMmap(output): %v", err)
	}
	defer outMmap.Close()
	outGGUF, err := ParseGGUF(outMmap.Bytes())
	if err != nil {
		t.Fatalf("ParseGGUF(output): %v", err)
	}

	var srcTensor, outTensor *TensorInfo
	for i := range srcGGUF.Tensors {
		if srcGGUF.Tensors[i].Name == "blk.0.attn_q.weight" {
			srcTensor = &srcGGUF.Tensors[i]
		}
	}
	for i := range outGGUF.Tensors {
		if outGGUF.Tensors[i].Name == "blk.0.attn_q.weight" {
			outTensor = &outGGUF.Tensors[i]
		}
	}
	if srcTensor == nil || outTensor == nil {
		t.Fatal("blk.0.attn_q.weight missing from source or output")
	}
	if outTensor.DType != GGMLTypeQ8_0 {
		t.Fatalf("blk.0.attn_q.weight: got type %s, want Q8_0", outTensor.DType)
	}

	srcBytes := srcMmap.Bytes()[srcGGUF.DataOffset+int(srcTensor.Offset):]
	srcF32 := make([]float32, srcTensor.Numel())
	for i := range srcF32 {
		bits := uint32(srcBytes[i*4]) | uint32(srcBytes[i*4+1])<<8 | uint32(srcBytes[i*4+2])<<16 | uint32(srcBytes[i*4+3])<<24
		srcF32[i] = math.Float32frombits(bits)
	}
	outSize, _ := GGMLTypeQ8_0.DataSize(outTensor.Numel())
	outBytes := outMmap.Bytes()[outGGUF.DataOffset+int(outTensor.Offset) : outGGUF.DataOffset+int(outTensor.Offset)+outSize]
	deq := DequantRowQ8_0(outBytes, outTensor.Numel())

	var maxErr float32
	for i := range srcF32 {
		e := abs32(srcF32[i] - deq[i])
		if e > maxErr {
			maxErr = e
		}
	}
	// smallWeights produces values in [-6/20, 6/20]; Q8_0's per-32-block
	// absmax quantization step at that magnitude is well under 0.01.
	if maxErr > 0.01 {
		t.Errorf("requantized blk.0.attn_q.weight diverges from source by %v (max), want < 0.01", maxErr)
	}
}
