package gopherllm

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// buildTinyQuantizedMistralGGUF builds a minimal but real Q4_K/Q6_K
// mistral3 GGUF -- unlike buildTinyStandardGGUF (gguf_synth_test.go), which
// uses F32 throughout, this one is large enough (dim=256, giving exactly
// one 256-element Q4_K/Q6_K block per row) to exercise the real quantized
// kernels, including the WebGPU backend, rather than only the F32 matvec
// path every architecture family also has to support.
func buildTinyQuantizedMistralGGUF() []byte {
	return buildTinyQuantizedMistralGGUFWithSpecials(nil)
}

// buildTinyQuantizedMistralGGUFWithSpecials is buildTinyQuantizedMistralGGUF
// plus extraSpecials appended to the vocabulary as additional single-purpose
// tokens (e.g. "[INST]"/"[IMG]") -- used by the vision fixture, which needs
// chatTemplateKind() to detect "mistral-inst" (via Tokenizer.SpecialID
// lookups, see runtime.go) so image content is accepted at all. Appending
// rather than inserting preserves every existing token ID, so
// buildTinyQuantizedMistralGGUF()'s own golden-output tests are unaffected
// by this function's existence.
func buildTinyQuantizedMistralGGUFWithSpecials(extraSpecials []string) []byte {
	const (
		dim    = 256
		heads  = 4
		kv     = 2
		hdim   = dim / heads // 64
		hidden = 256
		baseVocab = 32
	)
	vocab := baseVocab + len(extraSpecials)
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	special := []string{"<unk>", "<s>", "</s>"}
	for i := 0; i < baseVocab; i++ {
		if i < len(special) {
			toks[i] = special[i]
		} else {
			toks[i] = string(rune('a' + (i - len(special))))
		}
		scores[i] = float32(0)
	}
	for i, s := range extraSpecials {
		toks[baseVocab+i] = s
		scores[baseVocab+i] = float32(0)
	}
	arch := "mistral3"
	kvs := []ggufKV{
		{"general.architecture", ggufStr, arch},
		{"general.name", ggufStr, "tiny-quant"},
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

	q4t := func(name string, rows, cols, seed int) ggufTensor {
		var data []byte
		for r := 0; r < rows; r++ {
			data = append(data, QuantizeRowQ4K(smallWeights(cols, seed+r), cols)...)
		}
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeQ4_K, data: data}
	}
	q6t := func(name string, rows, cols, seed int) ggufTensor {
		var data []byte
		for r := 0; r < rows; r++ {
			data = append(data, QuantizeRowQ6K(smallWeights(cols, seed+r), cols)...)
		}
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeQ6_K, data: data}
	}
	vec := func(name string, n int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
	}

	tensors := []ggufTensor{
		q6t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim),
		q6t("output.weight", vocab, dim, 2),
		vec("blk.0.attn_norm.weight", dim),
		q4t("blk.0.attn_q.weight", heads*hdim, dim, 3),
		q4t("blk.0.attn_k.weight", kv*hdim, dim, 4),
		q4t("blk.0.attn_v.weight", kv*hdim, dim, 5),
		q4t("blk.0.attn_output.weight", dim, heads*hdim, 6),
		q4t("blk.0.ffn_gate.weight", hidden, dim, 7),
		q4t("blk.0.ffn_up.weight", hidden, dim, 8),
		q4t("blk.0.ffn_down.weight", dim, hidden, 9),
		vec("blk.0.ffn_norm.weight", dim),
	}
	return buildGGUF(3, kvs, tensors)
}

// TestGenerateWasmGPUFixture writes the tiny quantized mistral3 GGUF out to
// cmd/gopherllm-wasm/testdata/harness/tiny-quant-model.gguf for the
// WebGPU forward-pass browser harness, and prints the native (CPU-path)
// golden greedy-decode output the harness's expected result is pinned to.
// Opt-in (skipped by default) since it writes a file and prints instead of
// asserting; regenerate with:
//
//	GOPHERLLM_GEN_WASM_GPU_FIXTURE=1 go test -run TestGenerateWasmGPUFixture -v .
func TestGenerateWasmGPUFixture(t *testing.T) {
	if os.Getenv("GOPHERLLM_GEN_WASM_GPU_FIXTURE") != "1" {
		t.Skip("set GOPHERLLM_GEN_WASM_GPU_FIXTURE=1 to (re)generate cmd/gopherllm-wasm/testdata/harness/tiny-quant-model.gguf")
	}
	data := buildTinyQuantizedMistralGGUF()

	out := filepath.Join("cmd", "gopherllm-wasm", "testdata", "harness", "tiny-quant-model.gguf")
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(out, data, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %d bytes to %s", len(data), out)

	r, err := RunnerFromGGUFBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	res, err := r.GenerateChat([]ChatMessage{UserMessage("Hello")}, GenerationOptions{
		MaxTokens: 8,
		Sampler:   SamplerConfig{Temperature: 0, TopP: 1, RepeatPenalty: 1.1},
	})
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("GOLDEN_TEXT=%q\n", res.Text)
}
