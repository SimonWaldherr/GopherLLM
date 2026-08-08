package gopherllm

import (
	"math"
	"testing"
)

// This file exercises the classic-transformer and modern-dense-chat
// architectures added alongside it: GPT-2, GPT-NeoX, GPT-J, BLOOM, MPT,
// Falcon, StarCoder (v1), StarCoder2, ChatGLM, GLM4, Command-R, MiniCPM,
// Granite, and GraniteMoE. Each builder constructs a tiny synthetic GGUF
// (following gguf_synth_test.go's helpers) and each test both checks the
// resolved Config flags and, where possible, runs assertStandardBatchParity
// so the batched-prefill path (forward_batch.go) and the per-token decode
// path (ForwardBodyInto) are cross-checked against each other — this is what
// would have caught the missing position-embedding and ALiBi wiring in the
// batched path during development.

const (
	extraDim    = 8
	extraHeads  = 2
	extraKV     = 2
	extraHDim   = extraDim / extraHeads
	extraHidden = 16
	extraVocab  = 20
)

func extraTinyVocab() ([]any, []any) {
	toks := make([]any, extraVocab)
	scores := make([]any, extraVocab)
	for i := 0; i < extraVocab; i++ {
		if i == 0 {
			toks[i] = "<unk>"
		} else if i == 1 {
			toks[i] = "<s>"
		} else {
			toks[i] = string(rune('a' + i - 2))
		}
		scores[i] = float32(0)
	}
	return toks, scores
}

func extraBaseKVs(arch string) []ggufKV {
	toks, scores := extraTinyVocab()
	return []ggufKV{
		{"general.architecture", ggufStr, arch},
		{"general.name", ggufStr, "tiny-" + arch},
		{arch + ".embedding_length", ggufU32, uint32(extraDim)},
		{arch + ".block_count", ggufU32, uint32(1)},
		{arch + ".attention.head_count", ggufU32, uint32(extraHeads)},
		{arch + ".attention.head_count_kv", ggufU32, uint32(extraKV)},
		{arch + ".attention.key_length", ggufU32, uint32(extraHDim)},
		{arch + ".attention.value_length", ggufU32, uint32(extraHDim)},
		{arch + ".feed_forward_length", ggufU32, uint32(extraHidden)},
		{arch + ".context_length", ggufU32, uint32(64)},
		{"tokenizer.ggml.model", ggufStr, "gpt2"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.add_bos_token", ggufBool, false},
	}
}

func extraF32t(name string, rows, cols, seed int) ggufTensor {
	return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
}
func extraVec(name string, n, seed int) ggufTensor {
	return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(n, seed))}
}
func extraZeroVec(name string, n int) ggufTensor {
	return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(make([]float32, n))}
}
func extraOnesVec(name string, n int) ggufTensor {
	return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(onesF32(n))}
}

func runExtraArchSmoke(t *testing.T, data []byte, checkFlags func(*testing.T, *Runner)) {
	t.Helper()
	r, err := RunnerFromGGUFBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	checkFlags(t, r)
	tokens := r.tok.Encode("abcdefgh")
	if len(tokens) == 0 {
		tokens = []uint32{2, 3, 4, 5}
	}
	if r.canBatchPrefill() {
		assertStandardBatchParity(t, r, tokens)
		return
	}
	// MoE architectures fall back to a plain decode smoke test:
	// canBatchPrefill() is false whenever any layer has sparse experts.
	kDim, vDim, mh, mk, mv := r.cacheDims()
	cache := NewKVCache(r.config.NLayers, kDim, vDim, len(tokens)+1)
	buf := NewDecodeBuffer(r.config, mh, mk, mv)
	for pos, tok := range tokens {
		logits := Forward(r.config, r.standard, cache, buf, tok, pos)
		for i, v := range logits {
			if v != v || v > 1e30 || v < -1e30 {
				t.Fatalf("token %d logit %d is non-finite: %v", pos, i, v)
			}
		}
	}
}

// ---------------------------------------------------------------- GPT-2 ---

func buildTinyGPT2GGUF() []byte {
	kvs := append(extraBaseKVs("gpt2"),
		ggufKV{"gpt2.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraF32t("position_embd.weight", 64, extraDim, 2),
		extraOnesVec("output_norm.weight", extraDim),
		extraZeroVec("output_norm.bias", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 3),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraZeroVec("blk.0.attn_norm.bias", extraDim),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraKV*extraHDim, extraDim, 4),
		extraZeroVec("blk.0.attn_qkv.bias", extraDim+2*extraKV*extraHDim),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 5),
		extraZeroVec("blk.0.attn_output.bias", extraDim),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraZeroVec("blk.0.ffn_norm.bias", extraDim),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 6),
		extraZeroVec("blk.0.ffn_up.bias", extraHidden),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 7),
		extraZeroVec("blk.0.ffn_down.bias", extraDim),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestGPT2LoadsAbsolutePositionEmbdAndFusedQKVBias(t *testing.T) {
	runExtraArchSmoke(t, buildTinyGPT2GGUF(), func(t *testing.T, r *Runner) {
		if !r.config.UseLayerNorm || r.config.ParallelResidual || !r.config.usesAbsolutePositionEmbd() {
			t.Fatalf("unexpected GPT-2 graph flags: %+v", r.config)
		}
		if r.config.UseExactGELU {
			t.Fatal("GPT-2 must use tanh-approximation GELU, not exact GELU")
		}
		if r.standard.PositionEmbd.F32 == nil && r.standard.PositionEmbd.Raw == nil {
			t.Fatal("GPT-2 must load position_embd.weight")
		}
		layer := r.standard.Layers[0]
		if !layer.HasQKV {
			t.Fatal("GPT-2 must use the fused attn_qkv tensor")
		}
		if len(layer.BQ) != extraHeads*extraHDim || len(layer.BK) != extraKV*extraHDim || len(layer.BV) != extraKV*extraHDim {
			t.Fatal("GPT-2's fused attn_qkv.bias must be split into BQ/BK/BV")
		}
	})
}

// -------------------------------------------------------------- GPT-NeoX --

func buildTinyGPTNeoXGGUF() []byte {
	kvs := append(extraBaseKVs("gptneox"),
		ggufKV{"gptneox.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"gptneox.rope.freq_base", ggufF32, float32(10000)},
		ggufKV{"gptneox.rope.dimension_count", ggufU32, uint32(extraHDim / 2)},
		ggufKV{"gptneox.use_parallel_residual", ggufBool, false},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraZeroVec("output_norm.bias", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraZeroVec("blk.0.attn_norm.bias", extraDim),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraKV*extraHDim, extraDim, 3),
		extraZeroVec("blk.0.attn_qkv.bias", extraDim+2*extraKV*extraHDim),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 4),
		extraZeroVec("blk.0.attn_output.bias", extraDim),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraZeroVec("blk.0.ffn_norm.bias", extraDim),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 5),
		extraZeroVec("blk.0.ffn_up.bias", extraHidden),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 6),
		extraZeroVec("blk.0.ffn_down.bias", extraDim),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestGPTNeoXSequentialPartialRoPE(t *testing.T) {
	runExtraArchSmoke(t, buildTinyGPTNeoXGGUF(), func(t *testing.T, r *Runner) {
		if !r.config.UseLayerNorm || r.config.ParallelResidual {
			t.Fatalf("unexpected GPT-NeoX graph flags (expected sequential residual): %+v", r.config)
		}
		if r.config.RopeDimensionCount != extraHDim/2 {
			t.Fatalf("RopeDimensionCount = %d, want partial %d", r.config.RopeDimensionCount, extraHDim/2)
		}
		if ropeInterleaved(r.config.Arch) {
			t.Fatal("GPT-NeoX must use the NeoX (split-half) rope convention, not interleaved")
		}
	})
}

func TestGPTNeoXParallelResidualFlagFromGGUF(t *testing.T) {
	kvs := append(extraBaseKVs("gptneox"),
		ggufKV{"gptneox.use_parallel_residual", ggufBool, true},
	)
	g, err := ParseGGUFQuiet(buildGGUF(3, kvs, nil))
	if err != nil {
		t.Fatal(err)
	}
	cfg := ConfigFromGGUF(g)
	if !cfg.ParallelResidual {
		t.Fatal("gptneox.use_parallel_residual=true must set Config.ParallelResidual")
	}
}

// ---------------------------------------------------------------- GPT-J ---

func buildTinyGPTJGGUF() []byte {
	kvs := append(extraBaseKVs("gptj"),
		ggufKV{"gptj.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"gptj.rope.freq_base", ggufF32, float32(10000)},
		ggufKV{"gptj.rope.dimension_count", ggufU32, uint32(extraHDim / 2)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraZeroVec("output_norm.bias", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraZeroVec("blk.0.attn_norm.bias", extraDim),
		extraF32t("blk.0.attn_q.weight", extraHeads*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_k.weight", extraKV*extraHDim, extraDim, 4),
		extraF32t("blk.0.attn_v.weight", extraKV*extraHDim, extraDim, 5),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 6),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 7),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 8),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestGPTJSharedNormParallelInterleavedRoPE(t *testing.T) {
	runExtraArchSmoke(t, buildTinyGPTJGGUF(), func(t *testing.T, r *Runner) {
		if !r.config.UseLayerNorm || !r.config.ParallelResidual || !r.config.sharesParallelBranchNorm() {
			t.Fatalf("unexpected GPT-J graph flags: %+v", r.config)
		}
		if !ropeInterleaved(r.config.Arch) {
			t.Fatal("GPT-J must use the interleaved (rotate-every-two) rope convention")
		}
		layer := r.standard.Layers[0]
		if layer.HasQKV {
			t.Fatal("GPT-J's GGUF convention uses separate Q/K/V tensors, not a fused attn_qkv")
		}
		if layer.FFNNorm != nil {
			t.Fatal("GPT-J shares one norm across both branches and should not load a separate ffn_norm")
		}
	})
}

// --------------------------------------------------------------- BLOOM ---

func buildTinyBLOOMGGUF() []byte {
	kvs := append(extraBaseKVs("bloom"),
		ggufKV{"bloom.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("token_embd_norm.weight", extraDim),
		extraZeroVec("token_embd_norm.bias", extraDim),
		extraOnesVec("output_norm.weight", extraDim),
		extraZeroVec("output_norm.bias", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraZeroVec("blk.0.attn_norm.bias", extraDim),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraKV*extraHDim, extraDim, 3),
		extraZeroVec("blk.0.attn_qkv.bias", extraDim+2*extraKV*extraHDim),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 4),
		extraZeroVec("blk.0.attn_output.bias", extraDim),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraZeroVec("blk.0.ffn_norm.bias", extraDim),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 5),
		extraZeroVec("blk.0.ffn_up.bias", extraHidden),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 6),
		extraZeroVec("blk.0.ffn_down.bias", extraDim),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestBLOOMUsesALiBiNotRoPE(t *testing.T) {
	runExtraArchSmoke(t, buildTinyBLOOMGGUF(), func(t *testing.T, r *Runner) {
		if !r.config.usesALiBi() || r.config.ALiBiMaxBias != 8.0 {
			t.Fatalf("BLOOM must hardcode ALiBiMaxBias=8.0, got %+v", r.config)
		}
		if r.config.layerUsesRoPE(0) {
			t.Fatal("BLOOM must not apply RoPE")
		}
	})
}

// ---------------------------------------------------------------- MPT ----

func buildTinyMPTGGUF() []byte {
	kvs := append(extraBaseKVs("mpt"),
		ggufKV{"mpt.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"mpt.attention.max_alibi_bias", ggufF32, float32(8.0)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraKV*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 4),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 5),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 6),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestMPTALiBiFromGGUFKeyAndNoBiasTensors(t *testing.T) {
	runExtraArchSmoke(t, buildTinyMPTGGUF(), func(t *testing.T, r *Runner) {
		if !r.config.usesALiBi() || r.config.ALiBiMaxBias != 8.0 {
			t.Fatalf("MPT must read ALiBiMaxBias from its GGUF key, got %+v", r.config)
		}
	})
}

func TestMPTALiBiDisabledWhenMaxBiasAbsent(t *testing.T) {
	kvs := extraBaseKVs("mpt")
	g, err := ParseGGUFQuiet(buildGGUF(3, kvs, nil))
	if err != nil {
		t.Fatal(err)
	}
	cfg := ConfigFromGGUF(g)
	if cfg.usesALiBi() {
		t.Fatal("MPT without an explicit max_alibi_bias key must not enable ALiBi")
	}
}

// -------------------------------------------------------------- Falcon --

// buildTinyFalconGGUF models the 40B/180B "new decoder architecture" variant
// (attn_norm_2 present), the more structurally demanding of the two Falcon
// sizes: attn_norm_2 feeds attention while the base attn_norm feeds FFN.
func buildTinyFalconGGUF() []byte {
	kvs := append(extraBaseKVs("falcon"),
		ggufKV{"falcon.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"falcon.rope.freq_base", ggufF32, float32(10000)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraVec("blk.0.attn_norm_2.weight", extraDim, 99),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraKV*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 4),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 5),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 6),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestFalconNewDecoderArchAliasesNorms(t *testing.T) {
	runExtraArchSmoke(t, buildTinyFalconGGUF(), func(t *testing.T, r *Runner) {
		if !r.config.ParallelResidual {
			t.Fatal("Falcon must use the parallel residual graph")
		}
		layer := r.standard.Layers[0]
		if len(layer.AttnNorm) != extraDim || len(layer.FFNNorm) != extraDim {
			t.Fatal("Falcon must alias AttnNorm/FFNNorm to attn_norm_2/attn_norm when both are present")
		}
		same := true
		for i := range layer.AttnNorm {
			if layer.AttnNorm[i] != layer.FFNNorm[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatal("40B-style Falcon must use DIFFERENT weights for AttnNorm (attn_norm_2) and FFNNorm (attn_norm)")
		}
	})
}

func buildTinyFalcon7BGGUF() []byte {
	kvs := append(extraBaseKVs("falcon"),
		ggufKV{"falcon.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"falcon.rope.freq_base", ggufF32, float32(10000)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraKV*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 4),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 5),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 6),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestFalcon7BSharesOneNormAcrossBothBranches(t *testing.T) {
	runExtraArchSmoke(t, buildTinyFalcon7BGGUF(), func(t *testing.T, r *Runner) {
		layer := r.standard.Layers[0]
		if len(layer.AttnNorm) != extraDim || len(layer.FFNNorm) != extraDim {
			t.Fatal("Falcon 7B must load both AttnNorm and FFNNorm from the single attn_norm tensor")
		}
		for i := range layer.AttnNorm {
			if layer.AttnNorm[i] != layer.FFNNorm[i] {
				t.Fatal("Falcon 7B (no attn_norm_2) must use the SAME weights for AttnNorm and FFNNorm")
			}
		}
	})
}

// ------------------------------------------------------------ StarCoder --

func buildTinyStarCoderGGUF() []byte {
	kvs := append(extraBaseKVs("starcoder"),
		ggufKV{"starcoder.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
	)
	// StarCoder v1 hardcodes MQA (head_count_kv=1) at conversion time.
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraF32t("position_embd.weight", 64, extraDim, 2),
		extraOnesVec("output_norm.weight", extraDim),
		extraZeroVec("output_norm.bias", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 3),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraZeroVec("blk.0.attn_norm.bias", extraDim),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraHDim, extraDim, 4),
		extraZeroVec("blk.0.attn_qkv.bias", extraDim+2*extraHDim),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 5),
		extraZeroVec("blk.0.attn_output.bias", extraDim),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraZeroVec("blk.0.ffn_norm.bias", extraDim),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 6),
		extraZeroVec("blk.0.ffn_up.bias", extraHidden),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 7),
		extraZeroVec("blk.0.ffn_down.bias", extraDim),
	}
	kvs[5] = ggufKV{"starcoder.attention.head_count_kv", ggufU32, uint32(1)}
	return buildGGUF(3, kvs, tensors)
}

func TestStarCoderV1MQAWithAbsolutePositionEmbd(t *testing.T) {
	runExtraArchSmoke(t, buildTinyStarCoderGGUF(), func(t *testing.T, r *Runner) {
		if r.config.NKVHeads != 1 {
			t.Fatalf("StarCoder v1 must be MQA (head_count_kv=1), got %d", r.config.NKVHeads)
		}
		if !r.config.usesAbsolutePositionEmbd() || r.config.layerUsesRoPE(0) {
			t.Fatal("StarCoder v1 must use absolute position embeddings, not RoPE")
		}
	})
}

// ----------------------------------------------------------- StarCoder2 --

func buildTinyStarCoder2GGUF() []byte {
	kvs := append(extraBaseKVs("starcoder2"),
		ggufKV{"starcoder2.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"starcoder2.rope.freq_base", ggufF32, float32(10000)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraZeroVec("output_norm.bias", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraZeroVec("blk.0.attn_norm.bias", extraDim),
		extraF32t("blk.0.attn_q.weight", extraHeads*extraHDim, extraDim, 3),
		extraZeroVec("blk.0.attn_q.bias", extraHeads*extraHDim),
		extraF32t("blk.0.attn_k.weight", extraKV*extraHDim, extraDim, 4),
		extraZeroVec("blk.0.attn_k.bias", extraKV*extraHDim),
		extraF32t("blk.0.attn_v.weight", extraKV*extraHDim, extraDim, 5),
		extraZeroVec("blk.0.attn_v.bias", extraKV*extraHDim),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 6),
		extraZeroVec("blk.0.attn_output.bias", extraDim),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraZeroVec("blk.0.ffn_norm.bias", extraDim),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 7),
		extraZeroVec("blk.0.ffn_up.bias", extraHidden),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 8),
		extraZeroVec("blk.0.ffn_down.bias", extraDim),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestStarCoder2FullWidthNeoXRoPEWithGQA(t *testing.T) {
	runExtraArchSmoke(t, buildTinyStarCoder2GGUF(), func(t *testing.T, r *Runner) {
		if r.config.RopeDimensionCount != extraHDim {
			t.Fatalf("StarCoder2 must use full-width rope, got dim=%d want %d", r.config.RopeDimensionCount, extraHDim)
		}
		if ropeInterleaved(r.config.Arch) {
			t.Fatal("StarCoder2 must use the NeoX rope convention, not interleaved")
		}
		if r.standard.Layers[0].HasQKV {
			t.Fatal("StarCoder2 GGUFs carry separate Q/K/V tensors, not a fused attn_qkv")
		}
	})
}

// ------------------------------------------------------------- ChatGLM --

func buildTinyChatGLMGGUF() []byte {
	kvs := append(extraBaseKVs("chatglm"),
		ggufKV{"chatglm.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"chatglm.rope.freq_base", ggufF32, float32(10000)},
		ggufKV{"chatglm.rope.dimension_count", ggufU32, uint32(extraHDim / 2)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_qkv.weight", extraDim+2*extraKV*extraHDim, extraDim, 3),
		extraZeroVec("blk.0.attn_qkv.bias", extraDim+2*extraKV*extraHDim),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 4),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraF32t("blk.0.ffn_up.weight", 2*extraHidden, extraDim, 5),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 6),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestChatGLMFusedQKVGatedFFNInterleavedRoPE(t *testing.T) {
	runExtraArchSmoke(t, buildTinyChatGLMGGUF(), func(t *testing.T, r *Runner) {
		if !ropeInterleaved(r.config.Arch) {
			t.Fatal("ChatGLM must use the interleaved rope convention")
		}
		layer := r.standard.Layers[0]
		if !layer.HasQKV {
			t.Fatal("ChatGLM's real checkpoints use a fused attn_qkv tensor")
		}
		if !layer.HasGateUp {
			t.Fatal("ChatGLM's FFN is a fused gate+up tensor")
		}
	})
}

// --------------------------------------------------------------- GLM4 ---

func buildTinyGLM4GGUF() []byte {
	kvs := append(extraBaseKVs("glm4"),
		ggufKV{"glm4.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"glm4.rope.freq_base", ggufF32, float32(10000)},
		ggufKV{"glm4.rope.dimension_count", ggufU32, uint32(extraHDim / 2)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_q.weight", extraHeads*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_k.weight", extraKV*extraHDim, extraDim, 4),
		extraF32t("blk.0.attn_v.weight", extraKV*extraHDim, extraDim, 5),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 6),
		extraOnesVec("blk.0.post_attention_norm.weight", extraDim),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraF32t("blk.0.ffn_up.weight", 2*extraHidden, extraDim, 7),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 8),
		extraOnesVec("blk.0.post_ffw_norm.weight", extraDim),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestGLM4SandwichNormsAroundBothBranches(t *testing.T) {
	runExtraArchSmoke(t, buildTinyGLM4GGUF(), func(t *testing.T, r *Runner) {
		layer := r.standard.Layers[0]
		if layer.PostAttnNorm == nil || layer.PostFFNNorm == nil {
			t.Fatal("GLM4 must load post_attention_norm and post_ffw_norm")
		}
		if !ropeInterleaved(r.config.Arch) {
			t.Fatal("GLM4 (text-only, non-mrope) must use the interleaved rope convention")
		}
	})
}

// ------------------------------------------------------------ Command-R --

func buildTinyCommandRGGUF() []byte {
	kvs := append(extraBaseKVs("command-r"),
		ggufKV{"command-r.attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"command-r.rope.freq_base", ggufF32, float32(10000)},
		ggufKV{"command-r.logit_scale", ggufF32, float32(0.25)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_q.weight", extraHeads*extraHDim, extraDim, 2),
		extraF32t("blk.0.attn_k.weight", extraKV*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_v.weight", extraKV*extraHDim, extraDim, 4),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 5),
		extraF32t("blk.0.ffn_gate.weight", extraHidden, extraDim, 6),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 7),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 8),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestCommandRTiedOutputParallelResidualAndInvertedLogitScale(t *testing.T) {
	runExtraArchSmoke(t, buildTinyCommandRGGUF(), func(t *testing.T, r *Runner) {
		if !r.config.ParallelResidual || !r.config.sharesParallelBranchNorm() {
			t.Fatal("Command-R must use the shared-norm parallel residual graph (no ffn_norm tensor)")
		}
		if !ropeInterleaved(r.config.Arch) {
			t.Fatal("Command-R must use the interleaved rope convention")
		}
		// The GGUF stores 0.25 as a direct multiplier; GopherLLM applies
		// 1/LogitScale, so LogitScale must be stored as the reciprocal (4).
		if d := r.config.LogitScale - 4.0; d < -1e-4 || d > 1e-4 {
			t.Fatalf("Command-R LogitScale = %v, want reciprocal of GGUF value (4.0)", r.config.LogitScale)
		}
		// No output.weight tensor in the GGUF: output must tie to TokenEmbd.
		if r.standard.Output.F32 == nil && r.standard.Output.Raw == nil {
			t.Fatal("Command-R output must tie to token_embd when output.weight is absent")
		}
	})
}

// ------------------------------------------------------------- MiniCPM --

func buildTinyMiniCPMGGUF() []byte {
	// Deliberately omit embedding_scale/residual_scale/logit_scale to
	// exercise MiniCPM's hardcoded-default fallback path.
	kvs := append(extraBaseKVs("minicpm"),
		ggufKV{"minicpm.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"minicpm.rope.freq_base", ggufF32, float32(10000)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_q.weight", extraHeads*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_k.weight", extraKV*extraHDim, extraDim, 4),
		extraF32t("blk.0.attn_v.weight", extraKV*extraHDim, extraDim, 5),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 6),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraF32t("blk.0.ffn_gate.weight", extraHidden, extraDim, 7),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 8),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 9),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestMiniCPMHardcodedScaleDefaults(t *testing.T) {
	runExtraArchSmoke(t, buildTinyMiniCPMGGUF(), func(t *testing.T, r *Runner) {
		if d := r.config.EmbeddingScale - 12.0; d < -1e-4 || d > 1e-4 {
			t.Fatalf("MiniCPM EmbeddingScale = %v, want 12.0 default", r.config.EmbeddingScale)
		}
		wantResidual := float32(1.4 / math.Sqrt(float64(r.config.NLayers)))
		if d := r.config.ResidualScale - wantResidual; d < -1e-4 || d > 1e-4 {
			t.Fatalf("MiniCPM ResidualScale = %v, want %v", r.config.ResidualScale, wantResidual)
		}
		wantLogit := float32(256.0 / float32(extraDim))
		if d := r.config.LogitScale - wantLogit; d < -1e-4 || d > 1e-4 {
			t.Fatalf("MiniCPM LogitScale = %v, want %v", r.config.LogitScale, wantLogit)
		}
	})
}

// -------------------------------------------------------------- Granite --

func buildTinyGraniteGGUF() []byte {
	kvs := append(extraBaseKVs("granite"),
		ggufKV{"granite.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"granite.rope.freq_base", ggufF32, float32(10000)},
		ggufKV{"granite.embedding_scale", ggufF32, float32(2.0)},
		ggufKV{"granite.residual_scale", ggufF32, float32(0.5)},
		ggufKV{"granite.attention.scale", ggufF32, float32(0.1)},
		ggufKV{"granite.logit_scale", ggufF32, float32(8.0)},
	)
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_q.weight", extraHeads*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_k.weight", extraKV*extraHDim, extraDim, 4),
		extraF32t("blk.0.attn_v.weight", extraKV*extraHDim, extraDim, 5),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 6),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraF32t("blk.0.ffn_gate.weight", extraHidden, extraDim, 7),
		extraF32t("blk.0.ffn_up.weight", extraHidden, extraDim, 8),
		extraF32t("blk.0.ffn_down.weight", extraDim, extraHidden, 9),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestGraniteFourMultipliersAndInterleavedRoPE(t *testing.T) {
	runExtraArchSmoke(t, buildTinyGraniteGGUF(), func(t *testing.T, r *Runner) {
		if r.config.EmbeddingScale != 2.0 || r.config.ResidualScale != 0.5 ||
			r.config.AttentionScale != 0.1 || r.config.LogitScale != 8.0 {
			t.Fatalf("Granite's four multipliers were not read from GGUF: %+v", r.config)
		}
		if !ropeInterleaved(r.config.Arch) {
			t.Fatal("Granite must use the interleaved rope convention (regression: this was wired to false before)")
		}
	})
}

// ------------------------------------------------------------ GraniteMoE --

func buildTinyGraniteMoEGGUF() []byte {
	const experts, used, expertHidden = 4, 2, extraHidden
	kvs := append(extraBaseKVs("granitemoe"),
		ggufKV{"granitemoe.attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		ggufKV{"granitemoe.rope.freq_base", ggufF32, float32(10000)},
		ggufKV{"granitemoe.expert_count", ggufU32, uint32(experts)},
		ggufKV{"granitemoe.expert_used_count", ggufU32, uint32(used)},
	)
	expert3D := func(name string, input, output, count, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(input), uint64(output), uint64(count)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(input*output*count, seed))}
	}
	tensors := []ggufTensor{
		extraF32t("token_embd.weight", extraVocab, extraDim, 1),
		extraOnesVec("output_norm.weight", extraDim),
		extraF32t("output.weight", extraVocab, extraDim, 2),
		extraOnesVec("blk.0.attn_norm.weight", extraDim),
		extraF32t("blk.0.attn_q.weight", extraHeads*extraHDim, extraDim, 3),
		extraF32t("blk.0.attn_k.weight", extraKV*extraHDim, extraDim, 4),
		extraF32t("blk.0.attn_v.weight", extraKV*extraHDim, extraDim, 5),
		extraF32t("blk.0.attn_output.weight", extraDim, extraHeads*extraHDim, 6),
		extraOnesVec("blk.0.ffn_norm.weight", extraDim),
		extraF32t("blk.0.ffn_gate_inp.weight", experts, extraDim, 7),
		expert3D("blk.0.ffn_gate_exps.weight", extraDim, expertHidden, experts, 8),
		expert3D("blk.0.ffn_up_exps.weight", extraDim, expertHidden, experts, 9),
		expert3D("blk.0.ffn_down_exps.weight", expertHidden, extraDim, experts, 10),
		// Always-on, ungated shared expert (no ffn_gate_inp_shexp tensor).
		extraF32t("blk.0.ffn_gate_shexp.weight", expertHidden, extraDim, 11),
		extraF32t("blk.0.ffn_up_shexp.weight", expertHidden, extraDim, 12),
		extraF32t("blk.0.ffn_down_shexp.weight", extraDim, expertHidden, 13),
	}
	return buildGGUF(3, kvs, tensors)
}

func TestGraniteMoEUngatedSharedExpert(t *testing.T) {
	runExtraArchSmoke(t, buildTinyGraniteMoEGGUF(), func(t *testing.T, r *Runner) {
		layer := r.standard.Layers[0]
		if layer.MoE == nil {
			t.Fatal("GraniteMoE must load sparse MoE weights")
		}
		if !layer.MoE.SharedAlways || layer.MoE.SharedGate == nil || layer.MoE.SharedGateIn != nil {
			t.Fatal("GraniteMoE's shared expert must be always-on and ungated (SharedAlways, no SharedGateIn)")
		}
	})
}
