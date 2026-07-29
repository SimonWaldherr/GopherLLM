package gopherllm

import (
	"math"
	"testing"
)

func buildTinyBERTGGUF() []byte { return buildTinyBERTGGUFWithArch("bert") }

func buildTinyBERTGGUFWithArch(arch string) []byte {
	return buildTinyBERTGGUFWithArchAndFusedQKV(arch, false)
}

func buildTinyBERTGGUFWithFusedQKV(arch string) []byte {
	return buildTinyBERTGGUFWithArchAndFusedQKV(arch, true)
}

func buildTinyBERTGGUFWithArchAndFusedQKV(arch string, fusedQKV bool) []byte {
	const (
		dim    = 4
		heads  = 2
		hidden = 8
		vocab  = 12
	)
	tokens := make([]any, vocab)
	scores := make([]any, vocab)
	for i := range tokens {
		tokens[i] = string(rune('a' + i))
		scores[i] = float32(0)
	}
	tokens[0], tokens[1], tokens[2] = "<unk>", "[CLS]", "[SEP]"
	kvs := []ggufKV{
		{"general.architecture", ggufStr, arch},
		{"general.name", ggufStr, "tiny bert embedding"},
		{arch + ".embedding_length", ggufU32, uint32(dim)},
		{arch + ".block_count", ggufU32, uint32(1)},
		{arch + ".attention.head_count", ggufU32, uint32(heads)},
		{arch + ".feed_forward_length", ggufU32, uint32(hidden)},
		{arch + ".context_length", ggufU32, uint32(16)},
		{arch + ".attention.layer_norm_epsilon", ggufF32, float32(1e-5)},
		{arch + ".pooling_type", ggufU32, uint32(1)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, tokens}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
		{"tokenizer.ggml.add_bos_token", ggufBool, true},
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n int, values []float32) ggufTensor {
		if values == nil {
			values = onesF32(n)
		}
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(values)}
	}
	zeros := make([]float32, dim)
	qData := smallWeights(dim*dim, 3)
	kData := smallWeights(dim*dim, 4)
	vData := smallWeights(dim*dim, 5)
	matrix := func(name string, rows, cols int, values []float32) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(values)}
	}
	attention := []ggufTensor{
		matrix("blk.0.attn_q.weight", dim, dim, qData),
		matrix("blk.0.attn_k.weight", dim, dim, kData),
		matrix("blk.0.attn_v.weight", dim, dim, vData),
	}
	if fusedQKV {
		fused := make([]float32, 0, 3*dim*dim)
		fused = append(fused, qData...)
		fused = append(fused, kData...)
		fused = append(fused, vData...)
		attention = []ggufTensor{matrix("blk.0.attn_qkv.weight", 3*dim, dim, fused)}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		f32t("position_embd.weight", 16, dim, 2),
		vec("token_embd_norm.weight", dim, nil),
		vec("token_embd_norm.bias", dim, zeros),
		vec("token_types.weight", dim, zeros),
		f32t("blk.0.attn_output.weight", dim, dim, 6),
		vec("blk.0.attn_q.bias", dim, zeros),
		vec("blk.0.attn_k.bias", dim, zeros),
		vec("blk.0.attn_v.bias", dim, zeros),
		vec("blk.0.attn_output.bias", dim, zeros),
		vec("blk.0.attn_output_norm.weight", dim, nil),
		vec("blk.0.attn_output_norm.bias", dim, zeros),
		f32t("blk.0.ffn_up.weight", hidden, dim, 7),
		f32t("blk.0.ffn_down.weight", dim, hidden, 8),
		vec("blk.0.ffn_up.bias", hidden, make([]float32, hidden)),
		vec("blk.0.ffn_down.bias", dim, zeros),
		vec("blk.0.layer_output_norm.weight", dim, nil),
		vec("blk.0.layer_output_norm.bias", dim, zeros),
	}
	// Keep the Q/K/V group next to the rest of attention tensors. The exact
	// ordering is immaterial to GGUF, but putting it here makes the fused test
	// fixture resemble the real Nomic v1.5 files.
	tensors = append(tensors[:5], append(attention, tensors[5:]...)...)
	if arch == "nomic-bert" {
		// Nomic-BERT uses RoPE rather than absolute position embeddings.
		tensors = append(tensors[:1], tensors[2:]...)
		tensors = append(tensors, f32t("blk.0.ffn_gate.weight", hidden, dim, 9))
	}
	return buildGGUF(3, kvs, tensors)
}

func TestLoadAndEmbedTinyBERTModel(t *testing.T) {
	if !ArchitectureSupported("bert") {
		t.Fatal("bert must be supported")
	}
	r, err := RunnerFromGGUFBytes(buildTinyBERTGGUF())
	if err != nil {
		t.Fatalf("load bert: %v", err)
	}
	if r.kind != loadedBERT {
		t.Fatalf("kind = %v, want loadedBERT", r.kind)
	}
	emb, err := r.Embed("abc")
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(emb.Embedding) != r.Config().Dim || emb.TokenCount == 0 {
		t.Fatalf("embedding=%d tokens=%d", len(emb.Embedding), emb.TokenCount)
	}
	var norm float64
	for _, v := range emb.Embedding {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite embedding %v", v)
		}
		norm += float64(v * v)
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-4 {
		t.Fatalf("embedding norm = %v, want 1", math.Sqrt(norm))
	}
	if _, err := r.Generate("hello", DefaultGenerationOptions()); err == nil {
		t.Fatal("embedding model unexpectedly generated")
	}
}

func TestLoadAndEmbedTinyNomicBERTModel(t *testing.T) {
	if !ArchitectureSupported("nomic-bert") {
		t.Fatal("nomic-bert must be supported")
	}
	r, err := RunnerFromGGUFBytes(buildTinyBERTGGUFWithArch("nomic-bert"))
	if err != nil {
		t.Fatalf("load nomic-bert: %v", err)
	}
	if r.kind != loadedBERT {
		t.Fatalf("kind = %v, want loadedBERT", r.kind)
	}
	if _, err := r.Embed("nomic"); err != nil {
		t.Fatalf("embed: %v", err)
	}
}

func TestLoadAndEmbedTinyFusedQKVNomicBERTModel(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyBERTGGUFWithFusedQKV("nomic-bert"))
	if err != nil {
		t.Fatalf("load fused nomic-bert: %v", err)
	}
	if !r.bert.Layers[0].HasQKV {
		t.Fatal("fused Nomic QKV tensor was not retained as a fused projection")
	}
	if r.bert.Layers[0].Q.F32 != nil || r.bert.Layers[0].K.F32 != nil || r.bert.Layers[0].V.F32 != nil {
		t.Fatal("fused Nomic QKV unexpectedly loaded split Q/K/V weights")
	}
	emb, err := r.Embed("nomic")
	if err != nil {
		t.Fatalf("embed fused nomic-bert: %v", err)
	}
	if len(emb.Embedding) != r.config.Dim || emb.TokenCount == 0 {
		t.Fatalf("embedding=%d tokens=%d", len(emb.Embedding), emb.TokenCount)
	}
}

func TestEmbedFusedBERTQKVMatchesSplit(t *testing.T) {
	split, err := RunnerFromGGUFBytes(buildTinyBERTGGUFWithArch("nomic-bert"))
	if err != nil {
		t.Fatalf("load split nomic-bert: %v", err)
	}
	fused, err := RunnerFromGGUFBytes(buildTinyBERTGGUFWithFusedQKV("nomic-bert"))
	if err != nil {
		t.Fatalf("load fused nomic-bert: %v", err)
	}
	tokens := []uint32{3, 4, 5, 6}
	want, err := embedBERTWithMatvec(split.config, split.bert, tokens, matvecBERTSequential)
	if err != nil {
		t.Fatalf("embed split nomic-bert: %v", err)
	}
	got, err := EmbedBERT(fused.config, fused.bert, tokens)
	if err != nil {
		t.Fatalf("embed fused nomic-bert: %v", err)
	}
	if got.TokenCount != want.TokenCount || len(got.Embedding) != len(want.Embedding) {
		t.Fatalf("shape got=%d/%d want=%d/%d", got.TokenCount, len(got.Embedding), want.TokenCount, len(want.Embedding))
	}
	for i := range want.Embedding {
		if d := math.Abs(float64(got.Embedding[i] - want.Embedding[i])); d > 1e-5 {
			t.Fatalf("embedding[%d]: fused=%v split=%v", i, got.Embedding[i], want.Embedding[i])
		}
	}
}

func TestRunnerFusedBERTQKVScratchIsReused(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyBERTGGUFWithFusedQKV("nomic-bert"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Embed("nomic"); err != nil {
		t.Fatal(err)
	}
	if len(r.bertScratch.QKVFlat) == 0 {
		t.Fatal("fused BERT QKV scratch was not retained by Runner")
	}
	first := &r.bertScratch.QKVFlat[0]
	if _, err := r.Embed("nomic"); err != nil {
		t.Fatal(err)
	}
	if &r.bertScratch.QKVFlat[0] != first {
		t.Fatal("same-shape fused BERT embedding reallocated QKV scratch")
	}
}

func TestEmbedBERTBatchedMatchesSequential(t *testing.T) {
	for _, arch := range []string{"bert", "nomic-bert"} {
		t.Run(arch, func(t *testing.T) {
			r, err := RunnerFromGGUFBytes(buildTinyBERTGGUFWithArch(arch))
			if err != nil {
				t.Fatal(err)
			}
			// The tiny tokenizer intentionally has no merges; use valid token
			// IDs directly so this regression covers a genuinely batched pass.
			tokens := []uint32{3, 4, 5, 6}
			for _, pooling := range []uint32{1, 2} {
				r.bert.PoolingType = pooling
				want, err := embedBERTWithMatvec(r.config, r.bert, tokens, matvecBERTSequential)
				if err != nil {
					t.Fatal(err)
				}
				got, err := EmbedBERT(r.config, r.bert, tokens)
				if err != nil {
					t.Fatal(err)
				}
				if got.TokenCount != want.TokenCount || len(got.Embedding) != len(want.Embedding) {
					t.Fatalf("pooling=%d shape got=%d/%d want=%d/%d", pooling, got.TokenCount, len(got.Embedding), want.TokenCount, len(want.Embedding))
				}
				for i := range want.Embedding {
					if d := math.Abs(float64(got.Embedding[i] - want.Embedding[i])); d > 1e-5 {
						t.Fatalf("pooling=%d value %d: batched=%v sequential=%v", pooling, i, got.Embedding[i], want.Embedding[i])
					}
				}
			}
		})
	}
}

func TestRunnerBERTEmbeddingScratchIsReused(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyBERTGGUF())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Embed("abc"); err != nil {
		t.Fatal(err)
	}
	if len(r.bertScratch.XFlat) == 0 {
		t.Fatal("runner did not retain BERT embedding scratch")
	}
	first := &r.bertScratch.XFlat[0]
	if _, err := r.Embed("abc"); err != nil {
		t.Fatal(err)
	}
	if &r.bertScratch.XFlat[0] != first {
		t.Fatal("same-shape embedding reallocated BERT scratch")
	}
}

func TestEmbedBERTParallelAttentionMatchesSequential(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyBERTGGUFWithArch("nomic-bert"))
	if err != nil {
		t.Fatal(err)
	}
	// Nomic-BERT uses RoPE and therefore has no absolute-position table to
	// constrain this direct graph test. 64 tokens crosses the parallel
	// attention threshold and verifies each worker owns its score scratch.
	tokens := make([]uint32, 64)
	for i := range tokens {
		tokens[i] = uint32(3 + i%6)
	}
	want, err := embedBERTWithMatvec(r.config, r.bert, tokens, matvecBERTSequential)
	if err != nil {
		t.Fatal(err)
	}
	got, err := EmbedBERT(r.config, r.bert, tokens)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.Embedding {
		if d := math.Abs(float64(got.Embedding[i] - want.Embedding[i])); d > 1e-5 {
			t.Fatalf("value %d: batched=%v sequential=%v", i, got.Embedding[i], want.Embedding[i])
		}
	}
}

func TestEmbedBERTPoolingModes(t *testing.T) {
	config := Config{Dim: 2, NHeads: 1, HeadDim: 2}
	weights := BERTWeights{
		TokenEmbd:     Weight{F32: []float32{1, -1, -1, 1}},
		PositionEmbd:  Weight{F32: make([]float32, 4)},
		EmbeddingNorm: []float32{1, 1},
		Epsilon:       1e-5,
		PoolingType:   1,
	}
	mean, err := EmbedBERT(config, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(mean.Embedding[0])) > 1e-5 || math.Abs(float64(mean.Embedding[1])) > 1e-5 {
		t.Fatalf("mean pooling = %v, want zero", mean.Embedding)
	}
	weights.PoolingType = 2
	cls, err := EmbedBERT(config, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if cls.Embedding[0] <= 0 || cls.Embedding[1] >= 0 {
		t.Fatalf("CLS pooling = %v, want first token direction", cls.Embedding)
	}
}
