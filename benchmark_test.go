package gopherllm

import (
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func BenchmarkDotF32_4096(b *testing.B) {
	x := benchFloatSlice(4096)
	y := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(x) * 4 * 2))
	for b.Loop() {
		_ = DotF32(x, y)
	}
}

func BenchmarkMatvecF32_1024x1024(b *testing.B) {
	data := benchFloatSlice(1024 * 1024)
	x := benchFloatSlice(1024)
	out := make([]float32, 1024)
	b.ReportAllocs()
	b.SetBytes(int64((len(data) + len(x)) * 4))
	for b.Loop() {
		MatvecF32Into(data, x, 1024, 1024, &out)
	}
}

func BenchmarkAxpyF32_4096(b *testing.B) {
	out := benchFloatSlice(4096)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(out) * 4 * 2))
	for b.Loop() {
		AxpyF32(out, 0.75, x)
	}
}

func BenchmarkScaleF32_4096(b *testing.B) {
	out := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(out) * 4))
	for b.Loop() {
		ScaleF32(out, 0.999)
	}
}

func BenchmarkScaleAddF32_4096(b *testing.B) {
	out := benchFloatSlice(4096)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(out) * 4 * 2))
	for b.Loop() {
		ScaleAddF32(out, 0.999, x)
	}
}

func BenchmarkMulScaleF32_4096(b *testing.B) {
	x := benchFloatSlice(4096)
	weight := benchFloatSlice(4096)
	out := make([]float32, 4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(out) * 4 * 3))
	for b.Loop() {
		mulScaleF32(x, weight, 0.999, out)
	}
}

func BenchmarkRMSNorm_4096(b *testing.B) {
	x := benchFloatSlice(4096)
	weight := benchFloatSlice(4096)
	out := make([]float32, 4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(out) * 4 * 3))
	for b.Loop() {
		rmsNormInto(x, weight, 1e-5, &out)
	}
}

func BenchmarkLayerNorm_4096(b *testing.B) {
	x := benchFloatSlice(4096)
	weight := benchFloatSlice(4096)
	bias := benchFloatSlice(4096)
	out := make([]float32, 4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(out) * 4 * 4))
	for b.Loop() {
		layerNormInto(x, weight, bias, 1e-5, &out)
	}
}

func BenchmarkDotQ4K_4096(b *testing.B) {
	row := benchBytes((4096 / 256) * 144)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = DotQ4KF32(row, x, 4096)
	}
}

func BenchmarkMatvecQ4K_1024x1024(b *testing.B) {
	data := benchBytes(1024 * (1024 / 256) * 144)
	x := benchFloatSlice(1024)
	out := make([]float32, 1024)
	b.ReportAllocs()
	b.SetBytes(int64(len(data) + len(x)*4))
	for b.Loop() {
		MatvecQ4KInto(data, x, 1024, 1024, &out)
	}
}

func BenchmarkMatvecQ8K_1024x1024(b *testing.B) {
	data := benchBytes(1024 * (1024 / 256) * 292)
	for i := 0; i < len(data); i += 292 {
		data[i], data[i+1], data[i+2], data[i+3] = 0, 0, 0, 0x3f // scale 0.5
	}
	x := benchFloatSlice(1024)
	out := make([]float32, 1024)
	b.ReportAllocs()
	b.SetBytes(int64(len(data) + len(x)*4))
	for b.Loop() {
		MatvecQ8KInto(data, x, 1024, 1024, &out)
	}
}

func BenchmarkMatvecQ4_0QKVFused_1024x1024(b *testing.B) {
	data := benchQ4_0Data(1024, 1024)
	wq := Weight{Raw: data, Type: GGMLTypeQ4_0, Rows: 1024, Cols: 1024}
	wk := Weight{Raw: append([]byte(nil), data...), Type: GGMLTypeQ4_0, Rows: 1024, Cols: 1024}
	wv := Weight{Raw: append([]byte(nil), data...), Type: GGMLTypeQ4_0, Rows: 1024, Cols: 1024}
	x := benchFloatSlice(1024)
	q, k, v, sums := []float32{}, []float32{}, []float32{}, []float32{}
	b.ReportAllocs()
	b.SetBytes(int64(3*len(data) + len(x)*4))
	for b.Loop() {
		if !tryMatvec3Into(wq, wk, wv, x, &sums, &q, &k, &v) {
			b.Fatal("tryMatvec3Into declined Q4_0 weights")
		}
	}
}

func BenchmarkMatvecQ4_0QKVSeparate_1024x1024(b *testing.B) {
	data := benchQ4_0Data(1024, 1024)
	w := Weight{Raw: data, Type: GGMLTypeQ4_0, Rows: 1024, Cols: 1024}
	x := benchFloatSlice(1024)
	q, k, v := []float32{}, []float32{}, []float32{}
	b.ReportAllocs()
	b.SetBytes(int64(3*len(data) + len(x)*4))
	for b.Loop() {
		w.MatvecInto(x, &q)
		w.MatvecInto(x, &k)
		w.MatvecInto(x, &v)
	}
}

func benchQ4_0Data(rows, cols int) []byte {
	data := benchBytes(rows * (cols / 32) * 18)
	for block := 0; block < len(data)/18; block++ {
		data[block*18+1] = 0x3c // f16 1.0
	}
	return data
}

func BenchmarkMatvecPreparedQ4K_1024x1024(b *testing.B) {
	data := benchBytes(1024 * (1024 / 256) * 144)
	prepared := PrepareQuantizedWeight(data, GGMLTypeQ4_K, 1024, 1024)
	x := benchFloatSlice(1024)
	out := make([]float32, 1024)
	b.ReportAllocs()
	b.SetBytes(int64(len(data) + len(x)*4))
	for b.Loop() {
		if !MatvecPreparedQ4KInto(data, prepared, x, 1024, 1024, &out) {
			b.Fatal("MatvecPreparedQ4KInto returned false")
		}
	}
}

func BenchmarkMatvecQ4K3_1024x1024(b *testing.B) {
	qData := benchBytes(1024 * (1024 / 256) * 144)
	kData := benchBytes(1024 * (1024 / 256) * 144)
	vData := benchBytes(1024 * (1024 / 256) * 144)
	x := benchFloatSlice(1024)
	q := make([]float32, 1024)
	k := make([]float32, 1024)
	v := make([]float32, 1024)
	b.ReportAllocs()
	b.SetBytes(int64(len(qData) + len(kData) + len(vData) + len(x)*4))
	for b.Loop() {
		ok := Q4KMatvec3Into(
			Q4KMatrix{Data: qData, Rows: 1024, Cols: 1024},
			Q4KMatrix{Data: kData, Rows: 1024, Cols: 1024},
			Q4KMatrix{Data: vData, Rows: 1024, Cols: 1024},
			x,
			&q,
			&k,
			&v,
		)
		if !ok {
			b.Fatal("Q4KMatvec3Into returned false")
		}
	}
}

func BenchmarkMatvecPreparedQ4K3_1024x1024(b *testing.B) {
	qData := benchBytes(1024 * (1024 / 256) * 144)
	kData := benchBytes(1024 * (1024 / 256) * 144)
	vData := benchBytes(1024 * (1024 / 256) * 144)
	qPrep := PrepareQuantizedWeight(qData, GGMLTypeQ4_K, 1024, 1024)
	kPrep := PrepareQuantizedWeight(kData, GGMLTypeQ4_K, 1024, 1024)
	vPrep := PrepareQuantizedWeight(vData, GGMLTypeQ4_K, 1024, 1024)
	x := benchFloatSlice(1024)
	q := make([]float32, 1024)
	k := make([]float32, 1024)
	v := make([]float32, 1024)
	xs := []float32{}
	b.ReportAllocs()
	b.SetBytes(int64(len(qData) + len(kData) + len(vData) + len(x)*4))
	for b.Loop() {
		ok := MatvecPreparedQ4K3IntoWithXSums(
			qData, qPrep, 1024, 1024,
			kData, kPrep, 1024, 1024,
			vData, vPrep, 1024, 1024,
			x,
			&xs,
			&q,
			&k,
			&v,
		)
		if !ok {
			b.Fatal("MatvecPreparedQ4K3IntoWithXSums returned false")
		}
	}
}

func BenchmarkMatvecQ4K2Q6K_AttentionShape(b *testing.B) {
	const cols, qRows, kRows, vRows = 3072, 4096, 1024, 1024
	qData := benchBytes(qRows * (cols / 256) * 144)
	kData := benchBytes(kRows * (cols / 256) * 144)
	vData := benchBytes(vRows * (cols / 256) * 210)
	x := benchFloatSlice(cols)
	q := make([]float32, qRows)
	k := make([]float32, kRows)
	v := make([]float32, vRows)
	xs := []float32{}
	b.ReportAllocs()
	b.SetBytes(int64(len(qData) + len(kData) + len(vData) + len(x)*4))
	for b.Loop() {
		if !MatvecQ4K2Q6KIntoWithXSums(qData, qRows, cols, kData, kRows, cols, vData, vRows, cols, x, &xs, &q, &k, &v) {
			b.Fatal("MatvecQ4K2Q6KIntoWithXSums returned false")
		}
	}
}

func BenchmarkMatvecQ4K2Q6K_AttentionShapeSeparate(b *testing.B) {
	const cols, qRows, kRows, vRows = 3072, 4096, 1024, 1024
	wq := Weight{Raw: benchBytes(qRows * (cols / 256) * 144), Type: GGMLTypeQ4_K, Rows: qRows, Cols: cols}
	wk := Weight{Raw: benchBytes(kRows * (cols / 256) * 144), Type: GGMLTypeQ4_K, Rows: kRows, Cols: cols}
	wv := Weight{Raw: benchBytes(vRows * (cols / 256) * 210), Type: GGMLTypeQ6_K, Rows: vRows, Cols: cols}
	x := benchFloatSlice(cols)
	q := make([]float32, qRows)
	k := make([]float32, kRows)
	v := make([]float32, vRows)
	b.ReportAllocs()
	b.SetBytes(int64(len(wq.Raw) + len(wk.Raw) + len(wv.Raw) + len(x)*4))
	for b.Loop() {
		wq.MatvecInto(x, &q)
		wk.MatvecInto(x, &k)
		wv.MatvecInto(x, &v)
	}
}

func BenchmarkDotQ5K_4096(b *testing.B) {
	row := benchBytes((4096 / 256) * 176)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = DotQ5KF32(row, x, 4096)
	}
}

func BenchmarkDotQ6K_4096(b *testing.B) {
	row := benchBytes((4096 / 256) * 210)
	x := benchFloatSlice(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = DotQ6KF32(row, x, 4096)
	}
}

func BenchmarkDotQ4KWithXSums_4096(b *testing.B) {
	row := benchBytes((4096 / 256) * 144)
	x := benchFloatSlice(4096)
	scratch := []float32{}
	xs := fillQ4KXSums(x, 4096, &scratch)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		_ = dotQ4KF32WithXSums(row, x, xs, 4096)
	}
}

func BenchmarkFillQ4KXSums_4096(b *testing.B) {
	x := benchFloatSlice(4096)
	scratch := []float32{}
	b.ReportAllocs()
	b.SetBytes(int64(len(x) * 4))
	for b.Loop() {
		_ = fillQ4KXSums(x, 4096, &scratch)
	}
}

func BenchmarkFillQ6KXSums16_4096(b *testing.B) {
	x := benchFloatSlice(4096)
	scratch := []float32{}
	b.ReportAllocs()
	b.SetBytes(int64(len(x) * 4))
	for b.Loop() {
		_ = fillQ6KXSums16(x, 4096, &scratch)
	}
}

func BenchmarkDotQ6KWithXSums_4096(b *testing.B) {
	row := benchBytes((4096 / 256) * 210)
	x := benchFloatSlice(4096)
	scratch := []float32{}
	xs := fillQ6KXSums16(x, 4096, &scratch)
	out := make([]float32, 1)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		dotQ6KRowsWithXSums(row, x, xs, 4096, len(row), 0, 1, out)
	}
}

func BenchmarkDotQ6KWithXSums_3072(b *testing.B) {
	row := benchBytes((3072 / 256) * 210)
	x := benchFloatSlice(3072)
	scratch := []float32{}
	xs := fillQ6KXSums16(x, 3072, &scratch)
	out := make([]float32, 1)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(x)*4))
	for b.Loop() {
		dotQ6KRowsWithXSums(row, x, xs, 3072, len(row), 0, 1, out)
	}
}

func BenchmarkMatvecQ6K_1024x1024(b *testing.B) {
	data := benchBytes(1024 * (1024 / 256) * 210)
	x := benchFloatSlice(1024)
	out := make([]float32, 1024)
	b.ReportAllocs()
	b.SetBytes(int64(len(data) + len(x)*4))
	for b.Loop() {
		MatvecQ6KInto(data, x, 1024, 1024, &out)
	}
}

func BenchmarkArgmaxMatvecQ6K_OutputShape(b *testing.B) {
	const rows, cols = 131072, 3072
	data := benchBytes(rows * (cols / 256) * 210)
	x := benchFloatSlice(cols)
	w := Weight{Raw: data, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols}
	b.ReportAllocs()
	b.SetBytes(int64(len(data) + len(x)*4))
	for b.Loop() {
		if _, ok := w.ArgmaxMatvec(x); !ok {
			b.Fatal("ArgmaxMatvec returned false")
		}
	}
}

func BenchmarkMatvecQ6KOutputShapeThenArgmax(b *testing.B) {
	const rows, cols = 131072, 3072
	data := benchBytes(rows * (cols / 256) * 210)
	x := benchFloatSlice(cols)
	out := make([]float32, rows)
	w := Weight{Raw: data, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols}
	b.ReportAllocs()
	b.SetBytes(int64(len(data) + len(x)*4))
	for b.Loop() {
		w.MatvecInto(x, &out)
		_ = argmaxFiniteToken(out)
	}
}

func BenchmarkMatvecPreparedQ6K_1024x1024(b *testing.B) {
	data := benchBytes(1024 * (1024 / 256) * 210)
	prepared := PrepareQuantizedWeight(data, GGMLTypeQ6_K, 1024, 1024)
	x := benchFloatSlice(1024)
	out := make([]float32, 1024)
	b.ReportAllocs()
	b.SetBytes(int64(len(data) + len(x)*4))
	for b.Loop() {
		if !MatvecPreparedQ6KInto(data, prepared, x, 1024, 1024, &out) {
			b.Fatal("MatvecPreparedQ6KInto returned false")
		}
	}
}

func BenchmarkGenerationConfiguredModel(b *testing.B) {
	modelPath := os.Getenv("GOPHERLLM_BENCH_MODEL")
	if modelPath == "" {
		b.Skip("set GOPHERLLM_BENCH_MODEL=/path/to/model.gguf to benchmark end-to-end generation")
	}
	runner, _, err := RunnerFromPath(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	defer runner.Close()

	options := DefaultGenerationOptions()
	options.MaxTokens = 8
	options.SystemPrompt = ""
	options.Sampler.Temperature = 0
	options.Sampler.TopK = 1
	options.Sampler.TopP = 1

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := runner.Generate("Wer war Albert Einstein?", options); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEmbeddingConfiguredModel is an opt-in end-to-end embedding
// benchmark for a locally installed GGUF. BERT models additionally compare
// batched projections with the single-token reference route; this makes a
// real-model speedup reproducible without hard-coding a developer's model path.
func BenchmarkEmbeddingConfiguredModel(b *testing.B) {
	modelPath := os.Getenv("GOPHERLLM_BENCH_EMBED_MODEL")
	if modelPath == "" {
		b.Skip("set GOPHERLLM_BENCH_EMBED_MODEL=/path/to/embedding-model.gguf to benchmark embeddings")
	}
	if raw := os.Getenv("GOPHERLLM_BENCH_THREADS"); raw != "" {
		threads, err := strconv.Atoi(raw)
		if err != nil || threads < 1 {
			b.Fatalf("GOPHERLLM_BENCH_THREADS must be a positive integer, got %q", raw)
		}
		previousThreads := configuredThreads.Load()
		previousProcs := runtime.GOMAXPROCS(threads)
		SetNumThreads(threads)
		b.Cleanup(func() {
			configuredThreads.Store(previousThreads)
			runtime.GOMAXPROCS(previousProcs)
		})
	}
	runner, _, err := RunnerFromPath(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	defer runner.Close()
	// Tokenizer byte fallbacks can split each visible word into several IDs.
	// Trim the sample for small-context embedding models rather than failing
	// an otherwise useful local benchmark.
	units := 16
	if raw := os.Getenv("GOPHERLLM_BENCH_EMBED_UNITS"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 {
			b.Fatalf("GOPHERLLM_BENCH_EMBED_UNITS must be a positive integer, got %q", raw)
		}
		units = parsed
	}
	prompt := ""
	var tokens []uint32
	for units >= 1 {
		prompt = strings.Repeat("semantic retrieval benchmark ", units)
		tokens = runner.tok.Encode(prompt)
		if runner.config.MaxSeqLen <= 0 || len(tokens) <= runner.config.MaxSeqLen {
			break
		}
		units /= 2
	}
	if runner.config.MaxSeqLen > 0 && len(tokens) > runner.config.MaxSeqLen {
		b.Skip("benchmark prompt exceeds this model's context length")
	}
	b.Logf("embedding benchmark prompt: %d tokens (%d repeated units)", len(tokens), units)

	b.Run("production", func(b *testing.B) {
		if _, err := runner.Embed(prompt); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			if _, err := runner.Embed(prompt); err != nil {
				b.Fatal(err)
			}
		}
	})
	if runner.kind != loadedBERT {
		return
	}
	if runner.bert.UseRoPE {
		b.Run("bert_paired_ffn", func(b *testing.B) {
			var scratch bertEmbeddingScratch
			if _, err := embedBERTWithScratchWithPair(runner.config, runner.bert, tokens, matvecBERTBatch, matvecBERTBatch2, &scratch); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := embedBERTWithScratchWithPair(runner.config, runner.bert, tokens, matvecBERTBatch, matvecBERTBatch2, &scratch); err != nil {
					b.Fatal(err)
				}
			}
		})
		// Keep Nomic's paired gate/up projection separately measurable. This
		// route is otherwise identical to production, but deliberately omits
		// the batch-pair hook as a stable A/B baseline for the optimization.
		b.Run("bert_unpaired_ffn", func(b *testing.B) {
			var scratch bertEmbeddingScratch
			if _, err := embedBERTWithScratch(runner.config, runner.bert, tokens, matvecBERTBatch, &scratch); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				if _, err := embedBERTWithScratch(runner.config, runner.bert, tokens, matvecBERTBatch, &scratch); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	b.Run("bert_serial_projections", func(b *testing.B) {
		var scratch bertEmbeddingScratch
		if _, err := embedBERTWithScratch(runner.config, runner.bert, tokens, matvecBERTSequential, &scratch); err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for b.Loop() {
			if _, err := embedBERTWithScratch(runner.config, runner.bert, tokens, matvecBERTSequential, &scratch); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func benchFloatSlice(n int) []float32 {
	rng := rand.New(rand.NewSource(1))
	out := make([]float32, n)
	for i := range out {
		out[i] = rng.Float32()*2 - 1
	}
	return out
}

func benchBytes(n int) []byte {
	rng := rand.New(rand.NewSource(2))
	out := make([]byte, n)
	for i := range out {
		out[i] = byte(rng.Intn(256))
	}
	return out
}
