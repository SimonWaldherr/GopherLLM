package gopherllm

import (
	"math/rand"
	"os"
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
