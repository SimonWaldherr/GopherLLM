//go:build darwin && cgo && metal

package gopherllm

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	metalbackend "github.com/SimonWaldherr/GopherLLM/internal/metal"
)

func TestMetalMinistralSelectivePreparationThresholds(t *testing.T) {
	// Ministral-3 GQA has 3K/5K Q and O projections and 1K K/V
	// projections. Those matrices are deliberately CPU-side: no current Metal
	// operation can dispatch them, so preparing GPU buffers for them would only
	// extend model load and consume unified memory. Its 3B FFN and vocabulary
	// projection remain above the crossover.
	tests := []struct {
		name string
		typ  GGMLType
		rows int
		want bool
	}{
		{name: "3B query", typ: GGMLTypeQ4_K, rows: 3072, want: false},
		{name: "14B attention output", typ: GGMLTypeQ4_K, rows: 5120, want: false},
		{name: "GQA key", typ: GGMLTypeQ4_K, rows: 1024, want: false},
		{name: "GQA value", typ: GGMLTypeQ6_K, rows: 1024, want: false},
		{name: "3B FFN gate", typ: GGMLTypeQ4_K, rows: 9216, want: true},
		{name: "3B FFN down", typ: GGMLTypeQ6_K, rows: 3072, want: true},
		{name: "vocabulary output", typ: GGMLTypeQ6_K, rows: 131072, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := metalWeightMayUseDirect(tt.typ, tt.rows); got != tt.want {
				t.Fatalf("metalWeightMayUseDirect(%v, %d) = %v, want %v", tt.typ, tt.rows, got, tt.want)
			}
		})
	}
	if !MetalAvailable() {
		return
	}
	for _, tt := range tests {
		if tt.want {
			continue
		}
		// The data need not describe a usable matrix because the threshold must
		// reject it before the backend sees it. Before this guard, the same
		// shape allocated a Metal handle despite having no eligible dispatch.
		if w := prepareMetalWeight([]byte{0}, tt.typ, tt.rows, 256, false); w != nil {
			releaseMetalWeight(w)
			t.Fatalf("prepareMetalWeight retained ineligible %s matrix", tt.name)
		}
	}
}

func TestMetalQ4KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ4KDirectMinRows, 256
	rng := rand.New(rand.NewSource(91))
	data := make([]byte, 0, rows*144)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	x := metalTestVector(cols)
	want := []float32{}
	MatvecQ4KInto(data, x, rows, cols, &want)

	w := prepareMetalWeight(data, GGMLTypeQ4_K, rows, cols, false)
	if w == nil {
		t.Fatalf("prepare Q4_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)
	got := []float32{}
	if !matvecMetalQ4KInto(w, x, rows, cols, &got) {
		t.Fatalf("Q4_K Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, got, want)
}

func TestMetalBorrowedQ4KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ4KDirectMinRows, 256
	rng := rand.New(rand.NewSource(92))
	data := make([]byte, 0, rows*144)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	image := make([]byte, 32, 32+len(data))
	image = append(image, data...)
	path := filepath.Join(t.TempDir(), "borrowed-q4k.bin")
	if err := os.WriteFile(path, image, 0o600); err != nil {
		t.Fatal(err)
	}
	mapped, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	borrowed := mapped.Bytes()[32:]

	x := metalTestVector(cols)
	want := []float32{}
	MatvecQ4KInto(data, x, rows, cols, &want)
	w := prepareMetalWeight(borrowed, GGMLTypeQ4_K, rows, cols, true)
	if w == nil {
		t.Fatalf("prepare borrowed Q4_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)
	got := []float32{}
	if !matvecMetalQ4KInto(w, x, rows, cols, &got) {
		t.Fatalf("borrowed Q4_K Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, got, want)
}

func TestMetalQ4KMatvec2MatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ4KDirectMinRows, 256
	rng := rand.New(rand.NewSource(93))
	aData := make([]byte, 0, rows*144)
	bData := make([]byte, 0, rows*144)
	for range rows {
		aData = append(aData, randomQ4KRow(rng, cols)...)
		bData = append(bData, randomQ4KRow(rng, cols)...)
	}
	x := metalTestVector(cols)
	wantA, wantB := []float32{}, []float32{}
	MatvecQ4KInto(aData, x, rows, cols, &wantA)
	MatvecQ4KInto(bData, x, rows, cols, &wantB)

	a := prepareMetalWeight(aData, GGMLTypeQ4_K, rows, cols, false)
	b := prepareMetalWeight(bData, GGMLTypeQ4_K, rows, cols, false)
	if a == nil || b == nil {
		releaseMetalWeight(a)
		releaseMetalWeight(b)
		t.Fatalf("prepare Q4_K Metal weights: %s", MetalError())
	}
	defer releaseMetalWeight(a)
	defer releaseMetalWeight(b)
	gotA, gotB := []float32{}, []float32{}
	if !matvecMetalQ4K2Into(a, b, x, rows, rows, cols, &gotA, &gotB) {
		t.Fatalf("Q4_K Metal matvec2: %s", MetalError())
	}
	assertMetalMatvecClose(t, gotA, wantA)
	assertMetalMatvecClose(t, gotB, wantB)
}

func TestMetalQ5KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ5KDirectMinRows, 256
	rng := rand.New(rand.NewSource(94))
	data := make([]byte, 0, rows*176)
	for range rows {
		data = append(data, randomQ5KRow(rng, cols)...)
	}
	x := metalTestVector(cols)
	want := []float32{}
	MatvecQ5KInto(data, x, rows, cols, &want)

	w := prepareMetalWeight(data, GGMLTypeQ5_K, rows, cols, false)
	if w == nil {
		t.Fatalf("prepare Q5_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)
	got := []float32{}
	if !matvecMetalQ5KInto(w, x, rows, cols, &got) {
		t.Fatalf("Q5_K Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, got, want)
	releaseMetalWeight(w)
	if w.q5 != nil {
		t.Fatal("releaseMetalWeight retained the Q5_K Metal handle")
	}
}

func TestMetalBorrowedQ5KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ5KDirectMinRows, 256
	rng := rand.New(rand.NewSource(98))
	data := make([]byte, 0, rows*176)
	for range rows {
		data = append(data, randomQ5KRow(rng, cols)...)
	}
	image := make([]byte, 32, 32+len(data))
	image = append(image, data...)
	path := filepath.Join(t.TempDir(), "borrowed-q5k.bin")
	if err := os.WriteFile(path, image, 0o600); err != nil {
		t.Fatal(err)
	}
	mapped, err := OpenMmap(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	borrowed := mapped.Bytes()[32:]

	x := metalTestVector(cols)
	want := []float32{}
	MatvecQ5KInto(data, x, rows, cols, &want)
	w := prepareMetalWeight(borrowed, GGMLTypeQ5_K, rows, cols, true)
	if w == nil {
		t.Fatalf("prepare borrowed Q5_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)
	got := []float32{}
	if !matvecMetalQ5KInto(w, x, rows, cols, &got) {
		t.Fatalf("borrowed Q5_K Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, got, want)
}

func TestMetalQ4K2Q6KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	// Exercise a shape that the production dispatch heuristic admits. Narrow
	// GQA K/V projections deliberately remain on CPU because their Metal
	// command-buffer overhead exceeds the saved compute.
	const qRows, kRows, vRows, cols = metalQ4KDirectMinRows, metalQ4KDirectMinRows, metalQ6KDirectMinRows, 256
	rng := rand.New(rand.NewSource(95))
	qData := make([]byte, 0, qRows*144)
	kData := make([]byte, 0, kRows*144)
	vData := make([]byte, 0, vRows*210)
	for range qRows {
		qData = append(qData, randomQ4KRow(rng, cols)...)
	}
	for range kRows {
		kData = append(kData, randomQ4KRow(rng, cols)...)
	}
	for range vRows {
		vData = append(vData, randomQ6KRow(rng, cols)...)
	}
	x := metalTestVector(cols)
	wantQ, wantK, wantV := []float32{}, []float32{}, []float32{}
	MatvecQ4KInto(qData, x, qRows, cols, &wantQ)
	MatvecQ4KInto(kData, x, kRows, cols, &wantK)
	MatvecQ6KInto(vData, x, vRows, cols, &wantV)

	qWeight := prepareMetalWeight(qData, GGMLTypeQ4_K, qRows, cols, false)
	kWeight := prepareMetalWeight(kData, GGMLTypeQ4_K, kRows, cols, false)
	vWeight := prepareMetalWeight(vData, GGMLTypeQ6_K, vRows, cols, false)
	if qWeight == nil || kWeight == nil || vWeight == nil {
		releaseMetalWeight(qWeight)
		releaseMetalWeight(kWeight)
		releaseMetalWeight(vWeight)
		t.Fatalf("prepare mixed Metal weights: %s", MetalError())
	}
	defer releaseMetalWeight(qWeight)
	defer releaseMetalWeight(kWeight)
	defer releaseMetalWeight(vWeight)
	gotQ, gotK, gotV := []float32{}, []float32{}, []float32{}
	if !matvecMetalQ4K2Q6KInto(qWeight, kWeight, vWeight, x, qRows, kRows, vRows, cols, &gotQ, &gotK, &gotV) {
		t.Fatalf("mixed Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, gotQ, wantQ)
	assertMetalMatvecClose(t, gotK, wantK)
	assertMetalMatvecClose(t, gotV, wantV)
}

func TestMetalQ4K2SwiGLUQ6KMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const inputCols, hiddenRows, outputRows = 256, 1024, 256
	rng := rand.New(rand.NewSource(96))
	gateData := make([]byte, 0, hiddenRows*144)
	upData := make([]byte, 0, hiddenRows*144)
	for range hiddenRows {
		gateData = append(gateData, randomQ4KRow(rng, inputCols)...)
		upData = append(upData, randomQ4KRow(rng, inputCols)...)
	}
	downData := make([]byte, 0, outputRows*(hiddenRows/256)*210)
	for range outputRows {
		downData = append(downData, randomQ6KRow(rng, hiddenRows)...)
	}

	x := metalTestVector(inputCols)
	gate, up, hidden, want := []float32{}, []float32{}, make([]float32, hiddenRows), []float32{}
	MatvecQ4KInto(gateData, x, hiddenRows, inputCols, &gate)
	MatvecQ4KInto(upData, x, hiddenRows, inputCols, &up)
	siluMulF32(gate, up, hidden)
	MatvecQ6KInto(downData, hidden, outputRows, hiddenRows, &want)

	gateWeight := metalbackend.PrepareQ4K(gateData, hiddenRows, inputCols, false)
	upWeight := metalbackend.PrepareQ4K(upData, hiddenRows, inputCols, false)
	downWeight := metalbackend.PrepareQ6K(downData, outputRows, hiddenRows, false)
	if gateWeight == nil || upWeight == nil || downWeight == nil {
		metalbackend.Release(gateWeight)
		metalbackend.Release(upWeight)
		metalbackend.Release(downWeight)
		t.Fatalf("prepare fused FFN Metal weights: %s", MetalError())
	}
	defer metalbackend.Release(gateWeight)
	defer metalbackend.Release(upWeight)
	defer metalbackend.Release(downWeight)
	got := make([]float32, outputRows)
	if !metalbackend.MatvecQ4K2SwiGLUQ6K(gateWeight, upWeight, downWeight, x, got) {
		t.Fatalf("fused FFN Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, got, want)
}

// BenchmarkMetalMinistral3BFFN covers the dominant dense block in the
// Ministral-3 3B decode path: Q4_K gate/up projections, SwiGLU, then a Q6_K
// down projection. Keeping the production shape here makes Metal scheduling
// changes measurable without requiring a multi-gigabyte model fixture.
func BenchmarkMetalMinistral3BFFN(b *testing.B) {
	benchmarkMetalMinistralFFN(b, 3072, 9216, 3072)
}

func BenchmarkMetalMinistral14BFFN(b *testing.B) {
	benchmarkMetalMinistralFFN(b, 5120, 16384, 5120)
}

func benchmarkMetalMinistralFFN(b *testing.B, inputCols, hiddenRows, outputRows int) {
	if !MetalAvailable() {
		b.Skip(MetalError())
	}
	rng := rand.New(rand.NewSource(196))
	gateRow := randomQ4KRow(rng, inputCols)
	upRow := randomQ4KRow(rng, inputCols)
	downRow := randomQ6KRow(rng, hiddenRows)
	gateData := make([]byte, hiddenRows*len(gateRow))
	upData := make([]byte, hiddenRows*len(upRow))
	downData := make([]byte, outputRows*len(downRow))
	for r := range hiddenRows {
		copy(gateData[r*len(gateRow):], gateRow)
		copy(upData[r*len(upRow):], upRow)
	}
	for r := range outputRows {
		copy(downData[r*len(downRow):], downRow)
	}
	x := metalTestVector(inputCols)

	b.Run("CPU", func(b *testing.B) {
		gate, up := make([]float32, hiddenRows), make([]float32, hiddenRows)
		hidden := make([]float32, hiddenRows)
		out := make([]float32, outputRows)
		b.ReportAllocs()
		for b.Loop() {
			MatvecQ4K2Into(gateData, hiddenRows, inputCols, upData, hiddenRows, inputCols, x, &gate, &up)
			siluMulF32(gate, up, hidden)
			MatvecQ6KInto(downData, hidden, outputRows, hiddenRows, &out)
		}
	})

	gateWeight := metalbackend.PrepareQ4K(gateData, hiddenRows, inputCols, false)
	upWeight := metalbackend.PrepareQ4K(upData, hiddenRows, inputCols, false)
	downWeight := metalbackend.PrepareQ6K(downData, outputRows, hiddenRows, false)
	if gateWeight == nil || upWeight == nil || downWeight == nil {
		metalbackend.Release(gateWeight)
		metalbackend.Release(upWeight)
		metalbackend.Release(downWeight)
		b.Fatalf("prepare fused FFN Metal weights: %s", MetalError())
	}
	b.Cleanup(func() {
		metalbackend.Release(gateWeight)
		metalbackend.Release(upWeight)
		metalbackend.Release(downWeight)
	})
	gateOut, upOut := make([]float32, hiddenRows), make([]float32, hiddenRows)
	b.Run("MetalQ4K2", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if !metalbackend.MatvecQ4K2(gateWeight, upWeight, x, gateOut, upOut) {
				b.Fatal(MetalError())
			}
		}
	})
	downX := metalTestVector(hiddenRows)
	out := make([]float32, outputRows)
	b.Run("MetalQ6K", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if !metalbackend.MatvecQ6K(downWeight, downX, out) {
				b.Fatal(MetalError())
			}
		}
	})
	b.Run("Metal", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if !metalbackend.MatvecQ4K2SwiGLUQ6K(gateWeight, upWeight, downWeight, x, out) {
				b.Fatal(MetalError())
			}
		}
	})
}

func TestMetalQ6KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	forceExactMetalReference(t)
	const rows, cols = metalQ6KDirectMinRows, 256
	rng := rand.New(rand.NewSource(97))
	data := make([]byte, 0, rows*210)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	x := metalTestVector(cols)
	want := []float32{}
	MatvecQ6KInto(data, x, rows, cols, &want)

	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatalf("prepare Q6_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)
	got := []float32{}
	if !matvecMetalQ6KInto(w, x, rows, cols, &got) {
		t.Fatalf("Q6_K Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, got, want)
	if token, ok := argmaxMetalQ6K(w, x); !ok || token != argmaxFiniteToken(want) {
		t.Fatalf("Metal argmax token = %d, ok=%v, want %d", token, ok, argmaxFiniteToken(want))
	}
	penalized := append([]float32(nil), got...)
	winner := argmaxFiniteToken(penalized)
	recent := []uint32{winner, winner, uint32(len(penalized) + 7)}
	applyRepeatPenalty(penalized, recent, 4)
	if token, ok := argmaxMetalQ6KPenalized(w, x, recent, 4); !ok || token != argmaxFiniteToken(penalized) {
		t.Fatalf("Metal penalized argmax token = %d, ok=%v, want %d", token, ok, argmaxFiniteToken(penalized))
	}

	weights := ModelWeights{Output: Weight{Raw: data, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols, Metal: w}}
	logits := []float32{}
	token, ok := argmaxOutputTokenInto(Config{LogitScale: 1}, weights, &DecodeBuffer{XN: x}, &logits)
	if !ok || token != argmaxFiniteToken(got) {
		t.Fatalf("Metal greedy token = %d, ok=%v, want %d", token, ok, argmaxFiniteToken(got))
	}
}

func TestMetalQ6KArgmaxTieBreaksLowestToken(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	// The SIMD-group reduction in the Metal argmax kernel must retain the
	// sampler's deterministic lowest-index tie break. A zero Q6_K matrix makes
	// every vocabulary logit exactly equal without relying on random data.
	const rows, cols = metalQ6KDirectMinRows, 256
	data := make([]byte, rows*210)
	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols, false)
	if w == nil {
		t.Fatalf("prepare zero Q6_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)

	x := metalTestVector(cols)
	for _, recent := range [][]uint32{nil, {0, 17, 0}} {
		token, ok := argmaxMetalQ6KPenalized(w, x, recent, 1.7)
		if !ok || token != 0 {
			t.Fatalf("Metal tied argmax token = %d, ok=%v, want 0", token, ok)
		}
	}
}

func forceExactMetalReference(t *testing.T) {
	t.Helper()
	saved := useQ8Activations.Load()
	useQ8Activations.Store(false)
	t.Cleanup(func() { useQ8Activations.Store(saved) })
}

func metalTestVector(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i%29)-14) / 11
	}
	return x
}

func assertMetalMatvecClose(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		tol := 1e-3 * math.Max(1, math.Abs(float64(want[i])))
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("output[%d] = %g, want %g (tolerance %g)", i, got[i], want[i], tol)
		}
	}
}
