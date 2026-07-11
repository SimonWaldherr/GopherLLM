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

func TestMetalQ4KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
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

func TestMetalQ4K2Q6KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const qRows, kRows, vRows, cols = 4096, 1024, 1024, 256
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

func TestMetalQ6KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
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

	weights := ModelWeights{Output: Weight{Raw: data, Type: GGMLTypeQ6_K, Rows: rows, Cols: cols, Metal: w}}
	logits := []float32{}
	token, ok := argmaxOutputTokenInto(Config{LogitScale: 1}, weights, &DecodeBuffer{XN: x}, &logits)
	if !ok || token != argmaxFiniteToken(got) {
		t.Fatalf("Metal greedy token = %d, ok=%v, want %d", token, ok, argmaxFiniteToken(got))
	}
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
