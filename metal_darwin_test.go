//go:build darwin && cgo && metal

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func TestMetalQ4KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols = metalQ4KMinRows, 256
	rng := rand.New(rand.NewSource(91))
	data := make([]byte, 0, rows*144)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	x := metalTestVector(cols)
	want := []float32{}
	MatvecQ4KInto(data, x, rows, cols, &want)

	w := prepareMetalWeight(data, GGMLTypeQ4_K, rows, cols)
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

func TestMetalQ4KMatvec2MatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols = metalQ4KMinRows, 256
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

	a := prepareMetalWeight(aData, GGMLTypeQ4_K, rows, cols)
	b := prepareMetalWeight(bData, GGMLTypeQ4_K, rows, cols)
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

func TestMetalQ6KMatvecMatchesCPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip(MetalError())
	}
	const rows, cols = metalQ6KMinRows, 256
	rng := rand.New(rand.NewSource(97))
	data := make([]byte, 0, rows*210)
	for range rows {
		data = append(data, randomQ6KRow(rng, cols)...)
	}
	x := metalTestVector(cols)
	want := []float32{}
	MatvecQ6KInto(data, x, rows, cols, &want)

	w := prepareMetalWeight(data, GGMLTypeQ6_K, rows, cols)
	if w == nil {
		t.Fatalf("prepare Q6_K Metal weight: %s", MetalError())
	}
	defer releaseMetalWeight(w)
	got := []float32{}
	if !matvecMetalQ6KInto(w, x, rows, cols, &got) {
		t.Fatalf("Q6_K Metal matvec: %s", MetalError())
	}
	assertMetalMatvecClose(t, got, want)

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
