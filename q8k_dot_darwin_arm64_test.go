//go:build darwin && arm64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// THIS IS THE TEST THAT VALIDATES THE HAND-ENCODED SDOT KERNELS.
//
// The assembly in q8k_dot_darwin_arm64.s emits SDOT as a raw WORD because Go's
// arm64 assembler has no mnemonic for it, so the encoding is not checked by any
// assembler and cannot be checked by Go's disassembler either. It was also
// written on a non-arm64 machine, where it can be cross-compiled but not run.
//
// So this file is the gate. It differentially tests every assembly kernel
// against quant_q8k_portable.go, which is itself pinned against the amd64
// assembly by quant_q8k_portable_amd64_test.go. If these pass, the SDOT
// encoding and the unpack are right. If they fail, set
// GOPHERLLM_Q8_ACTIVATIONS=0 to fall back to the exact float kernels while the
// kernel is fixed.
//
//	go test -run Q8K ./...

// TestDotInt8AsmPinsSDOT checks SDOT in isolation, before any block format is
// layered on. A wrong WORD encoding shows up here as a wrong number (or a
// SIGILL) with nothing else in the way.
func TestDotInt8AsmPinsSDOT(t *testing.T) {
	// A hand-computed case first, so a systematic error (e.g. lanes summed
	// wrongly, or unsigned instead of signed) is unambiguous.
	a := make([]int8, 16)
	b := make([]int8, 16)
	for i := range a {
		a[i] = int8(i + 1) // 1..16
		b[i] = 1
	}
	// sum(1..16) = 136
	if got := dotInt8Asm(&a[0], &b[0], 16); got != 136 {
		t.Fatalf("dotInt8Asm(1..16, ones) = %d, want 136", got)
	}
	// Signedness: negative weights must stay negative.
	for i := range a {
		a[i] = -1
		b[i] = 1
	}
	if got := dotInt8Asm(&a[0], &b[0], 16); got != -16 {
		t.Fatalf("dotInt8Asm(-1s, ones) = %d, want -16 — SDOT may be decoding as UDOT", got)
	}
	// Extremes, to catch a widening mistake.
	for i := range a {
		a[i] = -128
		b[i] = -128
	}
	if got := dotInt8Asm(&a[0], &b[0], 16); got != 16*16384 {
		t.Fatalf("dotInt8Asm(-128s, -128s) = %d, want %d", got, 16*16384)
	}

	// Then random vectors at several lengths against a scalar reference,
	// covering the 32-wide loop, the 16-wide tail, and their combination.
	rng := rand.New(rand.NewSource(4242))
	for _, n := range []int{16, 32, 48, 64, 256, 4096} {
		x := make([]int8, n)
		y := make([]int8, n)
		for i := range x {
			x[i] = int8(rng.Intn(256) - 128)
			y[i] = int8(rng.Intn(256) - 128)
		}
		var want int32
		for i := range x {
			want += int32(x[i]) * int32(y[i])
		}
		if got := dotInt8Asm(&x[0], &y[0], n); got != want {
			t.Fatalf("n=%d: dotInt8Asm = %d, want %d", n, got, want)
		}
	}
}

// TestQ4KQ8Dots8AsmMatchesPortable checks the nibble unpack and the per
// sub-block accumulation, still before any scale is applied, so a layout error
// is separated from a scale error.
func TestQ4KQ8Dots8AsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4243))
	for iter := range 64 {
		q := make([]byte, 128)
		for i := range q {
			q[i] = byte(rng.Intn(256))
		}
		q8 := make([]int8, 256)
		for i := range q8 {
			q8[i] = int8(rng.Intn(256) - 128)
		}

		var got [8]int32
		q4kQ8Dots8Asm(&q[0], &q8[0], &got[0])

		// Reference: sub-block 2s takes the low nibbles of q[s*32:] against
		// q8[s*64:], sub-block 2s+1 the high nibbles against q8[s*64+32:].
		var want [8]int32
		for s := range 4 {
			var lo, hi int32
			for l := range 32 {
				qv := q[s*32+l]
				lo += int32(qv&0x0f) * int32(q8[s*64+l])
				hi += int32(qv>>4) * int32(q8[s*64+32+l])
			}
			want[2*s] = lo
			want[2*s+1] = hi
		}
		if got != want {
			t.Fatalf("iter %d: q4kQ8Dots8Asm = %v, want %v", iter, got, want)
		}
	}
}

// Q8_0's weights are already int8, so the only thing that can go wrong is the
// stride: each block is 34 bytes, a 2-byte f16 scale then 32 weights, and the
// kernel has to step over every scale. An off-by-two would read weights shifted
// by one byte and still produce plausible magnitudes.
func TestQ8_0Q8Dots8AsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4250))
	for iter := range 64 {
		row := make([]byte, 272)
		for i := range row {
			row[i] = byte(rng.Intn(256))
		}
		q8 := make([]int8, 256)
		for i := range q8 {
			q8[i] = int8(rng.Intn(256) - 128)
		}

		var got [8]int32
		q8_0Q8Dots8Asm(&row[0], &q8[0], &got[0])

		var want [8]int32
		for j := range 8 {
			off := j*34 + 2
			var dot int32
			for l := range 32 {
				dot += int32(int8(row[off+l])) * int32(q8[j*32+l])
			}
			want[j] = dot
		}
		if got != want {
			t.Fatalf("iter %d: q8_0Q8Dots8Asm = %v, want %v", iter, got, want)
		}
	}
}

// Poisoning only the scale bytes must not change any dot product — the sharpest
// test that the stride skips exactly the scale and nothing else.
func TestQ8_0Q8Dots8AsmIgnoresScaleBytes(t *testing.T) {
	row := make([]byte, 272)
	for i := range row {
		row[i] = 3
	}
	q8 := make([]int8, 256)
	for i := range q8 {
		q8[i] = 1
	}
	var before [8]int32
	q8_0Q8Dots8Asm(&row[0], &q8[0], &before[0])
	for j := range 8 {
		row[j*34], row[j*34+1] = 0xff, 0x7f
	}
	var after [8]int32
	q8_0Q8Dots8Asm(&row[0], &q8[0], &after[0])
	if before != after {
		t.Fatalf("scale bytes leaked into the dots: %v became %v", before, after)
	}
	// 32 weights of value 3 against activations of 1.
	for j, v := range before {
		if v != 96 {
			t.Fatalf("qdots[%d] = %d, want 96", j, v)
		}
	}
}

func TestQ8_0DotQ8KRowAsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4251))
	for _, cols := range []int{256, 1024, 3072, 4096} {
		blocks := cols / 256
		row := randomQ8_0Row(rng, cols)
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)

		got := q8_0DotQ8KRow(row, q8, xsc, blocks)
		want := q8_0DotQ8KRowPortable(row, q8, xsc, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != portable %v (diff %v)", cols, got, want, diff)
		}
	}
}

// TestQ5KQ8Dots8AsmMatchesPortable checks the fifth-bit plane merge. The four
// groups are unrolled with a different shift immediate each, so a single wrong
// immediate corrupts exactly one quarter of the block — which is why the qh
// pattern here has to exercise all eight bit positions.
func TestQ5KQ8Dots8AsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4248))
	for iter := range 64 {
		q := make([]byte, 128)
		qh := make([]byte, 32)
		for i := range q {
			q[i] = byte(rng.Intn(256))
		}
		for i := range qh {
			qh[i] = byte(rng.Intn(256))
		}
		q8 := make([]int8, 256)
		for i := range q8 {
			q8[i] = int8(rng.Intn(256) - 128)
		}

		var got [8]int32
		q5kQ8Dots8Asm(&q[0], &qh[0], &q8[0], &got[0])

		var want [8]int32
		for s := range 4 {
			var lo, hi int32
			for l := range 32 {
				qv := q[s*32+l]
				h1 := int32((qh[l] >> (2 * s)) & 1)
				h2 := int32((qh[l] >> (2*s + 1)) & 1)
				lo += (int32(qv&0x0f) + h1*16) * int32(q8[s*64+l])
				hi += (int32(qv>>4) + h2*16) * int32(q8[s*64+32+l])
			}
			want[2*s], want[2*s+1] = lo, hi
		}
		if got != want {
			t.Fatalf("iter %d: q5kQ8Dots8Asm = %v, want %v", iter, got, want)
		}
	}
}

// An all-ones qh must set the fifth bit of every quant, and an all-zero qh must
// set none — the cheapest way to catch a mask or shift direction mistake.
func TestQ5KQ8Dots8AsmBitplaneExtremes(t *testing.T) {
	q8 := make([]int8, 256)
	for i := range q8 {
		q8[i] = 1
	}
	q := make([]byte, 128) // all quants zero, so the result is purely the fifth bit
	for _, tc := range []struct {
		name string
		fill byte
		want int32
	}{
		{"qh all zero", 0x00, 0},
		{"qh all ones", 0xff, 32 * 16}, // every one of 32 elements contributes 16
	} {
		qh := make([]byte, 32)
		for i := range qh {
			qh[i] = tc.fill
		}
		var got [8]int32
		q5kQ8Dots8Asm(&q[0], &qh[0], &q8[0], &got[0])
		for j, v := range got {
			if v != tc.want {
				t.Fatalf("%s: qdots[%d] = %d, want %d (all eight sub-blocks should agree)", tc.name, j, v, tc.want)
			}
		}
	}
}

func TestQ5KDotQ8KRowAsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4249))
	for _, cols := range []int{256, 1024, 3072, 4096} {
		blocks := cols / 256
		row := randomQ5KRow(rng, cols)
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)
		var scratch []float32
		xsums := fillQ4KXSums(x, cols, &scratch)

		got := q5kDotQ8KRow(row, q8, xsc, xsums, blocks)
		want := q5kDotQ8KRowPortable(row, q8, xsc, xsums, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != portable %v (diff %v)", cols, got, want, diff)
		}
	}
}

// TestQ6KQ8Dots16AsmMatchesPortable checks Q6_K's two-plane unpack — low nibble
// from ql plus a 2-bit field from qh — and the half*8 + group*2 + l/16 output
// indexing that lets the Go side walk sc[0..15] straight against out[0..15].
// An off-by-one there is invisible in aggregate but wrong per scale.
func TestQ6KQ8Dots16AsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4246))
	for iter := range 64 {
		ql := make([]byte, 128)
		qh := make([]byte, 64)
		for i := range ql {
			ql[i] = byte(rng.Intn(256))
		}
		for i := range qh {
			qh[i] = byte(rng.Intn(256))
		}
		q8 := make([]int8, 256)
		for i := range q8 {
			q8[i] = int8(rng.Intn(256) - 128)
		}

		var got [16]int32
		q6kQ8Dots16Asm(&ql[0], &qh[0], &q8[0], &got[0])

		var want [16]int32
		for half := range 2 {
			qlh := ql[half*64 : half*64+64]
			qhh := qh[half*32 : half*32+32]
			q8h := q8[half*128 : half*128+128]
			for l := range 32 {
				is := l / 16
				q1 := int32((qlh[l] & 0x0f) | ((qhh[l] & 0x03) << 4))
				q2 := int32((qlh[l+32] & 0x0f) | (((qhh[l] >> 2) & 0x03) << 4))
				q3 := int32((qlh[l] >> 4) | (((qhh[l] >> 4) & 0x03) << 4))
				q4 := int32((qlh[l+32] >> 4) | (((qhh[l] >> 6) & 0x03) << 4))
				want[half*8+0*2+is] += q1 * int32(q8h[l])
				want[half*8+1*2+is] += q2 * int32(q8h[32+l])
				want[half*8+2*2+is] += q3 * int32(q8h[64+l])
				want[half*8+3*2+is] += q4 * int32(q8h[96+l])
			}
		}
		if got != want {
			t.Fatalf("iter %d: q6kQ8Dots16Asm = %v, want %v", iter, got, want)
		}
	}
}

func TestQ6KDotQ8KRowAsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4247))
	for _, cols := range []int{256, 1024, 3072, 4096} {
		blocks := cols / 256
		row := randomQ6KRow(rng, cols)
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)
		var scratch []float32
		xsums := fillQ6KXSums16(x, cols, &scratch)
		ScaleF32(xsums, 32)

		got := q6kDotQ8KRow(row, q8, xsc, xsums, blocks)
		want := q6kDotQ8KRowPortable(row, q8, xsc, xsums, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != portable %v (diff %v)", cols, got, want, diff)
		}
	}
}

// Every self-check must have passed, or the kernel silently degraded to the
// portable path and the differential tests here proved nothing about the
// assembly — they would be comparing the portable kernel against itself.
func TestQ8KSelfChecksPassed(t *testing.T) {
	for name, ok := range map[string]bool{
		"Q4_K":  q4kDotAsmOK,
		"Q5_K":  q5kDotAsmOK,
		"Q6_K":  q6kDotAsmOK,
		"Q8_0":  q8_0DotAsmOK,
		"Q4_0":  q4_0DotAsmOK,
		"Q4_1":  q4_1DotAsmOK,
		"MXFP4": mxfp4DotAsmOK,
	} {
		if !ok {
			t.Errorf("%s SDOT self-check failed at init — the assembly kernel is not in use", name)
		}
	}
}

// The three 32-element-block formats share one test shape: build a row, quantize
// activations, and compare the assembly-backed row kernel against the portable
// one. Table-driven so a fourth legacy format is a table entry, not a fourth
// copy of the body.
func TestLegacyFormatRowKernelsMatchPortable(t *testing.T) {
	cases := []struct {
		name string
		row  func(rng *rand.Rand, cols int) []byte
		asm  func(row []byte, q8 []int8, xsc, xsums []float32, blocks int) float32
		ref  func(row []byte, q8 []int8, xsc, xsums []float32, blocks int) float32
	}{
		{"Q4_0", randomQ4_0Row, q4_0DotQ8KRow, q4_0DotQ8KRowPortable},
		{"Q4_1", randomQ4_1Row, q4_1DotQ8KRow, q4_1DotQ8KRowPortable},
		{
			"MXFP4", randomMXFP4Row,
			func(row []byte, q8 []int8, xsc, _ []float32, blocks int) float32 {
				return mxfp4DotQ8KRow(row, q8, xsc, blocks)
			},
			func(row []byte, q8 []int8, xsc, _ []float32, blocks int) float32 {
				return mxfp4DotQ8KRowPortable(row, q8, xsc, blocks)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(4260))
			for _, cols := range []int{256, 1024, 3072, 4096} {
				blocks := cols / 256
				row := tc.row(rng, cols)
				x := randomVec(rng, cols)
				q8 := make([]int8, cols)
				xsc := make([]float32, blocks)
				q8kQuantizePortable(x, q8, xsc, blocks)
				var scratch []float32
				xsums := fillQ4KXSums(x, cols, &scratch)

				got := tc.asm(row, q8, xsc, xsums, blocks)
				want := tc.ref(row, q8, xsc, xsums, blocks)
				if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
					t.Fatalf("cols=%d: asm %v != portable %v (diff %v)", cols, got, want, diff)
				}
			}
		})
	}
}

// MXFP4 is the one format whose nibbles pair with INTERLEAVED activations (low
// nibble of byte i against activation 2i), which the kernel handles by
// de-interleaving with UZP1/UZP2. Getting that backwards would swap every pair
// and still produce plausible magnitudes, so it gets a direct check: a table
// where index 1 maps to +1 and everything else to 0 makes each output equal the
// activation the low or high nibble should have selected.
func TestMXFP4Q8Dots8AsmDeinterleaves(t *testing.T) {
	row := make([]byte, 136)
	q8 := make([]int8, 256)
	for i := range q8 {
		q8[i] = int8(i % 127)
	}
	// Block 0: every byte 0x01 — low nibble 1, high nibble 0. With the real
	// doubled table, index 1 is 2 and index 0 is 0, so each block dot must be
	// 2 * sum of the EVEN-indexed activations of that block.
	for i := range 16 {
		row[i] = 0x01
	}
	var got [8]int32
	mxfp4Q8Dots8Asm(&row[0], &q8[0], &mxfp4DoubledLUT8[0], &got[0])

	var wantEven int32
	for i := range 16 {
		wantEven += 2 * int32(q8[i*2])
	}
	if got[0] != wantEven {
		t.Fatalf("low-nibble block dot = %d, want %d (UZP1/UZP2 may be swapped)", got[0], wantEven)
	}

	// Now 0x10: high nibble 1, low nibble 0 — the ODD activations.
	for i := range 16 {
		row[i] = 0x10
	}
	mxfp4Q8Dots8Asm(&row[0], &q8[0], &mxfp4DoubledLUT8[0], &got[0])
	var wantOdd int32
	for i := range 16 {
		wantOdd += 2 * int32(q8[i*2+1])
	}
	if got[0] != wantOdd {
		t.Fatalf("high-nibble block dot = %d, want %d (UZP1/UZP2 may be swapped)", got[0], wantOdd)
	}
	if wantEven == wantOdd {
		t.Fatal("test is vacuous: even and odd activation sums coincide")
	}
}

// TestQ4KDotQ8KRowAsmMatchesPortable is the end-to-end check: the assembly-backed
// row kernel against the portable one, over realistic rows.
func TestQ4KDotQ8KRowAsmMatchesPortable(t *testing.T) {
	rng := rand.New(rand.NewSource(4244))
	for _, cols := range []int{256, 1024, 3072, 4096, 9216} {
		blocks := cols / 256
		row := randomQ4KRow(rng, cols)
		x := randomVec(rng, cols)
		for i := 256; i < min(512, cols); i++ {
			x[i] = 0 // exercise the zero-scale block
		}
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)
		var scratch []float32
		xsums := fillQ4KXSums(x, cols, &scratch)

		got := q4kDotQ8KRow(row, q8, xsc, xsums, blocks)
		want := q4kDotQ8KRowPortable(row, q8, xsc, xsums, blocks)
		if math.IsNaN(float64(got)) != math.IsNaN(float64(want)) {
			t.Fatalf("cols=%d: got %v, want %v", cols, got, want)
		}
		// Both sides apply the scales in the same order; the only difference is
		// how the integer dots are summed, which is exact. Allow a hair for the
		// f32 accumulation order of blockInt across sub-blocks.
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != portable %v (diff %v)", cols, got, want, diff)
		}
	}
}

// The whole point of the int8 path is that it agrees with the exact float path
// closely enough to be indistinguishable in output. This asserts the two full
// matvec paths agree on a real weight shape, which is the property that
// actually matters to generation quality.
func TestQ4KMatvecInt8MatchesFloatPath(t *testing.T) {
	rng := rand.New(rand.NewSource(4245))
	const rows, cols = 96, 1024
	data := make([]byte, 0, rows*(cols/256)*144)
	for range rows {
		data = append(data, randomQ4KRow(rng, cols)...)
	}
	x := randomVec(rng, cols)

	int8Out := make([]float32, 0, rows)
	floatOut := make([]float32, 0, rows)

	saved := useQ8Activations
	t.Cleanup(func() { useQ8Activations = saved })

	useQ8Activations = true
	MatvecQ4KInto(data, x, rows, cols, &int8Out)
	useQ8Activations = false
	MatvecQ4KInto(data, x, rows, cols, &floatOut)

	var maxRel float64
	for r := range rows {
		a, b := float64(int8Out[r]), float64(floatOut[r])
		rel := math.Abs(a-b) / (1 + math.Abs(b))
		maxRel = math.Max(maxRel, rel)
	}
	// int8 activations are a 7-bit-mantissa approximation of the input, so a
	// few parts per thousand is expected and is the same error the amd64 path
	// ships with. Anything larger means the kernel is wrong, not merely
	// quantized.
	if maxRel > 5e-3 {
		t.Fatalf("int8 vs float matvec max relative error %v exceeds 5e-3", maxRel)
	}
}
