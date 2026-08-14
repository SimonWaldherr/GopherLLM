//go:build amd64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

// dotQ3KQ8KRowRef is an independently-written scalar reference for
// q3kDotQ8KRow, deliberately not reusing q3kDotQ8KRowPortable (the function
// the assembly replaces) so a shared misreading of the format would not slip
// past both — the same role dotQ2KQ8KRowRef plays for the Q2_K kernel.
//
// In particular this walks the hmask with an explicit per-element bit index
// rather than the portable kernel's running `m <<= 1`, so an off-by-one in
// which hmask bit belongs to which sub-block shows up as a mismatch.
func dotQ3KQ8KRowRef(row []byte, q8 []int8, xscales []float32, blocks int) float32 {
	var sum float32
	for b := range blocks {
		block := row[b*110 : (b+1)*110]
		hmask := block[0:32]
		qs := block[32:96]
		d := F16ToF32(binaryLE16(block[108:]))

		// Unpack the 16 six-bit scales straight from the spec: the low
		// nibbles live in bytes 0..7, the high 2 bits in bytes 8..11.
		var scales [16]int32
		for i := range 8 {
			lo := int32(block[96+i] & 0x0f)
			hi := int32(block[96+8+(i%4)]>>(2*(i/4))) & 3
			scales[i] = lo | hi<<4
		}
		for i := range 8 {
			lo := int32(block[96+i] >> 4)
			hi := int32(block[96+8+(i%4)]>>(2*(i/4)+4)) & 3
			scales[8+i] = lo | hi<<4
		}

		q8b := q8[b*256 : b*256+256]
		var blockInt int32
		for is := range 16 {
			// Sub-block is covers elements [is*16, is*16+16). Its 2-bit
			// codes live in the (is/8)'th 32-byte qs chunk at shift
			// 2*((is/2)%4), byte offset (is%2)*16 + l; its high bit is bit
			// (is/2) of the same hmask byte.
			chunk := (is / 8) * 32
			shift := uint(2 * ((is / 2) % 4))
			half := (is % 2) * 16
			hbit := uint(is / 2)
			dl := scales[is] - 32
			var qdot int32
			for l := range 16 {
				v := int32((qs[chunk+half+l] >> shift) & 3)
				if hmask[half+l]&(1<<hbit) == 0 {
					v -= 4
				}
				qdot += v * int32(q8b[is*16+l])
			}
			blockInt += dl * qdot
		}
		sum += d * xscales[b] * float32(blockInt)
	}
	return sum
}

func randomQ3KRowForAsm(rng *rand.Rand, cols int) []byte {
	row := make([]byte, (cols/256)*110)
	for i := range row {
		row[i] = byte(rng.Intn(256))
	}
	return row
}

// TestQ3KDotQ8KRowAsmMatchesReference is the gate on the AVX2 Q3_K kernel: it
// is the only check that the Σ(u+4h)·a − 4·Σa factorisation in Q3KQ8CHUNK
// actually reproduces Q3_K's per-element −4 bias.
func TestQ3KDotQ8KRowAsmMatchesReference(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewSource(90210))
	for _, cols := range []int{256, 512, 1024, 3072, 4096} {
		blocks := cols / 256
		row := randomQ3KRowForAsm(rng, cols)
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)

		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
		want := dotQ3KQ8KRowRef(row, q8, xsc, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != reference %v (diff %v)", cols, got, want, diff)
		}
	}
}

// The portable kernel is the oracle the rest of the engine is pinned against,
// so the assembly has to agree with it too, not just with the test-local
// reference. Disagreement between the two references would fail the test
// above instead.
func TestQ3KDotQ8KRowAsmMatchesPortable(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewSource(90211))
	for _, cols := range []int{256, 1024, 2048} {
		blocks := cols / 256
		row := randomQ3KRowForAsm(rng, cols)
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)

		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
		want := q3kDotQ8KRowPortable(row, q8, xsc, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("cols=%d: asm %v != portable %v (diff %v)", cols, got, want, diff)
		}
	}
}

// The hmask plane is the whole difficulty of Q3_K, and a kernel that ignored
// it entirely would still pass a loose random test on average. These two
// extremes pin it: with every high bit set no element is biased, with every
// bit clear every element is biased by exactly -4.
func TestQ3KDotQ8KRowAsmHmaskExtremes(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewSource(90212))
	const cols = 1024
	blocks := cols / 256
	for _, fill := range []byte{0x00, 0xff} {
		row := randomQ3KRowForAsm(rng, cols)
		for b := range blocks {
			for i := range 32 {
				row[b*110+i] = fill
			}
		}
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, blocks)
		q8kQuantizePortable(x, q8, xsc, blocks)

		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
		want := dotQ3KQ8KRowRef(row, q8, xsc, blocks)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("hmask=%#02x: asm %v != reference %v (diff %v)", fill, got, want, diff)
		}
	}

	// And a plane that differs per bit position, so a kernel that used one
	// fixed hmask bit for all eight chunks cannot pass.
	row := randomQ3KRowForAsm(rng, cols)
	for b := range blocks {
		for i := range 32 {
			row[b*110+i] = byte(1 << (i % 8))
		}
	}
	x := randomVec(rng, cols)
	q8 := make([]int8, cols)
	xsc := make([]float32, blocks)
	q8kQuantizePortable(x, q8, xsc, blocks)
	got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
	want := dotQ3KQ8KRowRef(row, q8, xsc, blocks)
	if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
		t.Fatalf("striped hmask: asm %v != reference %v (diff %v)", got, want, diff)
	}
}

// The A/B pair for the Q3_K kernel. Run them together and compare medians —
// this machine throttles under sustained load, so a single run of each is not
// a reliable comparison:
//
//	go test -run XXX -bench 'Q3KRowDot' -count 5 .
func BenchmarkQ3KRowDotAsm(b *testing.B) {
	if !hasAVX2 {
		b.Skip("no AVX2 on this CPU")
	}
	row, q8, xsc, blocks := benchQ3KInputs(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(q8)))
	for b.Loop() {
		_ = q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], blocks)
	}
}

func BenchmarkQ3KRowDotPortable(b *testing.B) {
	row, q8, xsc, blocks := benchQ3KInputs(4096)
	b.ReportAllocs()
	b.SetBytes(int64(len(row) + len(q8)))
	for b.Loop() {
		_ = q3kDotQ8KRowPortable(row, q8, xsc, blocks)
	}
}

func benchQ3KInputs(cols int) ([]byte, []int8, []float32, int) {
	rng := rand.New(rand.NewSource(7))
	blocks := cols / 256
	row := randomQ3KRowForAsm(rng, cols)
	x := randomVec(rng, cols)
	q8 := make([]int8, cols)
	xsc := make([]float32, blocks)
	q8kQuantizePortable(x, q8, xsc, blocks)
	return row, q8, xsc, blocks
}

// A scale of 32 decodes to dl = 0, so a kernel that dropped the -32 bias would
// produce a nonzero result where the correct answer is exactly zero. Sweeping
// every six-bit scale value also exercises the packed-nibble unpack across its
// full range, including the high-2-bit half that the low nibbles alone miss.
func TestQ3KDotQ8KRowAsmScaleBiasAndUnpack(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2 on this CPU")
	}
	rng := rand.New(rand.NewSource(90213))
	const cols = 256
	for sc := range 64 {
		row := randomQ3KRowForAsm(rng, cols)
		// Write scale value sc into all 16 sub-blocks: low nibble in bytes
		// 96..103, high 2 bits spread over bytes 104..107.
		for i := range 8 {
			row[96+i] = byte(sc&0x0f) | byte(sc&0x0f)<<4
		}
		for i := range 4 {
			hi := byte(sc>>4) & 3
			row[104+i] = hi | hi<<2 | hi<<4 | hi<<6
		}
		x := randomVec(rng, cols)
		q8 := make([]int8, cols)
		xsc := make([]float32, 1)
		q8kQuantizePortable(x, q8, xsc, 1)

		got := q3kDotQ8KRow(&row[0], &q8[0], &xsc[0], 1)
		want := dotQ3KQ8KRowRef(row, q8, xsc, 1)
		if diff := math.Abs(float64(got - want)); diff > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("scale=%d: asm %v != reference %v (diff %v)", sc, got, want, diff)
		}
		if sc == 32 && got != 0 {
			t.Fatalf("scale=32 gives dl=0, so the row dot must be exactly 0, got %v", got)
		}
	}
}
