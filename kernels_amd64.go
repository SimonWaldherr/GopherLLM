//go:build amd64

package gopherllm

import (
	"os"
	"sync"
)

// hasAVX2 reports whether the CPU (and OS) support the AVX2 + FMA instructions
// used by the hand-written amd64 kernels. It is set once at startup; every SIMD
// entry point falls back to the portable scalar path when it is false. Set
// GOPHERLLM_DISABLE_SIMD to force the scalar path (useful for A/B benchmarking).
var hasAVX2 = detectAVX2()

// hasF16C reports CPUID F16C support (VCVTPH2PS), required by the int8
// activation kernels that convert f16 block scales in-register. Every
// AVX2+FMA CPU in practice also has F16C, but the check is cheap.
var hasF16C = detectF16C()

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() uint32

func detectF16C() bool {
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" {
		return false
	}
	const f16cBit = 1 << 29 // CPUID.1:ECX.F16C
	_, _, ecx1, _ := cpuid(1, 0)
	return ecx1&f16cBit != 0
}

func detectAVX2() bool {
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" {
		return false
	}
	const (
		fmaBit     = 1 << 12 // CPUID.1:ECX.FMA
		osxsaveBit = 1 << 27 // CPUID.1:ECX.OSXSAVE
		avxBit     = 1 << 28 // CPUID.1:ECX.AVX
		avx2Bit    = 1 << 5  // CPUID.7:EBX.AVX2
	)
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&(fmaBit|osxsaveBit|avxBit) != (fmaBit | osxsaveBit | avxBit) {
		return false
	}
	// Confirm the OS has enabled XMM (bit 1) and YMM (bit 2) state saving.
	if xgetbv()&0x6 != 0x6 {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&avx2Bit != 0
}

// dotF32AVX2 computes the dot product of the overlapping prefix of a and b
// using AVX2 + FMA. Implemented in kernels_amd64.s.
func dotF32AVX2(a, b []float32) float32

func dotF32(a, b []float32) float32 {
	if hasAVX2 {
		return dotF32AVX2(a, b)
	}
	return dotF32Scalar(a, b)
}

// useF16KVCache stores KV-cache rows as f16 where AVX2+F16C makes the
// on-the-fly conversion effectively free. Halves attention bandwidth and
// cache footprint; set GOPHERLLM_KV_F16=0 to keep the exact f32 cache
// (A/B testing, accuracy debugging).
var useF16KVCache = hasAVX2 && hasF16C && os.Getenv("GOPHERLLM_KV_F16") != "0"

//go:noescape
func dotF32F16AVX2(a []float32, b []uint16) float32

//go:noescape
func axpyF16AVX2(out []float32, alpha float32, x []uint16)

//go:noescape
func scaleAddF16AVX2(out []float32, alpha float32, x []uint16)

//go:noescape
func f32ToF16RowAVX2(dst []uint16, src []float32)

func dotF32F16(a []float32, b []uint16) float32 {
	if hasAVX2 && hasF16C {
		n8 := min(len(a), len(b)) &^ 7
		return dotF32F16AVX2(a, b) + dotF32F16Scalar(a, b, n8)
	}
	return dotF32F16Scalar(a, b, 0)
}

func axpyF16(out []float32, alpha float32, x []uint16) {
	if hasAVX2 && hasF16C {
		axpyF16AVX2(out, alpha, x)
		axpyF16Scalar(out, alpha, x, min(len(out), len(x))&^7)
		return
	}
	axpyF16Scalar(out, alpha, x, 0)
}

func scaleAddF16(out []float32, alpha float32, x []uint16) {
	if hasAVX2 && hasF16C {
		scaleAddF16AVX2(out, alpha, x)
		scaleAddF16Scalar(out, alpha, x, min(len(out), len(x))&^7)
		return
	}
	scaleAddF16Scalar(out, alpha, x, 0)
}

func f32ToF16Row(dst []uint16, src []float32) {
	if hasAVX2 && hasF16C {
		f32ToF16RowAVX2(dst, src)
		f32ToF16RowScalar(dst, src, min(len(dst), len(src))&^7)
		return
	}
	f32ToF16RowScalar(dst, src, 0)
}

// mxfp4DotQ8KRow is the MXFP4 (gpt-oss) member of the int8-activation
// full-row kernel family: FP4 e2m1 magnitudes are doubled into exact small
// integers via a VPSHUFB lookup, signs ride VPSIGNB on the activation
// operand, and the E8M0 block scale (x0.5 to undo the doubling) folds into
// the float accumulation. Symmetric format — no xsums offset term.
//
//go:noescape
func mxfp4DotQ8KRow(row *byte, q8 *int8, xscales *float32, blocks int) float32

func dotMXFP4RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = mxfp4DotQ8KRow(&data[off], &q8[0], &xscale[0], blocks)
	}
}

// q4_0DotQ8KRow and q4_1DotQ8KRow bring the legacy 32-element block formats
// onto the int8-activation full-row path (see q4kDotQ8KRow). blocks counts
// 256-element superchunks; xsums are the per-32-element float sums of the
// original activations (fillQ4KXSums), which carry Q4_0's -8 offset term and
// Q4_1's +m offset term exactly.
//
//go:noescape
func q4_0DotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

//go:noescape
func q4_1DotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

func dotQ4_0RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q4_0DotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

func dotQ4_1RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q4_1DotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

// useQ8Activations enables the int8-activation fast path for Q4_K and Q6_K
// matvecs: the activation vector is quantized once per matvec to int8 with
// one scale per 256-element block (llama.cpp's Q8_K convention) and each
// weight row is dotted against it by a full-row VPMADDUBSW kernel
// (q4kDotQ8KRow / q6kDotQ8KRow). This is the same activation quantization
// llama.cpp applies unconditionally to every K-quant matvec, so it is on by
// default; set GOPHERLLM_Q8_ACTIVATIONS=0 to force the exact float kernels
// (A/B testing, accuracy debugging).
var useQ8Activations = hasAVX2 && hasF16C && os.Getenv("GOPHERLLM_Q8_ACTIVATIONS") != "0"

var (
	q8ScratchPool     = sync.Pool{New: func() any { s := make([]int8, 0, 16384); return &s }}
	xscaleScratchPool = sync.Pool{New: func() any { s := make([]float32, 0, 64); return &s }}
)

// acquireQ8 quantizes x once (shared across all rows/matrices of a matvec)
// and returns the int8 activations, per-256-element-block scales, and a
// release func that returns the scratch buffers to their pools. cols must be
// a multiple of 256, which every Q4_K/Q6_K matvec entry point guarantees
// before taking this path.
func acquireQ8(x []float32, cols int) (q8 []int8, xscale []float32, release func()) {
	blocks := cols / 256
	q8s := q8ScratchPool.Get().(*[]int8)
	scs := xscaleScratchPool.Get().(*[]float32)
	ensureLenNoClear(q8s, cols)
	ensureLenNoClear(scs, blocks)
	q8 = *q8s
	xscale = *scs
	q8kQuantize(&x[0], &q8[0], &xscale[0], blocks)
	return q8, xscale, func() {
		*q8s = q8
		q8ScratchPool.Put(q8s)
		*scs = xscale
		xscaleScratchPool.Put(scs)
	}
}

// dotQ4KRowsQ8 fills out[start:end] with Q4_K row dots against Q8K-quantized
// activations. xsums must hold the per-32-element float sums of the ORIGINAL
// activations (fillQ4KXSums); the dmin term stays exact float.
func dotQ4KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q4kDotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

// dotQ6KRowsQ8 is the Q6_K analogue. xsums must hold the per-16-element sums
// of the original activations pre-scaled by 32 (fillQ6KXSums16 +
// ScaleF32(xs, 32)), matching the float path's offset folding.
func dotQ6KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q6kDotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

// argmaxQ6KRowsQ8 finds argmax(W·x) over a Q6_K matrix using the same
// int8-activation row kernel as the materializing matvec, without writing the
// logits vector. xsums must be the per-16-element sums pre-scaled by 32.
// Returns false when the int8 path is unavailable so callers keep the exact
// float kernel. Without this, the greedy fast path was the ONLY Q6_K matvec
// still on the old per-block float kernel: 131k rows took 51 ms instead of the
// 17 ms the int8 kernel needs, making "skip the logits writeback" a 3x loss.
func argmaxQ6KRowsQ8(data []byte, x, xsums []float32, rows, cols, rowBytes int) (uint32, bool) {
	if !useQ8Activations || rows <= 0 || cols <= 0 || cols%256 != 0 {
		return 0, false
	}
	blocks := cols / 256
	q8, xsc, release := acquireQ8(x, cols)
	// The dot runs in the inner loop directly rather than through an
	// argmaxMatvecRows closure: a func-value call per row costs ~50 ns against
	// the kernel's ~127 ns, which measured as 25 ms vs 17 ms over 131k rows.
	q8p, xscp, xsumsp := &q8[0], &xsc[0], &xsums[0]
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for r := start; r < end; r++ {
			// Rows are scanned ascending, so a strict > keeps the lowest token
			// id on ties, matching argmaxMatvecRows and argmaxFiniteToken.
			v := q6kDotQ8KRow(&data[r*rowBytes], q8p, xscp, xsumsp, blocks)
			if finiteLogit(v) && (!localFound || v > localValue) {
				localToken, localValue, localFound = r, v, true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken, bestValue, found = localToken, localValue, true
		}
		mu.Unlock()
	})
	release()
	return uint32(bestToken), true
}

type batchQ8Scratch struct {
	q8    []int8
	xsc   []float32
	xsums []float32
}

var batchQ8Pool = sync.Pool{New: func() any { return &batchQ8Scratch{} }}

// matvecBatchQ8 is the batched-prefill analogue of the int8-activation
// matvec: every prompt token's activation vector is quantized once, then each
// weight row is streamed from memory exactly once and dotted against all
// tokens in row tiles small enough to keep both the tile and one token's
// int8 activations cache-resident. Compared to the dequantize-to-f32 batch
// path this cuts activation traffic 4x and weight-decode instructions ~3-6x.
// Returns false (nothing written) for shapes or types it does not handle.
func matvecBatchQ8(w Weight, xs, outs [][]float32) bool {
	p := len(xs)
	if p == 0 {
		return false
	}
	cols := len(xs[0])
	if cols <= 0 || cols%256 != 0 || w.Rows <= 0 || w.Cols != cols {
		return false
	}
	blocks := cols / 256
	var rowBytes, sumsPerTok int
	var kernel func(*byte, *int8, *float32, *float32, int) float32
	switch w.Type {
	case GGMLTypeQ4_K:
		rowBytes = blocks * 144
		sumsPerTok = blocks * 8
		kernel = q4kDotQ8KRow
	case GGMLTypeQ5_K:
		rowBytes = blocks * 176
		sumsPerTok = blocks * 8
		kernel = q5kDotQ8KRow
	case GGMLTypeQ6_K:
		rowBytes = blocks * 210
		sumsPerTok = blocks * 16
		kernel = q6kDotQ8KRow
	case GGMLTypeQ8_0:
		rowBytes = blocks * 272
		// Q8_0 is symmetric (no dmin/offset term), so it needs no xsums;
		// keep one slot per token so the shared indexing stays valid.
		sumsPerTok = 1
		kernel = func(row *byte, q8 *int8, xsc *float32, _ *float32, blocks int) float32 {
			return q8_0DotQ8KRow(row, q8, xsc, blocks)
		}
	case GGMLTypeQ4_0:
		rowBytes = blocks * 144 // 8 x 18-byte legacy blocks per superchunk
		sumsPerTok = blocks * 8
		kernel = q4_0DotQ8KRow
	case GGMLTypeQ4_1:
		rowBytes = blocks * 160 // 8 x 20-byte legacy blocks per superchunk
		sumsPerTok = blocks * 8
		kernel = q4_1DotQ8KRow
	case GGMLTypeMXFP4:
		rowBytes = blocks * 136 // 8 x 17-byte blocks per superchunk
		// Symmetric like Q8_0: no offset term, one dummy xsums slot per token.
		sumsPerTok = 1
		kernel = func(row *byte, q8 *int8, xsc *float32, _ *float32, blocks int) float32 {
			return mxfp4DotQ8KRow(row, q8, xsc, blocks)
		}
	default:
		return false
	}
	if len(w.Raw) < w.Rows*rowBytes {
		return false
	}
	for t := range p {
		if len(xs[t]) < cols || len(outs[t]) < w.Rows {
			return false
		}
	}

	scratch := batchQ8Pool.Get().(*batchQ8Scratch)
	ensureLenNoClear(&scratch.q8, p*cols)
	ensureLenNoClear(&scratch.xsc, p*blocks)
	ensureLenNoClear(&scratch.xsums, p*sumsPerTok)
	q8All := scratch.q8
	xscAll := scratch.xsc
	xsumsAll := scratch.xsums
	for t := range p {
		q8kQuantize(&xs[t][0], &q8All[t*cols], &xscAll[t*blocks], blocks)
		sub := xsumsAll[t*sumsPerTok : (t+1)*sumsPerTok : (t+1)*sumsPerTok]
		switch w.Type {
		case GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ4_0, GGMLTypeQ4_1:
			fillQ4KXSums(xs[t], cols, &sub)
		case GGMLTypeQ6_K:
			fillQ6KXSums16(xs[t], cols, &sub)
			ScaleF32(sub, 32)
		}
	}

	// Row tile sized so a tile of raw rows (~80-120 KB) stays L2-resident
	// while the per-token int8 activations (cols bytes) run through L1.
	const rowTile = 16
	parallelRows(w.Rows, func(start, end int) {
		for tileStart := start; tileStart < end; tileStart += rowTile {
			tileEnd := min(tileStart+rowTile, end)
			for t := range p {
				q8 := q8All[t*cols:]
				xsc := xscAll[t*blocks:]
				xsum := xsumsAll[t*sumsPerTok:]
				out := outs[t]
				for r := tileStart; r < tileEnd; r++ {
					out[r] = kernel(&w.Raw[r*rowBytes], &q8[0], &xsc[0], &xsum[0], blocks)
				}
			}
		}
	})
	batchQ8Pool.Put(scratch)
	return true
}

// q5kDotQ8KRow is the Q5_K analogue of q4kDotQ8KRow: a full-row
// int8-activation dot product over 176-byte Q5_K superblocks (the Q4_K
// layout plus a 32-byte fifth-bit plane). xsums are the per-32-element float
// sums of the ORIGINAL activations (fillQ4KXSums — Q5_K shares Q4_K's
// per-sub-block dmin/min structure exactly).
//
//go:noescape
func q5kDotQ8KRow(q *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

// dotQ5KRowsQ8 fills out[start:end] with Q5_K row dots against Q8K-quantized
// activations.
func dotQ5KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q5kDotQ8KRow(&data[off], &q8[0], &xscale[0], &xsums[0], blocks)
	}
}

// q8_0DotQ8KRow is the Q8_0 analogue of q4kDotQ8KRow/q6kDotQ8KRow: a full-row
// int8-activation dot product against Q8_0-quantized weights (34-byte
// blocks: one f16 scale + 32 signed int8 values, no min/offset term). row
// must point at 8*34=272 bytes per "block" (one Q8K activation super-block
// of 256 elements); q8/xscales are as in the K-quant kernels.
//
//go:noescape
func q8_0DotQ8KRow(row *byte, q8 *int8, xscales *float32, blocks int) float32

// dotQ8_0RowsQ8 fills out[start:end] with Q8_0 row dots against
// Q8K-quantized activations.
func dotQ8_0RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = q8_0DotQ8KRow(&data[off], &q8[0], &xscale[0], blocks)
	}
}

// hasQuantSIMD gates the AVX2 quantized dot-product fast paths on amd64. It
// mirrors hasAVX2 so callers fall back to the portable scalar kernels when the
// CPU lacks AVX2 + FMA.
var hasQuantSIMD = hasAVX2

const hasPreparedQ4K = false

// q4kQDots8 computes the 8 per-sub-block dot products (unsigned 4-bit quants
// times activations) of one Q4_K block. q points at the 128 packed nibble
// bytes, x at 256 floats and qdots at 8 floats. Implemented in
// kernels_amd64.s (AVX2).
//
//go:noescape
func q4kQDots8(q *byte, x *float32, qdots *float32)

func q4kDotPrepared(q *byte, x *float32, scales *float32, mins *float32, xsums *float32, blocks int) float32 {
	panic("q4kDotPrepared is only available on arm64")
}

// q8kQuantize quantizes x to int8 per 256-element block (symmetric absmax,
// one float scale per block — llama.cpp's Q8_K convention), used by
// acquireQ8 for the default int8-activation matvec path.
//
//go:noescape
func q8kQuantize(x *float32, q8 *int8, scales *float32, blocks int)

// q4kDotQ8KRow computes one full Q4_K row dot product against Q8K-quantized
// activations via VPMADDUBSW, with in-register scale/min decode and a single
// horizontal reduction per row. xsums are the per-32-element float sums of
// the original activations (exact dmin term).
//
//go:noescape
func q4kDotQ8KRow(q *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

// q6kDotQ8KRow is the Q6_K full-row analogue. xsums are the per-16-element
// sums of the original activations pre-scaled by 32.
//
//go:noescape
func q6kDotQ8KRow(row *byte, q8 *int8, xscales *float32, xsums *float32, blocks int) float32

// q6kQDots16 computes the 16 per-16-element dot products (unsigned 6-bit
// quants, before the -32 offset, times activations) of one Q6_K block. ql
// points at 128 bytes, qh at 64 bytes, x at 256 floats and qdots at 16 floats.
//
//go:noescape
func q6kQDots16(ql *byte, qh *byte, x *float32, qdots *float32)

// sumF32Groups32 writes one sum per 32-float group from x into out.
//
//go:noescape
func sumF32Groups32(x *float32, out *float32, groups int)

// sumF32Groups16 writes one sum per 16-float group from x into out.
//
//go:noescape
func sumF32Groups16(x *float32, out *float32, groups int)

// Runtime-settable views of the amd64 kernel toggles, so the autotuner can A/B
// them on the live model instead of trusting a compile-time or env-var guess.
// On other targets these are compile-time false
// (kernels_portable_tunables.go).

func q8ActivationsAvailable() bool { return hasAVX2 && hasF16C }

func q8ActivationsEnabled() bool { return useQ8Activations }

func setQ8Activations(on bool) { useQ8Activations = on && q8ActivationsAvailable() }

func kvF16Available() bool { return hasAVX2 && hasF16C }

func kvF16Enabled() bool { return useF16KVCache }

func setKVF16(on bool) { useF16KVCache = on && kvF16Available() }

func cpuFeatureString() string {
	s := "amd64"
	if hasAVX2 {
		s += "+avx2"
	}
	if hasF16C {
		s += "+f16c"
	}
	return s
}

//go:noescape
func siluMulF32AVX2(gate, up, out []float32)

// siluMulF32 computes out[i] = silu(gate[i]) * up[i] (SwiGLU), vectorized
// on AVX2 with a scalar tail.
func siluMulF32(gate, up, out []float32) {
	n := min(len(gate), len(up), len(out))
	i := 0
	if hasAVX2 {
		n8 := n &^ 7
		if n8 > 0 {
			siluMulF32AVX2(gate[:n8], up[:n8], out[:n8])
		}
		i = n8
	}
	siluMulF32Scalar(gate, up, out, i, n)
}

//go:noescape
func axpyF32AVX2(out []float32, alpha float32, x []float32)

//go:noescape
func scaleF32AVX2(out []float32, alpha float32)

//go:noescape
func scaleAddF32AVX2(out []float32, alpha float32, x []float32)

//go:noescape
func mulScaleF32AVX2(x []float32, weight []float32, scale float32, out []float32)

func axpyF32(out []float32, alpha float32, x []float32) {
	if hasAVX2 {
		axpyF32AVX2(out, alpha, x)
		return
	}
	axpyF32Scalar(out, alpha, x)
}

func scaleF32(out []float32, alpha float32) {
	if hasAVX2 {
		scaleF32AVX2(out, alpha)
		return
	}
	scaleF32Scalar(out, alpha)
}

func scaleAddF32(out []float32, alpha float32, x []float32) {
	if hasAVX2 {
		scaleAddF32AVX2(out, alpha, x)
		return
	}
	scaleAddF32Scalar(out, alpha, x)
}

func mulScaleF32(x []float32, weight []float32, scale float32, out []float32) {
	if hasAVX2 {
		mulScaleF32AVX2(x, weight, scale, out)
		return
	}
	mulScaleF32Scalar(x, weight, scale, out)
}
