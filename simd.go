package gopherllm

// CPU compute kernels: dot products and matrix-vector products over the
// quantized block formats (layouts documented on GGMLType.DataSize),
// dequantization, and the worker pool that parallelizes matvecs across rows.
//
// Every kernel exists in up to three tiers, chosen at runtime:
//
//	portable Go scalar  (always present; the correctness reference the
//	                     differential tests compare against)
//	ARM64 NEON          (hasQuantSIMD const true on arm64; *_arm64.s)
//	x86-64 AVX2+FMA     (hasAVX2/hasQuantSIMD via CPUID; *_amd64.s,
//	                     GOPHERLLM_DISABLE_SIMD=1 forces scalar)
//
// The "xsums" trick used by the Q4_K/Q6_K fast paths: both formats apply a
// per-sub-block affine dequant (val = d*sc*q - dmin*m for Q4_K; a -32 offset
// for Q6_K), so dot(row, x) splits into a quant-dependent term and a term
// that only needs the SUM of x over each sub-block. Those sums are computed
// once per matvec (fillQ4KXSums / fillQ6KXSums16) and shared by every row,
// removing the offset handling from the inner loop.

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

var configuredThreads atomic.Int64

var mxfp4LUT = [...]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}

// SetNumThreads overrides the worker count used by the parallel matvec
// dispatch (default GOMAXPROCS). The CLI's --threads flag calls this and sets
// GOMAXPROCS to the same value.
func SetNumThreads(n int) {
	if n < 1 {
		n = 1
	}
	configuredThreads.Store(int64(n))
}

func numThreads() int {
	if n := int(configuredThreads.Load()); n > 0 {
		return n
	}
	return max(1, runtime.GOMAXPROCS(0))
}

// f16LUT maps every possible f16 bit pattern to its float32 value (256 KB,
// built once at startup). Block scales in every quant format are f16, so this
// lookup sits on the innermost dequant loops; a table beats bit manipulation
// there.
var f16LUT []float32

func init() {
	f16LUT = make([]float32, 65536)
	for i := 0; i < 65536; i++ {
		f16LUT[i] = f16ToF32Soft(uint16(i))
	}
}

func f16ToF32Soft(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		var e uint32
		m := mant
		for (m & 0x400) == 0 {
			m <<= 1
			e++
		}
		m &= 0x3ff
		return math.Float32frombits((sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13))
	}
	if exp == 31 {
		return math.Float32frombits((sign << 31) | (0xff << 23) | (mant << 13))
	}
	return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}

func F16ToF32(h uint16) float32 {
	return f16LUT[h]
}

func DotF32(a, b []float32) float32 {
	return dotF32(a, b)
}

func dotF32Scalar(a, b []float32) float32 {
	n := min(len(a), len(b))
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func AxpyF32(out []float32, alpha float32, x []float32) {
	axpyF32(out, alpha, x)
}

func axpyF32Scalar(out []float32, alpha float32, x []float32) {
	for i := 0; i < min(len(out), len(x)); i++ {
		out[i] += alpha * x[i]
	}
}

func ScaleF32(out []float32, alpha float32) {
	scaleF32(out, alpha)
}

func scaleF32Scalar(out []float32, alpha float32) {
	for i := range out {
		out[i] *= alpha
	}
}

func ScaleAddF32(out []float32, alpha float32, x []float32) {
	scaleAddF32(out, alpha, x)
}

func scaleAddF32Scalar(out []float32, alpha float32, x []float32) {
	for i := 0; i < min(len(out), len(x)); i++ {
		out[i] = out[i]*alpha + x[i]
	}
}

func mulScaleF32Scalar(x []float32, weight []float32, scale float32, out []float32) {
	for i := 0; i < min(len(x), len(weight), len(out)); i++ {
		out[i] = x[i] * weight[i] * scale
	}
}

func MatvecF32(data, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	MatvecF32Into(data, x, rows, cols, &out)
	return out
}

func MatvecF32Into(data, x []float32, rows, cols int, out *[]float32) {
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			row := data[r*cols : min((r+1)*cols, len(data))]
			(*out)[r] = DotF32(row, x)
		}
	})
}

func MatvecQ8_0(data []byte, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	MatvecQ8_0Into(data, x, rows, cols, &out)
	return out
}

func MatvecQ8_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 32) * 34
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ8_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func MatvecQ4_0(data []byte, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	MatvecQ4_0Into(data, x, rows, cols, &out)
	return out
}

func MatvecQ4_0Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 32) * 18
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ4_0F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

var xsumsScratchPool = sync.Pool{New: func() any {
	s := make([]float32, 0, 1024)
	return &s
}}

func MatvecQ4KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 144
	ensureLenNoClear(out, rows)
	if cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, cols, scratch)
		if useQ8Activations {
			q8, xsc, release := acquireQ8(x, cols)
			parallelRows(rows, func(start, end int) {
				dotQ4KRowsQ8(data, q8, xsc, xs, cols, rowBytes, start, end, *out)
			})
			release()
		} else {
			parallelRows(rows, func(start, end int) {
				dotQ4KRowsWithXSums(data, x, xs, cols, rowBytes, start, end, *out)
			})
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ4KF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func MatvecQ4K2Into(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, x []float32, aOut, bOut *[]float32) bool {
	scratch := []float32{}
	return MatvecQ4K2IntoWithXSums(aData, aRows, aCols, bData, bRows, bCols, x, &scratch, aOut, bOut)
}

func MatvecQ4K2IntoWithXSums(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, x []float32, xSums *[]float32, aOut, bOut *[]float32) bool {
	if aCols <= 0 || aCols != bCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := (aCols / 256) * 144
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	xs := fillQ4KXSums(x, aCols, xSums)
	totalRows := aRows + bRows
	if useQ8Activations {
		q8, xsc, release := acquireQ8(x, aCols)
		parallelRows(totalRows, func(start, end int) {
			if as, ae := clippedRange(start, end, 0, aRows); as < ae {
				dotQ4KRowsQ8(aData, q8, xsc, xs, aCols, rowBytes, as, ae, *aOut)
			}
			if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
				dotQ4KRowsQ8(bData, q8, xsc, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
			}
		})
		release()
		return true
	}
	parallelRows(totalRows, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, aRows); as < ae {
			dotQ4KRowsWithXSums(aData, x, xs, aCols, rowBytes, as, ae, *aOut)
		}
		if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
			dotQ4KRowsWithXSums(bData, x, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
		}
	})
	return true
}

func dotQ4KRowsWithXSums(data []byte, x, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = dotQ4KF32WithXSums(data[off:off+rowBytes], x, xsums, cols)
	}
}

func MatvecQ5KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 176
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ5KF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func MatvecQ6KInto(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 256) * 210
	ensureLenNoClear(out, rows)
	if cols > 0 && cols%256 == 0 && len(data) >= rows*rowBytes && len(x) >= cols {
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ6KXSums16(x, cols, scratch)
		parallelRows(rows, func(start, end int) {
			dotQ6KRowsWithXSums(data, x, xs, cols, rowBytes, start, end, *out)
		})
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return
	}
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotQ6KF32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func MatvecQ6K2Into(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, x []float32, aOut, bOut *[]float32) bool {
	if aCols <= 0 || aCols != bCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := (aCols / 256) * 210
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	scratch := xsumsScratchPool.Get().(*[]float32)
	xs := fillQ6KXSums16(x, aCols, scratch)
	totalRows := aRows + bRows
	parallelRows(totalRows, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, aRows); as < ae {
			dotQ6KRowsWithXSums(aData, x, xs, aCols, rowBytes, as, ae, *aOut)
		}
		if bs, be := clippedRange(start, end, aRows, totalRows); bs < be {
			dotQ6KRowsWithXSums(bData, x, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
		}
	})
	*scratch = xs
	xsumsScratchPool.Put(scratch)
	return true
}

func MatvecQ6K3Into(aData []byte, aRows, aCols int, bData []byte, bRows, bCols int, cData []byte, cRows, cCols int, x []float32, aOut, bOut, cOut *[]float32) bool {
	if aCols <= 0 || aCols != bCols || aCols != cCols || aCols != len(x) || aCols%256 != 0 {
		return false
	}
	rowBytes := (aCols / 256) * 210
	if len(aData) < aRows*rowBytes || len(bData) < bRows*rowBytes || len(cData) < cRows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, aRows)
	ensureLenNoClear(bOut, bRows)
	ensureLenNoClear(cOut, cRows)
	scratch := xsumsScratchPool.Get().(*[]float32)
	xs := fillQ6KXSums16(x, aCols, scratch)
	abRows := aRows + bRows
	totalRows := abRows + cRows
	parallelRows(totalRows, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, aRows); as < ae {
			dotQ6KRowsWithXSums(aData, x, xs, aCols, rowBytes, as, ae, *aOut)
		}
		if bs, be := clippedRange(start, end, aRows, abRows); bs < be {
			dotQ6KRowsWithXSums(bData, x, xs, bCols, rowBytes, bs-aRows, be-aRows, *bOut)
		}
		if cs, ce := clippedRange(start, end, abRows, totalRows); cs < ce {
			dotQ6KRowsWithXSums(cData, x, xs, cCols, rowBytes, cs-abRows, ce-abRows, *cOut)
		}
	})
	*scratch = xs
	xsumsScratchPool.Put(scratch)
	return true
}

func dotQ6KRows(data []byte, x []float32, cols, rowBytes, start, end int, out []float32) {
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = DotQ6KF32(data[off:off+rowBytes], x, cols)
	}
}

func dotQ6KRowsWithXSums(data []byte, x, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	if !hasQuantSIMD {
		dotQ6KRows(data, x, cols, rowBytes, start, end, out)
		return
	}
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = dotQ6KF32SIMDWithXSums(data[off:off+rowBytes], x, xsums, cols)
	}
}

// fillQ6KXSums16 computes per-16-element sums of x, used to fold the
// constant -32 offset of Q6_K quants out of the inner dot product.
func fillQ6KXSums16(x []float32, cols int, scratch *[]float32) []float32 {
	groups := cols / 16
	ensureLenNoClear(scratch, groups)
	out := *scratch
	if hasQuantSIMD && groups > 0 && len(x) >= groups*16 {
		sumF32Groups16(&x[0], &out[0], groups)
		return out
	}
	for g := range groups {
		base := g * 16
		if base+16 > len(x) {
			out[g] = 0
			continue
		}
		xBlock := x[base : base+16]
		_ = xBlock[15]
		var s0, s1, s2, s3 float32
		for i := 0; i < 16; i += 4 {
			s0 += xBlock[i]
			s1 += xBlock[i+1]
			s2 += xBlock[i+2]
			s3 += xBlock[i+3]
		}
		out[g] = (s0 + s1) + (s2 + s3)
	}
	return out
}

// dotQ6KF32SIMDWithXSums computes a Q6_K row dot product using the SIMD
// block kernel. xsums must hold per-16-element sums of x (fillQ6KXSums16).
func dotQ6KF32SIMDWithXSums(row []byte, x, xsums []float32, cols int) float32 {
	var qdots [16]float32
	var sum float32
	blocks := cols / 256
	blocks = min(blocks, len(row)/210)
	blocks = min(blocks, len(x)/256)
	blocks = min(blocks, len(xsums)/16)
	if blocks <= 0 {
		return 0
	}
	_ = row[blocks*210-1]
	_ = x[blocks*256-1]
	_ = xsums[blocks*16-1]
	for b := 0; b < blocks; b++ {
		base := b * 210
		d := F16ToF32(uint16(row[base+208]) | uint16(row[base+209])<<8)
		q6kQDots16(&row[base], &row[base+128], &x[b*256], &qdots[0])
		xs := xsums[b*16:]
		_ = xs[15]
		blockSum :=
			float32(int8(row[base+192]))*(qdots[0]-32*xs[0]) +
				float32(int8(row[base+193]))*(qdots[1]-32*xs[1]) +
				float32(int8(row[base+194]))*(qdots[2]-32*xs[2]) +
				float32(int8(row[base+195]))*(qdots[3]-32*xs[3]) +
				float32(int8(row[base+196]))*(qdots[4]-32*xs[4]) +
				float32(int8(row[base+197]))*(qdots[5]-32*xs[5]) +
				float32(int8(row[base+198]))*(qdots[6]-32*xs[6]) +
				float32(int8(row[base+199]))*(qdots[7]-32*xs[7]) +
				float32(int8(row[base+200]))*(qdots[8]-32*xs[8]) +
				float32(int8(row[base+201]))*(qdots[9]-32*xs[9]) +
				float32(int8(row[base+202]))*(qdots[10]-32*xs[10]) +
				float32(int8(row[base+203]))*(qdots[11]-32*xs[11]) +
				float32(int8(row[base+204]))*(qdots[12]-32*xs[12]) +
				float32(int8(row[base+205]))*(qdots[13]-32*xs[13]) +
				float32(int8(row[base+206]))*(qdots[14]-32*xs[14]) +
				float32(int8(row[base+207]))*(qdots[15]-32*xs[15])
		sum += d * blockSum
	}
	return sum
}

func MatvecMXFP4Into(data []byte, x []float32, rows, cols int, out *[]float32) {
	rowBytes := (cols / 32) * 17
	ensureLenNoClear(out, rows)
	parallelRows(rows, func(start, end int) {
		for r := start; r < end; r++ {
			off := r * rowBytes
			(*out)[r] = DotMXFP4F32(data[off:min(off+rowBytes, len(data))], x, cols)
		}
	})
}

func DotQ8_0F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 32
	for b := 0; b < blocks; b++ {
		base := b * 34
		if base+34 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		rBlock := row[base+2 : base+34]
		xBlock := x[b*32 : b*32+32]
		_ = rBlock[31]
		_ = xBlock[31]

		var s0, s1, s2, s3 float32
		for i := 0; i < 32; i += 8 {
			s0 += float32(int8(rBlock[i])) * xBlock[i]
			s1 += float32(int8(rBlock[i+1])) * xBlock[i+1]
			s2 += float32(int8(rBlock[i+2])) * xBlock[i+2]
			s3 += float32(int8(rBlock[i+3])) * xBlock[i+3]

			s0 += float32(int8(rBlock[i+4])) * xBlock[i+4]
			s1 += float32(int8(rBlock[i+5])) * xBlock[i+5]
			s2 += float32(int8(rBlock[i+6])) * xBlock[i+6]
			s3 += float32(int8(rBlock[i+7])) * xBlock[i+7]
		}
		sum += scale * ((s0 + s1) + (s2 + s3))
	}
	return sum
}

func DotQ4_0F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 32
	for b := 0; b < blocks; b++ {
		base := b * 18
		if base+18 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		rBlock := row[base+2 : base+18]
		xBlock := x[b*32 : b*32+32]
		_ = rBlock[15]
		_ = xBlock[31]

		var s0, s1, s2, s3 float32
		for i := 0; i < 16; i += 4 {
			p0 := rBlock[i]
			p1 := rBlock[i+1]
			p2 := rBlock[i+2]
			p3 := rBlock[i+3]

			lo0 := float32(int(p0&0x0f) - 8)
			hi0 := float32(int((p0>>4)&0x0f) - 8)
			lo1 := float32(int(p1&0x0f) - 8)
			hi1 := float32(int((p1>>4)&0x0f) - 8)
			lo2 := float32(int(p2&0x0f) - 8)
			hi2 := float32(int((p2>>4)&0x0f) - 8)
			lo3 := float32(int(p3&0x0f) - 8)
			hi3 := float32(int((p3>>4)&0x0f) - 8)

			s0 += lo0*xBlock[i] + hi0*xBlock[16+i]
			s1 += lo1*xBlock[i+1] + hi1*xBlock[17+i]
			s2 += lo2*xBlock[i+2] + hi2*xBlock[18+i]
			s3 += lo3*xBlock[i+3] + hi3*xBlock[19+i]
		}
		sum += scale * ((s0 + s1) + (s2 + s3))
	}
	return sum
}

func DotQ4KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		block := row[base : base+144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q := block[16:144]
		xBlock := x[b*256 : b*256+256]

		_ = q[127]
		_ = xBlock[255]
		_ = scales[11]

		for step := 0; step < 4; step++ {
			j := step * 64
			is := step * 2

			var sc1, m1, sc2, m2 byte
			if is < 4 {
				sc1, m1 = scales[is]&63, scales[is+4]&63
				sc2, m2 = scales[is+1]&63, scales[is+5]&63
			} else {
				sc1, m1 = (scales[is+4]&0x0f)|((scales[is-4]>>6)<<4), (scales[is+4]>>4)|((scales[is]>>6)<<4)
				sc2, m2 = (scales[is+5]&0x0f)|((scales[is-3]>>6)<<4), (scales[is+5]>>4)|((scales[is+1]>>6)<<4)
			}

			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)

			qSub := q[step*32 : step*32+32]
			xSub1 := xBlock[j : j+32]
			xSub2 := xBlock[j+32 : j+64]

			_ = qSub[31]
			_ = xSub1[31]
			_ = xSub2[31]

			var qd1_0, qd1_1, qd1_2, qd1_3 float32
			var qd2_0, qd2_1, qd2_2, qd2_3 float32
			var xs1_0, xs1_1, xs1_2, xs1_3 float32
			var xs2_0, xs2_1, xs2_2, xs2_3 float32

			for l := 0; l < 32; l += 8 {
				qv0 := qSub[l]
				qv1 := qSub[l+1]
				qv2 := qSub[l+2]
				qv3 := qSub[l+3]
				qv4 := qSub[l+4]
				qv5 := qSub[l+5]
				qv6 := qSub[l+6]
				qv7 := qSub[l+7]

				x1_0 := xSub1[l]
				x1_1 := xSub1[l+1]
				x1_2 := xSub1[l+2]
				x1_3 := xSub1[l+3]
				x1_4 := xSub1[l+4]
				x1_5 := xSub1[l+5]
				x1_6 := xSub1[l+6]
				x1_7 := xSub1[l+7]

				x2_0 := xSub2[l]
				x2_1 := xSub2[l+1]
				x2_2 := xSub2[l+2]
				x2_3 := xSub2[l+3]
				x2_4 := xSub2[l+4]
				x2_5 := xSub2[l+5]
				x2_6 := xSub2[l+6]
				x2_7 := xSub2[l+7]

				qd1_0 += float32(qv0&0x0f) * x1_0
				qd1_1 += float32(qv1&0x0f) * x1_1
				qd1_2 += float32(qv2&0x0f) * x1_2
				qd1_3 += float32(qv3&0x0f) * x1_3
				qd1_0 += float32(qv4&0x0f) * x1_4
				qd1_1 += float32(qv5&0x0f) * x1_5
				qd1_2 += float32(qv6&0x0f) * x1_6
				qd1_3 += float32(qv7&0x0f) * x1_7

				qd2_0 += float32(qv0>>4) * x2_0
				qd2_1 += float32(qv1>>4) * x2_1
				qd2_2 += float32(qv2>>4) * x2_2
				qd2_3 += float32(qv3>>4) * x2_3
				qd2_0 += float32(qv4>>4) * x2_4
				qd2_1 += float32(qv5>>4) * x2_5
				qd2_2 += float32(qv6>>4) * x2_6
				qd2_3 += float32(qv7>>4) * x2_7

				xs1_0 += x1_0 + x1_4
				xs1_1 += x1_1 + x1_5
				xs1_2 += x1_2 + x1_6
				xs1_3 += x1_3 + x1_7

				xs2_0 += x2_0 + x2_4
				xs2_1 += x2_1 + x2_5
				xs2_2 += x2_2 + x2_6
				xs2_3 += x2_3 + x2_7
			}

			qdot1 := (qd1_0 + qd1_1) + (qd1_2 + qd1_3)
			qdot2 := (qd2_0 + qd2_1) + (qd2_2 + qd2_3)
			xsum1 := (xs1_0 + xs1_1) + (xs1_2 + xs1_3)
			xsum2 := (xs2_0 + xs2_1) + (xs2_2 + xs2_3)

			sum += d1*qdot1 - min1*xsum1
			sum += d2*qdot2 - min2*xsum2
		}
	}
	return sum
}

func fillQ4KXSums(x []float32, cols int, scratch *[]float32) []float32 {
	groups := cols / 32
	ensureLenNoClear(scratch, groups)
	out := *scratch
	if hasQuantSIMD && groups > 0 && len(x) >= groups*32 {
		sumF32Groups32(&x[0], &out[0], groups)
		return out
	}
	for g := range groups {
		base := g * 32
		if base+32 > len(x) {
			out[g] = 0
			continue
		}
		xBlock := x[base : base+32]
		_ = xBlock[31]
		var s0, s1, s2, s3 float32
		for i := 0; i < 32; i += 8 {
			s0 += xBlock[i] + xBlock[i+4]
			s1 += xBlock[i+1] + xBlock[i+5]
			s2 += xBlock[i+2] + xBlock[i+6]
			s3 += xBlock[i+3] + xBlock[i+7]
		}
		out[g] = (s0 + s1) + (s2 + s3)
	}
	return out
}

func dotQ4KF32WithXSums(row []byte, x, xsums []float32, cols int) float32 {
	if hasQuantSIMD && cols > 0 && cols%256 == 0 && len(x) >= cols && len(xsums) >= cols/32 {
		return dotQ4KF32SIMDWithXSums(row, x, xsums, cols)
	}
	return dotQ4KF32ScalarWithXSums(row, x, xsums, cols)
}

// dotQ4KF32SIMDWithXSums computes a Q4_K row dot product using the SIMD
// block kernel. xsums must hold per-32-element sums of x (fillQ4KXSums).
func dotQ4KF32SIMDWithXSums(row []byte, x, xsums []float32, cols int) float32 {
	var qdots [8]float32
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		block := row[base : base+144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q4kQDots8(&block[16], &x[b*256], &qdots[0])
		// The 8-wide combine is left in scalar: benchmarking an AVX2 variant
		// showed the horizontal reductions cost more than 8 FMAs at this width.
		for step := 0; step < 4; step++ {
			is := step * 2
			var sc1, m1, sc2, m2 byte
			if is < 4 {
				sc1, m1 = scales[is]&63, scales[is+4]&63
				sc2, m2 = scales[is+1]&63, scales[is+5]&63
			} else {
				sc1, m1 = (scales[is+4]&0x0f)|((scales[is-4]>>6)<<4), (scales[is+4]>>4)|((scales[is]>>6)<<4)
				sc2, m2 = (scales[is+5]&0x0f)|((scales[is-3]>>6)<<4), (scales[is+5]>>4)|((scales[is+1]>>6)<<4)
			}
			xsumBase := b*8 + is
			sum += d*float32(sc1)*qdots[is] - dmin*float32(m1)*xsums[xsumBase]
			sum += d*float32(sc2)*qdots[is+1] - dmin*float32(m2)*xsums[xsumBase+1]
		}
	}
	return sum
}

func dotQ4KF32ScalarWithXSums(row []byte, x, xsums []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		block := row[base : base+144]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		q := block[16:144]
		xBlock := x[b*256 : b*256+256]

		_ = q[127]
		_ = xBlock[255]
		_ = scales[11]

		for step := 0; step < 4; step++ {
			j := step * 64
			is := step * 2

			var sc1, m1, sc2, m2 byte
			if is < 4 {
				sc1, m1 = scales[is]&63, scales[is+4]&63
				sc2, m2 = scales[is+1]&63, scales[is+5]&63
			} else {
				sc1, m1 = (scales[is+4]&0x0f)|((scales[is-4]>>6)<<4), (scales[is+4]>>4)|((scales[is]>>6)<<4)
				sc2, m2 = (scales[is+5]&0x0f)|((scales[is-3]>>6)<<4), (scales[is+5]>>4)|((scales[is+1]>>6)<<4)
			}

			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)

			qSub := q[step*32 : step*32+32]
			xSub1 := xBlock[j : j+32]
			xSub2 := xBlock[j+32 : j+64]

			_ = qSub[31]
			_ = xSub1[31]
			_ = xSub2[31]

			var qd1_0, qd1_1, qd1_2, qd1_3 float32
			var qd2_0, qd2_1, qd2_2, qd2_3 float32

			for l := 0; l < 32; l += 8 {
				qv0 := qSub[l]
				qv1 := qSub[l+1]
				qv2 := qSub[l+2]
				qv3 := qSub[l+3]
				qv4 := qSub[l+4]
				qv5 := qSub[l+5]
				qv6 := qSub[l+6]
				qv7 := qSub[l+7]

				x1_0 := xSub1[l]
				x1_1 := xSub1[l+1]
				x1_2 := xSub1[l+2]
				x1_3 := xSub1[l+3]
				x1_4 := xSub1[l+4]
				x1_5 := xSub1[l+5]
				x1_6 := xSub1[l+6]
				x1_7 := xSub1[l+7]

				x2_0 := xSub2[l]
				x2_1 := xSub2[l+1]
				x2_2 := xSub2[l+2]
				x2_3 := xSub2[l+3]
				x2_4 := xSub2[l+4]
				x2_5 := xSub2[l+5]
				x2_6 := xSub2[l+6]
				x2_7 := xSub2[l+7]

				qd1_0 += float32(qv0&0x0f) * x1_0
				qd1_1 += float32(qv1&0x0f) * x1_1
				qd1_2 += float32(qv2&0x0f) * x1_2
				qd1_3 += float32(qv3&0x0f) * x1_3
				qd1_0 += float32(qv4&0x0f) * x1_4
				qd1_1 += float32(qv5&0x0f) * x1_5
				qd1_2 += float32(qv6&0x0f) * x1_6
				qd1_3 += float32(qv7&0x0f) * x1_7

				qd2_0 += float32(qv0>>4) * x2_0
				qd2_1 += float32(qv1>>4) * x2_1
				qd2_2 += float32(qv2>>4) * x2_2
				qd2_3 += float32(qv3>>4) * x2_3
				qd2_0 += float32(qv4>>4) * x2_4
				qd2_1 += float32(qv5>>4) * x2_5
				qd2_2 += float32(qv6>>4) * x2_6
				qd2_3 += float32(qv7>>4) * x2_7
			}

			qdot1 := (qd1_0 + qd1_1) + (qd1_2 + qd1_3)
			qdot2 := (qd2_0 + qd2_1) + (qd2_2 + qd2_3)
			xsumBase := b*8 + step*2
			xsum1 := xsums[xsumBase]
			xsum2 := xsums[xsumBase+1]

			sum += d1*qdot1 - min1*xsum1
			sum += d2*qdot2 - min2*xsum2
		}
	}
	return sum
}

func DotQ5KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 176
		if base+176 > len(row) {
			break
		}
		block := row[base : base+176]
		d := F16ToF32(binaryLE16(block[0:]))
		dmin := F16ToF32(binaryLE16(block[2:]))
		scales := block[4:16]
		qh := block[16:48]
		q := block[48:176]
		xBlock := x[b*256 : b*256+256]

		_ = qh[31]
		_ = q[127]
		_ = xBlock[255]
		_ = scales[11]

		for step := 0; step < 4; step++ {
			j := step * 64
			is := step * 2

			var sc1, m1, sc2, m2 byte
			if is < 4 {
				sc1, m1 = scales[is]&63, scales[is+4]&63
				sc2, m2 = scales[is+1]&63, scales[is+5]&63
			} else {
				sc1, m1 = (scales[is+4]&0x0f)|((scales[is-4]>>6)<<4), (scales[is+4]>>4)|((scales[is]>>6)<<4)
				sc2, m2 = (scales[is+5]&0x0f)|((scales[is-3]>>6)<<4), (scales[is+5]>>4)|((scales[is+1]>>6)<<4)
			}

			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)

			qSub := q[step*32 : step*32+32]
			xSub1 := xBlock[j : j+32]
			xSub2 := xBlock[j+32 : j+64]

			_ = qSub[31]
			_ = xSub1[31]
			_ = xSub2[31]

			var qd1_0, qd1_1, qd1_2, qd1_3 float32
			var qd2_0, qd2_1, qd2_2, qd2_3 float32
			var xs1_0, xs1_1, xs1_2, xs1_3 float32
			var xs2_0, xs2_1, xs2_2, xs2_3 float32

			switch step {
			case 0:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := (qh_0 & 1) << 4
					hi2_0 := (qh_0 & 2) << 3
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := (qh_1 & 1) << 4
					hi2_1 := (qh_1 & 2) << 3
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := (qh_2 & 1) << 4
					hi2_2 := (qh_2 & 2) << 3
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := (qh_3 & 1) << 4
					hi2_3 := (qh_3 & 2) << 3
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			case 1:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := (qh_0 & 4) << 2
					hi2_0 := (qh_0 & 8) << 1
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := (qh_1 & 4) << 2
					hi2_1 := (qh_1 & 8) << 1
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := (qh_2 & 4) << 2
					hi2_2 := (qh_2 & 8) << 1
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := (qh_3 & 4) << 2
					hi2_3 := (qh_3 & 8) << 1
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			case 2:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := qh_0 & 16
					hi2_0 := (qh_0 & 32) >> 1
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := qh_1 & 16
					hi2_1 := (qh_1 & 32) >> 1
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := qh_2 & 16
					hi2_2 := (qh_2 & 32) >> 1
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := qh_3 & 16
					hi2_3 := (qh_3 & 32) >> 1
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			case 3:
				_ = qSub[31]
				_ = qh[31]
				_ = xSub1[31]
				_ = xSub2[31]
				for l := 0; l < 32; l += 4 {
					qh_0 := qh[l]
					hi1_0 := (qh_0 & 64) >> 2
					hi2_0 := (qh_0 & 128) >> 3
					qv_0 := qSub[l]
					x1_0 := xSub1[l]
					x2_0 := xSub2[l]
					qd1_0 += float32((qv_0&0x0f)+hi1_0) * x1_0
					qd2_0 += float32((qv_0>>4)+hi2_0) * x2_0
					xs1_0 += x1_0
					xs2_0 += x2_0

					qh_1 := qh[l+1]
					hi1_1 := (qh_1 & 64) >> 2
					hi2_1 := (qh_1 & 128) >> 3
					qv_1 := qSub[l+1]
					x1_1 := xSub1[l+1]
					x2_1 := xSub2[l+1]
					qd1_1 += float32((qv_1&0x0f)+hi1_1) * x1_1
					qd2_1 += float32((qv_1>>4)+hi2_1) * x2_1
					xs1_1 += x1_1
					xs2_1 += x2_1

					qh_2 := qh[l+2]
					hi1_2 := (qh_2 & 64) >> 2
					hi2_2 := (qh_2 & 128) >> 3
					qv_2 := qSub[l+2]
					x1_2 := xSub1[l+2]
					x2_2 := xSub2[l+2]
					qd1_2 += float32((qv_2&0x0f)+hi1_2) * x1_2
					qd2_2 += float32((qv_2>>4)+hi2_2) * x2_2
					xs1_2 += x1_2
					xs2_2 += x2_2

					qh_3 := qh[l+3]
					hi1_3 := (qh_3 & 64) >> 2
					hi2_3 := (qh_3 & 128) >> 3
					qv_3 := qSub[l+3]
					x1_3 := xSub1[l+3]
					x2_3 := xSub2[l+3]
					qd1_3 += float32((qv_3&0x0f)+hi1_3) * x1_3
					qd2_3 += float32((qv_3>>4)+hi2_3) * x2_3
					xs1_3 += x1_3
					xs2_3 += x2_3
				}
			}

			qdot1 := (qd1_0 + qd1_1) + (qd1_2 + qd1_3)
			qdot2 := (qd2_0 + qd2_1) + (qd2_2 + qd2_3)
			xsum1 := (xs1_0 + xs1_1) + (xs1_2 + xs1_3)
			xsum2 := (xs2_0 + xs2_1) + (xs2_2 + xs2_3)

			sum += d1*qdot1 - min1*xsum1
			sum += d2*qdot2 - min2*xsum2
		}
	}
	return sum
}

func DotQ6KF32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 256
	for b := 0; b < blocks; b++ {
		base := b * 210
		if base+210 > len(row) {
			break
		}
		block := row[base : base+210]
		ql := block[0:128]
		qh := block[128:192]
		sc := block[192:208]
		d := F16ToF32(binaryLE16(block[208:]))
		xBlock := x[b*256 : b*256+256]

		_ = ql[127]
		_ = qh[63]
		_ = sc[15]
		_ = xBlock[255]

		for step := 0; step < 2; step++ {
			n := step * 128
			qlSub := ql[step*64 : step*64+64]
			qhSub := qh[step*32 : step*32+32]
			scSub := sc[step*8 : step*8+8]
			xSub := xBlock[n : n+128]

			_ = qlSub[63]
			_ = qhSub[31]
			_ = scSub[7]
			_ = xSub[127]

			var s0, s1, s2, s3 float32

			// Precompute scales for l < 16
			d_sc0 := d * float32(int8(scSub[0]))
			d_sc2 := d * float32(int8(scSub[2]))
			d_sc4 := d * float32(int8(scSub[4]))
			d_sc6 := d * float32(int8(scSub[6]))

			for l := 0; l < 16; l += 4 {
				ql0 := qlSub[l]
				ql32_0 := qlSub[l+32]
				qh0 := qhSub[l]
				q1_0 := float32(int((ql0&0x0f)|((qh0&0x03)<<4)) - 32)
				q2_0 := float32(int((ql32_0&0x0f)|(((qh0>>2)&0x03)<<4)) - 32)
				q3_0 := float32(int((ql0>>4)|(((qh0>>4)&0x03)<<4)) - 32)
				q4_0 := float32(int((ql32_0>>4)|(((qh0>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_0 * xSub[l]
				s1 += d_sc2 * q2_0 * xSub[32+l]
				s2 += d_sc4 * q3_0 * xSub[64+l]
				s3 += d_sc6 * q4_0 * xSub[96+l]

				ql1 := qlSub[l+1]
				ql32_1 := qlSub[l+33]
				qh1 := qhSub[l+1]
				q1_1 := float32(int((ql1&0x0f)|((qh1&0x03)<<4)) - 32)
				q2_1 := float32(int((ql32_1&0x0f)|(((qh1>>2)&0x03)<<4)) - 32)
				q3_1 := float32(int((ql1>>4)|(((qh1>>4)&0x03)<<4)) - 32)
				q4_1 := float32(int((ql32_1>>4)|(((qh1>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_1 * xSub[l+1]
				s1 += d_sc2 * q2_1 * xSub[33+l]
				s2 += d_sc4 * q3_1 * xSub[65+l]
				s3 += d_sc6 * q4_1 * xSub[97+l]

				ql2 := qlSub[l+2]
				ql32_2 := qlSub[l+34]
				qh2 := qhSub[l+2]
				q1_2 := float32(int((ql2&0x0f)|((qh2&0x03)<<4)) - 32)
				q2_2 := float32(int((ql32_2&0x0f)|(((qh2>>2)&0x03)<<4)) - 32)
				q3_2 := float32(int((ql2>>4)|(((qh2>>4)&0x03)<<4)) - 32)
				q4_2 := float32(int((ql32_2>>4)|(((qh2>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_2 * xSub[l+2]
				s1 += d_sc2 * q2_2 * xSub[34+l]
				s2 += d_sc4 * q3_2 * xSub[66+l]
				s3 += d_sc6 * q4_2 * xSub[98+l]

				ql3 := qlSub[l+3]
				ql32_3 := qlSub[l+35]
				qh3 := qhSub[l+3]
				q1_3 := float32(int((ql3&0x0f)|((qh3&0x03)<<4)) - 32)
				q2_3 := float32(int((ql32_3&0x0f)|(((qh3>>2)&0x03)<<4)) - 32)
				q3_3 := float32(int((ql3>>4)|(((qh3>>4)&0x03)<<4)) - 32)
				q4_3 := float32(int((ql32_3>>4)|(((qh3>>6)&0x03)<<4)) - 32)

				s0 += d_sc0 * q1_3 * xSub[l+3]
				s1 += d_sc2 * q2_3 * xSub[35+l]
				s2 += d_sc4 * q3_3 * xSub[67+l]
				s3 += d_sc6 * q4_3 * xSub[99+l]
			}

			// Precompute scales for l >= 16
			d_sc1 := d * float32(int8(scSub[1]))
			d_sc3 := d * float32(int8(scSub[3]))
			d_sc5 := d * float32(int8(scSub[5]))
			d_sc7 := d * float32(int8(scSub[7]))

			for l := 16; l < 32; l += 4 {
				ql0 := qlSub[l]
				ql32_0 := qlSub[l+32]
				qh0 := qhSub[l]
				q1_0 := float32(int((ql0&0x0f)|((qh0&0x03)<<4)) - 32)
				q2_0 := float32(int((ql32_0&0x0f)|(((qh0>>2)&0x03)<<4)) - 32)
				q3_0 := float32(int((ql0>>4)|(((qh0>>4)&0x03)<<4)) - 32)
				q4_0 := float32(int((ql32_0>>4)|(((qh0>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_0 * xSub[l]
				s1 += d_sc3 * q2_0 * xSub[32+l]
				s2 += d_sc5 * q3_0 * xSub[64+l]
				s3 += d_sc7 * q4_0 * xSub[96+l]

				ql1 := qlSub[l+1]
				ql32_1 := qlSub[l+33]
				qh1 := qhSub[l+1]
				q1_1 := float32(int((ql1&0x0f)|((qh1&0x03)<<4)) - 32)
				q2_1 := float32(int((ql32_1&0x0f)|(((qh1>>2)&0x03)<<4)) - 32)
				q3_1 := float32(int((ql1>>4)|(((qh1>>4)&0x03)<<4)) - 32)
				q4_1 := float32(int((ql32_1>>4)|(((qh1>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_1 * xSub[l+1]
				s1 += d_sc3 * q2_1 * xSub[33+l]
				s2 += d_sc5 * q3_1 * xSub[65+l]
				s3 += d_sc7 * q4_1 * xSub[97+l]

				ql2 := qlSub[l+2]
				ql32_2 := qlSub[l+34]
				qh2 := qhSub[l+2]
				q1_2 := float32(int((ql2&0x0f)|((qh2&0x03)<<4)) - 32)
				q2_2 := float32(int((ql32_2&0x0f)|(((qh2>>2)&0x03)<<4)) - 32)
				q3_2 := float32(int((ql2>>4)|(((qh2>>4)&0x03)<<4)) - 32)
				q4_2 := float32(int((ql32_2>>4)|(((qh2>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_2 * xSub[l+2]
				s1 += d_sc3 * q2_2 * xSub[34+l]
				s2 += d_sc5 * q3_2 * xSub[66+l]
				s3 += d_sc7 * q4_2 * xSub[98+l]

				ql3 := qlSub[l+3]
				ql32_3 := qlSub[l+35]
				qh3 := qhSub[l+3]
				q1_3 := float32(int((ql3&0x0f)|((qh3&0x03)<<4)) - 32)
				q2_3 := float32(int((ql32_3&0x0f)|(((qh3>>2)&0x03)<<4)) - 32)
				q3_3 := float32(int((ql3>>4)|(((qh3>>4)&0x03)<<4)) - 32)
				q4_3 := float32(int((ql32_3>>4)|(((qh3>>6)&0x03)<<4)) - 32)

				s0 += d_sc1 * q1_3 * xSub[l+3]
				s1 += d_sc3 * q2_3 * xSub[35+l]
				s2 += d_sc5 * q3_3 * xSub[67+l]
				s3 += d_sc7 * q4_3 * xSub[99+l]
			}

			sum += (s0 + s1) + (s2 + s3)
		}
	}
	return sum
}

func DotMXFP4F32(row []byte, x []float32, cols int) float32 {
	var sum float32
	blocks := cols / 32
	for b := 0; b < blocks; b++ {
		base := b * 17
		if base+17 > len(row) {
			break
		}
		scale := float32(math.Pow(2, float64(int(row[base+16])-127)))
		rBlock := row[base : base+16]
		xBlock := x[b*32 : b*32+32]
		_ = rBlock[15]
		_ = xBlock[31]
		for i := 0; i < 16; i++ {
			v := rBlock[i]
			sum += mxfp4LUT[v&0x0f] * scale * xBlock[i*2]
			sum += mxfp4LUT[v>>4] * scale * xBlock[i*2+1]
		}
	}
	return sum
}

func DequantRowQ8_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	for b := range cols / 32 {
		base := b * 34
		if base+34 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		for i := range 32 {
			out[b*32+i] = scale * float32(int8(row[base+2+i]))
		}
	}
	return out
}

func DequantRowQ4_0(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	for b := range cols / 32 {
		base := b * 18
		if base+18 > len(row) {
			break
		}
		scale := F16ToF32(binaryLE16(row[base:]))
		for i := range 16 {
			packed := row[base+2+i]
			out[b*32+i] = scale * float32(int(packed&0x0f)-8)
			out[b*32+16+i] = scale * float32(int((packed>>4)&0x0f)-8)
		}
	}
	return out
}

func DequantRowQ4K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ4KInto(row, cols, out)
	return out
}

func DequantRowQ4KInto(row []byte, cols int, out []float32) {
	for b := range cols / 256 {
		base := b * 144
		if base+144 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		dmin := F16ToF32(binaryLE16(row[base+2:]))
		scales := row[base+4 : base+16]
		q := row[base+16 : base+144]
		yoff := b * 256
		is := 0
		for j := 0; j < 256; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)
			for l := range 32 {
				out[yoff+j+l] = d1*float32(q[l]&0x0f) - min1
				out[yoff+j+32+l] = d2*float32(q[l]>>4) - min2
			}
			q = q[32:]
			is += 2
		}
	}
}

func DequantRowQ5K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	for b := range cols / 256 {
		base := b * 176
		if base+176 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base:]))
		dmin := F16ToF32(binaryLE16(row[base+2:]))
		scales := row[base+4 : base+16]
		qh := row[base+16 : base+48]
		q := row[base+48 : base+176]
		yoff := b * 256
		is := 0
		u1 := byte(1)
		u2 := byte(2)
		for j := 0; j < 256; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1 := d * float32(sc1)
			d2 := d * float32(sc2)
			min1 := dmin * float32(m1)
			min2 := dmin * float32(m2)
			for l := range 32 {
				hi1 := byte(0)
				if qh[l]&u1 != 0 {
					hi1 = 16
				}
				hi2 := byte(0)
				if qh[l]&u2 != 0 {
					hi2 = 16
				}
				out[yoff+j+l] = d1*float32((q[l]&0x0f)+hi1) - min1
				out[yoff+j+32+l] = d2*float32((q[l]>>4)+hi2) - min2
			}
			q = q[32:]
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
	return out
}

func DequantRowQ6K(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	DequantRowQ6KInto(row, cols, out)
	return out
}

func DequantRowQ6KInto(row []byte, cols int, out []float32) {
	for b := range cols / 256 {
		base := b * 210
		if base+210 > len(row) {
			break
		}
		d := F16ToF32(binaryLE16(row[base+208:]))
		ql := row[base:]
		qh := row[base+128 : base+192]
		sc := row[base+192 : base+208]
		yoff := b * 256
		for n := 0; n < 256; n += 128 {
			for l := range 32 {
				is := l / 16
				q1 := float32(int((ql[l]&0x0f)|((qh[l]&0x03)<<4)) - 32)
				q2 := float32(int((ql[l+32]&0x0f)|(((qh[l]>>2)&0x03)<<4)) - 32)
				q3 := float32(int((ql[l]>>4)|(((qh[l]>>4)&0x03)<<4)) - 32)
				q4 := float32(int((ql[l+32]>>4)|(((qh[l]>>6)&0x03)<<4)) - 32)
				out[yoff+n+l] = d * float32(int8(sc[is])) * q1
				out[yoff+n+32+l] = d * float32(int8(sc[is+2])) * q2
				out[yoff+n+64+l] = d * float32(int8(sc[is+4])) * q3
				out[yoff+n+96+l] = d * float32(int8(sc[is+6])) * q4
			}
			ql = ql[64:]
			qh = qh[32:]
			sc = sc[8:]
		}
	}
}

func DequantRowMXFP4(row []byte, cols int) []float32 {
	out := make([]float32, cols)
	for b := range cols / 32 {
		base := b * 17
		if base+17 > len(row) {
			break
		}
		scale := float32(math.Pow(2, float64(int(row[base+16])-127)))
		rBlock := row[base : base+16]
		outBlock := out[b*32 : b*32+32]
		_ = rBlock[15]
		_ = outBlock[31]
		for i := range 16 {
			v := rBlock[i]
			outBlock[i*2] = scale * mxfp4LUT[v&0x0f]
			outBlock[i*2+1] = scale * mxfp4LUT[v>>4]
		}
	}
	return out
}

func getScaleMinK4(j int, q []byte) (byte, byte) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// parallelRows splits [0, rows) across the persistent worker pool, running
// fn(start, end) on each range concurrently and returning when all are done.
// Small row counts (< 8 rows per worker) run inline — the dispatch overhead
// would exceed the work.
func parallelRows(rows int, fn func(start, end int)) {
	threads := min(numThreads(), rows)
	if threads <= 1 || rows < threads*8 {
		fn(0, rows)
		return
	}
	dispatchParallel(threads, rows, fn)
}

// parallelChunks splits n items across the worker pool without the minimum
// per-thread row count required by parallelRows. Intended for coarse-grained
// items (e.g. attention heads) where even a single item is substantial work.
func parallelChunks(n int, fn func(start, end int)) {
	threads := min(numThreads(), n)
	if threads <= 1 {
		fn(0, n)
		return
	}
	dispatchParallel(threads, n, fn)
}

func dispatchParallel(threads, rows int, fn func(start, end int)) {
	pool := getRowWorkerPool(threads)
	// Issue more chunks than workers so faster cores naturally pick up the
	// slack of slower ones (e.g. efficiency cores on Apple Silicon).
	chunks := threads
	if rows >= threads*16 {
		chunks = min(threads*4, cap(pool.jobs))
	}
	done := rowDonePool.Get().(chan struct{})
	if cap(done) < chunks {
		done = make(chan struct{}, chunks)
	}
	for w := range chunks {
		start := rows * w / chunks
		end := rows * (w + 1) / chunks
		pool.jobs <- rowJob{start: start, end: end, fn: fn, done: done}
	}
	for range chunks {
		<-done
	}
	if cap(done) <= 128 {
		rowDonePool.Put(done)
	}
}

type rowJob struct {
	start int
	end   int
	fn    func(start, end int)
	done  chan<- struct{}
}

// rowWorkerPool is the process-wide pool of matvec worker goroutines. It is
// created lazily at the first parallel dispatch and rebuilt (old workers
// stopped) if SetNumThreads changes the thread count. Keeping the goroutines
// alive across calls avoids per-matvec spawn cost — matvecs run ~30x per
// generated token.
type rowWorkerPool struct {
	threads int
	jobs    chan rowJob
	stop    chan struct{}
}

var (
	rowPoolMu   sync.Mutex
	rowPool     *rowWorkerPool
	rowDonePool = sync.Pool{New: func() any {
		return make(chan struct{}, 64)
	}}
)

func getRowWorkerPool(threads int) *rowWorkerPool {
	rowPoolMu.Lock()
	defer rowPoolMu.Unlock()
	if rowPool != nil && rowPool.threads == threads {
		return rowPool
	}
	if rowPool != nil {
		close(rowPool.stop)
	}
	pool := &rowWorkerPool{
		threads: threads,
		jobs:    make(chan rowJob, threads*4),
		stop:    make(chan struct{}),
	}
	for range threads {
		go rowWorker(pool.jobs, pool.stop)
	}
	rowPool = pool
	return pool
}

func rowWorker(jobs <-chan rowJob, stop <-chan struct{}) {
	for {
		select {
		case job := <-jobs:
			job.fn(job.start, job.end)
			job.done <- struct{}{}
		case <-stop:
			return
		}
	}
}

// ensureLen resizes *s to length n, reusing capacity when possible, and
// zeroes the contents. ensureLenNoClear is the same without the zeroing, for
// buffers that are fully overwritten anyway — it is the standard idiom for
// every scratch buffer on the decode path.
func ensureLen[T any](s *[]T, n int) {
	if cap(*s) < n {
		*s = make([]T, n)
		return
	}
	*s = (*s)[:n]
	var zero T
	for i := range *s {
		(*s)[i] = zero
	}
}

func ensureLenNoClear[T any](s *[]T, n int) {
	if cap(*s) < n {
		*s = make([]T, n)
		return
	}
	*s = (*s)[:n]
}

func binaryLE16(b []byte) uint16 {
	if len(b) < 2 {
		return 0
	}
	return uint16(b[0]) | uint16(b[1])<<8
}
