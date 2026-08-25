package gopherllm

import (
	"fmt"
	"sync"
)

// Weight is one loaded tensor in one of three states: F32 non-nil (owned
// float32 values), Raw quantized bytes, or Raw scalar F32/F16/BF16/F64 bytes.
// The last state is used by out-of-core loads: it keeps a mmap-backed matrix
// in its on-disk representation and converts only values used by a row dot.
// Rows/Cols describe every Raw form; the owned F32 form infers rows from
// len(F32)/cols at the call site for compatibility with existing callers.
type Weight struct {
	F32      []float32
	Raw      []byte
	Type     GGMLType
	Rows     int
	Cols     int
	Prepared *PreparedQuantizedWeight
	Metal    *MetalWeight
	// GPU is this tensor's WebGPU-resident copy (Q4_K/Q6_K only), non-nil
	// only under GOOS=js with a successfully acquired WebGPU device -- see
	// webgpu_js.go/webgpu_stub.go. Mutually exclusive with Metal in
	// practice (disjoint build domains: darwin+cgo+metal vs. js), so
	// MatvecInto tries both with no conflict.
	GPU *GPUWeight
}

// Matvec computes out = W·x, allocating the result. MatvecInto is the
// allocation-free form used on the decode hot path; it dispatches to the
// quant-type-specific parallel kernel in simd.go.
func (w Weight) Matvec(x []float32) []float32 {
	out := make([]float32, max(0, w.Rows))
	w.MatvecInto(x, &out)
	return out
}

func (w Weight) MatvecInto(x []float32, out *[]float32) {
	if w.F32 != nil {
		cols := len(x)
		rows := 0
		if cols > 0 {
			rows = len(w.F32) / cols
		}
		MatvecF32Into(w.F32, x, rows, cols, out)
		return
	}
	if w.rawScalarMatvecInto(x, out) {
		return
	}
	switch w.Type {
	case GGMLTypeQ8_0:
		MatvecQ8_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_0:
		MatvecQ4_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_1:
		MatvecQ4_1Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ5_0:
		MatvecQ5_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ5_1:
		MatvecQ5_1Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ8_1:
		MatvecQ8_1Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ8_K:
		MatvecQ8KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ2_K:
		MatvecQ2KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ3_K:
		MatvecQ3KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ4_K:
		if matvecMetalQ4KInto(w.Metal, x, w.Rows, w.Cols, out) {
			return
		}
		if matvecWebGPUQ4KInto(w.GPU, x, w.Rows, w.Cols, out) {
			return
		}
		if MatvecPreparedQ4KInto(w.Raw, w.Prepared, x, w.Rows, w.Cols, out) {
			return
		}
		MatvecQ4KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ5_K:
		if matvecMetalQ5KInto(w.Metal, x, w.Rows, w.Cols, out) {
			return
		}
		MatvecQ5KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ6_K:
		if matvecMetalQ6KInto(w.Metal, x, w.Rows, w.Cols, out) {
			return
		}
		if matvecWebGPUQ6KInto(w.GPU, x, w.Rows, w.Cols, out) {
			return
		}
		if MatvecPreparedQ6KInto(w.Raw, w.Prepared, x, w.Rows, w.Cols, out) {
			return
		}
		MatvecQ6KInto(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeMXFP4:
		MatvecMXFP4Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeTQ1_0:
		MatvecTQ1_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeTQ2_0:
		MatvecTQ2_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ1_0:
		MatvecQ1_0Into(w.Raw, x, w.Rows, w.Cols, out)
	case GGMLTypeQ2_0:
		MatvecQ2_0Into(w.Raw, x, w.Rows, w.Cols, out)
	default:
		panic(fmt.Sprintf("unsupported quantized matvec: %v", w.Type))
	}
}

// ArgmaxMatvec returns argmax(W*x) without materializing the full logits
// vector. Observed bottleneck: Ministral-3 3B Q4_K_M spends most decode time in
// the 131k-row output projection. For deterministic decoding, the sampler only
// needs the winning token, so this saves the logits writeback and second full
// vocab scan. Risk is limited by using it only for exact greedy-compatible
// sampler settings; rollback is to disable the runtime fast-path. Covers the
// quantized formats llama.cpp's own quantize tool leaves on tied/output
// embeddings: Q6_K (the "_M"/"_L" floor), and Q4_K/Q5_K/Q8_0 (the "_S"
// presets and this project's own --compress skip that floor). Other
// quantized types fall through to the general logits path below.
func (w Weight) ArgmaxMatvec(x []float32) (uint32, bool) {
	if len(x) == 0 {
		return 0, false
	}
	if w.F32 != nil {
		rows := len(w.F32) / len(x)
		if rows <= 0 || rows*len(x) > len(w.F32) {
			return 0, false
		}
		return argmaxMatvecRows(rows, func(row int) float32 {
			off := row * len(x)
			return DotF32(w.F32[off:off+len(x)], x)
		}), true
	}
	if token, ok := w.rawScalarArgmaxMatvec(x); ok {
		return token, true
	}
	if w.Rows <= 0 || w.Cols != len(x) {
		return 0, false
	}
	switch w.Type {
	case GGMLTypeQ6_K:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 256) * 210
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ6KXSums16(x, w.Cols, scratch)
		ScaleF32(xs, 32)
		tok, ok := argmaxQ6KRowsQ8(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ6KRowsWithXSums(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return tok, true
	case GGMLTypeQ4_K:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 256) * 144
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, w.Cols, scratch)
		tok, ok := argmaxQ4KRowsQ8(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ4KRowsWithXSums(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return tok, true
	case GGMLTypeQ5_K:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 256) * 176
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		// Q5_K shares Q4_K's per-sub-block scale/min structure (see
		// MatvecQ5KInto), so the same per-32-element activation sums apply.
		scratch := xsumsScratchPool.Get().(*[]float32)
		xs := fillQ4KXSums(x, w.Cols, scratch)
		tok, ok := argmaxQ5KRowsQ8(w.Raw, x, xs, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ5KRowsFloat(w.Raw, x, w.Rows, w.Cols, rowBytes)
		}
		*scratch = xs
		xsumsScratchPool.Put(scratch)
		return tok, true
	case GGMLTypeQ8_0:
		if w.Cols%256 != 0 {
			return 0, false
		}
		rowBytes := (w.Cols / 32) * 34
		if len(w.Raw) < w.Rows*rowBytes {
			return 0, false
		}
		tok, ok := argmaxQ8_0RowsQ8(w.Raw, x, w.Rows, w.Cols, rowBytes)
		if !ok {
			tok = argmaxQ8_0RowsFloat(w.Raw, x, w.Rows, w.Cols, rowBytes)
		}
		return tok, true
	default:
		return 0, false
	}
}

// argmaxQ6KRowsWithXSums is the f32-activation counterpart of the amd64
// Q8-activation greedy path. It keeps the Q6_K row dot directly in the loop
// instead of passing it through argmaxMatvecRows' function value: on ARM64
// that removes an indirect call for every vocabulary row while preserving the
// exact floating-point dot product, finite-value filtering, and lowest-token
// tie behavior. xsums must already include the Q6_K offset factor (×32).
func argmaxQ6KRowsWithXSums(data []byte, x, xsums []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			var v float32
			if hasQuantSIMD {
				v = dotQ6KF32SIMDWithXSums(data[off:off+rowBytes], x, xsums, cols)
			} else {
				v = DotQ6KF32(data[off:off+rowBytes], x, cols)
			}
			if !finiteLogit(v) {
				continue
			}
			// Rows are visited in ascending order; strict > retains the
			// lowest row index within a worker. The reduction below retains it
			// globally too, matching argmaxFiniteToken.
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
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
	return uint32(bestToken)
}

// argmaxQ4KRowsWithXSums is the exact-float counterpart of argmaxQ4KRowsQ8,
// used when the int8-activation path is unavailable or disabled.
// dotQ4KF32WithXSums already picks the SIMD or scalar kernel internally.
func argmaxQ4KRowsWithXSums(data []byte, x, xsums []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			v := dotQ4KF32WithXSums(data[off:off+rowBytes], x, xsums, cols)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
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
	return uint32(bestToken)
}

// argmaxQ5KRowsFloat is the exact-float counterpart of argmaxQ5KRowsQ8. Q5_K
// has no SIMD-with-xsums float kernel (unlike Q4_K/Q6_K), so this always uses
// the plain per-row dot product, matching MatvecQ5KInto's float fallback.
func argmaxQ5KRowsFloat(data []byte, x []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			v := DotQ5KF32(data[off:off+rowBytes], x, cols)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
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
	return uint32(bestToken)
}

// argmaxQ8_0RowsFloat is the exact-float counterpart of argmaxQ8_0RowsQ8.
func argmaxQ8_0RowsFloat(data []byte, x []float32, rows, cols, rowBytes int) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			off := row * rowBytes
			v := DotQ8_0F32(data[off:off+rowBytes], x, cols)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue {
				localToken, localValue, localFound = row, v, true
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
	return uint32(bestToken)
}

func argmaxMatvecRows(rows int, dot func(row int) float32) uint32 {
	var mu sync.Mutex
	bestToken := 0
	bestValue := negInf32
	found := false
	parallelRows(rows, func(start, end int) {
		localToken := start
		localValue := negInf32
		localFound := false
		for row := start; row < end; row++ {
			v := dot(row)
			if !finiteLogit(v) {
				continue
			}
			if !localFound || v > localValue || (v == localValue && row < localToken) {
				localToken = row
				localValue = v
				localFound = true
			}
		}
		if !localFound {
			return
		}
		mu.Lock()
		if !found || localValue > bestValue || (localValue == bestValue && localToken < bestToken) {
			bestToken = localToken
			bestValue = localValue
			found = true
		}
		mu.Unlock()
	})
	return uint32(bestToken)
}

// Row dequantizes a single weight row (used for token-embedding lookups).
// RowInto is the allocation-free form.
func (w Weight) Row(row, cols int) []float32 {
	out := make([]float32, cols)
	w.RowInto(row, cols, &out)
	return out
}

func (w Weight) RowInto(row, cols int, out *[]float32) {
	ensureLenNoClear(out, cols)
	if w.F32 != nil {
		start := row * cols
		copy(*out, w.F32[start:min(start+cols, len(w.F32))])
		return
	}
	if w.rawScalarRowInto(row, cols, out) {
		return
	}
	switch w.Type {
	case GGMLTypeQ8_0:
		rowBytes := (cols / 32) * 34
		DequantRowQ8_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ4_0:
		rowBytes := (cols / 32) * 18
		DequantRowQ4_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ4_1:
		rowBytes := (cols / 32) * 20
		DequantRowQ4_1Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ5_0:
		rowBytes := (cols / 32) * 22
		DequantRowQ5_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ5_1:
		rowBytes := (cols / 32) * 24
		DequantRowQ5_1Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ8_1:
		rowBytes := (cols / 32) * 36
		DequantRowQ8_1Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ8_K:
		rowBytes := (cols / 256) * 292
		DequantRowQ8KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ2_K:
		rowBytes := (cols / 256) * 84
		DequantRowQ2KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ3_K:
		rowBytes := (cols / 256) * 110
		DequantRowQ3KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ4_K:
		rowBytes := (cols / 256) * 144
		DequantRowQ4KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ5_K:
		rowBytes := (cols / 256) * 176
		DequantRowQ5KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ6_K:
		rowBytes := (cols / 256) * 210
		DequantRowQ6KInto(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeMXFP4:
		rowBytes := (cols / 32) * 17
		DequantRowMXFP4Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeTQ1_0:
		rowBytes := (cols / 256) * 54
		DequantRowTQ1_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeTQ2_0:
		rowBytes := (cols / 256) * 66
		DequantRowTQ2_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ1_0:
		rowBytes := (cols / 128) * 18
		DequantRowQ1_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	case GGMLTypeQ2_0:
		rowBytes := (cols / 64) * 18
		DequantRowQ2_0Into(w.Raw[row*rowBytes:min((row+1)*rowBytes, len(w.Raw))], cols, *out)
	default:
		panic(fmt.Sprintf("unsupported quantized row extraction: %v", w.Type))
	}
}

func (w Weight) RowF32(row, cols int) []float32 {
	if w.F32 != nil {
		start := row * cols
		return w.F32[start : start+cols]
	}
	if rawScalarWeight(w) {
		// A raw scalar mapping has no stable []float32 view. Return a decoded
		// row instead; RowF32 is a convenience accessor, not a mutation API.
		return w.Row(row, cols)
	}
	panic("expected f32 row storage")
}
