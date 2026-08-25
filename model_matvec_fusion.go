package gopherllm

// tryMatvec3Into / tryMatvec2Into route same-typed Q4_K or Q6_K weight groups
// (Q/K/V projections; FFN gate+up) through the fused kernels that share one
// activation-sums pass and one worker-pool dispatch. They return false —
// having written nothing — whenever types or shapes don't line up, and the
// caller falls back to independent matvecs.
func tryMatvec3Into(wq, wk, wv Weight, x []float32, q4kXSums *[]float32, q, k, v *[]float32) bool {
	if wq.Type != wk.Type || wq.Type != wv.Type || wq.Cols != wk.Cols || wq.Cols != wv.Cols || wq.Cols != len(x) || wq.F32 != nil || wk.F32 != nil || wv.F32 != nil {
		return false
	}
	switch wq.Type {
	case GGMLTypeQ4_K:
		if wq.Prepared != nil && wk.Prepared != nil && wv.Prepared != nil {
			if MatvecPreparedQ4K3IntoWithXSums(
				wq.Raw, wq.Prepared, wq.Rows, wq.Cols,
				wk.Raw, wk.Prepared, wk.Rows, wk.Cols,
				wv.Raw, wv.Prepared, wv.Rows, wv.Cols,
				x,
				q4kXSums,
				q,
				k,
				v,
			) {
				return true
			}
		}
		return Q4KMatvec3IntoWithXSums(
			Q4KMatrix{Data: wq.Raw, Rows: wq.Rows, Cols: wq.Cols},
			Q4KMatrix{Data: wk.Raw, Rows: wk.Rows, Cols: wk.Cols},
			Q4KMatrix{Data: wv.Raw, Rows: wv.Rows, Cols: wv.Cols},
			x,
			q4kXSums,
			q,
			k,
			v,
		)
	case GGMLTypeQ6_K:
		return MatvecQ6K3Into(wq.Raw, wq.Rows, wq.Cols, wk.Raw, wk.Rows, wk.Cols, wv.Raw, wv.Rows, wv.Cols, x, q, k, v)
	default:
		return matvecSameType3Into(wq, wk, wv, x, q, k, v)
	}
}

// tryMatvecAttentionInto keeps the attention projection fast for mixed-quant
// GGUFs. Bottleneck: Ministral-3-3B-Q4_K_M stores Q/K as Q4_K but V as Q6_K,
// so the previous all-or-nothing QKV fusion missed the reusable Q4_K xsums and
// worker dispatch for Q+K, and still dispatched V separately. Change: try full
// QKV fusion first, then the common mixed Q4_K/Q4_K/Q6_K one-dispatch path,
// then pairwise fusion for other same-typed projection pairs. Expected effect:
// lower decode latency on mixed Q4_K/Q6_K attention blocks. Risk: small extra
// branch cost. Rollback: replace this call with the former three independent
// MatvecInto calls.
func tryMatvecAttentionInto(wq, wk, wv Weight, x []float32, q4kXSums *[]float32, q, k, v *[]float32) {
	if matvecMetalQ4K2Q6KInto(wq.Metal, wk.Metal, wv.Metal, x, wq.Rows, wk.Rows, wv.Rows, wq.Cols, q, k, v) {
		return
	}
	if tryMatvec3Into(wq, wk, wv, x, q4kXSums, q, k, v) {
		return
	}
	if wq.F32 == nil && wk.F32 == nil && wv.F32 == nil &&
		wq.Type == GGMLTypeQ4_K && wk.Type == GGMLTypeQ4_K && wv.Type == GGMLTypeQ6_K &&
		MatvecQ4K2Q6KIntoWithXSums(wq.Raw, wq.Rows, wq.Cols, wk.Raw, wk.Rows, wk.Cols, wv.Raw, wv.Rows, wv.Cols, x, q4kXSums, q, k, v) {
		return
	}
	if tryMatvec2Into(wq, wk, x, q4kXSums, q, k) {
		wv.MatvecInto(x, v)
		return
	}
	if tryMatvec2Into(wq, wv, x, q4kXSums, q, v) {
		wk.MatvecInto(x, k)
		return
	}
	if tryMatvec2Into(wk, wv, x, q4kXSums, k, v) {
		wq.MatvecInto(x, q)
		return
	}
	wq.MatvecInto(x, q)
	wk.MatvecInto(x, k)
	wv.MatvecInto(x, v)
}

func tryMatvec2Into(a, b Weight, x []float32, q4kXSums *[]float32, aOut, bOut *[]float32) bool {
	if a.Type != b.Type || a.Cols != b.Cols || a.Cols != len(x) || a.F32 != nil || b.F32 != nil {
		return false
	}
	switch a.Type {
	case GGMLTypeQ4_K:
		if matvecMetalQ4K2Into(a.Metal, b.Metal, x, a.Rows, b.Rows, a.Cols, aOut, bOut) {
			return true
		}
		if a.Prepared != nil && b.Prepared != nil {
			if MatvecPreparedQ4K2IntoWithXSums(a.Raw, a.Prepared, a.Rows, a.Cols, b.Raw, b.Prepared, b.Rows, b.Cols, x, q4kXSums, aOut, bOut) {
				return true
			}
		}
		return MatvecQ4K2IntoWithXSums(a.Raw, a.Rows, a.Cols, b.Raw, b.Rows, b.Cols, x, q4kXSums, aOut, bOut)
	case GGMLTypeQ6_K:
		return MatvecQ6K2Into(a.Raw, a.Rows, a.Cols, b.Raw, b.Rows, b.Cols, x, aOut, bOut)
	default:
		return matvecSameType2Into(a, b, x, aOut, bOut)
	}
}

// matvecSameType{2,3}Into fuse ordinary same-format projections into one
// worker-pool dispatch. Unlike the Q4_K/Q6_K specializations above they do
// not share activation preprocessing; they avoid only the repeated channel
// dispatch. That is still valuable for Q4_0-heavy StableLM/InternLM GGUFs,
// whose Q/K/V and gate/up projections otherwise launch independently.
func matvecSameType2Into(a, b Weight, x []float32, aOut, bOut *[]float32) bool {
	dot, rowBytes, ok := sameTypeQuantDot(a, b, x)
	if !ok {
		return false
	}
	ensureLenNoClear(aOut, a.Rows)
	ensureLenNoClear(bOut, b.Rows)
	total := a.Rows + b.Rows
	parallelRows(total, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, a.Rows); as < ae {
			matvecDotRows(a.Raw, rowBytes, x, as, ae, *aOut, dot)
		}
		if bs, be := clippedRange(start, end, a.Rows, total); bs < be {
			matvecDotRows(b.Raw, rowBytes, x, bs-a.Rows, be-a.Rows, *bOut, dot)
		}
	})
	return true
}

func matvecSameType3Into(a, b, c Weight, x []float32, aOut, bOut, cOut *[]float32) bool {
	dot, rowBytes, ok := sameTypeQuantDot(a, b, x)
	if !ok || c.F32 != nil || c.Type != a.Type || c.Cols != a.Cols || c.Rows < 0 || len(c.Raw) < c.Rows*rowBytes {
		return false
	}
	ensureLenNoClear(aOut, a.Rows)
	ensureLenNoClear(bOut, b.Rows)
	ensureLenNoClear(cOut, c.Rows)
	ab := a.Rows + b.Rows
	total := ab + c.Rows
	parallelRows(total, func(start, end int) {
		if as, ae := clippedRange(start, end, 0, a.Rows); as < ae {
			matvecDotRows(a.Raw, rowBytes, x, as, ae, *aOut, dot)
		}
		if bs, be := clippedRange(start, end, a.Rows, ab); bs < be {
			matvecDotRows(b.Raw, rowBytes, x, bs-a.Rows, be-a.Rows, *bOut, dot)
		}
		if cs, ce := clippedRange(start, end, ab, total); cs < ce {
			matvecDotRows(c.Raw, rowBytes, x, cs-ab, ce-ab, *cOut, dot)
		}
	})
	return true
}

type quantRowDot func(row []byte, x []float32, cols int) float32

func sameTypeQuantDot(a, b Weight, x []float32) (quantRowDot, int, bool) {
	if a.F32 != nil || b.F32 != nil || a.Type != b.Type || a.Cols <= 0 || a.Cols != b.Cols || a.Cols != len(x) || a.Rows < 0 || b.Rows < 0 {
		return nil, 0, false
	}
	rowBytes, ok := a.Type.DataSize(a.Cols)
	if !ok || rowBytes <= 0 || len(a.Raw) < a.Rows*rowBytes || len(b.Raw) < b.Rows*rowBytes {
		return nil, 0, false
	}
	var dot quantRowDot
	switch a.Type {
	case GGMLTypeQ8_0:
		dot = DotQ8_0F32
	case GGMLTypeQ4_0:
		dot = DotQ4_0F32
	case GGMLTypeQ4_1:
		dot = DotQ4_1F32
	case GGMLTypeQ5_0:
		dot = DotQ5_0F32
	case GGMLTypeQ5_1:
		dot = DotQ5_1F32
	case GGMLTypeQ8_1:
		dot = DotQ8_1F32
	case GGMLTypeQ8_K:
		dot = DotQ8KF32
	case GGMLTypeQ2_K:
		dot = DotQ2KF32
	case GGMLTypeQ3_K:
		dot = DotQ3KF32
	case GGMLTypeQ4_K:
		dot = DotQ4KF32
	case GGMLTypeQ5_K:
		dot = DotQ5KF32
	case GGMLTypeQ6_K:
		dot = DotQ6KF32
	case GGMLTypeMXFP4:
		dot = DotMXFP4F32
	case GGMLTypeTQ1_0:
		dot = DotTQ1_0F32
	case GGMLTypeTQ2_0:
		dot = DotTQ2_0F32
	case GGMLTypeQ1_0:
		dot = DotQ1_0F32
	case GGMLTypeQ2_0:
		dot = DotQ2_0F32
	default:
		return nil, 0, false
	}
	return dot, rowBytes, true
}

func matvecDotRows(data []byte, rowBytes int, x []float32, start, end int, out []float32, dot quantRowDot) {
	for r := start; r < end; r++ {
		off := r * rowBytes
		out[r] = dot(data[off:off+rowBytes], x, len(x))
	}
}
