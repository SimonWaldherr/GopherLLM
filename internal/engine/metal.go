package engine

// Q4KMatvec3Into computes three Q4_K matvecs (attention Q/K/V projections)
// against the same activation vector in one fused pass: the per-32-element
// activation sums are computed once and the combined row range is spread over
// the worker pool as a single parallel dispatch instead of three. Returns
// false (nothing written) if shapes are incompatible, in which case the
// caller falls back to three separate matvecs.
func Q4KMatvec3Into(wq, wk, wv Q4KMatrix, x []float32, q, k, v *[]float32) bool {
	scratch := []float32{}
	return Q4KMatvec3IntoWithXSums(wq, wk, wv, x, &scratch, q, k, v)
}

func Q4KMatvec3IntoWithXSums(wq, wk, wv Q4KMatrix, x []float32, xSums *[]float32, q, k, v *[]float32) bool {
	if wq.Cols <= 0 || wq.Cols != wk.Cols || wq.Cols != wv.Cols || wq.Cols != len(x) {
		return false
	}
	if wq.Cols%256 != 0 || wk.Cols%256 != 0 || wv.Cols%256 != 0 {
		return false
	}
	qRowBytes := (wq.Cols / 256) * 144
	kRowBytes := (wk.Cols / 256) * 144
	vRowBytes := (wv.Cols / 256) * 144
	if len(wq.Data) < wq.Rows*qRowBytes || len(wk.Data) < wk.Rows*kRowBytes || len(wv.Data) < wv.Rows*vRowBytes {
		return false
	}

	ensureLenNoClear(q, wq.Rows)
	ensureLenNoClear(k, wk.Rows)
	ensureLenNoClear(v, wv.Rows)

	xs := fillQ4KXSums(x, wq.Cols, xSums)
	qRows := wq.Rows
	kRows := wk.Rows
	totalRows := qRows + kRows + wv.Rows
	if useQ8Activations.Load() {
		q8, xsc, lease := acquireQ8(x, wq.Cols)
		parallelRows(totalRows, func(start, end int) {
			if qs, qe := clippedRange(start, end, 0, qRows); qs < qe {
				dotQ4KRowsQ8(wq.Data, q8, xsc, xs, wq.Cols, qRowBytes, qs, qe, *q)
			}
			if ks, ke := clippedRange(start, end, qRows, qRows+kRows); ks < ke {
				dotQ4KRowsQ8(wk.Data, q8, xsc, xs, wk.Cols, kRowBytes, ks-qRows, ke-qRows, *k)
			}
			if vs, ve := clippedRange(start, end, qRows+kRows, totalRows); vs < ve {
				dotQ4KRowsQ8(wv.Data, q8, xsc, xs, wv.Cols, vRowBytes, vs-qRows-kRows, ve-qRows-kRows, *v)
			}
		})
		releaseQ8(q8, xsc, lease)
		return true
	}
	parallelRows(totalRows, func(start, end int) {
		if qs, qe := clippedRange(start, end, 0, qRows); qs < qe {
			dotQ4KRowsWithXSums(wq.Data, x, xs, wq.Cols, qRowBytes, qs, qe, *q)
		}
		if ks, ke := clippedRange(start, end, qRows, qRows+kRows); ks < ke {
			dotQ4KRowsWithXSums(wk.Data, x, xs, wk.Cols, kRowBytes, ks-qRows, ke-qRows, *k)
		}
		if vs, ve := clippedRange(start, end, qRows+kRows, totalRows); vs < ve {
			dotQ4KRowsWithXSums(wv.Data, x, xs, wv.Cols, vRowBytes, vs-qRows-kRows, ve-qRows-kRows, *v)
		}
	})
	return true
}

func clippedRange(start, end, lo, hi int) (int, int) {
	if start < lo {
		start = lo
	}
	if end > hi {
		end = hi
	}
	return start, end
}

// Q4KMatrix is a borrowed view of a Q4_K weight tensor: Rows x Cols elements
// packed as (Cols/256) 144-byte superblocks per row (see GGMLType.DataSize).
type Q4KMatrix struct {
	Data []byte
	Rows int
	Cols int
}

// Shared with loader policies that avoid preparing CPU-only projection pairs.
const (
	metalQ4KDirectMinRows = 8192
	metalQ5KDirectMinRows = 3072
	metalQ6KDirectMinRows = 3072
)

// metalWeightMayUseDirect gates standalone matvec offload. Contracting Q4_K
// FFN weights have a separate preparation exception for fused FFNs; this
// must not lower the standalone threshold for narrow attention projections.
func metalWeightMayUseDirect(typ GGMLType, rows int) bool {
	switch typ {
	case GGMLTypeQ4_K:
		// Q4_K attention-output projections in Mistral-family GQA models are
		// typically 3K-5K rows. On an M2 Max their CPU Q8_K/NEON path is
		// 35-50% faster than a standalone Metal dispatch; reserve direct GPU
		// work for the large FFN/vocabulary-shaped matrices that amortize the
		// command-buffer boundary.
		return rows >= metalQ4KDirectMinRows
	case GGMLTypeQ5_K:
		// Q5_K carries Q4_K's scale structure with an extra bitplane, so its
		// CPU kernel is slightly slower per row than Q4_K's while the GPU cost
		// is nearly identical — the crossover therefore sits at the Q6_K
		// threshold rather than Q4_K's higher one.
		return rows >= metalQ5KDirectMinRows
	case GGMLTypeQ6_K:
		return rows >= metalQ6KDirectMinRows
	case GGMLTypeQ8_0:
		return rows >= 2048
	default:
		return false
	}
}
