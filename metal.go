package gopherllm

// Metal acceleration in the Rust project uses Objective-C and therefore CGO
// in a Go port. GopherLLM keeps the public shape but uses pure-Go fallbacks.

// MetalAvailable reports GPU availability; always false in the pure-Go build.
func MetalAvailable() bool { return false }

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
	if useQ8Activations {
		q8, xsc, release := acquireQ8(x, wq.Cols)
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
		release()
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
