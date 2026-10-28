//go:build darwin && arm64 && cgo

package engine

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func prefillBLASTestWeight(typ GGMLType, rows, cols int) Weight {
	rng := rand.New(rand.NewSource(731))
	w := Weight{Type: typ, Rows: rows, Cols: cols}
	if typ == GGMLTypeF32 {
		w.F32 = randomVec(rng, rows*cols)
		return w
	}
	for range rows {
		switch typ {
		case GGMLTypeQ4_K:
			w.Raw = append(w.Raw, randomQ4KRow(rng, cols)...)
		case GGMLTypeQ6_K:
			w.Raw = append(w.Raw, randomQ6KRow(rng, cols)...)
		case GGMLTypeQ8_0:
			w.Raw = append(w.Raw, randomQ8_0Row(rng, cols)...)
		}
	}
	return w
}

func prefillBLASReference(w Weight, xs [][]float32) [][]float32 {
	cols := len(xs[0])
	rows := w.Rows
	if w.F32 != nil {
		rows = len(w.F32) / cols
	}
	want := make([][]float32, len(xs))
	for token := range xs {
		want[token] = make([]float32, rows)
	}
	var row []float32
	for r := range rows {
		w.RowInto(r, cols, &row)
		for token := range xs {
			want[token][r] = DotF32(row, xs[token])
		}
	}
	return want
}

func checkPrefillBLASOutput(t *testing.T, got, want [][]float32) {
	t.Helper()
	var error2, norm2 float64
	for token := range want {
		for r, value := range want[token] {
			// GEMM may group float sums differently from the NEON row dot.
			diff := float64(got[token][r]) - float64(value)
			error2 += diff * diff
			norm2 += float64(value) * float64(value)
			if !finite32(got[token][r]) || math.Abs(diff) > 1e-3*math.Max(1, math.Abs(float64(value))) {
				t.Fatalf("token=%d row=%d: got %g, want %g", token, r, got[token][r], value)
			}
		}
	}
	if relative := math.Sqrt(error2 / math.Max(norm2, 1e-30)); relative > 2e-6 {
		t.Fatalf("relative projection error %g exceeds float reduction tolerance", relative)
	}
}

func TestPrefillBLASTiledProjections(t *testing.T) {
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
		tokenCounts := []int{128, 129}
		if typ == GGMLTypeF32 {
			tokenCounts = []int{16, 33}
		}
		for _, tokens := range tokenCounts {
			t.Run(fmt.Sprintf("type%d/P%d", typ, tokens), func(t *testing.T) {
				const rows, cols = 259, 256 // Two full tiles and a short tail.
				w := prefillBLASTestWeight(typ, rows, cols)
				xs, outs := make([][]float32, tokens), make([][]float32, tokens)
				rng := rand.New(rand.NewSource(732))
				for i := range xs {
					xs[i], outs[i] = randomVec(rng, cols), make([]float32, rows)
				}
				want := prefillBLASReference(w, xs)
				if !matvecBatchBLAS(w, xs, outs, false) {
					t.Fatal("valid projection declined")
				}
				checkPrefillBLASOutput(t, outs, want)
				matvecBatchWithQ8(w, xs, outs, false)
				checkPrefillBLASOutput(t, outs, want)
				if typ != GGMLTypeF32 && matvecBatchBLAS(w, xs, outs, true) {
					t.Fatal("Q8 activation route must retain its arithmetic")
				}
			})
		}
	}
}

func TestPrefillBLASInPlaceAndInferredGeometry(t *testing.T) {
	const tokens, width = 33, 256
	w := prefillBLASTestWeight(GGMLTypeF32, width, width)
	w.Rows, w.Cols = 0, 0 // Owned F32 weights need no explicit geometry.
	xs := make([][]float32, tokens)
	for i := range xs {
		xs[i] = benchFloatSlice(width)
	}
	want := prefillBLASReference(w, xs)
	if !matvecBatchBLAS(w, xs, xs, true) {
		t.Fatal("in-place F32 projection declined")
	}
	checkPrefillBLASOutput(t, xs, want)
	// A tile must not overwrite another token's input before it is consumed.
	for i := range xs {
		xs[i] = benchFloatSlice(width)
	}
	want = prefillBLASReference(w, xs)
	outs := make([][]float32, tokens)
	for i := range outs {
		outs[i] = xs[(i+1)%tokens]
	}
	if !matvecBatchBLAS(w, xs, outs, false) {
		t.Fatal("cross-token in-place projection declined")
	}
	checkPrefillBLASOutput(t, outs, want)
}

func TestPrefillBLASDeclinesInvalidBatches(t *testing.T) {
	const tokens, width = 16, 256
	w := prefillBLASTestWeight(GGMLTypeF32, width, width)
	xs, outs := make([][]float32, tokens), make([][]float32, tokens)
	for i := range xs {
		xs[i], outs[i] = make([]float32, width), make([]float32, width)
	}
	if matvecBatchBLAS(w, xs[:1], outs[:1], false) || matvecBatchBLAS(w, xs, outs[:15], false) {
		t.Fatal("small or inconsistent batch accepted")
	}
	xs[15] = xs[15][:width-1]
	if matvecBatchBLAS(w, xs, outs, false) {
		t.Fatal("short input accepted")
	}
	xs[15] = xs[15][:width]
	outs[15] = outs[15][:width-1]
	if matvecBatchBLAS(w, xs, outs, false) {
		t.Fatal("short output accepted")
	}
	outs[15] = w.F32[:width]
	if matvecBatchBLAS(w, xs, outs, false) {
		t.Fatal("output aliasing weights accepted")
	}
	outs[15] = make([]float32, width)
	w = prefillBLASTestWeight(GGMLTypeQ4_K, width, width)
	if matvecBatchBLAS(w, xs, outs, false) {
		t.Fatal("short quantized batch bypassed the NEON crossover")
	}
	longInputs, longOutputs := make([][]float32, 128), make([][]float32, 128)
	for i := range longInputs {
		longInputs[i], longOutputs[i] = xs[i%tokens], make([]float32, width)
	}
	w.Raw = w.Raw[:len(w.Raw)-1]
	if matvecBatchBLAS(w, longInputs, longOutputs, false) {
		t.Fatal("short quantized weights accepted")
	}
}

func TestPrefillBLASMatrixTileBudgets(t *testing.T) {
	for _, rows := range []int{32, 259, 3072, 1 << 20} {
		for _, cols := range []int{64, 768, 4096, 12288, 16384} {
			for _, tokens := range []int{16, 33, 128, 256, 32768} {
				if tokens > prefillBLASInputElements/cols {
					continue
				}
				for _, expanded := range []bool{false, true} {
					compute, scatter := prefillBLASRowTiles(rows, cols, tokens, expanded)
					if scatter < 1 || compute < scatter || compute > rows || compute*tokens > prefillBLASInputElements {
						t.Fatalf("rows=%d cols=%d P=%d expanded=%t: invalid tiles %d/%d", rows, cols, tokens, expanded, compute, scatter)
					}
					if compute < rows && compute%scatter != 0 {
						t.Fatal("compute tiles must preserve the existing scatter boundaries")
					}
					if expanded && compute*cols > prefillBLASTileElements {
						t.Fatal("weight-expansion tiles exceeded the existing memory bound")
					}
				}
			}
		}
	}
	if compute, _ := prefillBLASRowTiles(3072, 768, 128, false); compute != 3072 {
		t.Fatalf("F32 matrix still split into %d-row GEMMs", compute)
	}
	if compute, scatter := prefillBLASRowTiles(3072, 768, 16, false); compute != scatter {
		t.Fatal("short batches must retain the previous GEMM size")
	}
}

func TestPrefillBLASContiguousAndClippedViews(t *testing.T) {
	const tokens, rows, cols = 33, 259, 256
	w := prefillBLASTestWeight(GGMLTypeF32, rows, cols)
	for _, clipped := range []bool{false, true} {
		t.Run(fmt.Sprintf("clipped=%t", clipped), func(t *testing.T) {
			input, output := benchFloatSlice(tokens*cols+2), make([]float32, tokens*rows+2)
			output[0], output[len(output)-1] = 1234, 5678
			before := append([]float32(nil), input...)
			xs, outs := make([][]float32, tokens), make([][]float32, tokens)
			for token := range xs {
				xEnd, outEnd := 1+(token+1)*cols, 1+(token+1)*rows
				xs[token] = input[1+token*cols : xEnd]
				outs[token] = output[1+token*rows : outEnd]
				if clipped {
					xs[token], outs[token] = xs[token][:cols:cols], outs[token][:rows:rows]
				}
			}
			want := prefillBLASReference(w, xs)
			if !matvecBatchBLAS(w, xs, outs, false) {
				t.Fatal("valid contiguous matrix declined")
			}
			checkPrefillBLASOutput(t, outs, want)
			if output[0] != 1234 || output[len(output)-1] != 5678 {
				t.Fatal("output guards overwritten")
			}
			for i, value := range input {
				if value != before[i] {
					t.Fatalf("input or its guard changed at %d", i)
				}
			}
		})
	}
}

func TestPrefillBLASContiguousInPlace(t *testing.T) {
	const tokens, width = 33, 256
	w := prefillBLASTestWeight(GGMLTypeF32, width, width)
	data := benchFloatSlice(tokens*width + 2)
	first, last := data[0], data[len(data)-1]
	xs := make([][]float32, tokens)
	for token := range xs {
		xs[token] = data[1+token*width : 1+(token+1)*width]
	}
	want := prefillBLASReference(w, xs)
	if !matvecBatchBLAS(w, xs, xs, false) {
		t.Fatal("contiguous in-place projection declined")
	}
	checkPrefillBLASOutput(t, xs, want)
	if data[0] != first || data[len(data)-1] != last {
		t.Fatal("in-place projection changed guards")
	}
}

func TestPrefillBLASOverlappingOutputOrder(t *testing.T) {
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
		t.Run(fmt.Sprintf("type%d", typ), func(t *testing.T) {
			const tokens, rows, cols, overlap = 128, 259, 256, 13
			w := prefillBLASTestWeight(typ, rows, cols)
			xs, outs, expectedViews := make([][]float32, tokens), make([][]float32, tokens), make([][]float32, tokens)
			actual, expected := make([]float32, (tokens-1)*(rows-overlap)+rows), make([]float32, (tokens-1)*(rows-overlap)+rows)
			for token := range xs {
				xs[token] = benchFloatSlice(cols)
				start := token * (rows - overlap)
				outs[token], expectedViews[token] = actual[start:start+rows], expected[start:start+rows]
			}
			want := prefillBLASReference(w, xs)
			// Reproduce the former 128-row, tile-major copy order. Overlapping outputs
			// expose a different final writer if a larger GEMM scatters token-major.
			for start := 0; start < rows; start += 128 {
				end := min(start+128, rows)
				for token := range xs {
					copy(expectedViews[token][start:end], want[token][start:end])
				}
			}
			if !matvecBatchBLAS(w, xs, outs, false) {
				t.Fatal("overlapping output views declined")
			}
			checkPrefillBLASOutput(t, [][]float32{actual}, [][]float32{expected})
		})
	}
}

func BenchmarkApplePrefillMatrixLayout(b *testing.B) {
	const rows, cols, tokens = 3072, 768, 64
	w := prefillBLASTestWeight(GGMLTypeF32, rows, cols)
	for _, layout := range []string{"separate", "bounded_views", "contiguous"} {
		b.Run(layout, func(b *testing.B) {
			xs, outs := make([][]float32, tokens), make([][]float32, tokens)
			input, output := benchFloatSlice(tokens*cols), make([]float32, tokens*rows)
			for token := range xs {
				xs[token] = input[token*cols : (token+1)*cols]
				outs[token] = output[token*rows : (token+1)*rows]
				switch layout {
				case "bounded_views":
					xs[token], outs[token] = xs[token][:cols:cols], outs[token][:rows:rows]
				case "separate":
					xs[token], outs[token] = benchFloatSlice(cols), make([]float32, rows)
				}
			}
			if !matvecBatchBLAS(w, xs, outs, false) {
				b.Fatal("projection declined")
			}
			b.ReportAllocs()
			for b.Loop() {
				if !matvecBatchBLAS(w, xs, outs, false) {
					b.Fatal("projection declined")
				}
			}
		})
	}
}

func BenchmarkAppleFloatPrefillProjection(b *testing.B) {
	for _, typ := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeQ6_K} {
		for _, tokens := range []int{16, 32, 64, 128} {
			b.Run(fmt.Sprintf("type%d/P%d", typ, tokens), func(b *testing.B) {
				const rows, cols = 3072, 768
				w := prefillBLASTestWeight(typ, rows, cols)
				xs, outs := make([][]float32, tokens), make([][]float32, tokens)
				for i := range xs {
					xs[i], outs[i] = benchFloatSlice(cols), make([]float32, rows)
				}
				b.Run("neon", func(b *testing.B) {
					var task rowTask
					if w.F32 != nil {
						task = &batchF32MatvecTask{weights: w.F32, xs: xs, outs: outs, p: tokens, cols: cols, tiled: true}
					} else {
						rowBytes, _ := typ.DataSize(cols)
						task = &batchDequantMatvecTask{raw: w.Raw, dequant: dequantRowInto(w, cols), xs: xs, outs: outs, p: tokens, cols: cols, rowBytes: rowBytes, tiled: true}
					}
					b.ReportAllocs()
					for b.Loop() {
						parallelRowsBatchedTask(rows, task)
					}
				})
				b.Run("accelerate", func(b *testing.B) {
					if !matvecBatchBLAS(w, xs, outs, false) {
						b.Skip("below Accelerate dispatch threshold")
					}
					b.ReportAllocs()
					for b.Loop() {
						if !matvecBatchBLAS(w, xs, outs, false) {
							b.Fatal("projection declined")
						}
					}
				})
			})
		}
	}
}
