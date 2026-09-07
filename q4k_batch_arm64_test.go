//go:build arm64

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestQ4KBatch4MatchesSingleToken(t *testing.T) {
	requireDotProd(t)
	if !q4kBatchAsmOK {
		t.Fatal("batch kernel startup check failed")
	}
	rng := rand.New(rand.NewSource(4404))
	for _, cols := range []int{256, 3072, 9216} {
		for _, tokens := range []int{4, 5, 7, 8, 9} {
			t.Run(fmt.Sprintf("cols=%d/tokens=%d", cols, tokens), func(t *testing.T) {
				const rows = 19
				blocks := cols / 256
				raw := make([]byte, 0, rows*blocks*144)
				for range rows {
					raw = append(raw, randomQ4KRow(rng, cols)...)
				}
				w := Weight{Raw: raw, Type: GGMLTypeQ4_K, Rows: rows, Cols: cols}
				q8 := make([]int8, tokens*cols)
				scales := make([]float32, tokens*blocks)
				sums := make([]float32, tokens*blocks*8)
				outs := make([][]float32, tokens)
				for token := range tokens {
					x := randomVec(rng, cols)
					if token%2 == 0 {
						clear(x[:256])
					}
					q8kQuantizePortable(x, q8[token*cols:], scales[token*blocks:], blocks)
					sub := sums[token*blocks*8 : (token+1)*blocks*8]
					fillQ4KXSums(x, cols, &sub)
					outs[token] = make([]float32, rows)
					for row := range rows {
						outs[token][row] = -999
					}
				}
				if !batchQ4KRows4(w, outs, q8, scales, sums, 3, rows-1) {
					t.Fatal("valid batch declined")
				}
				for token := range tokens {
					for row := range rows {
						want := float32(-999)
						if row >= 3 && row < rows-1 {
							want = q4kDotQ8KRow(raw[row*blocks*144:], q8[token*cols:], scales[token*blocks:], sums[token*blocks*8:], blocks)
						}
						if math.Float32bits(outs[token][row]) != math.Float32bits(want) {
							t.Fatalf("token %d row %d: got %v (%08x), want %v (%08x)", token, row, outs[token][row], math.Float32bits(outs[token][row]), want, math.Float32bits(want))
						}
					}
				}
			})
		}
	}
}

func TestQ4KBatch4DeclinesUnsupportedInputs(t *testing.T) {
	for _, w := range []Weight{{Type: GGMLTypeQ6_K}, {Type: GGMLTypeQ4_K}} {
		if batchQ4KRows4(w, make([][]float32, 3), nil, nil, nil, 0, 1) {
			t.Fatal("short batch accepted")
		}
	}
	if batchQ4KRows4(Weight{Type: GGMLTypeQ6_K}, make([][]float32, 4), nil, nil, nil, 0, 1) {
		t.Fatal("Q6_K accepted")
	}
}
