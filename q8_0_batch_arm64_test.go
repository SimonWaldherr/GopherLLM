//go:build arm64

package gopherllm

import (
	"math"
	"math/rand"
	"testing"
)

func TestQ8_0Batch4MatchesRows(test *testing.T) {
	if !hasDotProd {
		test.Skip("SDOT unavailable")
	}
	if !q8_0BatchAsmOK {
		test.Fatal("batch self-check failed")
	}
	rng := rand.New(rand.NewSource(99))
	for _, cols := range []int{256, 2560, 4096} {
		blocks := cols / 256
		row := randomQ8_0Row(rng, cols)
		for trial := range 12 {
			q := make([]int8, 4*cols)
			scales := make([]float32, 4*blocks)
			for t := range 4 {
				x := randomVec(rng, cols)
				if trial == 0 && t == 0 {
					clear(x)
				}
				q8kQuantize(x, q[t*cols:], scales[t*blocks:], blocks)
			}
			got := q8_0RowBatch4(row, q, scales, cols, blocks)
			for t := range 4 {
				want := q8_0DotQ8KRow(row, q[t*cols:], scales[t*blocks:], blocks)
				if math.Float32bits(want) != math.Float32bits(got[t]) {
					test.Fatalf("cols=%d trial=%d token=%d: got %v want %v", cols, trial, t, got[t], want)
				}
			}
		}
	}
}

func BenchmarkQ8_0PrefillRow4(b *testing.B) {
	if !q8_0BatchAsmOK {
		b.Skip("SDOT unavailable")
	}
	const cols = 2560
	rng := rand.New(rand.NewSource(42))
	row := randomQ8_0Row(rng, cols)
	q := make([]int8, 4*cols)
	sc := make([]float32, 4*cols/256)
	for t := range 4 {
		q8kQuantize(randomVec(rng, cols), q[t*cols:], sc[t*cols/256:], cols/256)
	}
	b.Run("single", func(b *testing.B) {
		for b.Loop() {
			for t := range 4 {
				q8_0DotQ8KRow(row, q[t*cols:], sc[t*cols/256:], cols/256)
			}
		}
	})
	b.Run("batch4", func(b *testing.B) {
		for b.Loop() {
			q8_0RowBatch4(row, q, sc, cols, cols/256)
		}
	})
}

func TestQ8_0BatchRowsTailAndRange(t *testing.T) {
	requireDotProd(t)
	rng := rand.New(rand.NewSource(52))
	const rows, cols = 19, 512
	blocks := cols / 256
	for _, count := range []int{4, 5, 9} {
		raw := make([]byte, 0, rows*blocks*272)
		for range rows {
			raw = append(raw, randomQ8_0Row(rng, cols)...)
		}
		q := make([]int8, count*cols)
		scales := make([]float32, count*blocks)
		outs := make([][]float32, count)
		for token := range count {
			q8kQuantize(randomVec(rng, cols), q[token*cols:], scales[token*blocks:], blocks)
			outs[token] = make([]float32, rows)
			for i := range rows {
				outs[token][i] = -999
			}
		}
		if !batchQ8_0Rows4(Weight{Type: GGMLTypeQ8_0, Raw: raw, Rows: rows, Cols: cols}, outs, q, scales, 3, rows-1) {
			t.Fatal("valid batch declined")
		}
		for token := range count {
			for row := range rows {
				want := float32(-999)
				if row >= 3 && row < rows-1 {
					want = q8_0DotQ8KRow(raw[row*blocks*272:], q[token*cols:], scales[token*blocks:], blocks)
				}
				if math.Float32bits(outs[token][row]) != math.Float32bits(want) {
					t.Fatalf("token=%d row=%d got=%v want=%v", token, row, outs[token][row], want)
				}
			}
		}
	}
}
