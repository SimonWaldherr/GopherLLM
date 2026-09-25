//go:build darwin && cgo

package layablas

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func TestAttentionMatchesScalar(t *testing.T) {
	for _, n := range []int{1, 7, 127, 128, 129, 257} {
		for _, window := range []int{-1, 0, 1, 8, 128} {
			t.Run(fmt.Sprintf("n%d/window%d", n, window), func(t *testing.T) {
				const heads, dim = 3, 5
				width := heads * dim
				rng := rand.New(rand.NewSource(42))
				qkv := make([]float32, n*3*width)
				for i := range qkv {
					qkv[i] = float32(rng.NormFloat64())
				}
				out := make([]float32, n*width)
				for i := range out {
					out[i] = 123
				} // Backend must overwrite, not accumulate.
				scratch := make([]float32, attentionBlock*n)
				scale := float32(1 / math.Sqrt(dim))
				if err := Attention(context.Background(), n, heads, dim, window, scale, qkv, out, &scratch); err != nil {
					t.Fatal(err)
				}
				for p := 0; p < n; p++ {
					for h := 0; h < heads; h++ {
						lo, hi := 0, n
						if window >= 0 {
							lo = max(0, p-window)
							hi = min(n, p+window+1)
						}
						scores := make([]float64, hi-lo)
						largest := math.Inf(-1)
						for j := lo; j < hi; j++ {
							for k := 0; k < dim; k++ {
								scores[j-lo] += float64(qkv[p*3*width+h*dim+k]) * float64(qkv[j*3*width+width+h*dim+k])
							}
							scores[j-lo] *= float64(scale)
							largest = math.Max(largest, scores[j-lo])
						}
						sum := 0.0
						for j := range scores {
							scores[j] = math.Exp(scores[j] - largest)
							sum += scores[j]
						}
						for k := 0; k < dim; k++ {
							want := 0.0
							for j := lo; j < hi; j++ {
								want += scores[j-lo] / sum * float64(qkv[j*3*width+2*width+h*dim+k])
							}
							got := float64(out[p*width+h*dim+k])
							if math.IsNaN(got) || math.Abs(got-want) > 3e-6 {
								t.Fatalf("p%d h%d k%d: got %g want %g", p, h, k, got, want)
							}
						}
					}
				}
				if len(scratch) > attentionBlock*n {
					t.Fatalf("unbounded scratch: %d", len(scratch))
				}
			})
		}
	}
}

// Deterministic cancellation between blocks, without wall-clock timing races.
type cancelSecondBlock struct {
	context.Context
	calls int
}

func (c *cancelSecondBlock) Err() error {
	c.calls++
	if c.calls > 2 {
		return context.Canceled
	}
	return nil
}
func TestAttentionCancellation(t *testing.T) {
	ctx := &cancelSecondBlock{Context: context.Background()}
	const n = attentionBlock + 1
	out := make([]float32, n)
	for i := range out {
		out[i] = 123
	}
	var scratch []float32
	err := Attention(ctx, n, 1, 1, -1, 1, make([]float32, 3*n), out, &scratch)
	if !errors.Is(err, context.Canceled) || out[0] != 0 || out[attentionBlock] != 123 {
		t.Fatalf("err=%v first=%v last=%v", err, out[0], out[attentionBlock])
	}
}
func TestAttentionInvalidDimensions(t *testing.T) {
	var scratch []float32
	for _, args := range [][3]int{{0, 1, 1}, {8193, 1, 1}, {1, 0, 1}, {1, 3, 1}} {
		if err := Attention(context.Background(), args[0], args[1], args[2], -1, 1, make([]float32, 12), make([]float32, 4), &scratch); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
func TestStridedProjection(t *testing.T) {
	const n, rows, cols, stride = 7, 11, 5, 10
	rng := rand.New(rand.NewSource(43))
	x, w, out := make([]float32, n*stride), make([]float32, rows*cols), make([]float32, n*rows)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	for i := range out {
		out[i] = 123
	}
	Project(n, rows, cols, stride, w, x, out)
	for i := 0; i < n; i++ {
		for j := 0; j < rows; j++ {
			want := 0.0
			for k := 0; k < cols; k++ {
				want += float64(x[i*stride+k]) * float64(w[j*cols+k])
			}
			if math.Abs(float64(out[i*rows+j])-want) > 3e-6 {
				t.Fatalf("row%d col%d", i, j)
			}
		}
	}
}
