package gopherllm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"
)

func layaRows(n, d int) [][]float32 {
	var b layaBuffer
	return b.resize(n, d, false)
}

// Keep the original layer rotation as an independent numerical reference.
func layaRotateReference(qkv [][]float32, heads int, theta float64) {
	d := len(qkv[0]) / 3
	hd := d / heads
	for j := 0; j < hd/2; j++ {
		frequency := math.Pow(theta, -float64(2*j)/float64(hd))
		for p, row := range qkv {
			angle := float64(p) * frequency
			s, c := float32(math.Sin(angle)), float32(math.Cos(angle))
			for h := 0; h < heads; h++ {
				for _, base := range []int{h * hd, d + h*hd} {
					a, b := row[base+j], row[base+j+hd/2]
					row[base+j] = a*c - b*s
					row[base+j+hd/2] = b*c + a*s
				}
			}
		}
	}
}

func TestLayaRotaryWorkspaceMatchesReference(t *testing.T) {
	var work layaWorkspace
	for _, tc := range []struct {
		n, heads, dim int
		theta         float64
	}{
		{1, 2, 4, 31}, {7, 2, 4, 17}, {129, 2, 4, 31},
		{11, 2, 4, 17}, {257, 2, 4, 17}, {3, 2, 4, 31},
		{129, 12, 64, 10000}, {257, 12, 64, 160000},
		{7, 12, 64, 10000}, {129, 1, 2, 31},
	} {
		t.Run(fmt.Sprintf("n%d/heads%d/dim%d/theta%g", tc.n, tc.heads, tc.dim, tc.theta), func(t *testing.T) {
			width := 3 * tc.heads * tc.dim
			got, want := layaRows(tc.n, width), layaRows(tc.n, width)
			for p, row := range got {
				for i := range row {
					row[i] = float32((p*width+i)%113-56) / 17
				}
				copy(want[p], row)
			}
			layaRotate(got, tc.heads, tc.theta, &work)
			layaRotateReference(want, tc.heads, tc.theta)
			rotations := work.rotations(tc.n, tc.dim, tc.theta)
			for p := range tc.n {
				for j := 0; j < tc.dim/2; j++ {
					angle := float64(p) * math.Pow(tc.theta, -float64(2*j)/float64(tc.dim))
					if math.Float32bits(rotations[p*tc.dim+j]) != math.Float32bits(float32(math.Cos(angle))) ||
						math.Float32bits(rotations[p*tc.dim+j+tc.dim/2]) != math.Float32bits(float32(math.Sin(angle))) {
						t.Fatalf("position%d dimension%d: cached rotation differs", p, j)
					}
				}
			}
			for p, row := range got {
				for i, v := range row {
					// Go permits fused multiply-add; race instrumentation can
					// change which product is fused in these equivalent loops.
					tolerance := 4e-7 * (1 + math.Abs(float64(want[p][i])))
					if math.IsNaN(float64(v)) || math.Abs(float64(v-want[p][i])) > tolerance {
						t.Fatalf("position%d column%d: got %g want %g", p, i, v, want[p][i])
					}
					if i >= 2*tc.heads*tc.dim && math.Float32bits(v) != math.Float32bits(want[p][i]) {
						t.Fatalf("position%d column%d: value projection changed", p, i)
					}
				}
			}
		})
	}
}

func TestLayaNormInPlaceMatchesSeparateOutput(t *testing.T) {
	for _, dim := range []int{8, 768} {
		x, want := layaRows(7, dim), layaRows(7, dim)
		norm := layaNorm{weight: make([]float32, dim), bias: make([]float32, dim), epsilon: 1e-5}
		for i := range norm.weight {
			norm.weight[i] = float32(i%13+1) / 7
			norm.bias[i] = float32(i%17-8) / 19
		}
		for p, row := range x {
			for i := range row {
				row[i] = float32((p*dim+i)%113-56) / 17
			}
			layerNormInto(row, norm.weight, norm.bias, norm.epsilon, &want[p])
		}
		norm.inPlace(x)
		for p, row := range x {
			for i, v := range row {
				if math.Float32bits(v) != math.Float32bits(want[p][i]) {
					t.Fatalf("dim%d position%d column%d: got %g want %g", dim, p, i, v, want[p][i])
				}
			}
		}
	}
}

func BenchmarkLayaRotary(b *testing.B) {
	const heads, dim, layers = 12, 64, 12
	for _, n := range []int{128, 512} {
		for _, cached := range []bool{false, true} {
			b.Run(fmt.Sprintf("n%d/cached%t", n, cached), func(b *testing.B) {
				qkv := layaRows(n, 3*heads*dim)
				for _, row := range qkv {
					for j := range row {
						row[j] = float32(j%113-56) / 17
					}
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var work layaWorkspace
					for l := 0; l < layers; l++ {
						if cached {
							layaRotate(qkv, heads, 10000, &work)
						} else {
							layaRotateReference(qkv, heads, 10000)
						}
					}
				}
			})
		}
	}
}

// Preserve the previous algorithm, including its per-chunk score allocation,
// as a bit-exact reference and a reproducible allocation benchmark.
func layaAttentionReference(ctx context.Context, n, heads, dim, window int, scale float32, work *layaWorkspace) {
	width := heads * dim
	clear(work.attention.data)
	parallelChunks(n*heads, func(start, end int) {
		scores := make([]float32, n)
		for job := start; job < end; job++ {
			if ctx.Err() != nil {
				return
			}
			p, h := job/heads, job%heads
			lo, hi := 0, n
			if window >= 0 {
				lo = max(0, p-window)
				hi = min(n, p+window+1)
			}
			q := work.qkv.rows[p][h*dim : (h+1)*dim]
			for j := lo; j < hi; j++ {
				scores[j-lo] = DotF32(q, work.qkv.rows[j][width+h*dim:width+(h+1)*dim]) * scale
			}
			row := scores[:hi-lo]
			layaSoftmax(row)
			dest := work.attention.rows[p][h*dim : (h+1)*dim]
			for j := lo; j < hi; j++ {
				AxpyF32(dest, row[j-lo], work.qkv.rows[j][2*width+h*dim:2*width+(h+1)*dim])
			}
		}
	})
}

func TestLayaPortableAttentionMatchesReference(t *testing.T) {
	previous := numThreads()
	defer SetNumThreads(previous)
	previousOversubscribe := oversubscribeDispatch.Load()
	defer oversubscribeDispatch.Store(previousOversubscribe)
	SetNumThreads(3)
	oversubscribeDispatch.Store(true)
	for _, n := range []int{1, 7, 64, 129, 257} {
		for _, window := range []int{-1, 0, 1, 64, 512} {
			const heads, dim = 5, 7
			var work layaWorkspace
			work.qkv.resize(n, 3*heads*dim, false)
			work.attention.resize(n, heads*dim, false)
			for i := range work.qkv.data {
				work.qkv.data[i] = float32(math.Sin(float64(i) * .31))
			}
			layaAttentionReference(context.Background(), n, heads, dim, window, .37, &work)
			want := append([]float32(nil), work.attention.data...)
			for range 2 {
				for i := range work.attention.data {
					work.attention.data[i] = 123
				}
				if err := layaAttentionPortable(context.Background(), n, heads, dim, window, .37, &work); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(work.attention.data, want) {
					t.Fatalf("n%d window%d changed", n, window)
				}
			}
		}
	}
}

func TestLayaPortableAttentionConcurrentAndCanceled(t *testing.T) {
	const n, heads, dim = 129, 3, 5
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var work layaWorkspace
			work.qkv.resize(n, 3*heads*dim, false)
			work.attention.resize(n, heads*dim, false)
			for i := range work.qkv.data {
				work.qkv.data[i] = float32(math.Sin(float64(i) * .19))
			}
			layaAttentionReference(context.Background(), n, heads, dim, 8, .4, &work)
			want := append([]float32(nil), work.attention.data...)
			if err := layaAttentionPortable(context.Background(), n, heads, dim, 8, .4, &work); err != nil {
				t.Error(err)
			}
			if !reflect.DeepEqual(work.attention.data, want) {
				t.Error("concurrent output changed")
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if err := layaAttentionPortable(ctx, n, heads, dim, 8, .4, &work); !errors.Is(err, context.Canceled) {
				t.Errorf("cancel: %v", err)
			}
		}()
	}
	wg.Wait()
}

func BenchmarkLayaPortableAttention(b *testing.B) {
	for _, n := range []int{128, 512} {
		for _, window := range []int{-1, 64} {
			for _, reuse := range []bool{false, true} {
				b.Run(fmt.Sprintf("n%d/window%d/reuse%t", n, window, reuse), func(b *testing.B) {
					const heads, dim = 12, 64
					var work layaWorkspace
					work.qkv.resize(n, 3*heads*dim, false)
					work.attention.resize(n, heads*dim, false)
					for i := range work.qkv.data {
						work.qkv.data[i] = float32(math.Sin(float64(i) * .07))
					}
					ctx := context.Background()
					run := func() {
						if reuse {
							if err := layaAttentionPortable(ctx, n, heads, dim, window, .125, &work); err != nil {
								b.Fatal(err)
							}
						} else {
							layaAttentionReference(ctx, n, heads, dim, window, .125, &work)
						}
					}
					run()
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						run()
					}
				})
			}
		}
	}
}
