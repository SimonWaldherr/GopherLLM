package yolo

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/yolotest"
)

func TestYOLOIm2colOverwritesPadding(t *testing.T) {
	for _, shape := range [][2]int{{1, 1}, {2, 3}, {7, 9}} {
		for _, k := range []int{1, 3, 5} {
			for _, stride := range []int{1, 2, 3} {
				t.Run(fmt.Sprintf("%dx%d/k%d/s%d", shape[0], shape[1], k, stride), func(t *testing.T) {
					x := benchmarkYOLOTensor(3, shape[0], shape[1])
					pad := k / 2
					oh, ow := (x.h+2*pad-k)/stride+1, (x.w+2*pad-k)/stride+1
					n := oh * ow
					got := make([]float32, x.c*k*k*n)
					for i := range got {
						got[i] = 99 // reused scratch must clear every padded position
					}
					yoloIm2colInto(got, x, k, stride, pad, oh, ow)
					for c := range x.c {
						for ky := range k {
							for kx := range k {
								for oy := range oh {
									for ox := range ow {
										iy, ix := oy*stride-pad+ky, ox*stride-pad+kx
										var want float32
										if iy >= 0 && iy < x.h && ix >= 0 && ix < x.w {
											want = x.data[(c*x.h+iy)*x.w+ix]
										}
										i := ((c*k+ky)*k+kx)*n + oy*ow + ox
										if math.Float32bits(got[i]) != math.Float32bits(want) {
											t.Fatalf("column value %d = %v, want %v", i, got[i], want)
										}
									}
								}
							}
						}
					}
				})
			}
		}
	}
}

func TestYOLODepthwisePreservesAccumulationOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, shape := range [][2]int{{1, 1}, {2, 3}, {7, 9}} {
		for _, k := range []int{1, 3, 5} {
			for _, stride := range []int{1, 2, 3} {
				x := benchmarkYOLOTensor(3, shape[0], shape[1])
				cv := yoloConv{k: k, stride: stride, w: make([]float32, x.c*k*k)}
				for i := range cv.w {
					cv.w[i] = rng.Float32()*2 - 1
				}
				pad := k / 2
				out := benchmarkYOLOTensor(x.c, (x.h+2*pad-k)/stride+1, (x.w+2*pad-k)/stride+1)
				yoloDepthwise(x, &cv, pad, out)
				for c := range x.c {
					for oy := range out.h {
						for ox := range out.w {
							var want float32
							for ky := range k {
								for kx := range k {
									iy, ix := oy*stride-pad+ky, ox*stride-pad+kx
									if iy >= 0 && iy < x.h && ix >= 0 && ix < x.w {
										want += x.data[(c*x.h+iy)*x.w+ix] * cv.w[(c*k+ky)*k+kx]
									}
								}
							}
							i := (c*out.h+oy)*out.w + ox
							if math.Float32bits(out.data[i]) != math.Float32bits(want) {
								t.Fatalf("%dx%d/k%d/s%d: output %d = %v, want %v", x.h, x.w, k, stride, i, out.data[i], want)
							}
						}
					}
				}
			}
		}
	}
}

func TestYOLOAttentionScoresPreserveAccumulationOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	for _, shape := range [][2]int{{1, 1}, {7, 13}, {32, 400}} {
		dim, n := shape[0], shape[1]
		q, k := make([]float32, dim*n), make([]float32, dim*n)
		for i := range q {
			q[i], k[i] = rng.Float32()*2-1, rng.Float32()*2-1
		}
		scale := float32(1 / math.Sqrt(float64(dim)))
		scores := make([]float32, n)
		for _, query := range []int{0, n / 2, n - 1} {
			largest := yoloAttentionScores(scores, q, k, dim, n, query, scale)
			wantLargest := float32(math.Inf(-1))
			for j := range n {
				var want float32
				for d := range dim {
					want += q[d*n+query] * k[d*n+j]
				}
				want *= scale
				wantLargest = max(wantLargest, want)
				if math.Float32bits(scores[j]) != math.Float32bits(want) {
					t.Fatalf("dim=%d/n=%d/query=%d: score %d = %v, want %v", dim, n, query, j, scores[j], want)
				}
			}
			if largest != wantLargest {
				t.Fatalf("largest = %v, want %v", largest, wantLargest)
			}
		}
	}
}

func TestYOLOUpsampleDuplicatesPixels(t *testing.T) {
	x := benchmarkYOLOTensor(3, 2, 5)
	got := yoloUpsample2x(x)
	for c := range got.c {
		for y := range got.h {
			for xx := range got.w {
				want := x.data[(c*x.h+y/2)*x.w+xx/2]
				if got.data[(c*got.h+y)*got.w+xx] != want {
					t.Fatalf("output [%d,%d,%d] does not duplicate the input pixel", c, y, xx)
				}
			}
		}
	}
}

func TestYOLOSharedScratchConcurrentForward(t *testing.T) {
	m, err := LoadYOLOSafetensors(yolotest.V8().Safetensors(yoloUltralyticsNames))
	if err != nil {
		t.Fatal(err)
	}
	inputs := make([]YOLOImage, 2)
	want := make([][]float32, len(inputs))
	for i, size := range []int{32, 64} {
		inputs[i], err = PrepareYOLOImage(yoloSynthImage(), YOLOConfig{InputWidth: size, InputHeight: size})
		if err != nil {
			t.Fatal(err)
		}
		want[i], _, _, err = m.Forward(inputs[i])
		if err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	for worker := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iteration := range 3 {
				i := (worker + iteration) % len(inputs)
				got, _, _, err := m.Forward(inputs[i])
				if err != nil {
					t.Error(err)
					return
				}
				for j := range got {
					if got[j] != want[i][j] {
						t.Errorf("worker %d/iteration %d: output %d = %v, want %v", worker, iteration, j, got[j], want[i][j])
						return
					}
				}
			}
		}()
	}
	wg.Wait()
}
