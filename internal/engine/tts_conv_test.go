package engine

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func ttsConvTestData(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i+1)*.713)) * .2
	}
	return out
}

func ttsCheckConvolution(t *testing.T, got ttsMatrix, want []float64, out, length int) {
	t.Helper()
	if got.channels != out || got.length != length || len(got.data) != len(want) {
		t.Fatalf("output shape %dx%d, want %dx%d", got.channels, got.length, out, length)
	}
	for i, v := range want {
		if math.IsNaN(float64(got.data[i])) || math.Abs(float64(got.data[i])-v) > 2e-5+math.Abs(v)*2e-5 {
			t.Fatalf("output[%d]=%g, want %g", i, got.data[i], v)
		}
	}
}

func TestTTSConv1DMatchesScalar(t *testing.T) {
	setPerformanceThreads(t, 3)
	for _, length := range []int{1, 3, 127, 129, 257} {
		for _, groups := range []int{1, 2, 6} {
			for _, kernel := range []int{1, 3, 4} {
				for _, dilation := range []int{1, 3} {
					const in, out = 6, 12
					padding := (kernel - 1) * dilation / 2
					n := length + 2*padding - (kernel-1)*dilation
					if n <= 0 {
						continue
					}
					x := ttsMatrix{data: ttsConvTestData(in * length), channels: in, length: length}
					w := ttsWeight{data: ttsConvTestData(out * (in / groups) * kernel), shape: []int{out, in / groups, kernel}}
					bias := ttsConvTestData(out)
					got, err := ttsConv1D(context.Background(), x, w, bias, dilation, padding, groups)
					if err != nil {
						t.Fatal(err)
					}
					want := make([]float64, out*n)
					for oc := range out {
						for p := range n {
							sum := float64(bias[oc])
							g := oc / (out / groups)
							for ic := range in / groups {
								for k := range kernel {
									at := p - padding + k*dilation
									if at >= 0 && at < length {
										sum += float64(x.data[(g*(in/groups)+ic)*length+at]) * float64(w.data[(oc*(in/groups)+ic)*kernel+k])
									}
								}
							}
							want[oc*n+p] = sum
						}
					}
					ttsCheckConvolution(t, got, want, out, n)
				}
			}
		}
	}
}

func TestTTSConvTransposeMatchesScalar(t *testing.T) {
	setPerformanceThreads(t, 3)
	for _, length := range []int{1, 3, 127, 129, 257} {
		for _, shape := range [][3]int{{3, 1, 1}, {4, 2, 1}, {16, 8, 4}, {5, 3, 2}} {
			const in, out = 6, 4
			kernel, stride, padding := shape[0], shape[1], shape[2]
			n := (length-1)*stride - 2*padding + kernel
			x := ttsMatrix{data: ttsConvTestData(in * length), channels: in, length: length}
			w := ttsWeight{data: ttsConvTestData(in * out * kernel), shape: []int{in, out, kernel}}
			bias := ttsConvTestData(out)
			got, err := ttsConvTranspose1D(context.Background(), x, w, bias, stride, padding)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float64, out*n)
			for oc := range out {
				for p := range n {
					want[oc*n+p] = float64(bias[oc])
				}
			}
			for ic := range in {
				for p := range length {
					for oc := range out {
						for k := range kernel {
							at := p*stride - padding + k
							if at >= 0 && at < n {
								want[oc*n+at] += float64(x.data[ic*length+p]) * float64(w.data[(ic*out+oc)*kernel+k])
							}
						}
					}
				}
			}
			ttsCheckConvolution(t, got, want, out, n)
		}
	}
}

func TestTTSConvolutionValidationAndCancellation(t *testing.T) {
	x := ttsMatrix{data: []float32{1, 2, 3}, channels: 1, length: 3}
	w := ttsWeight{data: []float32{1, 2, 3}, shape: []int{1, 1, 3}}
	for _, params := range [][3]int{{0, 1, 1}, {1, -1, 1}, {1, 1, 0}, {1, 1, 2}, {int(^uint(0) >> 1), 1, 1}} {
		if _, err := ttsConv1D(context.Background(), x, w, nil, params[0], params[1], params[2]); err == nil {
			t.Fatalf("accepted invalid parameters %v", params)
		}
	}
	if _, err := ttsConvTranspose1D(context.Background(), x, w, nil, 0, 1); err == nil {
		t.Fatal("accepted zero stride")
	}
	if _, err := ttsConvTranspose1D(context.Background(), x, w, nil, 2, 100); err == nil {
		t.Fatal("accepted impossible padding")
	}
	if _, err := ttsConv1D(context.Background(), x, w, []float32{1, 2}, 1, 1, 1); err == nil {
		t.Fatal("accepted bias shape mismatch")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ttsConv1D(ctx, x, w, nil, 1, 1, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	if _, err := ttsConvTranspose1D(ctx, x, w, nil, 2, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
}

func TestTTSConvolutionBufferReuse(t *testing.T) {
	ctx := context.Background()
	var dst []float32
	var scratch ttsConvScratch
	// Large -> small -> large exercises reused tails and changing tile strides.
	for _, length := range []int{257, 7, 129} {
		x := ttsMatrix{data: ttsConvTestData(4 * length), channels: 4, length: length}
		original := append([]float32(nil), x.data...)
		for _, groups := range []int{1, 4} {
			w := ttsWeight{data: ttsConvTestData(4 * (4 / groups) * 3), shape: []int{4, 4 / groups, 3}}
			want, err := ttsConv1D(ctx, x, w, nil, 3, 3, groups)
			if err != nil {
				t.Fatal(err)
			}
			got, err := ttsConv1DInto(ctx, x, w, nil, 3, 3, groups, &dst, &scratch)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.data, want.data) {
				t.Fatalf("buffer reuse differs for length=%d groups=%d", length, groups)
			}
			if !slices.Equal(x.data, original) {
				t.Fatal("residual input modified")
			}
		}
		w := ttsWeight{data: ttsConvTestData(4 * 4 * 4), shape: []int{4, 4, 4}}
		want, err := ttsConvTranspose1D(ctx, x, w, nil, 2, 1)
		if err != nil {
			t.Fatal(err)
		}
		for repeat := 0; repeat < 2; repeat++ {
			got, err := ttsConvTranspose1DInto(ctx, x, w, nil, 2, 1, &dst, &scratch)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.data, want.data) {
				t.Fatal("transposed convolution retained previous output")
			}
		}
	}
	x := ttsMatrix{data: []float32{1, 2, 3}, channels: 1, length: 3}
	w := ttsWeight{data: []float32{1}, shape: []int{1, 1, 1}}
	dst = x.data
	if _, err := ttsConv1DInto(ctx, x, w, nil, 1, 0, 1, &dst, &scratch); err == nil {
		t.Fatal("accepted aliased convolution output")
	}
	if _, err := ttsConvTranspose1DInto(ctx, x, w, nil, 1, 0, &dst, &scratch); err == nil {
		t.Fatal("accepted aliased transposed output")
	}
}

func TestTTSConvolutionRejectsPartialOutputAliases(t *testing.T) {
	ctx := context.Background()
	for _, target := range []struct {
		name  string
		start int
	}{
		{"input", 3}, {"weight", 7}, {"bias", 10},
	} {
		for _, initialLength := range []int{0, 2} {
			for _, transpose := range []bool{false, true} {
				backing := ttsConvTestData(16)
				before := append([]float32(nil), backing...)
				x := ttsMatrix{data: backing[2:5], channels: 1, length: 3}
				w := ttsWeight{data: backing[6:9], shape: []int{1, 1, 3}}
				bias := backing[11:12]
				// Even an empty output slice can overlap after it is extended
				// to the three required output values using its capacity.
				dst := backing[target.start : target.start+initialLength]
				var scratch ttsConvScratch
				var err error
				if transpose {
					_, err = ttsConvTranspose1DInto(ctx, x, w, bias, 1, 1, &dst, &scratch)
				} else {
					_, err = ttsConv1DInto(ctx, x, w, bias, 1, 1, 1, &dst, &scratch)
				}
				if err == nil {
					t.Fatalf("accepted %s alias with initial length=%d, transpose=%v", target.name, initialLength, transpose)
				}
				if !slices.Equal(backing, before) {
					t.Fatalf("modified %s before rejecting overlap", target.name)
				}
			}
		}
	}
}
