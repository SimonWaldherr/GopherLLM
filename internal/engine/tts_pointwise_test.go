package engine

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"
	"time"
)

// The previous kernel=1 path packed each input tile and copied the GEMM output.
func ttsPointwisePackedReference(ctx context.Context, x ttsMatrix, weight ttsWeight, bias []float32, dilation, padding, groups int, dst *[]float32, scratch *ttsConvScratch) (ttsMatrix, error) {
	if err := ctx.Err(); err != nil {
		return ttsMatrix{}, err
	}
	in, out, kernel, err := ttsConvInputValid(x, weight, bias, false)
	if err != nil {
		return ttsMatrix{}, err
	}
	if dilation <= 0 || padding < 0 || groups <= 0 || x.channels%groups != 0 || out%groups != 0 || in != x.channels/groups {
		return ttsMatrix{}, fmt.Errorf("tts convolution: invalid dilation, padding or groups")
	}
	maxInt := int(^uint(0) >> 1)
	if kernel-1 > (maxInt-1)/dilation || padding > (maxInt-x.length)/2 {
		return ttsMatrix{}, fmt.Errorf("tts convolution: output size overflow")
	}
	length := x.length + 2*padding - dilation*(kernel-1)
	if length <= 0 || out > maxInt/length {
		return ttsMatrix{}, fmt.Errorf("tts convolution: invalid output size")
	}
	ensureLenNoClear(dst, out*length)
	if batchSlicesOverlap(*dst, x.data) || batchSlicesOverlap(*dst, weight.data) || batchSlicesOverlap(*dst, bias) {
		return ttsMatrix{}, fmt.Errorf("tts convolution: output overlaps input or weights")
	}
	if kernel != 1 || padding != 0 {
		return ttsMatrix{}, fmt.Errorf("pointwise benchmark requires an unpadded unit kernel")
	}
	rows := out / groups
	y := ttsMatrix{*dst, out, length}
	tile := min(ttsConvTileFrames, length)
	ensureLenNoClear(&scratch.patch, in*tile)
	ensureLenNoClear(&scratch.projected, rows*tile)
	for g := range groups {
		for start := 0; start < length; start += tile {
			n := min(tile, length-start)
			patch, projected := scratch.patch[:in*n], scratch.projected[:rows*n]
			for ic := range in {
				copyPaddedWindow(patch[ic*n:(ic+1)*n], x.data[(g*in+ic)*length:(g*in+ic+1)*length], start)
			}
			if err := scratch.gemm.multiply(ctx, rows, n, in, weight.data[g*rows*in:(g+1)*rows*in], patch, projected); err != nil {
				return ttsMatrix{}, err
			}
			for oc := range rows {
				var offset float32
				if bias != nil {
					offset = bias[g*rows+oc]
				}
				for j := range n {
					y.data[(g*rows+oc)*length+start+j] = projected[oc*n+j] + offset
				}
			}
		}
	}
	return y, nil
}

func TestTTSPointwiseDirectMatchesPacked(t *testing.T) {
	setPerformanceThreads(t, 3)
	for _, length := range []int{1, 31, 127, 128, 129, 257} {
		for _, groups := range []int{1, 2} {
			for _, out := range []int{2, 4, 12} {
				for _, withBias := range []bool{false, true} {
					t.Run(fmt.Sprintf("length=%d/out=%d/groups=%d/bias=%t", length, out, groups, withBias), func(t *testing.T) {
						x := ttsMatrix{ttsConvTestData(6 * length), 6, length}
						weight := ttsWeight{data: ttsConvTestData(out * 6 / groups), shape: []int{out, 6 / groups, 1}}
						var bias []float32
						if withBias {
							bias = ttsConvTestData(out)
						}
						original := slices.Clone(x.data)
						var gotData, wantData []float32
						var gotScratch, wantScratch ttsConvScratch
						got, err := ttsConv1DInto(context.Background(), x, weight, bias, 3, 0, groups, &gotData, &gotScratch)
						if err != nil {
							t.Fatal(err)
						}
						want, err := ttsPointwisePackedReference(context.Background(), x, weight, bias, 3, 0, groups, &wantData, &wantScratch)
						if err != nil {
							t.Fatal(err)
						}
						for i, value := range got.data {
							if math.Float32bits(value) != math.Float32bits(want.data[i]) {
								t.Fatalf("output[%d]=%g, want %g", i, value, want.data[i])
							}
						}
						if !slices.Equal(x.data, original) {
							t.Fatal("pointwise convolution changed input")
						}
						if length <= ttsConvTileFrames && out/groups <= 4 && (gotScratch.patch != nil || gotScratch.projected != nil) {
							t.Fatal("direct pointwise convolution allocated packing buffers")
						}
					})
				}
			}
		}
	}
}

func BenchmarkTTSPointwiseDirect(b *testing.B) {
	ctx := context.Background()
	for _, shape := range [][3]int{{192, 2, 32}, {192, 2, 64}, {192, 4, 128}, {192, 192, 32}} {
		in, out, length := shape[0], shape[1], shape[2]
		b.Run(fmt.Sprintf("in=%d/out=%d/frames=%d", in, out, length), func(b *testing.B) {
			x := ttsMatrix{ttsConvTestData(in * length), in, length}
			weight := ttsWeight{data: ttsConvTestData(out * in), shape: []int{out, in, 1}}
			bias := ttsConvTestData(out)
			var referenceData, directData []float32
			var referenceScratch, directScratch ttsConvScratch
			old := func() {
				if _, err := ttsPointwisePackedReference(ctx, x, weight, bias, 1, 0, 1, &referenceData, &referenceScratch); err != nil {
					b.Fatal(err)
				}
			}
			direct := func() {
				if _, err := ttsConv1DInto(ctx, x, weight, bias, 1, 0, 1, &directData, &directScratch); err != nil {
					b.Fatal(err)
				}
			}
			old()
			direct()
			referenceSamples, directSamples := make([]int64, b.N), make([]int64, b.N)
			const batch = 16
			measure := func(run func()) int64 {
				start := time.Now()
				for range batch {
					run()
				}
				return time.Since(start).Nanoseconds() / batch
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := range b.N {
				if i&1 == 0 {
					referenceSamples[i], directSamples[i] = measure(old), measure(direct)
				} else {
					directSamples[i], referenceSamples[i] = measure(direct), measure(old)
				}
			}
			b.StopTimer()
			slices.Sort(referenceSamples)
			slices.Sort(directSamples)
			b.ReportMetric(float64(referenceSamples[len(referenceSamples)/2]), "packed_ns/op")
			b.ReportMetric(float64(directSamples[len(directSamples)/2]), "direct_ns/op")
			b.ReportMetric(float64((cap(referenceScratch.patch)+cap(referenceScratch.projected))*4), "packed_scratch_B")
			b.ReportMetric(float64((cap(directScratch.patch)+cap(directScratch.projected))*4), "direct_scratch_B")
			b.Logf("packed_sorted_ns=%v direct_sorted_ns=%v", referenceSamples, directSamples)
		})
	}
}
