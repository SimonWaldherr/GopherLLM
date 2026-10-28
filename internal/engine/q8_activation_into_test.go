package engine

import (
	"math"
	"slices"
	"testing"
)

func TestQuantizeQ8IntoMatchesLeasedInput(t *testing.T) {
	if !q8ActivationsAvailable() {
		t.Skip("Q8 activation instructions unavailable")
	}
	for _, cols := range []int{256, 512, 2048} {
		x := randomExpertInput(cols, 42100)
		clear(x[:256])
		q8 := make([]int8, cols+2)
		scales := make([]float32, cols/256+2)
		q8[0], q8[len(q8)-1] = -71, 93
		scales[0], scales[len(scales)-1] = -123, 456
		quantizeQ8Into(x, q8[1:len(q8)-1], scales[1:len(scales)-1])
		wantQ, wantS, lease := acquireQ8(x, cols)
		func() {
			defer releaseQ8(wantQ, wantS, lease)
			if !slices.Equal(q8[1:len(q8)-1], wantQ) {
				t.Fatalf("width %d quantized input differs", cols)
			}
			for i, scale := range wantS {
				if math.Float32bits(scales[i+1]) != math.Float32bits(scale) {
					t.Fatalf("width %d scale %d differs", cols, i)
				}
			}
		}()
		if q8[0] != -71 || q8[len(q8)-1] != 93 || scales[0] != -123 || scales[len(scales)-1] != 456 {
			t.Fatal("quantization changed adjacent rows")
		}
	}
}

func TestQuantizeQ8IntoBufferBounds(t *testing.T) {
	quantizeQ8Into(nil, nil, nil)
	for _, sizes := range [][3]int{{255, 256, 1}, {256, 255, 1}, {512, 512, 1}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("accepted invalid buffers: %v", sizes)
				}
			}()
			quantizeQ8Into(make([]float32, sizes[0]), make([]int8, sizes[1]), make([]float32, sizes[2]))
		}()
	}
}
