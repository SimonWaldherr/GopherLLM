package yolo

import (
	"fmt"
	"testing"

	"github.com/SimonWaldherr/GopherLLM/internal/yolotest"
)

var yoloBenchmarkResult yoloTensor

func benchmarkYOLOTensor(c, h, w int) yoloTensor {
	x := yoloTensor{c: c, h: h, w: w, data: make([]float32, c*h*w)}
	for i := range x.data {
		x.data[i] = float32(i%37-18) / 19
	}
	return x
}

func BenchmarkYOLOIm2col(b *testing.B) {
	for _, stride := range []int{1, 2} {
		b.Run(fmt.Sprintf("Stride%d", stride), func(b *testing.B) {
			x := benchmarkYOLOTensor(32, 80, 80)
			oh, ow := (x.h-1)/stride+1, (x.w-1)/stride+1
			b.ReportAllocs()
			b.SetBytes(int64(x.c * 9 * oh * ow * 4))
			b.ResetTimer()
			for range b.N {
				yoloBenchmarkResult.data = yoloIm2col(x, 3, stride, 1, oh, ow)
			}
		})
	}
}

func BenchmarkYOLODepthwise(b *testing.B) {
	x := benchmarkYOLOTensor(32, 80, 80)
	cv := yoloConv{in: x.c, out: x.c, k: 3, stride: 1, depthwise: true, w: make([]float32, x.c*9)}
	for i := range cv.w {
		cv.w[i] = float32(i%7-3) / 9
	}
	out := benchmarkYOLOTensor(x.c, x.h, x.w)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		yoloDepthwise(x, &cv, 1, out)
	}
	yoloBenchmarkResult = out
}

func BenchmarkYOLOAttention(b *testing.B) {
	const heads, keyDim, headDim = 2, 32, 64
	dim := heads * headDim
	conv := func(out, in, k int, dw bool) yoloConv {
		per := in * k * k
		if dw {
			per = k * k
		}
		cv := yoloConv{in: in, out: out, k: k, stride: 1, depthwise: dw, w: make([]float32, out*per), b: make([]float32, out)}
		for i := range cv.w {
			cv.w[i] = float32(i%11-5) / 17
		}
		return cv
	}
	a := yoloAttention{heads: heads, keyDim: keyDim, headDim: headDim,
		qkv: conv(heads*(2*keyDim+headDim), dim, 1, false), proj: conv(dim, dim, 1, false), pe: conv(dim, dim, 3, true)}
	x := benchmarkYOLOTensor(dim, 20, 20)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		var err error
		yoloBenchmarkResult, err = a.forward(x)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkYOLOForward(b *testing.B) {
	for _, tc := range []struct {
		name  string
		model func() *yolotest.Checkpoint
		size  int
	}{
		{"V8", yolotest.V8, 128}, {"V11", yolotest.V11, 64},
	} {
		b.Run(tc.name, func(b *testing.B) {
			m, err := LoadYOLOSafetensors(tc.model().Safetensors(yoloUltralyticsNames))
			if err != nil {
				b.Fatal(err)
			}
			input, err := PrepareYOLOImage(yoloSynthImage(), YOLOConfig{InputWidth: tc.size, InputHeight: tc.size})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, _, _, err := m.Forward(input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
