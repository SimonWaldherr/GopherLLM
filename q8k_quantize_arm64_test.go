//go:build arm64

package gopherllm

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"

	workerthreads "github.com/SimonWaldherr/GopherLLM/internal/threads"
)

func TestQ8KQuantizeMatchesPortableBits(t *testing.T) {
	if !validateQ8KQuantizeNEON() {
		t.Fatal("NEON quantization self-check failed")
	}
	previous := q8kQuantizeNEONOK
	q8kQuantizeNEONOK = true
	t.Cleanup(func() { q8kQuantizeNEONOK = previous })
	rng := rand.New(rand.NewSource(1882))
	patterns := []struct {
		name string
		fill func([]float32)
	}{
		{"finite", func(x []float32) {
			for i := range x {
				x[i] = float32(rng.NormFloat64()) * float32(math.Ldexp(1, i/256%80-40))
			}
		}},
		{"random bits", func(x []float32) {
			for i := range x {
				x[i] = math.Float32frombits(rng.Uint32())
			}
		}},
		{"ties", func(x []float32) {
			for i := range x {
				v := float32(i%254-127) + .5
				switch i % 3 {
				case 0:
					v = math.Nextafter32(v, float32(math.Inf(-1)))
				case 2:
					v = math.Nextafter32(v, float32(math.Inf(1)))
				}
				x[i] = v
			}
			for i := 0; i < len(x); i += 256 {
				x[i], x[i+1] = 127, -127
			}
		}},
		{"signed zero", func(x []float32) {
			for i := range x {
				x[i] = math.Float32frombits(uint32(i&1) << 31)
			}
		}},
		{"all NaN", func(x []float32) {
			for i := range x {
				x[i] = math.Float32frombits(0x7f800001 + uint32(i%256) + uint32(i&1)<<31)
			}
		}},
		{"NaN and finite", func(x []float32) {
			for i := range x {
				x[i] = float32(i%131-65) / 16
				if i%7 == 0 {
					x[i] = math.Float32frombits(0xff800001 + uint32(i%256))
				}
			}
		}},
		{"infinity", func(x []float32) {
			for i := range x {
				x[i] = float32(i%131-65) / 16
				if i%17 == 0 {
					x[i] = float32(math.Inf(i%2*2 - 1))
				}
			}
		}},
		{"subnormal", func(x []float32) {
			for i := range x {
				x[i] = math.Float32frombits(uint32(i%255+1) | uint32(i&1)<<31)
			}
		}},
		{"tiny normal", func(x []float32) {
			for i := range x {
				x[i] = math.Float32frombits(0x00800000 + uint32(i%255) | uint32(i&1)<<31)
			}
		}},
		{"float extremes", func(x []float32) {
			bits := [...]uint32{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff}
			for i := range x {
				x[i] = math.Float32frombits(bits[i%len(bits)])
			}
		}},
	}
	for _, cols := range []int{256, 512, 3072, 9216} {
		for _, pattern := range patterns {
			t.Run(fmt.Sprintf("cols%d/%s", cols, pattern.name), func(t *testing.T) {
				x := make([]float32, cols+3)
				x[0], x[cols+1], x[cols+2] = 1e30, -1e30, 1e30
				pattern.fill(x[1 : cols+1])
				original := append([]float32(nil), x...)
				q8, want := make([]int8, cols+3), make([]int8, cols+3)
				for i := range q8 {
					q8[i], want[i] = 37, 37
				}
				scales, wantScales := make([]float32, cols/256+2), make([]float32, cols/256+2)
				for i := range scales {
					scales[i], wantScales[i] = -99, -99
				}
				q8kQuantizePortable(x[1:cols+1], want[1:cols+1], wantScales[1:len(wantScales)-1], cols/256)
				q8kQuantize(x[1:cols+1], q8[1:cols+1], scales[1:len(scales)-1], cols/256)
				for i, got := range q8 {
					if got != want[i] {
						t.Fatalf("q8[%d]=%d, want %d", i, got, want[i])
					}
				}
				for i, got := range scales {
					if math.Float32bits(got) != math.Float32bits(wantScales[i]) {
						t.Fatalf("scale[%d]=%g (%08x), want %g (%08x)", i, got, math.Float32bits(got), wantScales[i], math.Float32bits(wantScales[i]))
					}
				}
				for i, got := range x {
					if math.Float32bits(got) != math.Float32bits(original[i]) {
						t.Fatalf("input/guard[%d] changed", i)
					}
				}
			})
		}
	}
}

func TestQ8KQuantizeSIMDEnvironment(t *testing.T) {
	if os.Getenv("GOPHERLLM_DISABLE_SIMD") != "" && q8kQuantizeNEONOK {
		t.Fatal("NEON quantizer enabled despite GOPHERLLM_DISABLE_SIMD")
	}
}

func TestQ8KQuantizeBounds(t *testing.T) {
	q8kQuantize(nil, nil, nil, 0)
	q8kQuantize(nil, nil, nil, -1)
	for _, tc := range []struct {
		name                  string
		x, q8, scales, blocks int
	}{
		{"input", 255, 256, 1, 1},
		{"quants", 256, 255, 1, 1},
		{"scales", 256, 256, 0, 1},
		{"later block", 511, 512, 2, 2},
		{"overflow", 256, 256, 1, int(^uint(0) >> 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("short buffers accepted")
				}
			}()
			q8kQuantize(make([]float32, tc.x), make([]int8, tc.q8), make([]float32, tc.scales), tc.blocks)
		})
	}
	x := make([]float32, 513)
	q8, want := make([]int8, 513), make([]int8, 513)
	for i := range x {
		x[i], q8[i], want[i] = float32(i%127-63), 37, 37
	}
	scales, wantScales := []float32{-99, -99, -99}, []float32{-99, -99, -99}
	q8kQuantizePortable(x, want, wantScales, 1)
	q8kQuantize(x, q8, scales, 1)
	for i := range q8 {
		if q8[i] != want[i] {
			t.Fatalf("prefix/untouched tail at %d differs", i)
		}
	}
	for i := range scales {
		if math.Float32bits(scales[i]) != math.Float32bits(wantScales[i]) {
			t.Fatalf("scale prefix/untouched tail at %d differs", i)
		}
	}
}

func TestQ8KQuantizeProjectionParity(t *testing.T) {
	if !hasQ8KDotAsm {
		t.Skip("dot-product kernels unavailable")
	}
	if !validateQ8KQuantizeNEON() {
		t.Fatal("NEON quantization self-check failed")
	}
	previous := q8kQuantizeNEONOK
	t.Cleanup(func() { q8kQuantizeNEONOK = previous })
	rng := rand.New(rand.NewSource(1884))
	const rows, cols = 17, 512
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K, GGMLTypeQ8_0} {
		var raw []byte
		for range rows {
			switch typ {
			case GGMLTypeQ4_K:
				raw = append(raw, randomQ4KRow(rng, cols)...)
			case GGMLTypeQ6_K:
				raw = append(raw, randomQ6KRow(rng, cols)...)
			case GGMLTypeQ8_0:
				raw = append(raw, randomQ8_0Row(rng, cols)...)
			}
		}
		w := Weight{Raw: raw, Type: typ, Rows: rows, Cols: cols}
		for _, tokens := range []int{1, 4, 5, 32, 128} {
			xs, got, want := make([][]float32, tokens), make([][]float32, tokens), make([][]float32, tokens)
			for token := range tokens {
				xs[token] = randomVec(rng, cols)
				got[token], want[token] = make([]float32, rows), make([]float32, rows)
			}
			withQ8Activations(true, func() {
				q8kQuantizeNEONOK = false
				matvecBatchWithQ8(w, xs, want, true)
				q8kQuantizeNEONOK = true
				matvecBatchWithQ8(w, xs, got, true)
			})
			for token := range tokens {
				for row := range rows {
					if math.Float32bits(got[token][row]) != math.Float32bits(want[token][row]) {
						t.Fatalf("type=%v tokens=%d token=%d row=%d: got %g, want %g", typ, tokens, token, row, got[token][row], want[token][row])
					}
				}
			}
		}
	}
}

func BenchmarkARM64Q8KActivationQuantization(b *testing.B) {
	for _, cols := range []int{3072, 9216} {
		for _, tokens := range []int{1, 32, 128} {
			x := benchFloatSlice(cols * tokens)
			q8 := make([]int8, len(x))
			scales := make([]float32, len(x)/256)
			for _, mode := range []struct {
				name string
				fn   func([]float32, []int8, []float32, int)
			}{{"portable", q8kQuantizePortable}, {"dispatch", q8kQuantize}} {
				b.Run(fmt.Sprintf("cols%d/p%d/%s", cols, tokens, mode.name), func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(len(x) * 4))
					for b.Loop() {
						for token := range tokens {
							mode.fn(x[token*cols:(token+1)*cols], q8[token*cols:], scales[token*(cols/256):], cols/256)
						}
					}
				})
			}
		}
	}
}

func BenchmarkARM64Q8KQuantizationProjection(b *testing.B) {
	if !hasQ8KDotAsm {
		b.Skip("dot-product kernels unavailable")
	}
	previousThreads := configuredThreads.Load()
	previousProcs := runtime.GOMAXPROCS(8)
	SetNumThreads(8)
	b.Cleanup(func() {
		configuredThreads.Store(previousThreads)
		workerthreads.Set(int(previousThreads))
		runtime.GOMAXPROCS(previousProcs)
	})
	for _, shape := range []struct{ cols, rows int }{{3072, 9216}, {9216, 3072}} {
		rng := rand.New(rand.NewSource(1883))
		raw := make([]byte, 0, shape.rows*(shape.cols/256)*144)
		for range shape.rows {
			raw = append(raw, randomQ4KRow(rng, shape.cols)...)
		}
		w := Weight{Raw: raw, Type: GGMLTypeQ4_K, Rows: shape.rows, Cols: shape.cols}
		for _, tokens := range []int{1, 32, 128} {
			xs, outs := make([][]float32, tokens), make([][]float32, tokens)
			for token := range tokens {
				xs[token], outs[token] = randomVec(rng, shape.cols), make([]float32, shape.rows)
			}
			b.Run(fmt.Sprintf("cols%d/rows%d/p%d", shape.cols, shape.rows, tokens), func(b *testing.B) {
				withQ8Activations(true, func() {
					matvecBatchWithQ8(w, xs, outs, true)
					b.ReportAllocs()
					b.ResetTimer()
					for b.Loop() {
						matvecBatchWithQ8(w, xs, outs, true)
					}
				})
			})
		}
	}
}
