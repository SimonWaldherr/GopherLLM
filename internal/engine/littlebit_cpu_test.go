package engine

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	legacyrand "math/rand"
	"math/rand/v2"
	"sync"
	"testing"
)

func TestLittleBitGoldenPackedRows(t *testing.T) {
	// 33 columns cross both a byte and a word boundary. Word padding stays
	// positive even in the second row and at the sign bit of the first word.
	signs := make([]float32, 66)
	for i := range signs {
		signs[i] = 1
	}
	for _, i := range []int{0, 7, 8, 31, 32, 34, 64} {
		signs[i] = -1
	}
	golden := []byte{0x81, 0x01, 0x00, 0x80, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x80, 0, 0, 0, 0}
	if got := packLittleBitFixture(signs, 33); !bytes.Equal(got, golden) {
		t.Fatalf("packed rows = %x, want %x", got, golden)
	}
	x := make([]float32, 33)
	for row := range 2 {
		packed := golden[row*8 : row*8+8]
		for col := range x {
			x[col] = 1
			for name, dot := range map[string]func([]byte, []float32) float32{
				"scalar": littleBitDotScalar,
				"native": littleBitDot,
			} {
				if got, want := dot(packed, x), signs[row*33+col]; got != want {
					t.Fatalf("%s [%d,%d] = %v, want %v", name, row, col, got, want)
				}
			}
			x[col] = 0
		}
	}
}

type littleBitFixture struct {
	rank           int
	u, v           []float32
	u1, u2, v1, v2 []float32
}

func makeLittleBitFixture(rng *rand.Rand, in, out, rank int, typ GGMLType) (LittleBitFactors, littleBitFixture) {
	signs := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(2*rng.IntN(2) - 1)
		}
		return s
	}
	scales := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			// Exact in each of the three scale formats.
			s[i] = float32(rng.IntN(63)-31) / 32
		}
		return s
	}
	f := littleBitFixture{
		rank: rank, u: signs(out * rank), v: signs(rank * in),
		u1: scales(out), u2: scales(rank), v1: scales(rank), v2: scales(in),
	}
	return LittleBitFactors{
		Rank: rank,
		U:    packLittleBitFixture(f.u, rank),
		V:    packLittleBitFixture(f.v, in),
		U1:   encodeLittleBitScale(f.u1, typ),
		U2:   encodeLittleBitScale(f.u2, typ),
		V1:   encodeLittleBitScale(f.v1, typ),
		V2:   encodeLittleBitScale(f.v2, typ),
	}, f
}

// Test packing uses a word-oriented construction independent of the byte-wise
// inference implementation.
func packLittleBitFixture(signs []float32, cols int) []byte {
	words := (cols + 31) / 32
	packed := make([]byte, len(signs)/cols*words*4)
	for row := 0; row < len(signs)/cols; row++ {
		for word := range words {
			var bits uint32
			for bit := range 32 {
				col := word*32 + bit
				if col < cols && signs[row*cols+col] == -1 {
					bits |= 1 << bit
				}
			}
			binary.LittleEndian.PutUint32(packed[(row*words+word)*4:], bits)
		}
	}
	return packed
}

func encodeLittleBitScale(values []float32, typ GGMLType) LittleBitScale {
	width := scalarBytesPerElement(typ)
	raw := make([]byte, len(values)*width)
	for i, v := range values {
		switch typ {
		case GGMLTypeF32:
			binary.LittleEndian.PutUint32(raw[i*width:], math.Float32bits(v))
		case GGMLTypeF16:
			binary.LittleEndian.PutUint16(raw[i*width:], F32ToF16(v))
		case GGMLTypeBF16:
			binary.LittleEndian.PutUint16(raw[i*width:], uint16(math.Float32bits(v)>>16))
		}
	}
	return LittleBitScale{Raw: raw, Type: typ}
}

// The independent float64 reference computes coefficients of W explicitly
// from the unpacked sign fixtures. This path is used only by the test.
func (f littleBitFixture) denseDot(row int, x []float32) float64 {
	var sum float64
	for j, activation := range x {
		var coef float64
		for k := range f.rank {
			u := float64(f.u[row*f.rank+k]) * float64(f.u1[row]) * float64(f.u2[k])
			v := float64(f.v[k*len(x)+j]) * float64(f.v1[k]) * float64(f.v2[j])
			coef += u * v
		}
		sum += coef * float64(activation)
	}
	return sum
}

func TestLittleBitMatvec(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x2018, 0x2026))
	for _, shape := range []struct {
		in, out, rank, residual int
		typ                     GGMLType
	}{
		{1, 1, 1, 0, GGMLTypeF32},
		{7, 5, 3, 2, GGMLTypeF16},
		{33, 17, 31, 32, GGMLTypeBF16},
		{64, 37, 33, 15, GGMLTypeF32},
		{1025, 67, 65, 0, GGMLTypeF16},
	} {
		t.Run(fmt.Sprintf("%dx%dR%dExtra%d", shape.out, shape.in, shape.rank, shape.residual), func(t *testing.T) {
			main, mainFixture := makeLittleBitFixture(rng, shape.in, shape.out, shape.rank, shape.typ)
			w := Weight{Rows: shape.out, Cols: shape.in, LittleBit: &LittleBitWeight{Main: main}}
			var residualFixture littleBitFixture
			var bias []float32
			if shape.residual > 0 {
				residual, fixture := makeLittleBitFixture(rng, shape.in, shape.out, shape.residual, GGMLTypeF32)
				w.LittleBit.Residual, residualFixture = &residual, fixture
				bias = make([]float32, shape.out)
				for i := range bias {
					bias[i] = float32(i-5) / 16
				}
				w.LittleBit.Bias = encodeLittleBitScale(bias, shape.typ)
			}
			x := make([]float32, shape.in)
			for i := range x {
				x[i] = 2*rng.Float32() - 1
			}
			want := func(row int) float64 {
				value := mainFixture.denseDot(row, x)
				if shape.residual > 0 {
					value += residualFixture.denseDot(row, x) + float64(bias[row])
				}
				return value
			}
			var scalar, native []float32
			w.LittleBit.matvecInto(x, w.Rows, w.Cols, &scalar, false)
			w.MatvecInto(x, &native)
			for row := range scalar {
				expected := want(row)
				tolerance := 2e-4 * max(1, math.Abs(expected))
				if math.Abs(float64(scalar[row])-expected) > tolerance ||
					math.Abs(float64(native[row])-expected) > tolerance {
					t.Errorf("row %d native=%v scalar=%v dense=%v (tolerance %v)", row, native[row], scalar[row], expected, tolerance)
				}
			}
			if token, ok := w.ArgmaxMatvec(x); !ok || token != argmaxFiniteToken(native) {
				t.Errorf("argmax=%d, %v, want %d", token, ok, argmaxFiniteToken(native))
			}
			tokens := [][]float32{x, x}
			outs := [][]float32{make([]float32, shape.out), make([]float32, shape.out)}
			matvecBatch(w, tokens, outs)
			for i, out := range outs {
				for row, val := range out {
					if val != native[row] {
						t.Errorf("batch %d row %d = %v, want %v", i, row, val, native[row])
					}
				}
			}
			var argmax []uint32
			if !w.ArgmaxMatvecBatch(tokens, &argmax) || argmax[0] != argmaxFiniteToken(native) || argmax[1] != argmax[0] {
				t.Errorf("batch argmax=%v", argmax)
			}
		})
	}
}

func TestLittleBitConcurrentMatvec(t *testing.T) {
	const in, out, rank = 1025, 257, 193
	rng := rand.New(rand.NewPCG(6, 13))
	main, _ := makeLittleBitFixture(rng, in, out, rank, GGMLTypeF16)
	w := Weight{Rows: out, Cols: in, LittleBit: &LittleBitWeight{Main: main}}
	var done sync.WaitGroup
	for i := range 4 {
		x := make([]float32, in)
		for j := range x {
			x[j] = rng.Float32() - float32(i)/4
		}
		want := w.Matvec(x)
		done.Add(1)
		go func() {
			defer done.Done()
			y := make([]float32, out)
			for range 4 {
				w.MatvecInto(x, &y)
				for row, got := range y {
					if got != want[row] {
						t.Errorf("row %d = %v, want %v", row, got, want[row])
						return
					}
				}
			}
		}()
	}
	done.Wait()
}

func BenchmarkLittleBitMatvec(b *testing.B) {
	rng := rand.New(rand.NewPCG(8, 32))
	for _, rank := range []int{32, 256, 1024} {
		const in, out = 4096, 4096
		main, _ := makeLittleBitFixture(rng, in, out, rank, GGMLTypeF16)
		w := &LittleBitWeight{Main: main}
		x, y := make([]float32, in), make([]float32, out)
		for i := range x {
			x[i] = rng.Float32()
		}
		for _, optimized := range []bool{false, true} {
			name := "portable"
			if optimized {
				name = "native"
			}
			b.Run(fmt.Sprintf("rank=%d/%s", rank, name), func(b *testing.B) {
				w.matvecInto(x, out, in, &y, optimized)
				b.ReportAllocs()
				size := len(main.U) + len(main.V) + len(main.U1.Raw) + len(main.U2.Raw) + len(main.V1.Raw) + len(main.V2.Raw)
				b.SetBytes(int64(size))
				b.ResetTimer()
				for range b.N {
					w.matvecInto(x, out, in, &y, optimized)
				}
				b.ReportMetric(float64(8*size)/(in*out), "bpw")
			})
		}
	}
}

// BenchmarkLittleBitProjectionBaselines uses the same dimensions as
// BenchmarkLittleBitMatvec. Q4_K is the individual tensor type used by the
// model-level Q4_K_M preset alongside other types such as Q6_K.
func BenchmarkLittleBitProjectionBaselines(b *testing.B) {
	const in, out = 4096, 4096
	rng := legacyrand.New(legacyrand.NewSource(32))
	x, y := randomVec(rng, in), make([]float32, out)
	for _, format := range []struct {
		typ GGMLType
		row func(*legacyrand.Rand, int) []byte
	}{
		{GGMLTypeQ1_0, func(rng *legacyrand.Rand, cols int) []byte {
			data := make([]byte, (cols/128)*18)
			for offset := 0; offset < len(data); offset += 18 {
				binary.LittleEndian.PutUint16(data[offset:], F32ToF16(0.0625))
				_, _ = rng.Read(data[offset+2 : offset+18])
			}
			return data
		}},
		{GGMLTypeQ2_K, randomQ2KRow},
		{GGMLTypeQ4_K, randomQ4KRow},
	} {
		b.Run(fmt.Sprint(format.typ), func(b *testing.B) {
			size, _ := format.typ.DataSize(in * out)
			data := make([]byte, 0, size)
			for range out {
				data = append(data, format.row(rng, in)...)
			}
			w := Weight{Type: format.typ, Raw: data, Rows: out, Cols: in}
			w.MatvecInto(x, &y)
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for range b.N {
				w.MatvecInto(x, &y)
			}
			b.ReportMetric(float64(8*len(data))/(in*out), "bpw")
		})
	}
}
