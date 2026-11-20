package engine

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

func TestBF16WeightDotMatchesWidenedFloat32(t *testing.T) {
	if !bf16WeightSIMD() {
		t.Skip("native BF16 widening unavailable")
	}
	rng := rand.New(rand.NewPCG(60, 6))
	for n := range 257 {
		x := make([]float32, n)
		widened := make([]float32, n)
		// A tensor can begin at a byte offset without the native SIMD alignment.
		raw := make([]byte, n*2+1)[1:]
		for i := range n {
			x[i] = 2*rng.Float32() - 1
			word := uint16(math.Float32bits(2*rng.Float32()-1) >> 16)
			binary.LittleEndian.PutUint16(raw[i*2:], word)
			widened[i] = math.Float32frombits(uint32(word) << 16)
		}
		got, want := dotF32BF16(x, raw), DotF32(widened, x)
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("width %d: dot=%v, widened=%v", n, got, want)
		}
	}
}

func TestBF16WeightMatvecAndArgmax(t *testing.T) {
	const rows, cols = 137, 1025
	x, raw, f32 := benchmarkBF16WeightFixture(rows, cols)
	w := Weight{Raw: raw, Rows: rows, Cols: cols, Type: GGMLTypeBF16}
	dense := Weight{F32: f32, Rows: rows, Cols: cols}
	// One nonfinite row and two identical maxima test the finite and
	// lowest-token tie rules through the real ArgmaxMatvec entry point.
	for i := range cols {
		x[i] = float32(math.Abs(float64(x[i])))
		for row := range 2 {
			binary.LittleEndian.PutUint16(raw[2*(row*cols+i):], 0x3f80)
			f32[row*cols+i] = 1
		}
	}
	binary.LittleEndian.PutUint16(raw[4*cols:], 0x7fc1)
	f32[2*cols] = float32(math.NaN())
	got, want := w.Matvec(x), dense.Matvec(x)
	for i := range got {
		if math.IsNaN(float64(want[i])) && math.IsNaN(float64(got[i])) {
			continue
		}
		if !finite32(got[i]) || bf16WeightSIMD() && math.Float32bits(got[i]) != math.Float32bits(want[i]) ||
			math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("row %d: got %g, widened reference %g", i, got[i], want[i])
		}
	}
	if token, ok := w.ArgmaxMatvec(x); !ok || token != 0 {
		t.Fatalf("argmax=%d, ok=%t; want first maximum 0", token, ok)
	}
}

func TestLittleBitBF16HeadBorrowsMappedWeight(t *testing.T) {
	if !bf16WeightSIMD() {
		t.Skip("native BF16 widening unavailable")
	}
	for _, test := range []struct {
		name string
		tied bool
	}{{"separate", false}, {"tied", true}} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newLittleBitModelFixture(t, 40, 45, 1, 4, 2, 9, 0, 2, true)
			embedding := fixture.packed[0]
			embedding.dtype = GGMLTypeBF16
			embedding.data = make([]byte, len(embedding.data)/2)
			for i := range len(embedding.data) / 2 {
				bits := binary.LittleEndian.Uint32(fixture.packed[0].data[i*4:])
				binary.LittleEndian.PutUint16(embedding.data[i*2:], uint16(bits>>16))
			}
			fixture.packed[0] = embedding
			fixture.dense[0] = embedding
			headName := embedding.name
			if !test.tied {
				head := embedding
				head.name = "output.weight"
				headName = head.name
				fixture.packed = append(fixture.packed, head)
				fixture.dense = append(fixture.dense, head)
			}
			path := filepath.Join(t.TempDir(), "head.gguf")
			data := writeLittleBitFixture(t, fixture.meta, fixture.packed)
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			r, _, err := RunnerFromPath(path)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			w := r.standard.Output
			info := indexTensors(r.GGUF())[headName]
			off := r.GGUF().DataOffset + int(info.Offset)
			if w.F32 != nil || w.Type != GGMLTypeBF16 || len(w.Raw) != len(embedding.data) ||
				&w.Raw[0] != &r.mappedFile.Bytes()[off] {
				t.Fatal("expected the LM head to borrow its original BF16 tensor bytes")
			}
			compareLittleBitModel(t, r, fixture.reference(t))
		})
	}
}

func benchmarkBF16WeightFixture(rows, cols int) ([]float32, []byte, []float32) {
	rng := rand.New(rand.NewPCG(60, 6))
	x := make([]float32, cols)
	for i := range x {
		x[i] = 2*rng.Float32() - 1
	}
	raw := make([]byte, rows*cols*2)
	f32 := make([]float32, rows*cols)
	for i := range f32 {
		word := uint16(math.Float32bits(2*rng.Float32()-1) >> 16)
		binary.LittleEndian.PutUint16(raw[i*2:], word)
		f32[i] = math.Float32frombits(uint32(word) << 16)
	}
	return x, raw, f32
}

func BenchmarkBF16WeightArgmax(b *testing.B) {
	const rows, cols = 16384, 4096
	x, raw, f32 := benchmarkBF16WeightFixture(rows, cols)
	for _, tc := range []struct {
		name string
		w    Weight
	}{
		{"BF16", Weight{Raw: raw, Rows: rows, Cols: cols, Type: GGMLTypeBF16}},
		{"WidenedF32", Weight{F32: f32, Rows: rows, Cols: cols}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			tc.w.ArgmaxMatvec(x)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				tc.w.ArgmaxMatvec(x)
			}
		})
	}
}
