package gopherllm

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func f16BytesForTest(values []float32) []byte {
	// The values here are exactly representable in f16, so the existing helper
	// keeps this test focused on raw-scalar access rather than rounding.
	bits := []uint16{0x3c00, 0xbc00, 0x3800, 0xb800, 0x3800, 0x4200}
	b := make([]byte, len(values)*2)
	for i := range values {
		binary.LittleEndian.PutUint16(b[i*2:], bits[i%len(bits)])
	}
	return b
}

func TestRawScalarWeightMatvecRowAndArgmax(t *testing.T) {
	x := []float32{1, -2, 0.5}
	values := []float32{1, -1, 0.5, -0.5, 0.5, 3}
	for _, tc := range []struct {
		name string
		w    Weight
		want []float32
	}{
		{
			name: "f32",
			w:    Weight{Raw: f32Bytes(values), Type: GGMLTypeF32, Rows: 2, Cols: 3},
			want: []float32{3.25, 0},
		},
		{
			name: "f16",
			w:    Weight{Raw: f16BytesForTest(values), Type: GGMLTypeF16, Rows: 2, Cols: 3},
			want: []float32{3.25, 0},
		},
		{
			name: "bf16",
			w: Weight{Raw: []byte{
				0x80, 0x3f, 0x80, 0xbf, 0x00, 0x3f,
				0x00, 0xbf, 0x00, 0x3f, 0x40, 0x40,
			}, Type: GGMLTypeBF16, Rows: 2, Cols: 3},
			want: []float32{3.25, 0},
		},
		{
			name: "f64",
			w: Weight{Raw: func() []byte {
				b := make([]byte, len(values)*8)
				for i, v := range values {
					binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(float64(v)))
				}
				return b
			}(), Type: GGMLTypeF64, Rows: 2, Cols: 3},
			want: []float32{3.25, 0},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.w.Matvec(x)
			if len(got) != len(tc.want) {
				t.Fatalf("matvec len = %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if math.Abs(float64(got[i]-tc.want[i])) > 1e-5 {
					t.Fatalf("matvec[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
			if row := tc.w.RowF32(1, 3); len(row) != 3 || math.Abs(float64(row[0]+0.5)) > 1e-5 {
				t.Fatalf("decoded row = %v", row)
			}
			if token, ok := tc.w.ArgmaxMatvec(x); !ok || token != 0 {
				t.Fatalf("argmax = (%d, %v), want (0, true)", token, ok)
			}
		})
	}
}

func TestOutOfCorePathKeepsScalarWeightsMapped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, buildTinyLlamaGGUF(), 0o600); err != nil {
		t.Fatal(err)
	}
	r, info, err := RunnerFromPathWithOptions(path, LoadOptions{OutOfCore: true})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !info.OutOfCore || !r.OutOfCore() {
		t.Fatalf("out-of-core state was not retained: info=%+v runner=%v", info, r.OutOfCore())
	}
	if r.standard.TokenEmbd.F32 != nil || !rawScalarWeight(r.standard.TokenEmbd) {
		t.Fatalf("token embedding was expanded instead of mmap-backed: %+v", r.standard.TokenEmbd)
	}
	if r.standard.Layers[0].WQ.F32 != nil || !rawScalarWeight(r.standard.Layers[0].WQ) {
		t.Fatal("projection was expanded instead of mmap-backed")
	}
	opts := DefaultGenerationOptions()
	opts.MaxTokens = 2
	opts.SystemPrompt = ""
	opts.Sampler.Temperature = 0
	opts.Sampler.TopK = 1
	if _, err := r.Generate("a b", opts); err != nil {
		t.Fatalf("out-of-core generation: %v", err)
	}
	if _, err := r.AutoTune(AutoTuneOptions{Rounds: 1, DecodeSteps: 1, Context: 1}); err == nil {
		t.Fatal("out-of-core auto-tuning must not warm the full model")
	}
}

func TestOutOfCoreRejectsInMemoryAndCopyingModes(t *testing.T) {
	data := buildTinyLlamaGGUF()
	if _, err := RunnerFromGGUFBytesWithOptions(data, LoadOptions{OutOfCore: true}); err == nil {
		t.Fatal("expected byte-backed out-of-core load to fail")
	}
	if _, err := OpenBytes(context.Background(), data, WithOutOfCore(true)); err == nil {
		t.Fatal("expected OpenBytes out-of-core load to fail")
	}
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, options := range []LoadOptions{
		{OutOfCore: true, UseMetal: true},
		{OutOfCore: true, PrepareQuantized: true},
	} {
		if _, _, err := RunnerFromPathWithOptions(path, options); err == nil {
			t.Fatalf("expected incompatible options to fail: %+v", options)
		}
	}
}

func TestCorePrefaultRangesSkipOnlySparseExperts(t *testing.T) {
	data := buildTinySparseMoEGGUF("mixtral", false, 0)
	gguf, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	ranges, skipped := corePrefaultRanges(data, gguf)
	if skipped != 3 || len(ranges) == 0 {
		t.Fatalf("skipped=%d ranges=%v, want three experts and core ranges", skipped, ranges)
	}
	for _, tensor := range gguf.Tensors {
		if !isSparseExpertTensor(tensor.Name, tensor) {
			continue
		}
		start := gguf.DataOffset + int(tensor.Offset)
		for _, r := range ranges {
			if start >= r.start && start < r.end {
				t.Fatalf("expert tensor %s starts in prefault range %+v", tensor.Name, r)
			}
		}
	}
}

func TestNormalizeMmapRanges(t *testing.T) {
	got := normalizeMmapRanges(100, []mmapByteRange{{80, 120}, {5, 10}, {-3, 6}, {12, 12}, {20, 40}, {35, 60}})
	want := []mmapByteRange{{0, 10}, {20, 60}, {80, 100}}
	if len(got) != len(want) {
		t.Fatalf("ranges = %#v, want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ranges[%d] = %#v, want %#v", i, got[i], want[i])
		}
	}
}

func BenchmarkRawScalarF16Matvec_1024x1024(b *testing.B) {
	const rows, cols = 1024, 1024
	w := Weight{Raw: f16BytesForTest(make([]float32, rows*cols)), Type: GGMLTypeF16, Rows: rows, Cols: cols}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%17)-8) / 8
	}
	out := make([]float32, rows)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.MatvecInto(x, &out)
	}
}
