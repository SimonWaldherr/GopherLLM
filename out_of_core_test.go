package gopherllm

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
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

func TestQwen35OutOfCoreKeepsDeltaConvHot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := os.WriteFile(path, buildTinyQwen35MoEGGUF(false), 0o600); err != nil {
		t.Fatal(err)
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{OutOfCore: true})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if !r.OutOfCore() || r.qwen35.TokenEmbd.F32 != nil {
		t.Fatal("Qwen3.5 model did not retain its main scalar weights as out-of-core views")
	}
	conv := r.qwen35.Layers[0].DeltaNet.ConvKernel
	if conv.F32 == nil || len(conv.F32) == 0 {
		t.Fatal("Qwen3.5 DeltaNet convolution should remain a small hot F32 tensor")
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

func TestCorePrefaultRangesSkipFusedSparseGateUpExperts(t *testing.T) {
	data := buildTinySparseMoEGGUFWithExpertLayout("mixtral", false, 0, false, true)
	gguf, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	ranges, skipped := corePrefaultRanges(data, gguf)
	if skipped != 2 || len(ranges) == 0 {
		t.Fatalf("skipped=%d ranges=%v, want fused gate/up and down expert banks", skipped, ranges)
	}
	for _, tensor := range gguf.Tensors {
		if !isSparseExpertTensor(tensor.Name, tensor) {
			continue
		}
		start := gguf.DataOffset + int(tensor.Offset)
		for _, r := range ranges {
			if start >= r.start && start < r.end {
				t.Fatalf("fused expert tensor %s starts in prefault range %+v", tensor.Name, r)
			}
		}
	}
}

func TestCorePrefaultRangesKeepQwenNextNCold(t *testing.T) {
	for _, tc := range []struct {
		name    string
		arch    string
		layers  uint32
		nextN   uint32
		markers []string
		cold    bool
	}{
		{"dense-metadata", "qwen35", 3, 1, []string{"blk.2.nextn.eh_proj.weight"}, true},
		{"moe-metadata", "qwen35moe", 3, 1, []string{"blk.2.nextn.eh_proj.weight"}, true},
		{"dense-marker-fallback", "qwen35", 3, 0, []string{"blk.2.nextn.eh_proj.weight"}, true},
		{"moe-marker-fallback", "qwen35moe", 3, 0, []string{"blk.2.nextn.eh_proj.weight"}, true},
		{"metadata-without-marker", "qwen35", 3, 1, nil, false},
		{"nontrailing-marker", "qwen35", 3, 0, []string{"blk.1.nextn.eh_proj.weight"}, false},
		{"unsupported-nextn-count", "qwen35", 3, 2, []string{"blk.1.nextn.eh_proj.weight", "blk.2.nextn.eh_proj.weight"}, false},
		{"unsupported-marker-chain", "qwen35", 3, 0, []string{"blk.1.nextn.eh_proj.weight", "blk.2.nextn.eh_proj.weight"}, false},
		{"missing-trunk", "qwen35", 1, 1, []string{"blk.0.nextn.eh_proj.weight"}, false},
		{"missing-layer-count", "qwen35", 0, 1, []string{"blk.2.nextn.eh_proj.weight"}, false},
		{"other-architecture", "exaone4", 3, 1, []string{"blk.2.nextn.eh_proj.weight"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := &GGUFFile{Metadata: map[string]MetaValue{
				"general.architecture":            {Kind: "str", Value: tc.arch},
				tc.arch + ".block_count":          {Kind: "u32", Value: tc.layers},
				tc.arch + ".nextn_predict_layers": {Kind: "u32", Value: tc.nextN},
			}}
			if tc.nextN == 0 {
				delete(g.Metadata, tc.arch+".nextn_predict_layers")
			}
			names := []string{
				"token_embd.weight", "output.weight", "output_norm.weight",
				"blk.0.attn_q.weight", "blk.1.ffn_up.weight",
				"blk.2.attn_q.weight", "blk.2.ffn_down.weight", "blk.2.attn_norm.weight",
				// A similar layer number or a companion encoder is not the draft block.
				"blk.20.attn_q.weight", "vision.blk.2.attn_q.weight",
			}
			names = append(names, tc.markers...)
			const tensorBytes = 64
			for i, name := range names {
				g.Tensors = append(g.Tensors, TensorInfo{
					Name: name, Dims: []uint64{16}, DType: GGMLTypeF32, Offset: uint64(i * tensorBytes),
				})
			}
			data := make([]byte, len(names)*tensorBytes)
			ranges, skippedExperts := corePrefaultRanges(data, g)
			if skippedExperts != 0 {
				t.Fatalf("NextN tensors counted as sparse experts: %d", skippedExperts)
			}
			for i, name := range names {
				wantWarm := !(tc.cold && strings.HasPrefix(name, "blk.2."))
				// Check both ends so a neighboring warm tensor cannot hide a
				// partial overlap with an optional tensor after range merging.
				for _, offset := range []int{i * tensorBytes, (i+1)*tensorBytes - 1} {
					warm := false
					for _, r := range ranges {
						warm = warm || (offset >= r.start && offset < r.end)
					}
					if warm != wantWarm {
						t.Fatalf("%s byte %d: warm=%t, want %t (ranges=%v)", name, offset, warm, wantWarm, ranges)
					}
				}
			}
		})
	}
}

func TestCorePrefaultRangesKeepUntiedInputEmbeddingCold(t *testing.T) {
	for _, tc := range []struct {
		name     string
		arch     string
		typ      GGMLType
		dims     []uint64
		separate bool
		cold     bool
	}{
		{"dense-q4k", "qwen35", GGMLTypeQ4_K, []uint64{256, 3}, true, true},
		{"moe-q6k", "qwen35moe", GGMLTypeQ6_K, []uint64{256, 3}, true, true},
		{"dense-q8", "qwen35", GGMLTypeQ8_0, []uint64{256, 3}, true, true},
		{"tied-output", "qwen35", GGMLTypeQ4_K, []uint64{256, 3}, false, false},
		{"standard-architecture", "llama", GGMLTypeQ4_K, []uint64{256, 3}, true, true},
		{"ministral14b-untied", "mistral3", GGMLTypeQ4_K, []uint64{5120, 3}, true, true},
		{"ministral3b-tied", "mistral3", GGMLTypeQ6_K, []uint64{3072, 3}, false, false},
		{"f32-expanded", "qwen35", GGMLTypeF32, []uint64{256, 3}, true, false},
		{"f16-expanded", "qwen35", GGMLTypeF16, []uint64{256, 3}, true, false},
		{"bf16-expanded", "qwen35", GGMLTypeBF16, []uint64{256, 3}, true, false},
		{"f64-expanded", "qwen35", GGMLTypeF64, []uint64{256, 3}, true, false},
		{"unknown-format", "qwen35", GGMLTypeUnknown, []uint64{256, 3}, true, false},
		{"invalid-row-packing", "qwen35", GGMLTypeQ4_K, []uint64{257, 3}, true, false},
		{"not-a-matrix", "qwen35", GGMLTypeQ4_K, []uint64{768}, true, false},
		{"empty-matrix", "qwen35", GGMLTypeQ4_K, []uint64{256, 0}, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := &GGUFFile{Metadata: map[string]MetaValue{
				"general.architecture": {Kind: "str", Value: tc.arch},
			}}
			embedding := TensorInfo{Name: "token_embd.weight", Dims: tc.dims, DType: tc.typ}
			bytes, _ := tc.typ.DataSize(embedding.Numel())
			// Give malformed descriptors a small nonempty inferred range so the
			// test verifies conservative retention rather than dropping no bytes.
			embeddingBytes := max(64, bytes)
			g.Tensors = append(g.Tensors, embedding,
				TensorInfo{Name: "output_norm.weight", Dims: []uint64{16}, DType: GGMLTypeF32, Offset: uint64(embeddingBytes)},
				TensorInfo{Name: "blk.0.attn_q.weight", Dims: []uint64{16}, DType: GGMLTypeF32, Offset: uint64(embeddingBytes + 64)})
			if tc.separate {
				g.Tensors = append(g.Tensors, TensorInfo{Name: "output.weight", Dims: []uint64{256, 3}, DType: GGMLTypeQ6_K, Offset: uint64(embeddingBytes + 128)})
			}
			data := make([]byte, embeddingBytes+128+630)
			ranges, skippedExperts := corePrefaultRanges(data, g)
			if skippedExperts != 0 {
				t.Fatalf("embedding counted as sparse experts: %d", skippedExperts)
			}
			for i, tensor := range g.Tensors {
				wantWarm := i != 0 || !tc.cold
				start := int(tensor.Offset)
				length, _ := tensor.DType.DataSize(tensor.Numel())
				if i == 0 && length <= 0 {
					length = embeddingBytes
				}
				for _, offset := range []int{start, start + length - 1} {
					warm := false
					for _, r := range ranges {
						warm = warm || (offset >= r.start && offset < r.end)
					}
					if warm != wantWarm {
						t.Fatalf("%s byte %d: warm=%t, want %t (ranges=%v)", tensor.Name, offset, warm, wantWarm, ranges)
					}
				}
			}
		})
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
