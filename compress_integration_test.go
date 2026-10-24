package gopherllm

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCompressMixedMetadataAndLoad(t *testing.T) {
	input := writeTempGGUF(t, buildRequantizableLlamaGGUF())
	for _, preset := range []string{"Q4_K_M", "Q5_K_M"} {
		t.Run(preset, func(t *testing.T) {
			output := filepath.Join(t.TempDir(), "quantized.gguf")
			opts, _ := ParseCompressOptions(preset)
			if err := CompressModel(input, output, opts); err != nil {
				t.Fatal(err)
			}
			g := parseGGUFFile(t, output)
			wantFileType := uint32(15)
			if preset == "Q5_K_M" {
				wantFileType = 17
			}
			if g.GetU32("general.file_type", 0) != wantFileType || g.GetU32("general.quantization_version", 0) != 2 {
				t.Fatal("incorrect quantization metadata")
			}
			if method, _ := g.GetString("gopherllm.quantization.method"); method != "rtn" {
				t.Fatal("missing encoder provenance")
			}
			if tensorType(g, "token_embd.weight") != opts.TargetFormat || tensorType(g, "output.weight") != GGMLTypeQ6_K {
				t.Fatal("wrong mixed embedding/head policy")
			}
			encoded, err := os.ReadFile(output)
			if err != nil {
				t.Fatal(err)
			}
			r, err := RunnerFromGGUFBytes(encoded)
			if err != nil {
				t.Fatalf("load mixed GGUF: %v", err)
			}
			defer r.Close()
			generation := DefaultGenerationOptions()
			generation.MaxTokens = 2
			if _, err := r.Generate("a b", generation); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCompressMetadataDoesNotMutateSource(t *testing.T) {
	source := map[string]MetaValue{
		"general.file_type":   {Kind: "u32", Value: uint32(1)},
		"split.no":            {Kind: "u16", Value: uint16(0)},
		"split.count":         {Kind: "u16", Value: uint16(1)},
		"split.tensors.count": {Kind: "i32", Value: int32(2)},
		"custom.note":         {Kind: "str", Value: "keep"},
	}
	out := compressionMetadata(source, CompressOptions{TargetFormat: GGMLTypeQ4_K, Preset: "Q4_K_M"})
	if len(source) != 5 || source["general.file_type"].Value != uint32(1) {
		t.Fatal("source metadata mutated")
	}
	for _, key := range splitKeys {
		if _, ok := out[key]; ok {
			t.Errorf("stale %s", key)
		}
	}
	if out["custom.note"].Value != "keep" {
		t.Fatal("custom metadata lost")
	}
}

func TestCompressFailurePreservesDestination(t *testing.T) {
	for _, mode := range []string{"truncated", "nonfinite", "overflow", "cancel-during-write"} {
		t.Run(mode, func(t *testing.T) {
			data := buildRequantizableLlamaGGUF()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			opts := CompressOptions{Preset: "Q4_K_M", Context: ctx}
			g, err := ParseGGUF(data)
			if err != nil {
				t.Fatal(err)
			}
			switch mode {
			case "truncated":
				data = data[:len(data)-128]
			case "nonfinite":
				binary.LittleEndian.PutUint32(data[g.DataOffset:], math.Float32bits(float32(math.NaN())))
			case "overflow":
				binary.LittleEndian.PutUint32(data[g.DataOffset:], math.Float32bits(math.MaxFloat32))
			case "cancel-during-write":
				opts.LogWriter = cancelCompressionLog{cancel: cancel}
			}
			input := writeTempGGUF(t, data)
			dir := t.TempDir()
			output := filepath.Join(dir, "existing.gguf")
			original := []byte("existing output")
			if err := os.WriteFile(output, original, 0o600); err != nil {
				t.Fatal(err)
			}
			err = CompressModel(input, output, opts)
			if err == nil {
				t.Fatal("expected conversion error")
			}
			if mode == "cancel-during-write" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation lost: %v", err)
			}
			got, err := os.ReadFile(output)
			if err != nil || !bytes.Equal(got, original) {
				t.Fatal("existing destination changed")
			}
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 1 {
				t.Fatalf("temporary output leaked: %v, %v", entries, err)
			}
		})
	}
}

type cancelCompressionLog struct{ cancel context.CancelFunc }

func (l cancelCompressionLog) Write(p []byte) (int, error) {
	// Planning completes, then cancellation is observed before tensor output.
	if bytes.HasPrefix(p, []byte("total:")) {
		l.cancel()
	}
	return len(p), nil
}

func TestVerifyCompressedOutputChecksExactShape(t *testing.T) {
	input := writeTempGGUF(t, buildRequantizableLlamaGGUF())
	g := parseGGUFFile(t, input)
	planned := make([]PlannedTensor, len(g.Tensors))
	for i, tensor := range g.Tensors {
		planned[i] = PlannedTensor{Name: tensor.Name, Dims: append([]uint64(nil), tensor.Dims...), DType: tensor.DType}
	}
	planned[0].Dims = []uint64{128, 64} // same element count as 256x32
	if err := verifyCompressedOutput(input, planned); err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("shape mismatch accepted: %v", err)
	}
}

func TestCompressionTensorSizeRejectsInvalidGeometry(t *testing.T) {
	for _, dims := range [][]uint64{nil, {0, 256}, {^uint64(0), 2}, {256, ^uint64(0)}, {33, 32}} {
		if _, err := compressionTensorSize(TensorInfo{Name: "bad", Dims: dims, DType: GGMLTypeQ4_K}); err == nil {
			t.Fatalf("accepted invalid dimensions %v", dims)
		}
	}
}

func TestCompressF16ExpertBankAndFallbackRows(t *testing.T) {
	// 3-D expert storage plus narrow rows exercise both K-block fallbacks
	// through actual GGUF writing/reading, rather than the planner alone.
	for _, preset := range []string{"Q4_K_M", "Q5_K_M"} {
		t.Run(preset, func(t *testing.T) {
			plans := []PlannedTensor{
				{Name: "blk.0.ffn_up_exps.weight", Dims: []uint64{256, 2, 3}, DType: GGMLTypeF16},
				{Name: "blk.0.attn_q.weight", Dims: []uint64{96, 2}, DType: GGMLTypeF16},
				{Name: "blk.0.attn_k.weight", Dims: []uint64{65, 2}, DType: GGMLTypeF16},
				{Name: "blk.0.ffn_gate_inp.weight", Dims: []uint64{256, 3}, DType: GGMLTypeF16},
			}
			var source bytes.Buffer
			writer, err := NewGGUFWriter(&source, map[string]MetaValue{
				"general.architecture": {Kind: "str", Value: "llama"},
				"llama.block_count":    {Kind: "u32", Value: uint32(1)},
			}, plans, 32)
			if err != nil {
				t.Fatal(err)
			}
			var values [][]float32
			for _, plan := range plans {
				row := smallWeights((TensorInfo{Dims: plan.Dims}).Numel(), 28)
				raw := quantizeRowF16(row, len(row))
				decoded, _ := tensorToF32(GGMLTypeF16, raw, len(row))
				values = append(values, decoded)
				if err := writer.WriteTensor(raw); err != nil {
					t.Fatal(err)
				}
			}
			if err := writer.Close(); err != nil {
				t.Fatal(err)
			}
			input := writeTempGGUF(t, source.Bytes())
			output := filepath.Join(t.TempDir(), "mixed.gguf")
			// Exercise successful replacement of an existing destination too.
			if err := os.WriteFile(output, []byte("old destination"), 0o600); err != nil {
				t.Fatal(err)
			}
			opts, _ := ParseCompressOptions(preset)
			if err := CompressModel(input, output, opts); err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(output)
			if err != nil {
				t.Fatal(err)
			}
			model, err := ParseGGUF(raw)
			if err != nil {
				t.Fatal(err)
			}
			fallback := GGMLTypeQ5_0
			if preset == "Q5_K_M" {
				fallback = GGMLTypeQ5_1
			}
			wantTypes := []GGMLType{opts.TargetFormat, fallback, GGMLTypeF16, GGMLTypeF16}
			for i, tensor := range model.Tensors {
				if tensor.DType != wantTypes[i] {
					t.Fatalf("%s: type %s, want %s", tensor.Name, tensor.DType, wantTypes[i])
				}
				start := model.DataOffset + int(tensor.Offset)
				size, _ := tensor.DType.DataSize(tensor.Numel())
				decoded, ok := tensorToF32(tensor.DType, raw[start:start+size], tensor.Numel())
				if !ok {
					t.Fatal("could not decode result")
				}
				for j, value := range decoded {
					bound := float32(0.09)
					if tensor.DType == GGMLTypeF16 {
						bound = 0
					}
					if math.IsNaN(float64(value)) || abs32(value-values[i][j]) > bound {
						t.Fatalf("%s[%d]: error %g", tensor.Name, j, value-values[i][j])
					}
				}
			}
		})
	}
}
