package engine

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func qi21TestSafetensors(header string, data []byte) []byte {
	out := make([]byte, 8, 8+len(header)+len(data))
	binary.LittleEndian.PutUint64(out, uint64(len(header)))
	out = append(out, header...)
	return append(out, data...)
}

func qi21TestWriteWeights(t *testing.T, files map[string][]byte) string {
	t.Helper()
	dir := t.TempDir()
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestQI21WeightsMappedMatricesAndOwnedFloats(t *testing.T) {
	for _, dtype := range []string{"F32", "F16", "BF16", "Q8_0"} {
		t.Run(dtype, func(t *testing.T) {
			rows, cols := 2, 2
			want := []float32{1, 2, 3, 4}
			raw := f32Bytes(want)
			if dtype == "F16" || dtype == "BF16" {
				raw = make([]byte, 2*len(want))
				for i, v := range want {
					bits := uint16(math.Float32bits(v) >> 16)
					if dtype == "F16" {
						bits = F32ToF16(v)
					}
					binary.LittleEndian.PutUint16(raw[i*2:], bits)
				}
			}
			file := "model.safetensors"
			header := fmt.Sprintf(`{"w":{"dtype":%q,"shape":[2,2],"data_offsets":[0,%d]}}`, dtype, len(raw))
			data := qi21TestSafetensors(header, raw)
			offset := 8 + len(header)
			if dtype == "Q8_0" {
				cols = 32
				want = make([]float32, rows*cols)
				raw = make([]byte, rows*34)
				for row := range rows {
					binary.LittleEndian.PutUint16(raw[row*34:], F32ToF16(1))
					for col := range cols {
						v := row*cols + col - 32
						raw[row*34+2+col] = byte(int8(v))
						want[row*cols+col] = float32(v)
					}
				}
				file = "model.gguf"
				data = buildGGUF(3, nil, []ggufTensor{{"w", []uint64{32, 2}, GGMLTypeQ8_0, raw}})
				g, err := ParseGGUFQuiet(data)
				if err != nil {
					t.Fatal(err)
				}
				offset = g.DataOffset
			}
			source, err := qi21OpenWeights(context.Background(), qi21TestWriteWeights(t, map[string][]byte{file: data}))
			if err != nil {
				t.Fatal(err)
			}
			defer source.close()
			matrix, err := source.matrix("w", rows, cols)
			if err != nil {
				t.Fatal(err)
			}
			if &matrix.Raw[0] != &source.mappings[0].Bytes()[offset] {
				t.Fatal("matrix was copied instead of borrowing its mapped bytes")
			}
			row := make([]float32, cols)
			matrix.RowInto(1, cols, &row)
			if !slices.Equal(row, want[cols:]) {
				t.Fatalf("decoded row = %v, want %v", row, want[cols:])
			}
			values, err := source.floats("w", rows, cols)
			if err != nil {
				t.Fatal(err)
			}
			for _, call := range []func() error{
				func() error { _, e := source.matrix("missing", rows, cols); return e },
				func() error { _, e := source.matrix("w", rows, cols+1); return e },
				func() error { _, e := source.floats("missing", rows, cols); return e },
				func() error { _, e := source.floats("w", rows*cols); return e },
			} {
				if err := call(); err == nil {
					t.Fatal("accepted missing tensor or wrong shape")
				}
			}
			source.close()
			if !slices.Equal(values, want) {
				t.Fatalf("owned values after unmap = %v, want %v", values, want)
			}
		})
	}
}

func TestQI21WeightsRejectMalformedStorage(t *testing.T) {
	f32 := f32Bytes([]float32{1, 2})
	safe := func(dtype, shape, offsets string) []byte {
		return qi21TestSafetensors(`{"w":{"dtype":"`+dtype+`","shape":`+shape+`,"data_offsets":`+offsets+`}}`, f32)
	}
	gguf := func(dims []uint64, dtype GGMLType, raw []byte) []byte {
		return buildGGUF(3, nil, []ggufTensor{{"w", dims, dtype, raw}})
	}
	oversizedHeader := make([]byte, 8)
	binary.LittleEndian.PutUint64(oversizedHeader, 16<<20+1)
	badOffset := gguf([]uint64{2}, GGMLTypeF32, f32)
	// One tensor: header (24), name (8+1), rank (4), dim (8), type (4).
	binary.LittleEndian.PutUint64(badOffset[49:57], math.MaxUint64)
	for _, tc := range []struct {
		name, file string
		data       []byte
	}{
		{"header", "model.safetensors", oversizedHeader},
		{"rank", "model.safetensors", safe("F32", "[]", "[0,8]")},
		{"zero_dimension", "model.safetensors", safe("F32", "[0,2]", "[0,8]")},
		{"negative_dimension", "model.safetensors", safe("F32", "[-1,2]", "[0,8]")},
		{"shape_overflow", "model.safetensors", safe("F32", "[16777216,16777216,16777216]", "[0,8]")},
		{"short_storage", "model.safetensors", safe("F32", "[3]", "[0,8]")},
		{"dtype", "model.safetensors", safe("I32", "[2]", "[0,8]")},
		{"negative_range", "model.safetensors", safe("F32", "[2]", "[-1,7]")},
		{"reversed_range", "model.safetensors", safe("F32", "[2]", "[8,0]")},
		{"range_past_end", "model.safetensors", safe("F32", "[2]", "[4,12]")},
		{"empty_safetensors", "model.safetensors", qi21TestSafetensors("{}", nil)},
		{"gguf_shape_overflow", "model.gguf", gguf([]uint64{1 << 24, 1 << 24, 1 << 24}, GGMLTypeF32, f32)},
		{"gguf_byte_overflow", "model.gguf", gguf([]uint64{1 << 24, 1 << 24, 1 << 13}, GGMLTypeF64, f32)},
		{"gguf_oversized_dimension", "model.gguf", gguf([]uint64{math.MaxUint64}, GGMLTypeF32, f32)},
		{"gguf_partial_quant_row", "model.gguf", gguf([]uint64{31, 2}, GGMLTypeQ8_0, make([]byte, 68))},
		{"gguf_short_storage", "model.gguf", gguf([]uint64{3}, GGMLTypeF32, f32)},
		{"gguf_range", "model.gguf", badOffset},
		{"empty_gguf", "model.gguf", buildGGUF(3, nil, nil)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source, err := qi21OpenWeights(context.Background(), qi21TestWriteWeights(t, map[string][]byte{tc.file: tc.data}))
			if err == nil {
				source.close()
				t.Fatal("accepted malformed storage")
			}
		})
	}
}

func TestQI21WeightsShardIndex(t *testing.T) {
	shard := qi21TestSafetensors(`{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`, f32Bytes([]float32{1, 2}))
	for _, tc := range []struct {
		name, index string
		shards      map[string][]byte
		valid       bool
	}{
		{"valid", `{"weight_map":{"w":"a.safetensors"}}`, map[string][]byte{"a.safetensors": shard}, true},
		{"missing_tensor", `{"weight_map":{"w":"a.safetensors","missing":"a.safetensors"}}`, map[string][]byte{"a.safetensors": shard}, false},
		{"wrong_shard", `{"weight_map":{"w":"b.safetensors","other":"a.safetensors"}}`, map[string][]byte{"a.safetensors": shard, "b.safetensors": shard}, false},
		{"missing_file", `{"weight_map":{"w":"a.safetensors"}}`, nil, false},
		{"empty", `{"weight_map":{}}`, nil, false},
		{"outside_directory", `{"weight_map":{"w":"../a.safetensors"}}`, nil, false},
		{"duplicate_tensor", "", map[string][]byte{"a.safetensors": shard, "b.safetensors": shard}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			files := tc.shards
			if files == nil {
				files = make(map[string][]byte)
			}
			if tc.index != "" {
				files["weights.index.json"] = []byte(tc.index)
			}
			source, err := qi21OpenWeights(context.Background(), qi21TestWriteWeights(t, files))
			if (err == nil) != tc.valid {
				t.Fatalf("open error = %v, valid = %v", err, tc.valid)
			}
			if source != nil {
				source.close()
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := qi21OpenWeights(ctx, qi21TestWriteWeights(t, map[string][]byte{"model.safetensors": shard})); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled open = %v", err)
	}
}

func TestQI21WeightsScalarExpansionBudget(t *testing.T) {
	source := &qi21Weights{tensors: map[string]qi21StoredTensor{
		"oversized": {shape: []int{2, qi21MaxFloatTensorBytes/8 + 1}, w: Weight{Type: GGMLTypeF16}},
		"small":     {shape: []int{1}, w: Weight{Type: GGMLTypeF32, Raw: f32Bytes([]float32{1}), Rows: 1, Cols: 1}},
		"nonfinite": {shape: []int{1}, w: Weight{Type: GGMLTypeF32, Raw: f32Bytes([]float32{float32(math.Inf(1))}), Rows: 1, Cols: 1}},
	}}
	if _, err := source.floats("oversized", 2, qi21MaxFloatTensorBytes/8+1); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("oversized allocation = %v", err)
	}
	if _, err := source.floats("nonfinite", 1); err == nil {
		t.Fatal("accepted non-finite scalar data")
	}
	source.expandedBytes = qi21MaxFloatTotalBytes - 4
	if _, err := source.floats("small", 1); err != nil {
		t.Fatal(err)
	}
	if _, err := source.floats("small", 1); err == nil || !strings.Contains(err.Error(), "budget") {
		t.Fatalf("total expansion limit = %v", err)
	}
}
