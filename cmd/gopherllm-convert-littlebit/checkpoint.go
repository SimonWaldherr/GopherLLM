package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
	"github.com/SimonWaldherr/GopherLLM/internal/formats/safetensors"
	"github.com/SimonWaldherr/GopherLLM/internal/mmapfile"
	"github.com/SimonWaldherr/GopherLLM/internal/outputfile"
)

type hfTensor struct {
	dtype safetensors.DType
	shape []int
	raw   []byte
}

func readJSON(dir, name string, dest any) error {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, dest); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

type tensorIndex map[string]hfTensor

func openCheckpoint(dir string) (tensorIndex, func(), error) {
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	shards := map[string]bool{}
	switch err := readJSON(dir, "model.safetensors.index.json", &idx); {
	case errors.Is(err, os.ErrNotExist):
		shards["model.safetensors"] = true
	case err != nil:
		return nil, nil, err
	default:
		if len(idx.WeightMap) == 0 {
			return nil, nil, fmt.Errorf("empty safetensors weight_map")
		}
		for _, shard := range idx.WeightMap {
			if !filepath.IsLocal(shard) || filepath.Base(shard) != shard || !strings.HasSuffix(shard, ".safetensors") {
				return nil, nil, fmt.Errorf("invalid safetensors shard name %q", shard)
			}
			shards[shard] = true
		}
	}
	var files []*mmapfile.File
	closeAll := func() {
		for _, f := range files {
			_ = f.Close()
		}
	}
	tensors := make(tensorIndex)
	for shard := range shards {
		f, err := mmapfile.Open(filepath.Join(dir, shard))
		if err != nil {
			closeAll()
			return nil, nil, err
		}
		files = append(files, f)
		st, err := safetensors.Parse(f.Bytes())
		if err == nil {
			err = indexShard(tensors, st, shard, idx.WeightMap)
		}
		if err != nil {
			closeAll()
			return nil, nil, fmt.Errorf("%s: %w", shard, err)
		}
	}
	if idx.WeightMap != nil && len(tensors) != len(idx.WeightMap) {
		closeAll()
		return nil, nil, fmt.Errorf("safetensors weight_map lists missing tensors")
	}
	return tensors, closeAll, nil
}

func indexShard(dst tensorIndex, st *safetensors.SafetensorsFile, shard string, weightMap map[string]string) error {
	data := st.Data[st.DataOffset:]
	type span struct{ begin, end int }
	spans := make([]span, 0, len(st.Tensors))
	for name, info := range st.Tensors {
		if weightMap != nil && weightMap[name] != shard {
			return fmt.Errorf("tensor %s disagrees with safetensors weight_map", name)
		}
		if _, ok := dst[name]; ok {
			return fmt.Errorf("duplicate tensor %s", name)
		}
		size := 0
		switch info.DType {
		case safetensors.F32, "I32":
			size = 4
		case safetensors.F16, safetensors.BF16:
			size = 2
		case "I64":
			size = 8
		case "I8":
			size = 1
		default:
			return fmt.Errorf("tensor %s: unsupported dtype %s", name, info.DType)
		}
		for _, dim := range info.Shape {
			if dim <= 0 || size > math.MaxInt/dim {
				return fmt.Errorf("tensor %s: invalid dimensions %v", name, info.Shape)
			}
			size *= dim
		}
		if info.Begin < 0 || info.End < info.Begin || info.End > len(data) || info.End-info.Begin != size {
			return fmt.Errorf("tensor %s: invalid byte range", name)
		}
		dst[name] = hfTensor{info.DType, info.Shape, data[info.Begin:info.End]}
		spans = append(spans, span{info.Begin, info.End})
	}
	slices.SortFunc(spans, func(a, b span) int { return a.begin - b.begin })
	end := 0
	for _, region := range spans {
		if region.begin != end {
			return fmt.Errorf("safetensors data ranges overlap or contain gaps")
		}
		end = region.end
	}
	if end != len(data) {
		return fmt.Errorf("unexpected data after last safetensors tensor")
	}
	return nil
}

type exportTensor struct {
	gguf.PlannedTensor
	raw []byte
	// Nonzero for Llama Q/K output axes that need a RoPE permutation.
	headDim int
	pack    int // unpacked factor columns; 0 means bytes are ready to stream
	source  safetensors.DType
}

func convert(dir, out string, variant uint32) error {
	if err := outputfile.CheckNew(out); err != nil {
		return err
	}
	cfg, variant, err := loadConfig(dir, variant)
	if err != nil {
		return err
	}
	index, closeAll, err := openCheckpoint(dir)
	if err != nil {
		return err
	}
	defer closeAll()
	meta, err := cfg.metadata(dir)
	if err != nil {
		return err
	}
	tensors, params, factorBytes, err := planTensors(index, cfg)
	if err != nil {
		return err
	}
	const prefix = "gopherllm.littlebit."
	meta[prefix+"version"] = gguf.MetaValue{Kind: "u32", Value: uint32(1)}
	meta[prefix+"variant"] = gguf.MetaValue{Kind: "u32", Value: variant}
	meta[prefix+"original_parameters"] = gguf.MetaValue{Kind: "u64", Value: params}
	meta[prefix+"effective_bpw"] = gguf.MetaValue{Kind: "f64", Value: 8 * float64(factorBytes) / float64(params)}
	plan := make([]gguf.PlannedTensor, len(tensors))
	for i, tensor := range tensors {
		plan[i] = tensor.PlannedTensor
	}
	return outputfile.Write(out, func(w io.Writer) error {
		writer, err := gguf.NewGGUFWriter(w, meta, plan, 32)
		if err != nil {
			return err
		}
		for _, tensor := range tensors {
			if err := writer.WriteTensorStream(tensor.write); err != nil {
				return fmt.Errorf("%s: %w", tensor.Name, err)
			}
		}
		return writer.Close()
	})
}

func (t exportTensor) write(w io.Writer) error {
	if t.headDim == 0 && t.pack == 0 {
		_, err := w.Write(t.raw)
		return err
	}
	rows := int(t.Dims[len(t.Dims)-1])
	stride := len(t.raw) / rows
	var packed []byte
	if t.pack > 0 {
		packed = make([]byte, wordCount(t.pack)*4)
	}
	for i := range rows {
		rowIndex := i
		if t.headDim > 0 {
			// HF RoPE uses consecutive coordinate halves per head. GGUF
			// Llama expects each corresponding pair to be adjacent.
			pos := i % t.headDim
			rowIndex = i - pos + pos/2 + (pos%2)*(t.headDim/2)
		}
		row := t.raw[rowIndex*stride : (rowIndex+1)*stride]
		if t.pack > 0 {
			clear(packed)
			if err := packRow(packed, row, t.source); err != nil {
				return err
			}
			row = packed
		}
		if _, err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}

func wordCount(cols int) int { return (cols-1)/32 + 1 }

func packRow(dst, src []byte, dtype safetensors.DType) error {
	width, sign, magnitude, inf := 0, uint32(0), uint32(0), uint32(0)
	switch dtype {
	case "I8":
		for i, b := range src {
			switch int8(b) {
			case -1:
				dst[i/8] |= 1 << (i % 8)
			case 1:
			default:
				return fmt.Errorf("unpacked I8 factor must contain only -1 and +1")
			}
		}
		return nil
	case safetensors.F32:
		width, sign, magnitude, inf = 4, 0x80000000, 0x7fffffff, 0x7f800000
	case safetensors.F16:
		width, sign, magnitude, inf = 2, 0x8000, 0x7fff, 0x7c00
	case safetensors.BF16:
		width, sign, magnitude, inf = 2, 0x8000, 0x7fff, 0x7f80
	default:
		return fmt.Errorf("unsupported unpacked factor dtype %s", dtype)
	}
	for i := range len(src) / width {
		b := src[i*width:]
		value := uint32(binary.LittleEndian.Uint16(b))
		if width == 4 {
			value = binary.LittleEndian.Uint32(b)
		}
		if value&magnitude >= inf {
			return fmt.Errorf("non-finite value in unpacked factor")
		}
		if value&sign != 0 && value&magnitude != 0 {
			dst[i/8] |= 1 << (i % 8)
		}
	}
	return nil
}
