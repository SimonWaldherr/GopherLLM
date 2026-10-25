package gopherllm

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

// Model components share this interface so projections can retain packed
// weights while small normalization and convolution tensors are decoded once.
type qi21WeightSource interface {
	matrix(name string, rows, cols int) (Weight, error)
	floats(name string, shape ...int) ([]float32, error)
}

type qi21StoredTensor struct {
	shape []int
	w     Weight
}

type qi21Weights struct {
	mappings      []*MmapFile
	tensors       map[string]qi21StoredTensor
	expandedBytes uint64
}

const (
	qi21MaxFloatTensorBytes = 64 << 20
	qi21MaxFloatTotalBytes  = uint64(2 << 30)
	qi21MaxStoredTensors    = 65536
)

// Check both products before calling code that assumes valid tensor shapes.
// DataSize on one block cannot overflow; multiplying that result also avoids
// overflowing DataSize's internal arithmetic on corrupt metadata.
func qi21TensorStorage(shape []int, typ GGMLType) (numel, size int, err error) {
	if len(shape) == 0 || len(shape) > 5 {
		return 0, 0, errors.New("invalid tensor rank")
	}
	maxInt := int(^uint(0) >> 1)
	numel = 1
	for _, dim := range shape {
		if dim <= 0 || dim > 1<<24 || numel > maxInt/dim {
			return 0, 0, errors.New("oversized tensor dimensions")
		}
		numel *= dim
	}
	block := typ.BlockSize()
	blockBytes, ok := typ.DataSize(block)
	if !ok || blockBytes <= 0 || shape[len(shape)-1]%block != 0 || numel/block > maxInt/blockBytes {
		return 0, 0, errors.New("invalid tensor storage")
	}
	return numel, numel / block * blockBytes, nil
}

func (s *qi21Weights) close() {
	for _, mapping := range s.mappings {
		_ = mapping.Close()
	}
	s.mappings, s.tensors = nil, nil
}

func (s *qi21Weights) matrix(name string, rows, cols int) (Weight, error) {
	t, ok := s.tensors[name]
	if !ok || !slices.Equal(t.shape, []int{rows, cols}) {
		return Weight{}, fmt.Errorf("qwen-image-2.1: tensor %s shape %v, want [%d %d]", name, t.shape, rows, cols)
	}
	return t.w, nil
}

func (s *qi21Weights) floats(name string, shape ...int) ([]float32, error) {
	t, ok := s.tensors[name]
	if !ok || !slices.Equal(t.shape, shape) {
		return nil, fmt.Errorf("qwen-image-2.1: tensor %s shape %v, want %v", name, t.shape, shape)
	}
	n, size, err := qi21TensorStorage(t.shape, t.w.Type)
	if err != nil {
		return nil, fmt.Errorf("qwen-image-2.1: invalid scalar tensor %s", name)
	}
	// Only normalization and VAE tensors are expanded. The largest official
	// VAE convolution is below 64 MiB; transformer matrices stay mapped.
	if n > qi21MaxFloatTensorBytes/4 {
		return nil, fmt.Errorf("qwen-image-2.1: scalar tensor %s exceeds the expansion budget", name)
	}
	bytes := uint64(n) * 4
	if s.expandedBytes > qi21MaxFloatTotalBytes-bytes {
		return nil, fmt.Errorf("qwen-image-2.1: scalar tensor %s exceeds the expansion budget", name)
	}
	if size != len(t.w.Raw) {
		return nil, fmt.Errorf("qwen-image-2.1: invalid scalar tensor %s", name)
	}
	values, ok := tensorToF32(t.w.Type, t.w.Raw, n)
	if !ok {
		return nil, fmt.Errorf("qwen-image-2.1: unsupported scalar tensor %s", name)
	}
	for _, v := range values {
		if !finite32(v) {
			return nil, fmt.Errorf("qwen-image-2.1: non-finite tensor %s", name)
		}
	}
	s.expandedBytes += bytes
	return values, nil
}

func qi21OpenWeights(ctx context.Context, dir string) (_ *qi21Weights, err error) {
	s := &qi21Weights{tensors: make(map[string]qi21StoredTensor)}
	defer func() {
		if err != nil {
			s.close()
		}
	}()
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	for _, name := range []string{"weights.index.json", "model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"} {
		path := filepath.Join(dir, name)
		if _, e := os.Stat(path); errors.Is(e, os.ErrNotExist) {
			continue
		}
		if err = readModelJSON(path, 8<<20, &index); err != nil {
			return nil, err
		}
		if len(index.WeightMap) == 0 {
			return nil, errors.New("qwen-image-2.1: empty weight index")
		}
		break
	}
	files := make(map[string]bool)
	if len(index.WeightMap) != 0 {
		for _, name := range index.WeightMap {
			if !filepath.IsLocal(name) || filepath.Base(name) != name {
				return nil, errors.New("qwen-image-2.1: invalid shard path")
			}
			files[name] = true
		}
	} else {
		entries, e := os.ReadDir(dir)
		if e != nil {
			return nil, e
		}
		for _, entry := range entries {
			if !entry.IsDir() && (filepath.Ext(entry.Name()) == ".safetensors" || filepath.Ext(entry.Name()) == ".gguf") {
				files[entry.Name()] = true
			}
		}
	}
	if len(files) == 0 || len(files) > 128 {
		return nil, errors.New("qwen-image-2.1: expected 1..128 weight shards")
	}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		path := filepath.Join(dir, name)
		info, e := os.Stat(path)
		if e != nil {
			return nil, e
		}
		if !info.Mode().IsRegular() || info.Size() < 8 || info.Size() > 32<<30 || uint64(info.Size()) > uint64(^uint(0)>>1) {
			return nil, fmt.Errorf("qwen-image-2.1: invalid shard %s", name)
		}
		mapping, e := OpenMmap(path)
		if e != nil {
			return nil, e
		}
		s.mappings = append(s.mappings, mapping)
		data := mapping.Bytes()
		add := func(tensor string, shape []int, typ GGMLType, raw []byte) error {
			if len(s.tensors) >= qi21MaxStoredTensors {
				return errors.New("qwen-image-2.1: too many tensors")
			}
			if _, exists := s.tensors[tensor]; exists {
				return fmt.Errorf("qwen-image-2.1: duplicate tensor %s", tensor)
			}
			if len(index.WeightMap) != 0 && index.WeightMap[tensor] != name {
				return fmt.Errorf("qwen-image-2.1: index disagrees with shard for %s", tensor)
			}
			n, size, e := qi21TensorStorage(shape, typ)
			if e != nil {
				return fmt.Errorf("qwen-image-2.1: tensor %s: %w", tensor, e)
			}
			cols := shape[len(shape)-1]
			if size != len(raw) {
				return fmt.Errorf("qwen-image-2.1: invalid tensor storage %s", tensor)
			}
			s.tensors[tensor] = qi21StoredTensor{shape: slices.Clone(shape), w: Weight{Raw: raw, Type: typ, Rows: n / cols, Cols: cols}}
			return nil
		}
		if filepath.Ext(name) == ".gguf" {
			// Component containers have no tokenizer metadata. Limiting the
			// parse window bounds allocations independently of shard size.
			g, e := ParseGGUFQuiet(data[:min(len(data), 16<<20)])
			if e != nil {
				return nil, e
			}
			for _, t := range g.Tensors {
				shape := make([]int, len(t.Dims))
				for i, d := range t.Dims {
					if d > 1<<24 {
						return nil, errors.New("qwen-image-2.1: oversized GGUF dimension")
					}
					shape[len(shape)-1-i] = int(d)
				}
				_, size, e := qi21TensorStorage(shape, t.DType)
				if e != nil {
					return nil, fmt.Errorf("qwen-image-2.1: tensor %s: %w", t.Name, e)
				}
				if g.DataOffset < 0 || g.DataOffset > len(data) || t.Offset > uint64(len(data)-g.DataOffset) || size > len(data)-g.DataOffset-int(t.Offset) {
					return nil, errors.New("qwen-image-2.1: invalid GGUF tensor range")
				}
				begin := g.DataOffset + int(t.Offset)
				if err = add(t.Name, shape, t.DType, data[begin:begin+size]); err != nil {
					return nil, err
				}
			}
		} else {
			if len(data) < 8 || binary.LittleEndian.Uint64(data[:8]) > 16<<20 {
				return nil, errors.New("qwen-image-2.1: oversized safetensors header")
			}
			f, e := ParseSafetensors(data)
			if e != nil {
				return nil, e
			}
			for tensor, t := range f.Tensors {
				typ := GGMLTypeUnknown
				switch t.DType {
				case SafetensorsF32:
					typ = GGMLTypeF32
				case SafetensorsF16:
					typ = GGMLTypeF16
				case SafetensorsBF16:
					typ = GGMLTypeBF16
				}
				if t.Begin < 0 || t.End < t.Begin || t.End > len(data)-f.DataOffset {
					return nil, fmt.Errorf("qwen-image-2.1: invalid range for %s", tensor)
				}
				if err = add(tensor, t.Shape, typ, data[f.DataOffset+t.Begin:f.DataOffset+t.End]); err != nil {
					return nil, err
				}
			}
		}
	}
	if len(s.tensors) == 0 {
		return nil, errors.New("qwen-image-2.1: weight shards contain no tensors")
	}
	if len(index.WeightMap) != 0 && len(index.WeightMap) != len(s.tensors) {
		return nil, errors.New("qwen-image-2.1: missing indexed tensors")
	}
	return s, nil
}
