package engine

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strings"
)

const littleBitNamespace = "gopherllm.littlebit."

const littleBitTensorMarker = ".weight.lb."

func hasLittleBitGGUF(gguf *GGUFFile) bool {
	for key := range gguf.Metadata {
		if strings.HasPrefix(key, littleBitNamespace) {
			return true
		}
	}
	for _, tensor := range gguf.Tensors {
		if strings.Contains(tensor.Name, littleBitTensorMarker) {
			return true
		}
	}
	return false
}

func littleBitArchitectureSupported(arch string) bool {
	return arch == "llama" || arch == "llama2" || arch == "llama3" || arch == "qwen3"
}

// loadLittleBitProjections validates the composite layout before any weight
// uploads. The ordinary model loader then handles the rest of the decoder.
func loadLittleBitProjections(data []byte, gguf *GGUFFile, config Config, tensors map[string]TensorInfo, inferred map[string]int, borrow bool) (map[string]Weight, error) {
	version, _ := gguf.Metadata[littleBitNamespace+"version"].Value.(uint32)
	if version != 1 {
		return nil, fmt.Errorf("LittleBit GGUF: expected %sversion (u32) = 1", littleBitNamespace)
	}
	variant, _ := gguf.Metadata[littleBitNamespace+"variant"].Value.(uint32)
	if variant != 1 && variant != 2 {
		return nil, fmt.Errorf("LittleBit GGUF: variant must be u32 1 or 2, got %v", gguf.Metadata[littleBitNamespace+"variant"].Value)
	}
	if config.ExpertCount != 0 || config.Dim <= 0 || config.HiddenDim <= 0 || config.VocabSize <= 0 ||
		config.MaxSeqLen <= 0 || config.NHeads <= 0 || config.NKVHeads <= 0 ||
		config.NHeads%config.NKVHeads != 0 || config.HeadDim <= 0 || config.ValueDim <= 0 ||
		config.HeadDim > math.MaxInt/config.NHeads || config.ValueDim > math.MaxInt/config.NHeads ||
		config.RopeDimensionCount <= 0 || config.RopeDimensionCount > config.HeadDim || config.RopeDimensionCount%2 != 0 {
		return nil, fmt.Errorf("LittleBit GGUF: invalid dense %s decoder dimensions", config.Arch)
	}
	if len(tensors) != len(gguf.Tensors) {
		return nil, fmt.Errorf("LittleBit GGUF: duplicate tensor name")
	}

	// List the projections in the exact layout consumed by the standard
	// decoder. A v1 file uses separate Q/K/V and separate SwiGLU gate/up.
	type shape struct{ rows, cols int }
	allowed := map[string]shape{"output.weight": {config.VocabSize, config.Dim}}
	checks := map[string][]uint64{
		"token_embd.weight":  {uint64(config.Dim), uint64(config.VocabSize)},
		"output_norm.weight": {uint64(config.Dim)},
	}
	qRows := config.NHeads * config.HeadDim
	kRows := config.NKVHeads * config.HeadDim
	vRows := config.NKVHeads * config.ValueDim
	for layer := range config.NLayers {
		prefix := fmt.Sprintf("blk.%d.", layer)
		for name, dims := range map[string]shape{
			"attn_q": {qRows, config.Dim}, "attn_k": {kRows, config.Dim}, "attn_v": {vRows, config.Dim},
			"attn_output": {config.Dim, config.NHeads * config.ValueDim},
			"ffn_gate":    {config.HiddenDim, config.Dim}, "ffn_up": {config.HiddenDim, config.Dim},
			"ffn_down": {config.Dim, config.HiddenDim},
		} {
			allowed[prefix+name+".weight"] = dims
		}
		checks[prefix+"attn_norm.weight"] = []uint64{uint64(config.Dim)}
		checks[prefix+"ffn_norm.weight"] = []uint64{uint64(config.Dim)}
		if config.Arch == "qwen3" {
			checks[prefix+"attn_q_norm.weight"] = []uint64{uint64(config.HeadDim)}
			checks[prefix+"attn_k_norm.weight"] = []uint64{uint64(config.HeadDim)}
		}
		for _, name := range []string{"attn_qkv.weight", "attn_qkv.bias", "ffn_gate_inp.weight", "ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"} {
			if _, ok := tensors[prefix+name]; ok {
				return nil, fmt.Errorf("LittleBit GGUF: unsupported tensor %s%s", prefix, name)
			}
		}
	}
	components := make(map[string]bool)
	for _, tensor := range gguf.Tensors {
		prefix, suffix, found := strings.Cut(tensor.Name, littleBitTensorMarker)
		if !found {
			continue
		}
		name := prefix + ".weight"
		if _, ok := allowed[name]; !ok {
			return nil, fmt.Errorf("LittleBit GGUF: unsupported factorized projection %s", name)
		}
		switch suffix {
		case "u", "v", "u1", "u2", "v1", "v2", "bias",
			"residual.u", "residual.v", "residual.u1", "residual.u2", "residual.v1", "residual.v2":
			components[name] = components[name] || strings.HasPrefix(suffix, "residual.")
		default:
			return nil, fmt.Errorf("LittleBit GGUF: unknown component %s", tensor.Name)
		}
	}
	if len(components) == 0 {
		return nil, fmt.Errorf("LittleBit GGUF: no factorized projections")
	}
	reader := littleBitTensorReader{data: data, offset: gguf.DataOffset, tensors: tensors, inferred: inferred, borrow: borrow}
	weights := make(map[string]Weight, len(components))
	var parameters uint64
	for name, dims := range allowed {
		hasResidual, factored := components[name]
		if !factored {
			_, explicitOutput := tensors[name]
			if name != "output.weight" || explicitOutput {
				checks[name] = []uint64{uint64(dims.cols), uint64(dims.rows)}
			}
			biasName := strings.TrimSuffix(name, ".weight") + ".bias"
			if _, hasBias := tensors[biasName]; hasBias {
				checks[biasName] = []uint64{uint64(dims.rows)}
			}
			continue
		}
		if _, exists := tensors[name]; exists {
			return nil, fmt.Errorf("LittleBit GGUF: both ordinary and factorized tensors for %s", name)
		}
		prefix := name + ".lb."
		main, err := reader.factors(prefix, dims.rows, dims.cols)
		if err != nil {
			return nil, err
		}
		projection := &LittleBitWeight{Main: main}
		if hasResidual {
			residual, err := reader.factors(prefix+"residual.", dims.rows, dims.cols)
			if err != nil {
				return nil, err
			}
			projection.Residual = &residual
		}
		biasName := strings.TrimSuffix(name, ".weight") + ".bias"
		if _, hasBias := tensors[biasName]; hasBias {
			return nil, fmt.Errorf("LittleBit GGUF: use %sbias for the bias of %s", prefix, name)
		}
		if _, hasBias := tensors[prefix+"bias"]; hasBias {
			projection.Bias, err = reader.scale(prefix+"bias", dims.rows)
			if err != nil {
				return nil, err
			}
		}
		rows, cols := uint64(dims.rows), uint64(dims.cols)
		if rows > (math.MaxUint64-parameters)/cols {
			return nil, fmt.Errorf("LittleBit GGUF: original parameter count overflow")
		}
		parameters += rows * cols
		weights[name] = Weight{LittleBit: projection, Type: GGMLTypeI32, Rows: dims.rows, Cols: dims.cols}
	}
	wantParameters, _ := gguf.Metadata[littleBitNamespace+"original_parameters"].Value.(uint64)
	if wantParameters != parameters {
		return nil, fmt.Errorf("LittleBit GGUF: original_parameters must be u64 %d, got %v", parameters, gguf.Metadata[littleBitNamespace+"original_parameters"].Value)
	}
	bpw, _ := gguf.Metadata[littleBitNamespace+"effective_bpw"].Value.(float64)
	wantBPW := 8 * float64(reader.payloadBytes) / float64(parameters)
	if !(math.Abs(bpw-wantBPW) <= 1e-9*wantBPW) {
		return nil, fmt.Errorf("LittleBit GGUF: effective_bpw must be f64 %.12g, got %v", wantBPW, gguf.Metadata[littleBitNamespace+"effective_bpw"].Value)
	}
	for name, dims := range checks {
		info := tensors[name]
		if _, err := reader.tensor(name, info.DType, dims); err != nil {
			return nil, err
		}
	}
	return weights, nil
}

type littleBitTensorReader struct {
	data         []byte
	offset       int
	tensors      map[string]TensorInfo
	inferred     map[string]int
	borrow       bool
	payloadBytes uint64
}

func (r *littleBitTensorReader) factors(prefix string, rows, cols int) (LittleBitFactors, error) {
	rankDims := r.tensors[prefix+"v1"].Dims
	if len(rankDims) != 1 || rankDims[0] == 0 || rankDims[0] > math.MaxInt {
		return LittleBitFactors{}, fmt.Errorf("LittleBit GGUF: %sv1 must declare a positive rank", prefix)
	}
	f := LittleBitFactors{Rank: int(rankDims[0])}
	for _, factor := range []struct {
		name       string
		rows, cols int
		dst        *[]byte
	}{{"u", rows, f.Rank, &f.U}, {"v", f.Rank, cols, &f.V}} {
		words := (factor.cols-1)/32 + 1
		name := prefix + factor.name
		raw, err := r.tensor(name, GGMLTypeI32, []uint64{uint64(words), uint64(factor.rows)})
		if err != nil {
			return LittleBitFactors{}, err
		}
		if used := factor.cols % 32; used != 0 {
			stride := words * 4
			for row := range factor.rows {
				i := (row+1)*stride - 4
				if binary.LittleEndian.Uint32(raw[i:])>>used != 0 {
					return LittleBitFactors{}, fmt.Errorf("LittleBit GGUF: %s has nonzero padding in row %d", name, row)
				}
			}
		}
		*factor.dst = r.keep(raw)
	}
	for _, scale := range []struct {
		name   string
		length int
		dst    *LittleBitScale
	}{{"u1", rows, &f.U1}, {"u2", f.Rank, &f.U2}, {"v1", f.Rank, &f.V1}, {"v2", cols, &f.V2}} {
		s, err := r.scale(prefix+scale.name, scale.length)
		if err != nil {
			return LittleBitFactors{}, err
		}
		*scale.dst = s
	}
	return f, nil
}

func (r *littleBitTensorReader) keep(raw []byte) []byte {
	r.payloadBytes += uint64(len(raw))
	if !r.borrow {
		return bytes.Clone(raw)
	}
	return raw
}

func (r *littleBitTensorReader) scale(name string, n int) (LittleBitScale, error) {
	info, ok := r.tensors[name]
	if !ok {
		return LittleBitScale{}, fmt.Errorf("LittleBit GGUF: missing tensor %s", name)
	}
	if info.DType != GGMLTypeF16 && info.DType != GGMLTypeBF16 && info.DType != GGMLTypeF32 {
		return LittleBitScale{}, fmt.Errorf("LittleBit GGUF: %s must be F16, BF16 or F32, got %s", name, info.DType)
	}
	raw, err := r.tensor(name, info.DType, []uint64{uint64(n)})
	if err != nil {
		return LittleBitScale{}, err
	}
	scale := LittleBitScale{Raw: raw, Type: info.DType}
	for i := range n {
		if !finite32(scale.at(i)) {
			return LittleBitScale{}, fmt.Errorf("LittleBit GGUF: %s contains a nonfinite scale at %d", name, i)
		}
	}
	scale.Raw = r.keep(raw)
	return scale, nil
}

func (r *littleBitTensorReader) tensor(name string, dtype GGMLType, dims []uint64) ([]byte, error) {
	info, ok := r.tensors[name]
	if !ok {
		return nil, fmt.Errorf("LittleBit GGUF: missing tensor %s", name)
	}
	if info.DType != dtype || !slices.Equal(info.Dims, dims) {
		return nil, fmt.Errorf("LittleBit GGUF: %s is %s %v, want %s %v", name, info.DType, info.Dims, dtype, dims)
	}
	n := 1
	for _, d := range dims {
		if d == 0 || d > uint64(math.MaxInt/n) {
			return nil, fmt.Errorf("LittleBit GGUF: %s dimension overflow", name)
		}
		n *= int(d)
	}
	blockSize := dtype.BlockSize()
	blockBytes, ok := dtype.DataSize(blockSize)
	if !ok || blockBytes <= 0 || dims[0]%uint64(blockSize) != 0 || n/blockSize > math.MaxInt/blockBytes {
		return nil, fmt.Errorf("LittleBit GGUF: invalid byte size for %s", name)
	}
	size := n / blockSize * blockBytes
	region := r.data[r.offset:]
	if info.Shard != nil {
		region = info.Shard
	}
	if size > r.inferred[name] || info.Offset > uint64(len(region)) || uint64(size) > uint64(len(region))-info.Offset {
		return nil, fmt.Errorf("LittleBit GGUF: %s exceeds its tensor byte range", name)
	}
	start := int(info.Offset)
	return region[start : start+size : start+size], nil
}
