package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/internal/formats/gguf"
	"github.com/SimonWaldherr/GopherLLM/internal/formats/safetensors"
)

type tensorPlan struct {
	index       tensorIndex
	tensors     []exportTensor
	params      uint64
	factorBytes uint64
}

func (p *tensorPlan) take(name string) (hfTensor, bool) {
	tensor, ok := p.index[name]
	delete(p.index, name)
	return tensor, ok
}

func (p *tensorPlan) hasModule(name string) bool {
	prefix := name + "."
	for key := range p.index {
		if strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}

func planTensors(index tensorIndex, cfg hfConfig) ([]exportTensor, uint64, uint64, error) {
	p := tensorPlan{index: index}
	if err := p.model(cfg); err != nil {
		return nil, 0, 0, err
	}
	if len(p.index) > 0 {
		var names []string
		for name := range p.index {
			names = append(names, name)
		}
		slices.Sort(names)
		return nil, 0, 0, fmt.Errorf("unrecognized or unused checkpoint tensor %s (%d total)", names[0], len(names))
	}
	if p.params == 0 {
		return nil, 0, 0, fmt.Errorf("checkpoint contains no trained LittleBit projections")
	}
	return p.tensors, p.params, p.factorBytes, nil
}

func (p *tensorPlan) model(cfg hfConfig) error {
	if err := p.floats("model.embed_tokens.weight", "token_embd.weight", false, 0, cfg.Vocab, cfg.Hidden); err != nil {
		return err
	}
	if err := p.floats("model.norm.weight", "output_norm.weight", false, 0, cfg.Hidden); err != nil {
		return err
	}
	qkPermute := cfg.HeadDim
	if cfg.ModelType == "qwen3" {
		qkPermute = 0
	}
	for layer := range cfg.Layers {
		src := fmt.Sprintf("model.layers.%d.", layer)
		dst := fmt.Sprintf("blk.%d.", layer)
		if err := p.floats(src+"input_layernorm.weight", dst+"attn_norm.weight", false, 0, cfg.Hidden); err != nil {
			return err
		}
		if err := p.floats(src+"post_attention_layernorm.weight", dst+"ffn_norm.weight", false, 0, cfg.Hidden); err != nil {
			return err
		}
		if cfg.ModelType == "qwen3" {
			if err := p.floats(src+"self_attn.q_norm.weight", dst+"attn_q_norm.weight", false, 0, cfg.HeadDim); err != nil {
				return err
			}
			if err := p.floats(src+"self_attn.k_norm.weight", dst+"attn_k_norm.weight", false, 0, cfg.HeadDim); err != nil {
				return err
			}
		}
		for _, proj := range []struct {
			src, dst   string
			in, out    int
			headDim    int
			denseBias  bool
			expectBias bool
		}{
			{"self_attn.q_proj", "attn_q", cfg.Hidden, cfg.Heads * cfg.HeadDim, qkPermute, true, cfg.AttentionBias},
			{"self_attn.k_proj", "attn_k", cfg.Hidden, cfg.KVHeads * cfg.HeadDim, qkPermute, true, cfg.AttentionBias},
			{"self_attn.v_proj", "attn_v", cfg.Hidden, cfg.KVHeads * cfg.HeadDim, 0, true, cfg.AttentionBias},
			{"self_attn.o_proj", "attn_output", cfg.Heads * cfg.HeadDim, cfg.Hidden, 0, true, cfg.AttentionBias},
			{"mlp.gate_proj", "ffn_gate", cfg.Hidden, cfg.Intermediate, 0, false, cfg.MLPBias},
			{"mlp.up_proj", "ffn_up", cfg.Hidden, cfg.Intermediate, 0, false, cfg.MLPBias},
			{"mlp.down_proj", "ffn_down", cfg.Intermediate, cfg.Hidden, 0, false, cfg.MLPBias},
		} {
			if err := p.projection(src+proj.src, dst+proj.dst, proj.in, proj.out, proj.headDim, proj.denseBias, proj.expectBias); err != nil {
				return err
			}
		}
	}
	if p.hasModule("lm_head") || !cfg.TieEmbeddings {
		if err := p.projection("lm_head", "output", cfg.Hidden, cfg.Vocab, 0, true, false); err != nil {
			return err
		}
	}
	if raw, err := cfg.ropeFactors(); err != nil {
		return err
	} else if raw != nil {
		p.tensors = append(p.tensors, exportTensor{
			PlannedTensor: gguf.PlannedTensor{Name: "rope_freqs.weight", Dims: []uint64{uint64(len(raw) / 4)}, DType: gguf.GGMLTypeF32},
			raw:           raw,
		})
	}
	return nil
}

func (p *tensorPlan) projection(src, dst string, in, out, headDim int, denseBias, expectBias bool) error {
	_, dense := p.index[src+".weight"]
	main := p.hasFactors(src, "")
	residual := p.hasFactors(src, "_R")
	if dense && (main || residual) {
		return fmt.Errorf("%s: both dense and LittleBit weights are present", src)
	}
	if !main {
		if residual {
			return fmt.Errorf("%s: residual factors require main factors", src)
		}
		if err := p.floats(src+".weight", dst+".weight", false, headDim, out, in); err != nil {
			return err
		}
		if _, present := p.index[src+".bias"]; !denseBias && (present || expectBias) {
			return fmt.Errorf("%s: runtime does not support this dense projection bias", src)
		}
		return p.floats(src+".bias", dst+".bias", !expectBias, headDim, out)
	}
	first := len(p.tensors)
	rank, err := p.factors(src, dst+".weight.lb", "", in, out, headDim)
	if err != nil {
		return err
	}
	if residual {
		if _, err := p.factors(src, dst+".weight.lb.residual", "_R", in, out, headDim); err != nil {
			return err
		}
	}
	if err := p.floats(src+".bias", dst+".weight.lb.bias", !expectBias, headDim, out); err != nil {
		return err
	}
	if err := p.telemetry(src, rank); err != nil {
		return err
	}
	if uint64(out) > (math.MaxUint64-p.params)/uint64(in) {
		return fmt.Errorf("%s: original parameter count overflows", src)
	}
	p.params += uint64(in) * uint64(out)
	for _, part := range p.tensors[first:] {
		size := uint64(len(part.raw))
		if part.pack > 0 {
			size = part.Dims[0] * 4 * part.Dims[1]
		}
		if size == 0 || size > math.MaxUint64-p.factorBytes {
			return fmt.Errorf("%s: factor payload size overflows", part.Name)
		}
		p.factorBytes += size
	}
	return nil
}

var factorNames = []string{"U", "V", "u1", "u2", "v1", "v2"}

func (p *tensorPlan) hasFactors(src, suffix string) bool {
	for _, part := range factorNames {
		name := src + "." + part + suffix
		if _, ok := p.index[name]; ok {
			return true
		}
		for _, extra := range []string{"_packed", "_shape"} {
			if _, ok := p.index[name+extra]; ok {
				return true
			}
		}
	}
	return false
}

func vectorLen(shape []int) int {
	switch {
	case len(shape) == 1:
		return shape[0]
	case len(shape) == 2 && shape[0] == 1:
		return shape[1]
	default:
		return 0
	}
}

func (p *tensorPlan) factors(src, dst, suffix string, in, out, headDim int) (int, error) {
	rank := vectorLen(p.index[src+".v1"+suffix].shape)
	if rank == 0 {
		return 0, fmt.Errorf("%s.v1%s: missing or invalid rank scale", src, suffix)
	}
	if err := p.signMatrix(src+".U"+suffix, dst+".u", out, rank, headDim); err != nil {
		return 0, err
	}
	if err := p.signMatrix(src+".V"+suffix, dst+".v", rank, in, 0); err != nil {
		return 0, err
	}
	for _, scale := range []struct {
		name    string
		n, head int
	}{
		{"u1", out, headDim}, {"u2", rank, 0}, {"v1", rank, 0}, {"v2", in, 0},
	} {
		srcName, dstName := src+"."+scale.name+suffix, dst+"."+scale.name
		if err := p.floats(srcName, dstName, false, scale.head, scale.n); err != nil {
			return 0, err
		}
	}
	return rank, nil
}

func (p *tensorPlan) signMatrix(src, dst string, rows, cols, headDim int) error {
	t, unpacked := p.take(src)
	packed, exists := p.take(src + "_packed")
	shape, hasShape := p.take(src + "_shape")
	if exists == unpacked || hasShape && !exists {
		return fmt.Errorf("%s: expected packed words or unpacked signs", src)
	}
	count := wordCount(cols)
	part := exportTensor{
		PlannedTensor: gguf.PlannedTensor{Name: dst, Dims: []uint64{uint64(count), uint64(rows)}, DType: gguf.GGMLTypeI32},
		headDim:       headDim,
	}
	if exists {
		if packed.dtype != "I32" || !slices.Equal(packed.shape, []int{rows, count}) {
			return fmt.Errorf("%s_packed: want I32 with shape [%d,%d], got %s %v", src, rows, count, packed.dtype, packed.shape)
		}
		if hasShape && (shape.dtype != "I64" || !slices.Equal(shape.shape, []int{2}) ||
			binary.LittleEndian.Uint64(shape.raw) != uint64(rows) ||
			binary.LittleEndian.Uint64(shape.raw[8:]) != uint64(cols)) {
			return fmt.Errorf("%s_shape: disagrees with expected logical shape [%d,%d]", src, rows, cols)
		}
		if tail := cols % 32; tail != 0 {
			stride := count * 4
			for i := stride - 4; i < len(packed.raw); i += stride {
				if binary.LittleEndian.Uint32(packed.raw[i:])>>tail != 0 {
					return fmt.Errorf("%s_packed: nonzero padding in row %d", src, i/stride)
				}
			}
		}
		part.raw = packed.raw
	} else {
		_, _, floatOK := ggufFloatType(t.dtype)
		if (!floatOK && t.dtype != "I8") || !slices.Equal(t.shape, []int{rows, cols}) {
			return fmt.Errorf("%s: invalid unpacked factor %s %v, want [%d,%d]", src, t.dtype, t.shape, rows, cols)
		}
		part.raw, part.pack, part.source = t.raw, cols, t.dtype
	}
	p.tensors = append(p.tensors, part)
	return nil
}

func ggufFloatType(dtype safetensors.DType) (gguf.GGMLType, int, bool) {
	switch dtype {
	case safetensors.F32:
		return gguf.GGMLTypeF32, 4, true
	case safetensors.F16:
		return gguf.GGMLTypeF16, 2, true
	case safetensors.BF16:
		return gguf.GGMLTypeBF16, 2, true
	default:
		return 0, 0, false
	}
}

func (p *tensorPlan) floats(src, dst string, optional bool, headDim int, shape ...int) error {
	tensor, ok := p.take(src)
	if !ok {
		if optional {
			return nil
		}
		return fmt.Errorf("missing checkpoint tensor %s", src)
	}
	dtype, width, isFloat := ggufFloatType(tensor.dtype)
	if !isFloat || !slices.Equal(tensor.shape, shape) && (len(shape) != 1 || vectorLen(tensor.shape) != shape[0]) {
		return fmt.Errorf("%s: got %s %v, expected floating point %v", src, tensor.dtype, tensor.shape, shape)
	}
	if strings.Contains(dst, ".lb.") {
		for i := 0; i < len(tensor.raw); i += width {
			val := tensor.raw[i:]
			var nonfinite bool
			switch dtype {
			case gguf.GGMLTypeF32:
				nonfinite = binary.LittleEndian.Uint32(val)&0x7f800000 == 0x7f800000
			case gguf.GGMLTypeF16:
				nonfinite = binary.LittleEndian.Uint16(val)&0x7c00 == 0x7c00
			case gguf.GGMLTypeBF16:
				nonfinite = binary.LittleEndian.Uint16(val)&0x7f80 == 0x7f80
			}
			if nonfinite {
				return fmt.Errorf("%s: non-finite scale or bias", src)
			}
		}
	}
	var dims []uint64
	for _, n := range slices.Backward(shape) {
		dims = append(dims, uint64(n))
	}
	p.tensors = append(p.tensors, exportTensor{
		PlannedTensor: gguf.PlannedTensor{Name: dst, Dims: dims, DType: dtype},
		raw:           tensor.raw, headDim: headDim,
	})
	return nil
}

func (p *tensorPlan) telemetry(src string, rank int) error {
	if t, ok := p.take(src + "._split_dim_final"); ok {
		if len(t.shape) != 0 && !slices.Equal(t.shape, []int{1}) || t.dtype != "I64" ||
			binary.LittleEndian.Uint64(t.raw) != uint64(rank) {
			return fmt.Errorf("%s._split_dim_final: does not match factor rank %d", src, rank)
		}
	}
	for _, suffix := range []string{"._eff_bit_actual", "._eff_bit_target"} {
		if t, ok := p.take(src + suffix); ok {
			if _, _, floating := ggufFloatType(t.dtype); !floating ||
				len(t.shape) != 0 && !slices.Equal(t.shape, []int{1}) {
				return fmt.Errorf("%s%s: invalid training statistic", src, suffix)
			}
		}
	}
	return nil
}
