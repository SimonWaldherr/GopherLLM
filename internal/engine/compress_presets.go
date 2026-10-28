package engine

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ParseCompressOptions parses a compression target without losing the mixed
// tensor policy carried by Q4_K_M and Q5_K_M. ParseCompressFormat remains useful
// for callers that only need the corresponding base GGML encoding.
func ParseCompressOptions(name string) (CompressOptions, bool) {
	name = strings.ToUpper(strings.TrimSpace(name))
	switch name {
	case "Q4_K_M":
		return CompressOptions{TargetFormat: GGMLTypeQ4_K, Preset: name}, true
	case "Q5_K_M":
		return CompressOptions{TargetFormat: GGMLTypeQ5_K, Preset: name}, true
	default:
		target, ok := ParseCompressFormat(name)
		if !ok || !isCompressionBaseTarget(target) {
			return CompressOptions{}, false
		}
		return CompressOptions{TargetFormat: target}, true
	}
}

func isCompressionBaseTarget(target GGMLType) bool {
	switch target {
	case GGMLTypeQ8_0, GGMLTypeQ4_0, GGMLTypeQ2_K, GGMLTypeQ3_K,
		GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K:
		return true
	default:
		return false
	}
}

func resolveCompressOptions(opts CompressOptions) (CompressOptions, error) {
	opts.Preset = strings.ToUpper(strings.TrimSpace(opts.Preset))
	if opts.Preset == "" {
		if !isCompressionBaseTarget(opts.TargetFormat) {
			return opts, fmt.Errorf("compress: unsupported target format %s (supported: Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_K_M, Q5_K_M)", opts.TargetFormat)
		}
		return opts, nil
	}
	var base GGMLType
	switch opts.Preset {
	case "Q4_K_M":
		base = GGMLTypeQ4_K
	case "Q5_K_M":
		base = GGMLTypeQ5_K
	default:
		return opts, fmt.Errorf("compress: unknown mixed preset %q (supported: Q4_K_M, Q5_K_M)", opts.Preset)
	}
	if opts.Uniform {
		return opts, fmt.Errorf("compress: mixed preset %s cannot be combined with uniform quantization; use %s for a uniform target", opts.Preset, base)
	}
	if opts.TargetFormat != 0 && opts.TargetFormat != base {
		return opts, fmt.Errorf("compress: preset %s requires base target %s, got %s", opts.Preset, base, opts.TargetFormat)
	}
	opts.TargetFormat = base
	return opts, nil
}

// compressionPlanner separates the model-wide mixed-precision recipe from
// encoding. The policy follows llama.cpp's Q4_K_M/Q5_K_M tensor categories;
// using these presets does not promise byte-identical quantizer output.
// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp
type compressionPlanner struct {
	opts          CompressOptions
	arch          string
	nLayers       int
	nExperts      int
	tiedEmbedding bool
	llama70B      bool
	attentionRank map[string]int
	nAttention    int
}

func newCompressionPlanner(src *GGUFFile, opts CompressOptions) *compressionPlanner {
	p := &compressionPlanner{opts: opts, tiedEmbedding: true}
	if src == nil || opts.Preset == "" {
		return p
	}
	p.arch, _ = src.GetString("general.architecture")
	p.nLayers = int(src.GetU32(p.arch+".block_count", 0))
	p.nExperts = int(src.GetU32(p.arch+".expert_count", 0))
	var attention []string
	for _, tensor := range src.Tensors {
		if tensor.Name == "output.weight" {
			p.tiedEmbedding = false
		}
		if layer, ok := compressionLayer(tensor.Name); ok && layer >= p.nLayers {
			p.nLayers = layer + 1
		}
		if len(tensor.Dims) == 3 && strings.Contains(tensor.Name, "_exps.weight") && int(tensor.Dims[2]) > p.nExperts {
			p.nExperts = int(tensor.Dims[2])
		}
		if compressionAttentionValue(tensor.Name) {
			attention = append(attention, tensor.Name)
		}
	}
	// Hybrid architectures can have attention in only a subset of layers.
	// Rank that subset by layer, independently of descriptor/file order.
	sort.Slice(attention, func(i, j int) bool {
		a, aOK := compressionLayer(attention[i])
		b, bOK := compressionLayer(attention[j])
		if aOK && bOK && a != b {
			return a < b
		}
		return attention[i] < attention[j]
	})
	p.attentionRank = make(map[string]int, len(attention))
	for i, name := range attention {
		p.attentionRank[name] = i
	}
	p.nAttention = len(attention)
	nHeads := src.GetU32(p.arch+".attention.head_count", 0)
	nKVHeads := src.GetU32(p.arch+".attention.head_count_kv", nHeads)
	p.llama70B = p.arch == "llama" && p.nLayers == 80 && nHeads != 0 && nKVHeads != nHeads
	return p
}

func compressionLayer(name string) (int, bool) {
	if !strings.HasPrefix(name, "blk.") {
		return 0, false
	}
	number, _, found := strings.Cut(strings.TrimPrefix(name, "blk."), ".")
	if !found {
		return 0, false
	}
	value, err := strconv.Atoi(number)
	return value, err == nil && value >= 0
}

func compressionMoreBits(index, count int) bool {
	return count > 0 && (index < count/8 || index >= 7*count/8 || (index-count/8)%3 == 2)
}

func compressionAttentionValue(name string) bool {
	return strings.Contains(name, "attn_v.weight") ||
		strings.Contains(name, "attn_qkv.weight") ||
		strings.Contains(name, "attn_kv_b.weight")
}

// compressionProtectedTensor leaves small or precision-sensitive tensors in
// their source format, including expert routers and recurrent convolution
// kernels. Integer lookup tables and unknown types also remain untouched.
func compressionProtectedTensor(t TensorInfo, arch string) bool {
	if len(t.Dims) < 2 || len(t.Dims) > 3 || !strings.HasSuffix(t.Name, ".weight") || t.Dims[0] == 0 || ggmlTypeFromUint32(uint32(t.DType)) == GGMLTypeUnknown {
		return true
	}
	name := t.Name
	for _, protected := range []string{
		"_norm.weight", "ffn_gate_inp", "ffn_gate_tid2eid", "altup", "laurel",
		"per_layer_model_proj", "position_embd", "position_embeddings", "token_types",
		"ssm_conv1d", "shortconv.conv.weight", "indexer.k_proj.weight", "indexer.q_proj.weight",
		"time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight", "time_mix_w2.weight",
		"time_mix_v0.weight", "time_mix_v1.weight", "time_mix_v2.weight", "time_mix_a0.weight",
		"time_mix_a1.weight", "time_mix_a2.weight", "time_mix_g1.weight", "time_mix_g2.weight",
		"time_mix_decay_w1.weight", "time_mix_decay_w2.weight", "time_mix_lerp_fused.weight",
		"attn_rel_b.weight", "sam.pos_embd", "sam.neck.", "sam.net_", ".rel_pos",
		".patch_embd", ".patch_merger", "a.rvq.codebook", "mm.a.code_embd",
	} {
		if strings.Contains(name, protected) {
			return true
		}
	}
	if arch == "vllm_decision2" && strings.HasPrefix(name, "decision.") {
		return true
	}
	if arch == "glm5_next" {
		for _, protected := range []string{"hc_", "indexer.attn_q_b", "indexer.attn_k", "indexer.proj", "indexer_compressor_gate", "indexer_compressor_ape", "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight", "ssm_beta.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight"} {
			if strings.Contains(name, protected) {
				return true
			}
		}
	}
	return false
}

func compressionShapeFallback(target GGMLType, columns uint64) GGMLType {
	if columns%uint64(target.BlockSize()) == 0 {
		return target
	}
	switch target {
	case GGMLTypeQ4_K:
		target = GGMLTypeQ5_0
	case GGMLTypeQ5_K:
		target = GGMLTypeQ5_1
	case GGMLTypeQ6_K:
		target = GGMLTypeQ8_0
	}
	if columns%uint64(target.BlockSize()) != 0 {
		return GGMLTypeF16
	}
	return target
}

func (p *compressionPlanner) plan(t TensorInfo) (GGMLType, bool) {
	if p.opts.Preset == "" {
		return planTensor(t, p.opts)
	}
	if compressionProtectedTensor(t, p.arch) {
		return t.DType, false
	}
	target := p.opts.TargetFormat
	name := t.Name
	embedding := name == "token_embd.weight" || name == "per_layer_token_embd.weight"
	switch {
	case name == "output.weight" || embedding && p.tiedEmbedding:
		if p.arch == "falcon" || t.Dims[0]%uint64(target.BlockSize()) != 0 {
			target = GGMLTypeQ8_0
		} else {
			target = GGMLTypeQ6_K
		}
	case embedding:
		// A separate input table does not receive the output-head upgrade.
	case compressionAttentionValue(name):
		index, ok := p.attentionRank[name]
		if !ok {
			index, _ = compressionLayer(name)
		}
		if compressionMoreBits(index, p.nAttention) {
			target = GGMLTypeQ6_K
		} else if p.llama70B && target == GGMLTypeQ4_K {
			target = GGMLTypeQ5_K
		}
		if p.nExperts == 8 {
			target = GGMLTypeQ8_0
		}
	case strings.Contains(name, "attn_k.weight") && p.nExperts == 8:
		target = GGMLTypeQ8_0
	case strings.Contains(name, "ffn_down"):
		if layer, ok := compressionLayer(name); ok {
			more := compressionMoreBits(layer, p.nLayers)
			if p.opts.Preset == "Q4_K_M" && p.arch == "falcon" {
				if layer < p.nLayers/16 {
					target = GGMLTypeQ6_K
				} else if more {
					target = GGMLTypeQ5_K
				}
			} else if more {
				target = GGMLTypeQ6_K
			}
		}
	case strings.Contains(name, "attn_output.weight") && p.nExperts == 8 && p.arch != "falcon":
		target = GGMLTypeQ5_K
	case p.arch == "glm5_next" && (strings.Contains(name, "attn_q_a") || strings.Contains(name, "attn_q_b") || strings.Contains(name, "nextn.eh_proj")):
		target = GGMLTypeQ8_0
	case p.arch == "muse_spark" && (strings.Contains(name, "ffn_latent") || strings.Contains(name, "attn_gate.weight")):
		target = GGMLTypeQ8_0
	}
	target = compressionShapeFallback(target, t.Dims[0])
	return target, target != t.DType
}
