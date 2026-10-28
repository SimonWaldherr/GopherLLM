package engine

import (
	"fmt"
	"strings"
	"testing"
)

func TestParseCompressOptions(t *testing.T) {
	for _, tc := range []struct {
		name   string
		target GGMLType
		preset string
	}{
		{"Q4_K_M", GGMLTypeQ4_K, "Q4_K_M"},
		{" q5_k_m ", GGMLTypeQ5_K, "Q5_K_M"},
		{"q4_k", GGMLTypeQ4_K, ""},
		{"Q5_K", GGMLTypeQ5_K, ""},
		{"Q8_0", GGMLTypeQ8_0, ""},
		{"q4_0", GGMLTypeQ4_0, ""},
		{"Q2_K", GGMLTypeQ2_K, ""},
		{"Q3_K", GGMLTypeQ3_K, ""},
		{"Q6_K", GGMLTypeQ6_K, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts, ok := ParseCompressOptions(tc.name)
			if !ok || opts.TargetFormat != tc.target || opts.Preset != tc.preset {
				t.Fatalf("parse %q = %+v, %v", tc.name, opts, ok)
			}
			if _, err := resolveCompressOptions(opts); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, name := range []string{"", "Q4_K_SMALL", "Q4_K_S", "F16", "Q5_0", "Q5_1", "Q8_K"} {
		if _, ok := ParseCompressOptions(name); ok {
			t.Errorf("accepted unsupported target %q", name)
		}
	}
}

func TestResolveCompressOptions(t *testing.T) {
	for _, tc := range []struct {
		opts CompressOptions
		want string
	}{
		{CompressOptions{}, "unsupported target"},
		{CompressOptions{TargetFormat: GGMLTypeF16}, "unsupported target"},
		{CompressOptions{TargetFormat: GGMLTypeQ5_0}, "unsupported target"},
		{CompressOptions{TargetFormat: GGMLTypeQ5_1}, "unsupported target"},
		{CompressOptions{Preset: "unknown"}, "unknown mixed preset"},
		{CompressOptions{Preset: "Q4_K_M", Uniform: true}, "cannot be combined"},
		{CompressOptions{Preset: "Q5_K_M", TargetFormat: GGMLTypeQ4_K}, "requires base target"},
	} {
		if _, err := resolveCompressOptions(tc.opts); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("resolve %+v: got %v, want %q", tc.opts, err, tc.want)
		}
	}
	for _, name := range []string{"q4_k_m", " q5_k_m "} {
		opts, err := resolveCompressOptions(CompressOptions{Preset: name})
		if err != nil {
			t.Fatal(err)
		}
		if opts.Preset != strings.ToUpper(strings.TrimSpace(name)) || opts.TargetFormat == 0 {
			t.Fatalf("unresolved options: %+v", opts)
		}
	}
}

func compressionPolicyFixture(arch string, layers, experts int) *GGUFFile {
	return &GGUFFile{Metadata: map[string]MetaValue{
		"general.architecture":            {Kind: "str", Value: arch},
		arch + ".block_count":             {Kind: "u32", Value: uint32(layers)},
		arch + ".expert_count":            {Kind: "u32", Value: uint32(experts)},
		arch + ".attention.head_count":    {Kind: "u32", Value: uint32(32)},
		arch + ".attention.head_count_kv": {Kind: "u32", Value: uint32(32)},
	}}
}

func compressionPolicyTensor(name string, cols uint64) TensorInfo {
	return TensorInfo{Name: name, Dims: []uint64{cols, 256}, DType: GGMLTypeF16}
}

func compressionPolicyOptions(t *testing.T, name string) CompressOptions {
	t.Helper()
	opts, ok := ParseCompressOptions(name)
	if !ok {
		t.Fatal("invalid test preset", name)
	}
	return opts
}

func checkCompressionPlan(t *testing.T, p *compressionPlanner, tensor TensorInfo, want GGMLType) {
	t.Helper()
	got, convert := p.plan(tensor)
	if got != want || convert != (want != tensor.DType) {
		t.Fatalf("%s: got %s/convert=%v, want %s/convert=%v", tensor.Name, got, convert, want, want != tensor.DType)
	}
	// Looking at a decision for logging or size estimation must not advance
	// counters and silently change the actual tensor precision on the next pass.
	again, convertAgain := p.plan(tensor)
	if again != got || convertAgain != convert {
		t.Fatalf("%s: planning is not repeatable", tensor.Name)
	}
}

func TestCompressionPresetLayerMixture(t *testing.T) {
	for _, preset := range []string{"Q4_K_M", "Q5_K_M"} {
		t.Run(preset, func(t *testing.T) {
			src := compressionPolicyFixture("llama", 32, 0)
			// Store tensors in reverse order to exercise layer ranking without
			// accidentally relying on the physical GGUF descriptor order.
			for layer := 31; layer >= 0; layer-- {
				src.Tensors = append(src.Tensors, compressionPolicyTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer), 256))
			}
			opts := compressionPolicyOptions(t, preset)
			p := newCompressionPlanner(src, opts)
			for layer := range 32 {
				want := opts.TargetFormat
				// First and last four layers, plus every third interior layer.
				if layer < 4 || layer >= 28 || layer == 6 || layer == 9 || layer == 12 || layer == 15 || layer == 18 || layer == 21 || layer == 24 || layer == 27 {
					want = GGMLTypeQ6_K
				}
				for _, role := range []string{"attn_v", "ffn_down"} {
					checkCompressionPlan(t, p, compressionPolicyTensor(fmt.Sprintf("blk.%d.%s.weight", layer, role), 256), want)
				}
				for _, role := range []string{"attn_q", "attn_k", "ffn_up", "ffn_gate"} {
					checkCompressionPlan(t, p, compressionPolicyTensor(fmt.Sprintf("blk.%d.%s.weight", layer, role), 256), opts.TargetFormat)
				}
			}
		})
	}
}

func TestCompressionPresetEmbeddingsAndFallbacks(t *testing.T) {
	for _, preset := range []string{"Q4_K_M", "Q5_K_M"} {
		opts := compressionPolicyOptions(t, preset)
		for _, tied := range []bool{true, false} {
			src := compressionPolicyFixture("qwen35", 32, 0)
			if !tied {
				src.Tensors = append(src.Tensors, compressionPolicyTensor("output.weight", 256))
			}
			p := newCompressionPlanner(src, opts)
			want := opts.TargetFormat
			if tied {
				want = GGMLTypeQ6_K
			}
			checkCompressionPlan(t, p, compressionPolicyTensor("token_embd.weight", 256), want)
			checkCompressionPlan(t, p, compressionPolicyTensor("per_layer_token_embd.weight", 256), want)
			checkCompressionPlan(t, p, compressionPolicyTensor("output.weight", 256), GGMLTypeQ6_K)
			checkCompressionPlan(t, p, compressionPolicyTensor("output.weight", 96), GGMLTypeQ8_0)
			checkCompressionPlan(t, p, compressionPolicyTensor("output.weight", 65), GGMLTypeF16)
			baseFallback := GGMLTypeQ5_0
			if opts.TargetFormat == GGMLTypeQ5_K {
				baseFallback = GGMLTypeQ5_1
			}
			checkCompressionPlan(t, p, compressionPolicyTensor("blk.5.attn_q.weight", 96), baseFallback)
			checkCompressionPlan(t, p, compressionPolicyTensor("blk.0.ffn_down.weight", 96), GGMLTypeQ8_0)
			checkCompressionPlan(t, p, compressionPolicyTensor("blk.5.attn_q.weight", 65), GGMLTypeF16)
			already := compressionPolicyTensor("output.weight", 256)
			already.DType = GGMLTypeQ6_K
			checkCompressionPlan(t, p, already, GGMLTypeQ6_K)
		}
	}
}

func TestCompressionPresetMoEAndFalcon(t *testing.T) {
	opts := compressionPolicyOptions(t, "Q4_K_M")
	for _, experts := range []int{8, 256} {
		src := compressionPolicyFixture("qwen35moe", 32, experts)
		for layer := range 32 {
			src.Tensors = append(src.Tensors, compressionPolicyTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer), 256))
		}
		p := newCompressionPlanner(src, opts)
		valueType, keyType, outputType := GGMLTypeQ4_K, GGMLTypeQ4_K, GGMLTypeQ4_K
		if experts == 8 {
			valueType, keyType, outputType = GGMLTypeQ8_0, GGMLTypeQ8_0, GGMLTypeQ5_K
		}
		checkCompressionPlan(t, p, compressionPolicyTensor("blk.5.attn_v.weight", 256), valueType)
		checkCompressionPlan(t, p, compressionPolicyTensor("blk.5.attn_k.weight", 256), keyType)
		checkCompressionPlan(t, p, compressionPolicyTensor("blk.5.attn_output.weight", 256), outputType)
		for _, layer := range []int{0, 5, 6, 31} {
			want := GGMLTypeQ4_K
			if layer != 5 {
				want = GGMLTypeQ6_K
			}
			for _, role := range []string{"ffn_down_exps", "ffn_down_shexp"} {
				tensor := compressionPolicyTensor(fmt.Sprintf("blk.%d.%s.weight", layer, role), 256)
				if role == "ffn_down_exps" {
					tensor.Dims = append(tensor.Dims, uint64(experts))
				}
				checkCompressionPlan(t, p, tensor, want)
			}
		}
	}
	src := compressionPolicyFixture("falcon", 32, 0)
	p := newCompressionPlanner(src, opts)
	for _, tc := range []struct {
		layer int
		want  GGMLType
	}{
		{0, GGMLTypeQ6_K}, {1, GGMLTypeQ6_K}, {2, GGMLTypeQ5_K},
		{5, GGMLTypeQ4_K}, {6, GGMLTypeQ5_K}, {31, GGMLTypeQ5_K},
	} {
		checkCompressionPlan(t, p, compressionPolicyTensor(fmt.Sprintf("blk.%d.ffn_down.weight", tc.layer), 256), tc.want)
	}
	checkCompressionPlan(t, p, compressionPolicyTensor("output.weight", 256), GGMLTypeQ8_0)
	p = newCompressionPlanner(src, compressionPolicyOptions(t, "Q5_K_M"))
	checkCompressionPlan(t, p, compressionPolicyTensor("blk.2.ffn_down.weight", 256), GGMLTypeQ6_K)
}

func TestCompressionPresetHybridAttentionAndMetadataFallback(t *testing.T) {
	src := compressionPolicyFixture("qwen35moe", 0, 0)
	for layer := 31; layer >= 3; layer -= 4 {
		src.Tensors = append(src.Tensors, compressionPolicyTensor(fmt.Sprintf("blk.%d.attn_qkv.weight", layer), 256))
	}
	expert := compressionPolicyTensor("blk.31.ffn_down_exps.weight", 256)
	expert.Dims = append(expert.Dims, 256)
	src.Tensors = append(src.Tensors, expert)
	p := newCompressionPlanner(src, compressionPolicyOptions(t, "Q4_K_M"))
	if p.nLayers != 32 || p.nExperts != 256 || p.nAttention != 8 {
		t.Fatalf("wrong inferred model shape: layers=%d experts=%d attention=%d", p.nLayers, p.nExperts, p.nAttention)
	}
	for _, tc := range []struct {
		layer int
		want  GGMLType
	}{
		{3, GGMLTypeQ6_K}, {7, GGMLTypeQ4_K}, {11, GGMLTypeQ4_K},
		{15, GGMLTypeQ6_K}, {19, GGMLTypeQ4_K}, {23, GGMLTypeQ4_K},
		{27, GGMLTypeQ6_K}, {31, GGMLTypeQ6_K},
	} {
		checkCompressionPlan(t, p, compressionPolicyTensor(fmt.Sprintf("blk.%d.attn_qkv.weight", tc.layer), 256), tc.want)
	}
	checkCompressionPlan(t, p, expert, GGMLTypeQ6_K)
}

func TestCompressionPresetArchitecturePrecisionFloors(t *testing.T) {
	opts := compressionPolicyOptions(t, "Q4_K_M")
	src := compressionPolicyFixture("llama", 80, 0)
	src.Metadata["llama.attention.head_count_kv"] = MetaValue{Kind: "u32", Value: uint32(8)}
	for layer := range 80 {
		src.Tensors = append(src.Tensors, compressionPolicyTensor(fmt.Sprintf("blk.%d.attn_v.weight", layer), 256))
	}
	p := newCompressionPlanner(src, opts)
	checkCompressionPlan(t, p, compressionPolicyTensor("blk.0.attn_v.weight", 256), GGMLTypeQ6_K)
	checkCompressionPlan(t, p, compressionPolicyTensor("blk.10.attn_v.weight", 256), GGMLTypeQ5_K)
	checkCompressionPlan(t, p, compressionPolicyTensor("blk.10.attn_q.weight", 256), GGMLTypeQ4_K)
	for _, tc := range []struct {
		arch, name string
		want       GGMLType
	}{
		{"glm5_next", "blk.0.attn_q_a.weight", GGMLTypeQ8_0},
		{"glm5_next", "blk.0.attn_q_b.weight", GGMLTypeQ8_0},
		{"glm5_next", "blk.0.attn_kv_a_mqa.weight", GGMLTypeF16},
		{"glm5_next", "blk.0.ssm_beta.weight", GGMLTypeF16},
		{"muse_spark", "blk.0.ffn_latent.weight", GGMLTypeQ8_0},
		{"muse_spark", "blk.0.attn_gate.weight", GGMLTypeQ8_0},
		{"vllm_decision2", "decision.output.weight", GGMLTypeF16},
	} {
		p = newCompressionPlanner(compressionPolicyFixture(tc.arch, 32, 0), opts)
		checkCompressionPlan(t, p, compressionPolicyTensor(tc.name, 256), tc.want)
	}
}

func TestCompressionPresetProtectedTensorsAndLegacyPolicy(t *testing.T) {
	opts := compressionPolicyOptions(t, "Q4_K_M")
	p := newCompressionPlanner(compressionPolicyFixture("qwen35moe", 32, 256), opts)
	for _, name := range []string{
		"blk.0.attn_norm.weight", "blk.0.ssm_norm.weight", "blk.0.ffn_gate_inp.weight",
		"blk.0.ffn_gate_inp_shexp.weight", "blk.0.ssm_conv1d.weight", "blk.0.ffn_gate_tid2eid.weight",
		"position_embd.weight", "token_types.weight", "blk.0.attn_rel_b.weight",
		"blk.0.time_mix_w1.weight", "blk.0.altup_proj.weight", "blk.0.laurel_proj.weight",
		"blk.0.per_layer_model_proj.weight", "v.blk.0.patch_embd.weight", "blk.0.attn_q.bias",
	} {
		checkCompressionPlan(t, p, compressionPolicyTensor(name, 256), GGMLTypeF16)
	}
	norm := TensorInfo{Name: "output_norm.weight", Dims: []uint64{256}, DType: GGMLTypeF32}
	checkCompressionPlan(t, p, norm, norm.DType)
	zero := compressionPolicyTensor("blk.0.attn_q.weight", 0)
	checkCompressionPlan(t, p, zero, zero.DType)
	unknown := compressionPolicyTensor("other.weight", 256)
	unknown.DType = GGMLTypeUnknown
	checkCompressionPlan(t, p, unknown, unknown.DType)
	// The existing Base-Type API retains its historical decisions, including
	// the embedding floor and passthrough for incompatible row lengths.
	for _, uniform := range []bool{false, true} {
		legacy := CompressOptions{TargetFormat: GGMLTypeQ4_K, Uniform: uniform}
		p = newCompressionPlanner(nil, legacy)
		for _, tensor := range []TensorInfo{
			compressionPolicyTensor("token_embd.weight", 256),
			compressionPolicyTensor("blk.0.ffn_down.weight", 256),
			compressionPolicyTensor("blk.0.ffn_gate_inp.weight", 256),
			compressionPolicyTensor("unusual", 96), norm,
		} {
			want, wantConvert := planTensor(tensor, legacy)
			got, gotConvert := p.plan(tensor)
			if got != want || gotConvert != wantConvert {
				t.Fatalf("base policy changed for %s", tensor.Name)
			}
		}
	}
}
