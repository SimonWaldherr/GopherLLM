package engine

import "testing"

func TestQwen35CPUOnlyWeightPolicy(t *testing.T) {
	for _, tc := range []struct {
		name               string
		a, b               GGMLType
		rowsA, rowsB, cols uint64
		qkvCPU, alphaCPU   bool
	}{
		{"Q4 narrow gate", GGMLTypeQ4_K, GGMLTypeQ4_K, 12288, 4096, 2048, false, true},
		{"Q4 direct pair", GGMLTypeQ4_K, GGMLTypeQ4_K, 12288, 8192, 2048, false, false},
		{"Q4 CPU pair", GGMLTypeQ4_K, GGMLTypeQ4_K, 4096, 2048, 2048, true, true},
		{"Q6 direct QKV", GGMLTypeQ6_K, GGMLTypeQ6_K, 12288, 8192, 2048, false, true},
		{"Q6 CPU pair", GGMLTypeQ6_K, GGMLTypeQ6_K, 2048, 1024, 2048, true, true},
		{"Q8 direct QKV", GGMLTypeQ8_0, GGMLTypeQ8_0, 12288, 8192, 2048, false, true},
		{"Q8 CPU pair", GGMLTypeQ8_0, GGMLTypeQ8_0, 1024, 512, 2048, true, true},
		{"mixed types", GGMLTypeQ4_K, GGMLTypeQ6_K, 12288, 4096, 2048, false, false},
		{"float", GGMLTypeF32, GGMLTypeF32, 12288, 4096, 2048, false, false},
		{"Q8 short blocks", GGMLTypeQ8_0, GGMLTypeQ8_0, 12288, 4096, 128, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tensors := map[string]TensorInfo{
				"blk.0.attn_qkv.weight":  {Dims: []uint64{tc.cols, tc.rowsA}, DType: tc.a},
				"blk.0.attn_gate.weight": {Dims: []uint64{tc.cols, tc.rowsB}, DType: tc.b},
				"blk.0.ssm_alpha.weight": {Dims: []uint64{tc.cols, tc.rowsA}, DType: tc.a},
				"blk.0.ssm_beta.weight":  {Dims: []uint64{tc.cols, tc.rowsB}, DType: tc.b},
				"output.weight":          {},
			}
			cpu := qwen35CPUOnlyWeights(Config{Arch: "qwen35moe", NLayers: 1}, tensors)
			if cpu["blk.0.attn_qkv.weight"] != tc.qkvCPU || cpu["blk.0.attn_gate.weight"] != tc.qkvCPU ||
				cpu["blk.0.ssm_alpha.weight"] != tc.alphaCPU || cpu["blk.0.ssm_beta.weight"] != tc.alphaCPU || !cpu["token_embd.weight"] || cpu["output.weight"] {
				t.Fatalf("unexpected CPU-only set: %v", cpu)
			}
			delete(tensors, "output.weight")
			if qwen35CPUOnlyWeights(Config{Arch: "qwen35moe", NLayers: 1}, tensors)["token_embd.weight"] {
				t.Fatal("tied vocabulary head lost acceleration")
			}
			if qwen35CPUOnlyWeights(Config{Arch: "qwen35", NLayers: 1}, tensors)["blk.0.attn_qkv.weight"] {
				t.Fatal("dense batching policy changed")
			}
		})
	}
}
