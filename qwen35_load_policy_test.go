package gopherllm

import "testing"

func TestQwen35CPUOnlyWeightPolicy(t *testing.T) {
	for _, tc := range []struct {
		name               string
		a, b               GGMLType
		rowsA, rowsB, cols uint64
		cpu                bool
	}{
		{"Q4 narrow gate", GGMLTypeQ4_K, GGMLTypeQ4_K, 12288, 4096, 2048, true},
		{"Q4 direct pair", GGMLTypeQ4_K, GGMLTypeQ4_K, 12288, 8192, 2048, false},
		{"Q6 CPU fusion", GGMLTypeQ6_K, GGMLTypeQ6_K, 12288, 8192, 2048, true},
		{"Q8 CPU fusion", GGMLTypeQ8_0, GGMLTypeQ8_0, 12288, 8192, 2048, true},
		{"mixed types", GGMLTypeQ4_K, GGMLTypeQ6_K, 12288, 4096, 2048, false},
		{"float", GGMLTypeF32, GGMLTypeF32, 12288, 4096, 2048, false},
		{"Q8 short blocks", GGMLTypeQ8_0, GGMLTypeQ8_0, 12288, 4096, 128, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tensors := map[string]TensorInfo{
				"blk.0.attn_qkv.weight":  {Dims: []uint64{tc.cols, tc.rowsA}, DType: tc.a},
				"blk.0.attn_gate.weight": {Dims: []uint64{tc.cols, tc.rowsB}, DType: tc.b},
				"output.weight":          {},
			}
			cpu := qwen35CPUOnlyWeights(Config{Arch: "qwen35moe", NLayers: 1}, tensors)
			if cpu["blk.0.attn_qkv.weight"] != tc.cpu || cpu["blk.0.attn_gate.weight"] != tc.cpu || !cpu["token_embd.weight"] || cpu["output.weight"] {
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
