package gopherllm

import "fmt"

// These tensors are read by RowInto or a fused CPU projection even in a
// Metal model. Preparing accelerator views for them reserves resources that
// generation never submits. Decide before loading, so no copy is allocated
// and then immediately released. Tied vocabulary heads retain their Metal view.
func qwen35CPUOnlyWeights(cfg Config, tensors map[string]TensorInfo) map[string]bool {
	cpu := make(map[string]bool)
	if _, separateOutput := tensors["output.weight"]; separateOutput {
		cpu["token_embd.weight"] = true
	}
	if cfg.Arch != "qwen35moe" {
		return cpu
	}
	for layer := range cfg.NLayers {
		prefix := fmt.Sprintf("blk.%d.", layer)
		for _, pair := range [][2]string{{"attn_qkv.weight", "attn_gate.weight"}, {"ssm_alpha.weight", "ssm_beta.weight"}} {
			a, aOK := tensors[prefix+pair[0]]
			b, bOK := tensors[prefix+pair[1]]
			if !aOK || !bOK || len(a.Dims) != 2 || len(b.Dims) != 2 || a.Dims[0] != b.Dims[0] || a.DType != b.DType || a.Dims[0]%256 != 0 {
				continue
			}
			switch a.DType {
			case GGMLTypeQ4_K:
				// Only a pair of direct Q4_K weights can take the fused GPU
				// route; a narrow gate keeps the whole pair on the CPU.
				if a.Dims[1] >= metalQ4KDirectMinRows && b.Dims[1] >= metalQ4KDirectMinRows {
					continue
				}
			case GGMLTypeQ6_K, GGMLTypeQ8_0:
				// tryMatvec2Into always computes these pairs on the CPU.
			default:
				continue
			}
			cpu[prefix+pair[0]], cpu[prefix+pair[1]] = true, true
		}
	}
	return cpu
}
