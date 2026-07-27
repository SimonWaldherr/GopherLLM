package gopherllm

// Native pure Mamba-2 inference.
//
// Mamba2 GGUFs use the same convolution/SSM state layout as the Mamba blocks
// in Nemotron-H, but have no attention KV cache and use y*SiLU(z) rather than
// Nemotron's z*SiLU(y) gate. Keeping this loader and forward path separate
// makes that graph distinction explicit instead of accepting an architecture
// label and silently running the wrong recurrence.

import (
	"fmt"
	"io"
)

type Mamba2LayerWeights struct {
	Norm  []float32
	Mamba NemotronMambaWeights
}

type Mamba2Weights struct {
	TokenEmbd  Weight
	OutputNorm []float32
	Output     Weight
	Layers     []Mamba2LayerWeights
}

// LoadMamba2Model loads the canonical pure Mamba-2 graph. Models containing an
// additional MLP branch intentionally fail here: that branch has a different
// input projection layout and accepting it would produce plausible but wrong
// logits.
func LoadMamba2Model(data []byte, gguf *GGUFFile, borrow, prepareQuantized, useMetal bool, logw io.Writer, outOfCore ...bool) (Config, Mamba2Weights, error) {
	if logw == nil {
		logw = io.Discard
	}
	cfg := ConfigFromGGUF(gguf)
	if cfg.Arch != "mamba2" {
		return cfg, Mamba2Weights{}, fmt.Errorf("not a Mamba2 GGUF: %s", cfg.Arch)
	}
	if cfg.Dim <= 0 || cfg.NLayers <= 0 || cfg.SSMConv <= 0 || cfg.SSMInner <= 0 || cfg.SSMState <= 0 || cfg.SSMHeads <= 0 || cfg.SSMGroups <= 0 {
		return cfg, Mamba2Weights{}, fmt.Errorf("mamba2: incomplete SSM metadata")
	}
	if cfg.NHeads != 0 || cfg.HiddenDim != 0 {
		return cfg, Mamba2Weights{}, fmt.Errorf("mamba2: only pure Mamba-2 (without attention or an MLP branch) is supported")
	}
	if cfg.SSMInner%cfg.SSMHeads != 0 || cfg.SSMHeads%cfg.SSMGroups != 0 {
		return cfg, Mamba2Weights{}, fmt.Errorf("mamba2: invalid SSM dimensions inner=%d heads=%d groups=%d", cfg.SSMInner, cfg.SSMHeads, cfg.SSMGroups)
	}

	tensors := indexTensors(gguf)
	inferred := inferTensorSizes(data, gguf)
	lazyScalarWeights := len(outOfCore) > 0 && outOfCore[0]
	load := func(name string) (Weight, error) {
		return loadWeight(data, gguf.DataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
	}
	loadVec := func(name string) ([]float32, error) {
		return loadF32Vec(data, gguf.DataOffset, name, tensors, inferred)
	}
	requireMatrix := func(name string, cols, rows int) error {
		info, ok := tensors[name]
		if !ok {
			return fmt.Errorf("missing tensor: %s", name)
		}
		if len(info.Dims) != 2 || int(info.Dims[0]) != cols || int(info.Dims[1]) != rows {
			return fmt.Errorf("tensor %s has dimensions %v, want [%d %d]", name, info.Dims, cols, rows)
		}
		return nil
	}

	token, err := load("token_embd.weight")
	if err != nil {
		return cfg, Mamba2Weights{}, err
	}
	outNorm, err := loadVec("output_norm.weight")
	if err != nil {
		return cfg, Mamba2Weights{}, err
	}
	out := token
	if _, ok := tensors["output.weight"]; ok {
		out, err = load("output.weight")
		if err != nil {
			return cfg, Mamba2Weights{}, err
		}
	}

	channels := cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState
	inRows := 2*cfg.SSMInner + 2*cfg.SSMGroups*cfg.SSMState + cfg.SSMHeads
	weights := Mamba2Weights{TokenEmbd: token, OutputNorm: outNorm, Output: out, Layers: make([]Mamba2LayerWeights, cfg.NLayers)}
	for i := range cfg.NLayers {
		prefix := fmt.Sprintf("blk.%d.", i)
		if _, ok := tensors[prefix+"ssm_in.bias"]; ok {
			return cfg, Mamba2Weights{}, fmt.Errorf("mamba2 layer %d: ssm_in.bias indicates an unsupported MLP branch", i)
		}
		if _, ok := tensors[prefix+"ssm_out.bias"]; ok {
			return cfg, Mamba2Weights{}, fmt.Errorf("mamba2 layer %d: ssm_out.bias is not part of the canonical graph", i)
		}
		for _, check := range []struct {
			name       string
			cols, rows int
		}{
			{prefix + "ssm_in.weight", cfg.Dim, inRows},
			{prefix + "ssm_conv1d.weight", cfg.SSMConv, channels},
			{prefix + "ssm_out.weight", cfg.SSMInner, cfg.Dim},
		} {
			if err := requireMatrix(check.name, check.cols, check.rows); err != nil {
				return cfg, Mamba2Weights{}, fmt.Errorf("mamba2 layer %d: %w", i, err)
			}
		}

		layer := Mamba2LayerWeights{}
		layer.Norm, err = loadVec(prefix + "attn_norm.weight")
		if err == nil {
			layer.Mamba.In, err = load(prefix + "ssm_in.weight")
		}
		if err == nil {
			layer.Mamba.Conv, err = load(prefix + "ssm_conv1d.weight")
		}
		if err == nil {
			layer.Mamba.ConvBias, err = loadVec(prefix + "ssm_conv1d.bias")
		}
		if err == nil {
			layer.Mamba.DtBias, err = loadVec(prefix + "ssm_dt.bias")
		}
		if err == nil {
			layer.Mamba.A, err = loadVec(prefix + "ssm_a")
		}
		if err == nil {
			layer.Mamba.D, err = loadVec(prefix + "ssm_d")
		}
		if err == nil {
			layer.Mamba.Norm, err = loadVec(prefix + "ssm_norm.weight")
		}
		if err == nil {
			layer.Mamba.Out, err = load(prefix + "ssm_out.weight")
		}
		if err != nil {
			return cfg, Mamba2Weights{}, fmt.Errorf("mamba2 layer %d: %w", i, err)
		}
		if len(layer.Norm) != cfg.Dim || len(layer.Mamba.ConvBias) != channels || len(layer.Mamba.DtBias) != cfg.SSMHeads ||
			len(layer.Mamba.A) != cfg.SSMHeads || len(layer.Mamba.D) != cfg.SSMHeads || len(layer.Mamba.Norm) != cfg.SSMInner {
			return cfg, Mamba2Weights{}, fmt.Errorf("mamba2 layer %d: incompatible SSM vector dimensions", i)
		}
		weights.Layers[i] = layer
		if i == 0 || (i+1)%8 == 0 || i+1 == cfg.NLayers {
			fmt.Fprintf(logw, "  Loaded Mamba2 layer %d/%d\n", i+1, cfg.NLayers)
		}
	}
	return cfg, weights, nil
}

func ForwardMamba2Into(cfg Config, weights Mamba2Weights, cache *KVCache, buf *DecodeBuffer, token uint32, pos int, logits *[]float32) {
	ForwardMamba2BodyInto(cfg, weights, cache, buf, token, pos)
	weights.Output.MatvecInto(buf.XN, logits)
}

func ForwardMamba2BodyInto(cfg Config, weights Mamba2Weights, cache *KVCache, buf *DecodeBuffer, token uint32, _ int) {
	if cache == nil || cache.Nemotron == nil {
		panic("Mamba2 forward requires a recurrent cache")
	}
	weights.TokenEmbd.RowInto(int(token), cfg.Dim, &buf.X)
	for i, layer := range weights.Layers {
		rmsNormInto(buf.X, layer.Norm, cfg.RMSNormEps, &buf.XN)
		mambaSSMForward(cfg, layer.Mamba, cache.Nemotron, buf.XN, i, buf, false)
		addInPlace(buf.X[:cfg.Dim], buf.Proj[:cfg.Dim])
	}
	rmsNormInto(buf.X, weights.OutputNorm, cfg.RMSNormEps, &buf.XN)
}

func releaseMamba2MetalWeights(weights *Mamba2Weights) {
	if weights == nil {
		return
	}
	seen := map[*MetalWeight]bool{}
	release := func(w *Weight) {
		if w == nil || w.Metal == nil {
			return
		}
		if !seen[w.Metal] {
			releaseMetalWeight(w.Metal)
			seen[w.Metal] = true
		}
		w.Metal = nil
	}
	release(&weights.TokenEmbd)
	release(&weights.Output)
	for i := range weights.Layers {
		m := &weights.Layers[i].Mamba
		release(&m.In)
		release(&m.Conv)
		release(&m.Out)
	}
}
