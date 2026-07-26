package gopherllm

import (
	"fmt"
	"math"
)

// ExpertWeight retains the third GGUF tensor dimension. Weight itself stores
// matrices, while expert tensors are laid out as [input, output, expert].
// Keeping the plane geometry explicitly prevents a three-dimensional tensor
// from accidentally being treated as one very wide dense matrix.
type ExpertWeight struct {
	Weight  Weight
	Input   int
	Output  int
	Experts int
}

// ExpertBias is the optional [output, expert] bias plane paired with an
// ExpertWeight. It is used by GPT-OSS-style MoE checkpoints, but is harmless
// for the bias-free Mixtral and Qwen layouts.
type ExpertBias struct {
	Values  []float32
	Output  int
	Experts int
}

func (b ExpertBias) addTo(expert int, dst []float32) {
	if len(b.Values) == 0 {
		return
	}
	if expert < 0 || expert >= b.Experts || len(dst) != b.Output {
		panic("invalid MoE expert bias")
	}
	AxpyF32(dst, 1, b.Values[expert*b.Output:(expert+1)*b.Output])
}

// SparseMoEWeights describes the common llama/Mixtral/Qwen sparse FFN graph.
// The router is followed by selected SwiGLU experts. Qwen2-MoE additionally
// supplies a gated shared expert; GPT-OSS uses the same sparse layout with a
// slightly different SwiGLU activation and expert biases.
type SparseMoEWeights struct {
	Router     Weight
	RouterBias []float32
	Gate       ExpertWeight
	Up         ExpertWeight
	Down       ExpertWeight
	GateBias   ExpertBias
	UpBias     ExpertBias
	DownBias   ExpertBias

	// NormalizeTopK changes full-router softmax weights into a softmax over
	// just the selected experts. This is algebraically equivalent to
	// softmax-all + selected-weight normalization, and avoids exponentiating
	// every expert on Mixtral/Qwen3/GPT-OSS decode.
	NormalizeTopK bool
	Scale         float32
	OAIActivation bool
	ExpertUsed    int

	// Qwen2-MoE shared expert: sigmoid(SharedGateIn(x)) times a normal
	// SwiGLU FFN. All four pointers are either present together or absent.
	SharedGateIn *Weight
	SharedGate   *Weight
	SharedUp     *Weight
	SharedDown   *Weight
}

const moeWeightFloor = float32(6.103515625e-5) // smallest normalized F16

func loadExpertWeight(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, borrow bool) (ExpertWeight, error) {
	info, ok := tensors[name]
	if !ok {
		return ExpertWeight{}, fmt.Errorf("missing tensor: %s", name)
	}
	if len(info.Dims) != 3 || info.Dims[0] == 0 || info.Dims[1] == 0 || info.Dims[2] == 0 {
		return ExpertWeight{}, fmt.Errorf("tensor %s must have [input, output, expert] dimensions, got %v", name, info.Dims)
	}
	// Metal/Prepared weights describe one contiguous 2-D matrix. An expert
	// tensor contains many such planes, so preparing it as a single plane is
	// both wasteful and invalid. expertMatvecInto builds a lightweight plane
	// view and reuses the CPU SIMD/int8-activation kernels instead.
	w, err := loadWeight(data, dataOffset, name, tensors, inferred, false, borrow, false, false)
	if err != nil {
		return ExpertWeight{}, err
	}
	return ExpertWeight{Weight: w, Input: int(info.Dims[0]), Output: int(info.Dims[1]), Experts: int(info.Dims[2])}, nil
}

func validateExpertWeight(name string, w ExpertWeight, input, output, experts int) error {
	if w.Input != input || w.Output != output || w.Experts != experts {
		return fmt.Errorf("tensor %s has [%d, %d, %d], want [%d, %d, %d]", name, w.Input, w.Output, w.Experts, input, output, experts)
	}
	return nil
}

func loadOptionalMoEVec(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, length int) ([]float32, error) {
	if _, ok := tensors[name]; !ok {
		return nil, nil
	}
	v, err := loadF32Vec(data, dataOffset, name, tensors, inferred)
	if err != nil {
		return nil, err
	}
	if len(v) != length {
		return nil, fmt.Errorf("tensor %s has length %d, want %d", name, len(v), length)
	}
	return v, nil
}

func loadOptionalExpertBias(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, output, experts int) (ExpertBias, error) {
	info, ok := tensors[name]
	if !ok {
		return ExpertBias{}, nil
	}
	if len(info.Dims) != 2 || int(info.Dims[0]) != output || int(info.Dims[1]) != experts {
		return ExpertBias{}, fmt.Errorf("tensor %s has dimensions %v, want [%d %d]", name, info.Dims, output, experts)
	}
	w, err := loadWeight(data, dataOffset, name, tensors, inferred, true, false, false, false)
	if err != nil {
		return ExpertBias{}, err
	}
	if len(w.F32) != output*experts {
		return ExpertBias{}, fmt.Errorf("tensor %s has %d values, want %d", name, len(w.F32), output*experts)
	}
	return ExpertBias{Values: w.F32, Output: output, Experts: experts}, nil
}

func hasAnyTensor(tensors map[string]TensorInfo, names ...string) bool {
	for _, name := range names {
		if _, ok := tensors[name]; ok {
			return true
		}
	}
	return false
}

func loadSparseMoEWeights(data []byte, dataOffset int, prefix string, cfg Config, tensors map[string]TensorInfo, inferred map[string]int, borrow, prepareQuantized, useMetal bool) (*SparseMoEWeights, error) {
	if cfg.ExpertCount <= 0 || cfg.ExpertUsedCount <= 0 || cfg.ExpertUsedCount > cfg.ExpertCount {
		return nil, fmt.Errorf("%s: invalid MoE metadata experts=%d used=%d", cfg.Arch, cfg.ExpertCount, cfg.ExpertUsedCount)
	}
	routerName := prefix + "ffn_gate_inp.weight"
	routerInfo, ok := tensors[routerName]
	if !ok {
		return nil, fmt.Errorf("missing tensor: %s", routerName)
	}
	if len(routerInfo.Dims) != 2 || int(routerInfo.Dims[0]) != cfg.Dim || int(routerInfo.Dims[1]) != cfg.ExpertCount {
		return nil, fmt.Errorf("tensor %s has dimensions %v, want [%d %d]", routerName, routerInfo.Dims, cfg.Dim, cfg.ExpertCount)
	}
	load := func(name string) (Weight, error) {
		return loadWeight(data, dataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal)
	}
	router, err := load(routerName)
	if err != nil {
		return nil, err
	}
	w := &SparseMoEWeights{
		Router:        router,
		NormalizeTopK: cfg.ExpertWeightsNorm,
		Scale:         cfg.ExpertWeightsScale,
		OAIActivation: cfg.Arch == "gpt-oss",
		ExpertUsed:    cfg.ExpertUsedCount,
	}
	if w.Scale == 0 {
		w.Scale = 1
	}
	if w.RouterBias, err = loadOptionalMoEVec(data, dataOffset, prefix+"ffn_gate_inp.bias", tensors, inferred, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if w.Gate, err = loadExpertWeight(data, dataOffset, prefix+"ffn_gate_exps.weight", tensors, inferred, borrow); err != nil {
		return nil, err
	}
	if w.Up, err = loadExpertWeight(data, dataOffset, prefix+"ffn_up_exps.weight", tensors, inferred, borrow); err != nil {
		return nil, err
	}
	if w.Down, err = loadExpertWeight(data, dataOffset, prefix+"ffn_down_exps.weight", tensors, inferred, borrow); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(prefix+"ffn_gate_exps.weight", w.Gate, cfg.Dim, w.Gate.Output, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(prefix+"ffn_up_exps.weight", w.Up, cfg.Dim, w.Gate.Output, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(prefix+"ffn_down_exps.weight", w.Down, w.Gate.Output, cfg.Dim, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if w.GateBias, err = loadOptionalExpertBias(data, dataOffset, prefix+"ffn_gate_exps.bias", tensors, inferred, w.Gate.Output, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if w.UpBias, err = loadOptionalExpertBias(data, dataOffset, prefix+"ffn_up_exps.bias", tensors, inferred, w.Up.Output, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if w.DownBias, err = loadOptionalExpertBias(data, dataOffset, prefix+"ffn_down_exps.bias", tensors, inferred, cfg.Dim, cfg.ExpertCount); err != nil {
		return nil, err
	}

	sharedNames := []string{
		prefix + "ffn_gate_inp_shexp.weight",
		prefix + "ffn_gate_shexp.weight",
		prefix + "ffn_up_shexp.weight",
		prefix + "ffn_down_shexp.weight",
	}
	if hasAnyTensor(tensors, sharedNames...) {
		for _, name := range sharedNames {
			if _, ok := tensors[name]; !ok {
				return nil, fmt.Errorf("%s: unsupported partial shared-expert layout (missing %s)", cfg.Arch, name)
			}
		}
		gateInInfo := tensors[sharedNames[0]]
		if !((len(gateInInfo.Dims) == 1 && int(gateInInfo.Dims[0]) == cfg.Dim) ||
			(len(gateInInfo.Dims) == 2 && int(gateInInfo.Dims[0]) == cfg.Dim && int(gateInInfo.Dims[1]) == 1)) {
			return nil, fmt.Errorf("tensor %s has dimensions %v, want [%d] or [%d 1]", sharedNames[0], gateInInfo.Dims, cfg.Dim, cfg.Dim)
		}
		gateIn, err := load(sharedNames[0])
		if err != nil {
			return nil, err
		}
		gate, err := load(sharedNames[1])
		if err != nil {
			return nil, err
		}
		up, err := load(sharedNames[2])
		if err != nil {
			return nil, err
		}
		down, err := load(sharedNames[3])
		if err != nil {
			return nil, err
		}
		gateInfo, upInfo, downInfo := tensors[sharedNames[1]], tensors[sharedNames[2]], tensors[sharedNames[3]]
		if len(gateInfo.Dims) != 2 || len(upInfo.Dims) != 2 || len(downInfo.Dims) != 2 ||
			int(gateInfo.Dims[0]) != cfg.Dim || int(upInfo.Dims[0]) != cfg.Dim ||
			gateInfo.Dims[1] != upInfo.Dims[1] || int(downInfo.Dims[0]) != int(gateInfo.Dims[1]) || int(downInfo.Dims[1]) != cfg.Dim {
			return nil, fmt.Errorf("%s: inconsistent shared-expert dimensions", cfg.Arch)
		}
		w.SharedGateIn, w.SharedGate, w.SharedUp, w.SharedDown = &gateIn, &gate, &up, &down
	}
	return w, nil
}

func selectTopExperts(logits []float32, used int, dst *[]ExpertScore) []ExpertScore {
	if used <= 0 || used > len(logits) {
		panic("invalid MoE top-k")
	}
	ensureLenNoClear(dst, used)
	selected := *dst
	for i := 0; i < used; i++ {
		selected[i] = ExpertScore{Index: i, Score: logits[i]}
	}
	minAt := 0
	findMin := func() int {
		at := 0
		for j := 1; j < used; j++ {
			if selected[j].Score < selected[at].Score {
				at = j
			}
		}
		return at
	}
	minAt = findMin()
	for i := used; i < len(logits); i++ {
		score := logits[i]
		// Strict comparison retains the lowest expert index when logits tie
		// and treats NaNs exactly like the previous selection loop.
		if !(score > selected[minAt].Score) {
			continue
		}
		selected[minAt] = ExpertScore{Index: i, Score: score}
		minAt = findMin()
	}
	return selected
}

func sparseMoERoutingWeights(logits []float32, selected []ExpertScore, normalizeTopK bool, out *[]float32) []float32 {
	ensureLenNoClear(out, len(selected))
	weights := *out
	if normalizeTopK {
		maxLogit := selected[0].Score
		for _, choice := range selected[1:] {
			if choice.Score > maxLogit {
				maxLogit = choice.Score
			}
		}
		var sum float32
		for i, choice := range selected {
			v := float32(math.Exp(float64(choice.Score - maxLogit)))
			weights[i] = v
			sum += v
		}
		if sum < moeWeightFloor {
			sum = moeWeightFloor
		}
		ScaleF32(weights, 1/sum)
		return weights
	}
	maxLogit := logits[0]
	for _, score := range logits[1:] {
		if score > maxLogit {
			maxLogit = score
		}
	}
	var sum float32
	for _, score := range logits {
		sum += float32(math.Exp(float64(score - maxLogit)))
	}
	if sum < moeWeightFloor {
		sum = moeWeightFloor
	}
	for i, choice := range selected {
		weights[i] = float32(math.Exp(float64(choice.Score-maxLogit))) / sum
	}
	return weights
}

func sparseMoEForward(w *SparseMoEWeights, x []float32, buf *DecodeBuffer) {
	if w == nil || len(x) != w.Gate.Input || w.Gate.Experts <= 0 || w.ExpertUsed <= 0 || w.ExpertUsed > w.Gate.Experts ||
		w.Gate.Experts != w.Up.Experts || w.Gate.Experts != w.Down.Experts {
		panic("invalid sparse MoE weights")
	}
	w.Router.MatvecInto(x, &buf.RouterLogits)
	if len(buf.RouterLogits) != w.Gate.Experts {
		panic("sparse MoE router has an invalid shape")
	}
	if len(w.RouterBias) != 0 {
		addInPlace(buf.RouterLogits, w.RouterBias)
	}
	selected := selectTopExperts(buf.RouterLogits, w.ExpertUsed, &buf.TopExperts)
	routing := sparseMoERoutingWeights(buf.RouterLogits, selected, w.NormalizeTopK, &buf.ExpertProbs)
	ensureLenNoClear(&buf.Proj, w.Down.Output)
	clear(buf.Proj[:w.Down.Output])
	for i, choice := range selected {
		expertMatvecInto(w.Gate, choice.Index, x, &buf.Gate, &buf.ExpertRow)
		w.GateBias.addTo(choice.Index, buf.Gate)
		expertMatvecInto(w.Up, choice.Index, x, &buf.Up, &buf.ExpertRow)
		w.UpBias.addTo(choice.Index, buf.Up)
		ensureLenNoClear(&buf.Hidden, w.Gate.Output)
		if w.OAIActivation {
			for j := range w.Gate.Output {
				gate := min(buf.Gate[j], float32(7))
				up := clamp(buf.Up[j], -7, 7)
				buf.Hidden[j] = gate * nemotronSigmoid(1.702*gate) * (up + 1)
			}
		} else {
			siluMulF32(buf.Gate, buf.Up, buf.Hidden)
		}
		expertMatvecInto(w.Down, choice.Index, buf.Hidden, &buf.MOE, &buf.ExpertRow)
		w.DownBias.addTo(choice.Index, buf.MOE)
		AxpyF32(buf.Proj[:w.Down.Output], routing[i]*w.Scale, buf.MOE)
	}
	if w.SharedGateIn != nil {
		if w.SharedGate == nil || w.SharedUp == nil || w.SharedDown == nil {
			panic("partial shared MoE expert")
		}
		w.SharedGate.MatvecInto(x, &buf.Gate)
		w.SharedUp.MatvecInto(x, &buf.Up)
		ensureLenNoClear(&buf.Hidden, len(buf.Gate))
		siluMulF32(buf.Gate, buf.Up, buf.Hidden)
		w.SharedDown.MatvecInto(buf.Hidden, &buf.MOE)
		w.SharedGateIn.MatvecInto(x, &buf.RouterLogits)
		if len(buf.RouterLogits) != 1 {
			panic("shared MoE gate has an invalid shape")
		}
		ScaleF32(buf.MOE, nemotronSigmoid(buf.RouterLogits[0]))
		addInPlace(buf.Proj[:w.Down.Output], buf.MOE)
	}
}

// expertMatvecInto computes one expert plane against x. Quantized GGUF
// tensors store planes contiguously, so constructing a 2-D view lets the
// existing SIMD/int8-activation/parallel matvec kernels process all rows at
// once. The old row-by-row dequantization fallback remains for an unknown
// layout, preserving compatibility with future tensor types.
func expertMatvecInto(w ExpertWeight, expert int, x []float32, out *[]float32, row *[]float32) {
	if expert < 0 || expert >= w.Experts || len(x) != w.Input {
		panic("invalid expert matvec")
	}
	ensureLenNoClear(out, w.Output)
	if w.Weight.F32 != nil {
		base := expert * w.Output * w.Input
		MatvecF32Into(w.Weight.F32[base:base+w.Output*w.Input], x, w.Output, w.Input, out)
		return
	}
	if rowBytes, ok := w.Weight.Type.DataSize(w.Input); ok && rowBytes > 0 {
		planeBytes := w.Output * rowBytes
		start := expert * planeBytes
		if start >= 0 && start+planeBytes <= len(w.Weight.Raw) {
			plane := Weight{Raw: w.Weight.Raw[start : start+planeBytes], Type: w.Weight.Type, Rows: w.Output, Cols: w.Input}
			plane.MatvecInto(x, out)
			return
		}
	}
	ensureLenNoClear(row, w.Input)
	for r := range w.Output {
		w.Weight.RowInto(expert*w.Output+r, w.Input, row)
		(*out)[r] = DotF32(*row, x)
	}
}
