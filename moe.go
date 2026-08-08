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

	// StorageOutput and RowOffset describe a view into one expert plane. They
	// are normally zero, which means this logical matrix occupies a complete
	// [input, Output, expert] tensor. Fused gate/up tensors instead store both
	// projections in a [input, 2*Output, expert] plane; the two ExpertWeights
	// share Weight and select their own contiguous row range without copying or
	// dequantizing the backing tensor.
	StorageOutput int
	RowOffset     int
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
	// RouterCorrectionBias is DeepSeek/Kimi's exp_probs_b. It changes only
	// the top-k selection score; the routed probability remains sigmoid(router)
	// as in the original noaux router.
	RouterCorrectionBias []float32
	Gate                 ExpertWeight
	Up                   ExpertWeight
	Down                 ExpertWeight
	GateBias             ExpertBias
	UpBias               ExpertBias
	DownBias             ExpertBias

	// NormalizeTopK changes full-router softmax weights into a softmax over
	// just the selected experts. This is algebraically equivalent to
	// softmax-all + selected-weight normalization, and avoids exponentiating
	// every expert on Mixtral/Qwen3/GPT-OSS decode.
	NormalizeTopK  bool
	Scale          float32
	OAIActivation  bool
	ExpertUsed     int
	RoutingSigmoid bool
	// GroupCount/GroupUsed implement DeepSeek-V3's group-limited noaux
	// routing.  Before selecting ExpertUsed experts, the router keeps only
	// GroupUsed of GroupCount equally sized expert groups. Groups are ranked
	// by their best corrected sigmoid score, or by the sum of their best two
	// scores when a correction bias is present. A value of 1/1 is the Kimi-K2
	// / ungrouped DeepSeek behaviour.
	GroupCount int
	GroupUsed  int

	// Qwen2-MoE shared expert: sigmoid(SharedGateIn(x)) times a normal
	// SwiGLU FFN. For that layout all four pointers are present together;
	// DeepSeek/Kimi instead sets the three FFN weights plus SharedAlways.
	SharedGateIn *Weight
	SharedGate   *Weight
	SharedUp     *Weight
	SharedDown   *Weight
	// SharedAlways is DeepSeek-V2/V3/Kimi's always-on shared SwiGLU branch.
	// Qwen2-MoE instead has SharedGateIn and gates its shared output.
	SharedAlways bool
}

const moeWeightFloor = float32(6.103515625e-5) // smallest normalized F16

func loadExpertWeight(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, borrow bool, lazyScalars ...bool) (ExpertWeight, error) {
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
	w, err := loadWeight(data, dataOffset, name, tensors, inferred, false, borrow, false, false, lazyScalars...)
	if err != nil {
		return ExpertWeight{}, err
	}
	return ExpertWeight{Weight: w, Input: int(info.Dims[0]), Output: int(info.Dims[1]), Experts: int(info.Dims[2])}, nil
}

// loadFusedExpertGateUpWeight loads llama.cpp's optional fused expert tensor
// [input, 2*hidden, expert] and returns zero-copy views of its gate and up
// halves. Keeping a shared backing Weight is important for quantized and
// out-of-core MoE banks, where splitting it would otherwise duplicate many GB.
func loadFusedExpertGateUpWeight(data []byte, dataOffset int, name string, tensors map[string]TensorInfo, inferred map[string]int, borrow bool, lazyScalars ...bool) (ExpertWeight, ExpertWeight, error) {
	fused, err := loadExpertWeight(data, dataOffset, name, tensors, inferred, borrow, lazyScalars...)
	if err != nil {
		return ExpertWeight{}, ExpertWeight{}, err
	}
	if fused.Output <= 0 || fused.Output%2 != 0 {
		return ExpertWeight{}, ExpertWeight{}, fmt.Errorf("tensor %s has output width %d, want an even fused gate/up width", name, fused.Output)
	}
	storageOutput := fused.Output
	fused.Output /= 2
	fused.StorageOutput = storageOutput
	gate := fused
	up := fused
	up.RowOffset = fused.Output
	return gate, up, nil
}

func (w ExpertWeight) storageRows() int {
	if w.StorageOutput != 0 {
		return w.StorageOutput
	}
	return w.Output
}

func (w ExpertWeight) f32Plane(expert int) ([]float32, bool) {
	if w.Weight.F32 == nil || expert < 0 || expert >= w.Experts || w.Input <= 0 || w.Output <= 0 {
		return nil, false
	}
	storageRows := w.storageRows()
	if storageRows <= 0 || w.RowOffset < 0 || w.RowOffset+w.Output > storageRows {
		return nil, false
	}
	start := (expert*storageRows + w.RowOffset) * w.Input
	end := start + w.Output*w.Input
	if start < 0 || end < start || end > len(w.Weight.F32) {
		return nil, false
	}
	return w.Weight.F32[start:end], true
}

func validateExpertWeight(name string, w ExpertWeight, input, output, experts int) error {
	if w.Input != input || w.Output != output || w.Experts != experts {
		return fmt.Errorf("tensor %s has [%d, %d, %d], want [%d, %d, %d]", name, w.Input, w.Output, w.Experts, input, output, experts)
	}
	storageRows := w.storageRows()
	if storageRows < w.Output || w.RowOffset < 0 || w.RowOffset+w.Output > storageRows {
		return fmt.Errorf("tensor %s has invalid expert row view: offset=%d output=%d storage_output=%d", name, w.RowOffset, w.Output, storageRows)
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

func loadSparseMoEWeights(data []byte, dataOffset int, prefix string, cfg Config, tensors map[string]TensorInfo, inferred map[string]int, borrow, prepareQuantized, useMetal, lazyScalarWeights bool) (*SparseMoEWeights, error) {
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
		return loadWeight(data, dataOffset, name, tensors, inferred, false, borrow, prepareQuantized, useMetal, lazyScalarWeights)
	}
	router, err := load(routerName)
	if err != nil {
		return nil, err
	}
	w := &SparseMoEWeights{
		Router:         router,
		NormalizeTopK:  cfg.ExpertWeightsNorm,
		Scale:          cfg.ExpertWeightsScale,
		OAIActivation:  cfg.Arch == "gpt-oss",
		ExpertUsed:     cfg.ExpertUsedCount,
		RoutingSigmoid: deepSeek2Family(cfg.Arch) && cfg.ExpertGatingFunc == deepSeekExpertGatingSigmoid,
		GroupCount:     1,
		GroupUsed:      1,
	}
	if w.Scale == 0 {
		w.Scale = 1
	}
	if w.RouterBias, err = loadOptionalMoEVec(data, dataOffset, prefix+"ffn_gate_inp.bias", tensors, inferred, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if deepSeek2Family(cfg.Arch) {
		if cfg.ExpertGatingFunc != 0 && cfg.ExpertGatingFunc != deepSeekExpertGatingSigmoid {
			return nil, fmt.Errorf("%s: unsupported expert_gating_func=%d (supported: softmax=0, sigmoid/noaux=%d)", cfg.Arch, cfg.ExpertGatingFunc, deepSeekExpertGatingSigmoid)
		}
		// DeepSeek/Kimi's correction bias is not a normal router bias: it is
		// added after sigmoid only to rank candidates for top-k selection.
		w.RouterCorrectionBias, err = loadOptionalMoEVec(data, dataOffset, prefix+"exp_probs_b.bias", tensors, inferred, cfg.ExpertCount)
		if err != nil {
			return nil, err
		}
		if w.RouterCorrectionBias == nil {
			w.RouterCorrectionBias, err = loadOptionalMoEVec(data, dataOffset, prefix+"exp_probs_b", tensors, inferred, cfg.ExpertCount)
			if err != nil {
				return nil, err
			}
		}
		w.GroupCount = max(1, cfg.ExpertGroupCount)
		w.GroupUsed = max(1, cfg.ExpertGroupUsedCount)
		if w.GroupCount > cfg.ExpertCount || cfg.ExpertCount%w.GroupCount != 0 || w.GroupUsed > w.GroupCount ||
			cfg.ExpertUsedCount > (cfg.ExpertCount/w.GroupCount)*w.GroupUsed {
			return nil, fmt.Errorf("%s: invalid grouped MoE metadata experts=%d groups=%d groups_used=%d", cfg.Arch, cfg.ExpertCount, w.GroupCount, w.GroupUsed)
		}
		if w.GroupCount > 1 && !w.RoutingSigmoid {
			return nil, fmt.Errorf("%s: grouped MoE routing requires sigmoid/noaux expert_gating_func=%d", cfg.Arch, deepSeekExpertGatingSigmoid)
		}
	}
	gateName := prefix + "ffn_gate_exps.weight"
	upName := prefix + "ffn_up_exps.weight"
	fusedGateUpName := prefix + "ffn_gate_up_exps.weight"
	_, hasGate := tensors[gateName]
	_, hasUp := tensors[upName]
	_, hasFusedGateUp := tensors[fusedGateUpName]
	gateTensorName, upTensorName := gateName, upName
	switch {
	case hasGate && hasUp:
		// Prefer separate expert tensors when both layouts are available. This
		// is the canonical GGUF layout and also permits independent types.
		if w.Gate, err = loadExpertWeight(data, dataOffset, gateName, tensors, inferred, borrow, lazyScalarWeights); err != nil {
			return nil, err
		}
		if w.Up, err = loadExpertWeight(data, dataOffset, upName, tensors, inferred, borrow, lazyScalarWeights); err != nil {
			return nil, err
		}
	case !hasGate && !hasUp && hasFusedGateUp:
		w.Gate, w.Up, err = loadFusedExpertGateUpWeight(data, dataOffset, fusedGateUpName, tensors, inferred, borrow, lazyScalarWeights)
		if err != nil {
			return nil, err
		}
		gateTensorName, upTensorName = fusedGateUpName, fusedGateUpName
	case !hasGate && !hasUp:
		// Retain the established missing-tensor diagnostic for checkpoints
		// which provide neither representation.
		if w.Gate, err = loadExpertWeight(data, dataOffset, gateName, tensors, inferred, borrow, lazyScalarWeights); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("%s: partial MoE gate/up layout (need both %s and %s, or %s)", cfg.Arch, gateName, upName, fusedGateUpName)
	}
	if w.Down, err = loadExpertWeight(data, dataOffset, prefix+"ffn_down_exps.weight", tensors, inferred, borrow, lazyScalarWeights); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(gateTensorName, w.Gate, cfg.Dim, w.Gate.Output, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(upTensorName, w.Up, cfg.Dim, w.Gate.Output, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if err := validateExpertWeight(prefix+"ffn_down_exps.weight", w.Down, w.Gate.Output, cfg.Dim, cfg.ExpertCount); err != nil {
		return nil, err
	}
	if deepSeek2Family(cfg.Arch) && cfg.ExpertFeedForwardDim > 0 && w.Gate.Output != cfg.ExpertFeedForwardDim {
		return nil, fmt.Errorf("%s: expert tensor width=%d, want expert_feed_forward_length=%d", cfg.Arch, w.Gate.Output, cfg.ExpertFeedForwardDim)
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

	if usesUngatedSharedExpert(cfg.Arch) {
		// Kimi, DeepSeek-V2/V3, and GraniteMoE all use a plain, always-on
		// shared SwiGLU expert added unweighted to the routed output. There
		// is intentionally no ffn_gate_inp_shexp routing/gating tensor.
		sharedNames := []string{prefix + "ffn_gate_shexp.weight", prefix + "ffn_up_shexp.weight", prefix + "ffn_down_shexp.weight"}
		if cfg.ExpertSharedCount > 0 && !hasAnyTensor(tensors, sharedNames...) {
			return nil, fmt.Errorf("%s: expert_shared_count=%d requires ffn_gate_shexp, ffn_up_shexp, and ffn_down_shexp", cfg.Arch, cfg.ExpertSharedCount)
		}
		if hasAnyTensor(tensors, sharedNames...) {
			for _, name := range sharedNames {
				if _, ok := tensors[name]; !ok {
					return nil, fmt.Errorf("%s: partial DeepSeek/Kimi shared expert (missing %s)", cfg.Arch, name)
				}
			}
			gateInfo, upInfo, downInfo := tensors[sharedNames[0]], tensors[sharedNames[1]], tensors[sharedNames[2]]
			if len(gateInfo.Dims) != 2 || len(upInfo.Dims) != 2 || len(downInfo.Dims) != 2 ||
				int(gateInfo.Dims[0]) != cfg.Dim || int(upInfo.Dims[0]) != cfg.Dim ||
				gateInfo.Dims[1] != upInfo.Dims[1] || int(downInfo.Dims[0]) != int(gateInfo.Dims[1]) || int(downInfo.Dims[1]) != cfg.Dim {
				return nil, fmt.Errorf("%s: inconsistent shared-expert dimensions", cfg.Arch)
			}
			sharedWidth := int(gateInfo.Dims[1])
			if cfg.ExpertFeedForwardDim > 0 && sharedWidth != cfg.ExpertFeedForwardDim*max(1, cfg.ExpertSharedCount) {
				return nil, fmt.Errorf("%s: shared expert width=%d, want expert_feed_forward_length*expert_shared_count=%d", cfg.Arch, sharedWidth, cfg.ExpertFeedForwardDim*max(1, cfg.ExpertSharedCount))
			}
			gate, err := load(sharedNames[0])
			if err != nil {
				return nil, err
			}
			up, err := load(sharedNames[1])
			if err != nil {
				return nil, err
			}
			down, err := load(sharedNames[2])
			if err != nil {
				return nil, err
			}
			w.SharedGate, w.SharedUp, w.SharedDown, w.SharedAlways = &gate, &up, &down, true
		}
	} else {
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
	}
	if cfg.Arch == "qwen2moe" && w.SharedGateIn == nil {
		return nil, fmt.Errorf("qwen2moe requires ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, and ffn_down_shexp")
	}
	if cfg.Arch == "gpt-oss" {
		if len(w.RouterBias) != cfg.ExpertCount || len(w.GateBias.Values) != w.Gate.Output*cfg.ExpertCount ||
			len(w.UpBias.Values) != w.Up.Output*cfg.ExpertCount || len(w.DownBias.Values) != cfg.Dim*cfg.ExpertCount {
			return nil, fmt.Errorf("gpt-oss requires router and gate/up/down expert biases")
		}
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

// sparseMoESigmoidRoutingWeights implements DeepSeek-V3/Kimi's noaux router.
// selected contains indices ranked by score + exp_probs_b; scores deliberately
// contains the *uncorrected* sigmoid(router) values used as mixture weights.
func sparseMoESigmoidRoutingWeights(scores []float32, selected []ExpertScore, normalizeTopK bool, out *[]float32) []float32 {
	ensureLenNoClear(out, len(selected))
	weights := *out
	var sum float32
	for i, choice := range selected {
		if choice.Index < 0 || choice.Index >= len(scores) {
			panic("invalid sigmoid MoE expert selection")
		}
		weight := scores[choice.Index]
		weights[i] = weight
		sum += weight
	}
	if normalizeTopK {
		if sum < moeWeightFloor {
			sum = moeWeightFloor
		}
		ScaleF32(weights, 1/sum)
	}
	return weights
}

// selectDeepSeekGroupedExperts implements the group-limited greedy router
// used by DeepSeek-V3. The caller supplies corrected sigmoid scores: the
// correction bias participates in choosing experts and groups, while the
// uncorrected sigmoid values remain the mixture weights (see
// sparseMoESigmoidRoutingWeights).
//
// A correction bias changes group scoring from the largest score to the sum
// of the two largest scores. This seemingly small distinction is part of
// DeepSeek-V3's noaux algorithm and matches the reference implementation.
func selectDeepSeekGroupedExperts(scores []float32, groupCount, groupUsed, expertUsed int, useTopTwo bool, groupScratch *[]float32, groupTopScratch *[]ExpertScore, expertScratch *[]ExpertScore) []ExpertScore {
	if groupCount <= 0 || groupUsed <= 0 || groupUsed > groupCount || expertUsed <= 0 || expertUsed > len(scores) ||
		groupCount > len(scores) || len(scores)%groupCount != 0 ||
		expertUsed > (len(scores)/groupCount)*groupUsed {
		panic("invalid grouped DeepSeek MoE routing")
	}
	if groupCount == 1 {
		return selectTopExperts(scores, expertUsed, expertScratch)
	}
	groupWidth := len(scores) / groupCount
	ensureLenNoClear(groupScratch, groupCount)
	groups := *groupScratch
	for group := range groupCount {
		start := group * groupWidth
		best, second := float32(math.Inf(-1)), float32(math.Inf(-1))
		for _, score := range scores[start : start+groupWidth] {
			if score > best {
				second, best = best, score
			} else if score > second {
				second = score
			}
		}
		if useTopTwo && groupWidth > 1 {
			groups[group] = best + second
		} else {
			groups[group] = best
		}
	}
	selectedGroups := selectTopExperts(groups, groupUsed, groupTopScratch)

	// Disallow experts outside of the selected groups without allocating a
	// mask. A negative infinity sentinel retains group-limited semantics even
	// for a checkpoint whose correction bias makes a valid score negative.
	for group := range groupCount {
		keep := false
		for _, selected := range selectedGroups {
			if selected.Index == group {
				keep = true
				break
			}
		}
		if !keep {
			start := group * groupWidth
			for i := start; i < start+groupWidth; i++ {
				scores[i] = float32(math.Inf(-1))
			}
		}
	}
	return selectTopExperts(scores, expertUsed, expertScratch)
}

func sparseMoEForward(w *SparseMoEWeights, x []float32, buf *DecodeBuffer) {
	groupCount, groupUsed := 1, 1
	if w != nil {
		// Keep manually assembled SparseMoEWeights (including small unit-test
		// fixtures and API consumers) backward compatible with the pre-grouped
		// layout, where omitted group fields meant one unpartitioned group.
		if w.GroupCount != 0 {
			groupCount = w.GroupCount
		}
		if w.GroupUsed != 0 {
			groupUsed = w.GroupUsed
		}
	}
	if w == nil || len(x) != w.Gate.Input || w.Gate.Experts <= 0 || w.ExpertUsed <= 0 || w.ExpertUsed > w.Gate.Experts ||
		w.Gate.Experts != w.Up.Experts || w.Gate.Experts != w.Down.Experts ||
		(len(w.RouterCorrectionBias) != 0 && len(w.RouterCorrectionBias) != w.Gate.Experts) ||
		groupCount <= 0 || groupUsed <= 0 || groupUsed > groupCount ||
		groupCount > w.Gate.Experts || w.Gate.Experts%groupCount != 0 ||
		w.ExpertUsed > (w.Gate.Experts/groupCount)*groupUsed ||
		(groupCount > 1 && !w.RoutingSigmoid) {
		panic("invalid sparse MoE weights")
	}
	w.Router.MatvecInto(x, &buf.RouterLogits)
	if len(buf.RouterLogits) != w.Gate.Experts {
		panic("sparse MoE router has an invalid shape")
	}
	if len(w.RouterBias) != 0 {
		addInPlace(buf.RouterLogits, w.RouterBias)
	}
	var selected []ExpertScore
	var routing []float32
	if w.RoutingSigmoid {
		ensureLenNoClear(&buf.RouterSelection, len(buf.RouterLogits))
		for i, logit := range buf.RouterLogits {
			score := nemotronSigmoid(logit)
			buf.RouterLogits[i] = score
			selection := score
			if len(w.RouterCorrectionBias) != 0 {
				selection += w.RouterCorrectionBias[i]
			}
			buf.RouterSelection[i] = selection
		}
		if groupCount > 1 {
			selected = selectDeepSeekGroupedExperts(buf.RouterSelection, groupCount, groupUsed, w.ExpertUsed, len(w.RouterCorrectionBias) != 0, &buf.RouterGroups, &buf.TopGroups, &buf.TopExperts)
		} else {
			selected = selectTopExperts(buf.RouterSelection, w.ExpertUsed, &buf.TopExperts)
		}
		routing = sparseMoESigmoidRoutingWeights(buf.RouterLogits, selected, w.NormalizeTopK, &buf.ExpertProbs)
	} else {
		selected = selectTopExperts(buf.RouterLogits, w.ExpertUsed, &buf.TopExperts)
		routing = sparseMoERoutingWeights(buf.RouterLogits, selected, w.NormalizeTopK, &buf.ExpertProbs)
	}
	ensureLenNoClear(&buf.Proj, w.Down.Output)
	clear(buf.Proj[:w.Down.Output])
	for i, choice := range selected {
		if !expertMatvec2Into(w.Gate, w.Up, choice.Index, x, &buf.Q4KXSums, &buf.Gate, &buf.Up) {
			expertMatvecInto(w.Gate, choice.Index, x, &buf.Gate, &buf.ExpertRow)
			expertMatvecInto(w.Up, choice.Index, x, &buf.Up, &buf.ExpertRow)
		}
		w.GateBias.addTo(choice.Index, buf.Gate)
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
	if w.SharedGateIn != nil || w.SharedGate != nil || w.SharedUp != nil || w.SharedDown != nil {
		if w.SharedGate == nil || w.SharedUp == nil || w.SharedDown == nil {
			panic("partial shared MoE expert")
		}
		if !w.SharedAlways && w.SharedGateIn == nil {
			panic("shared MoE expert is missing its gate")
		}
		w.SharedGate.MatvecInto(x, &buf.Gate)
		w.SharedUp.MatvecInto(x, &buf.Up)
		ensureLenNoClear(&buf.Hidden, len(buf.Gate))
		siluMulF32(buf.Gate, buf.Up, buf.Hidden)
		w.SharedDown.MatvecInto(buf.Hidden, &buf.MOE)
		if !w.SharedAlways {
			w.SharedGateIn.MatvecInto(x, &buf.RouterLogits)
			if len(buf.RouterLogits) != 1 {
				panic("shared MoE gate has an invalid shape")
			}
			ScaleF32(buf.MOE, nemotronSigmoid(buf.RouterLogits[0]))
		}
		addInPlace(buf.Proj[:w.Down.Output], buf.MOE)
	}
}

// expertMatvec2Into computes an expert's gate/up projections together when
// their layout permits it. Gate and up always share x, so one worker dispatch
// and one activation-sum preparation are enough for common Q4_K/Q6_K GGUFs.
// Other type pairs deliberately fall back to the independently optimized
// kernel rather than forcing an inferior generic path.
func expertMatvec2Into(a, b ExpertWeight, expert int, x []float32, q4Sums *[]float32, aOut, bOut *[]float32) bool {
	if expert < 0 || expert >= a.Experts || expert >= b.Experts || len(x) != a.Input || a.Input != b.Input || a.Output != b.Output {
		return false
	}
	if a.Weight.F32 != nil && b.Weight.F32 != nil {
		aData, okA := a.f32Plane(expert)
		bData, okB := b.f32Plane(expert)
		if !okA || !okB {
			return false
		}
		ensureLenNoClear(aOut, a.Output)
		ensureLenNoClear(bOut, b.Output)
		totalRows := a.Output + b.Output
		parallelRows(totalRows, func(start, end int) {
			if as, ae := clippedRange(start, end, 0, a.Output); as < ae {
				for r := as; r < ae; r++ {
					(*aOut)[r] = DotF32(aData[r*a.Input:(r+1)*a.Input], x)
				}
			}
			if bs, be := clippedRange(start, end, a.Output, totalRows); bs < be {
				for r := bs; r < be; r++ {
					br := r - a.Output
					(*bOut)[br] = DotF32(bData[br*b.Input:(br+1)*b.Input], x)
				}
			}
		})
		return true
	}
	if a.Weight.F32 != nil || b.Weight.F32 != nil || a.Weight.Type != b.Weight.Type {
		return false
	}
	aPlane, okA := expertPlaneWeight(a, expert)
	bPlane, okB := expertPlaneWeight(b, expert)
	if !okA || !okB {
		return false
	}
	switch aPlane.Type {
	case GGMLTypeQ4_K:
		if q4Sums == nil {
			scratch := []float32{}
			q4Sums = &scratch
		}
		return MatvecQ4K2IntoWithXSums(aPlane.Raw, aPlane.Rows, aPlane.Cols, bPlane.Raw, bPlane.Rows, bPlane.Cols, x, q4Sums, aOut, bOut)
	case GGMLTypeQ6_K:
		return MatvecQ6K2Into(aPlane.Raw, aPlane.Rows, aPlane.Cols, bPlane.Raw, bPlane.Rows, bPlane.Cols, x, aOut, bOut)
	default:
		return false
	}
}

func expertPlaneWeight(w ExpertWeight, expert int) (Weight, bool) {
	if expert < 0 || expert >= w.Experts || w.Weight.F32 != nil {
		return Weight{}, false
	}
	storageRows := w.storageRows()
	if w.Input <= 0 || w.Output <= 0 || storageRows <= 0 || w.RowOffset < 0 || w.RowOffset+w.Output > storageRows {
		return Weight{}, false
	}
	rowBytes, ok := w.Weight.Type.DataSize(w.Input)
	if !ok || rowBytes <= 0 {
		return Weight{}, false
	}
	storageBytes := storageRows * rowBytes
	planeBytes := w.Output * rowBytes
	start := expert*storageBytes + w.RowOffset*rowBytes
	if start < 0 || start+planeBytes < start || start+planeBytes > len(w.Weight.Raw) {
		return Weight{}, false
	}
	// Prepared and Metal descriptors refer to the complete rank-3 tensor's
	// first plane, so a selected expert must use this raw 2-D view instead.
	return Weight{Raw: w.Weight.Raw[start : start+planeBytes], Type: w.Weight.Type, Rows: w.Output, Cols: w.Input}, true
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
		plane, ok := w.f32Plane(expert)
		if !ok {
			panic("invalid F32 expert plane")
		}
		MatvecF32Into(plane, x, w.Output, w.Input, out)
		return
	}
	if plane, ok := expertPlaneWeight(w, expert); ok {
		plane.MatvecInto(x, out)
		return
	}
	ensureLenNoClear(row, w.Input)
	storageRows := w.storageRows()
	for r := range w.Output {
		w.Weight.RowInto(expert*storageRows+w.RowOffset+r, w.Input, row)
		(*out)[r] = DotF32(*row, x)
	}
}
