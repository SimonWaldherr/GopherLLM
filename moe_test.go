package gopherllm

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// forceExpertGateUpBatchDisabledForTest routes sparseMoEForward through the
// original per-expert gate/up path for the duration of the calling test,
// restoring the batched fast path afterward.
func forceExpertGateUpBatchDisabledForTest(t *testing.T) func() {
	t.Helper()
	disableExpertGateUpBatchForTest.Store(true)
	return func() { disableExpertGateUpBatchForTest.Store(false) }
}

func closeMoEFloat(t *testing.T, name string, got, want float32) {
	t.Helper()
	if d := math.Abs(float64(got - want)); d > 2e-5*math.Max(1, math.Abs(float64(want))) {
		t.Fatalf("%s = %g, want %g (delta %g)", name, got, want, d)
	}
}

func TestSparseMoERoutingWeights(t *testing.T) {
	logits := []float32{2, 1, 0}
	var selectedScratch []ExpertScore
	var routingScratch []float32
	selected := selectTopExperts(logits, 2, &selectedScratch)
	if len(selected) != 2 || selected[0].Index != 0 || selected[1].Index != 1 {
		t.Fatalf("selected = %#v, want experts 0 and 1", selected)
	}

	topK := sparseMoERoutingWeights(logits, selected, true, &routingScratch)
	denomTopK := float32(math.Exp(2) + math.Exp(1))
	closeMoEFloat(t, "top-k weight 0", topK[0], float32(math.Exp(2))/denomTopK)
	closeMoEFloat(t, "top-k weight 1", topK[1], float32(math.Exp(1))/denomTopK)
	closeMoEFloat(t, "top-k sum", topK[0]+topK[1], 1)

	full := sparseMoERoutingWeights(logits, selected, false, &routingScratch)
	denomFull := float32(math.Exp(2) + math.Exp(1) + 1)
	closeMoEFloat(t, "full-router weight 0", full[0], float32(math.Exp(2))/denomFull)
	closeMoEFloat(t, "full-router weight 1", full[1], float32(math.Exp(1))/denomFull)
	if full[0]+full[1] >= 1 {
		t.Fatalf("full-router selected mass = %g, want < 1", full[0]+full[1])
	}
	// Selection is stable when logits tie: the earlier expert remains chosen.
	ties := selectTopExperts([]float32{1, 1, 1}, 2, &selectedScratch)
	if ties[0].Index != 0 || ties[1].Index != 1 {
		t.Fatalf("tie selection = %#v, want experts 0 and 1", ties)
	}
}

func TestSparseMoEForwardTopKAndSharedExpert(t *testing.T) {
	// x selects experts 0 and 1. They produce orthogonal output vectors so
	// the mixture weights are directly observable in the result.
	w := &SparseMoEWeights{
		Router:        Weight{F32: []float32{1, 0, 0, 1, -1, 0}},
		NormalizeTopK: true,
		Scale:         1,
		ExpertUsed:    2,
		Gate: ExpertWeight{Weight: Weight{F32: []float32{
			1, 0, // expert 0
			1, 0, // expert 1
			1, 0, // expert 2
		}}, Input: 2, Output: 1, Experts: 3},
		Up: ExpertWeight{Weight: Weight{F32: []float32{
			1, 0, // expert 0
			1, 0, // expert 1
			1, 0, // expert 2
		}}, Input: 2, Output: 1, Experts: 3},
		Down: ExpertWeight{Weight: Weight{F32: []float32{
			1, 0, // expert 0
			0, 2, // expert 1
			9, 9, // expert 2 (not selected)
		}}, Input: 1, Output: 2, Experts: 3},
		SharedGateIn: &Weight{F32: []float32{0, 0}}, // sigmoid(0) = 0.5
		SharedGate:   &Weight{F32: []float32{1, 0}},
		SharedUp:     &Weight{F32: []float32{1, 0}},
		SharedDown:   &Weight{F32: []float32{1, 1}},
	}
	buf := &DecodeBuffer{}
	sparseMoEForward(w, []float32{1, 0}, buf)

	hidden := float32(1 / (1 + math.Exp(-1))) // SiLU(1)
	p0 := float32(math.Exp(1) / (math.Exp(1) + 1))
	p1 := 1 - p0
	// The shared branch adds 0.5 * SiLU(1) to both coordinates.
	closeMoEFloat(t, "mixture x", buf.Proj[0], p0*hidden+0.5*hidden)
	closeMoEFloat(t, "mixture y", buf.Proj[1], p1*2*hidden+0.5*hidden)
}

func TestSparseMoEForwardGPTOSSActivationAndBiases(t *testing.T) {
	// GPT-OSS clamps gate at +7 and up to [-7, +7], and applies its expert
	// biases before the OAI-SwiGLU activation. A zero input makes the bias
	// contribution unambiguous.
	w := &SparseMoEWeights{
		Router:        Weight{F32: []float32{0}},
		NormalizeTopK: true,
		Scale:         1,
		OAIActivation: true,
		ExpertUsed:    1,
		Gate:          ExpertWeight{Weight: Weight{F32: []float32{0}}, Input: 1, Output: 1, Experts: 1},
		Up:            ExpertWeight{Weight: Weight{F32: []float32{0}}, Input: 1, Output: 1, Experts: 1},
		Down:          ExpertWeight{Weight: Weight{F32: []float32{1}}, Input: 1, Output: 1, Experts: 1},
		GateBias:      ExpertBias{Values: []float32{10}, Output: 1, Experts: 1},
		UpBias:        ExpertBias{Values: []float32{20}, Output: 1, Experts: 1},
	}
	buf := &DecodeBuffer{}
	sparseMoEForward(w, []float32{0}, buf)
	want := float32(7 * (1 / (1 + math.Exp(-1.702*7))) * 8)
	closeMoEFloat(t, "GPT-OSS OAI-SwiGLU", buf.Proj[0], want)
}

func buildTinySparseMoEGGUF(arch string, shared bool, metadataExperts int) []byte {
	return buildTinySparseMoEGGUFWithExpertLayout(arch, shared, metadataExperts, true, false)
}

func buildTinySparseMoEGGUFWithExpertLayout(arch string, shared bool, metadataExperts int, separateGateUp, fusedGateUp bool) []byte {
	const (
		dim, heads, kv, hdim = 8, 2, 2, 4
		hidden, expertHidden = 16, 4
		experts, used, vocab = 3, 2, 16
	)
	if metadataExperts == 0 {
		metadataExperts = experts
	}
	toks := make([]any, vocab)
	scores := make([]any, vocab)
	for i := range toks {
		if i == 0 {
			toks[i] = "<unk>"
		} else if i == 1 {
			toks[i] = "<s>"
		} else if i == 2 {
			toks[i] = "</s>"
		} else {
			toks[i] = string(rune('a' + i - 3))
		}
		scores[i] = float32(0)
	}
	kvs := []ggufKV{
		{"general.architecture", ggufStr, arch},
		{"general.name", ggufStr, "tiny-sparse-moe"},
		{arch + ".embedding_length", ggufU32, uint32(dim)},
		{arch + ".block_count", ggufU32, uint32(1)},
		{arch + ".attention.head_count", ggufU32, uint32(heads)},
		{arch + ".attention.head_count_kv", ggufU32, uint32(kv)},
		{arch + ".attention.key_length", ggufU32, uint32(hdim)},
		{arch + ".attention.value_length", ggufU32, uint32(hdim)},
		{arch + ".feed_forward_length", ggufU32, uint32(hidden)},
		{arch + ".context_length", ggufU32, uint32(32)},
		{arch + ".attention.layer_norm_rms_epsilon", ggufF32, float32(1e-5)},
		{arch + ".rope.freq_base", ggufF32, float32(10000)},
		{arch + ".rope.dimension_count", ggufU32, uint32(hdim)},
		{arch + ".expert_count", ggufU32, uint32(metadataExperts)},
		{arch + ".expert_used_count", ggufU32, uint32(used)},
		{"tokenizer.ggml.model", ggufStr, "llama"},
		{"tokenizer.ggml.tokens", ggufArr, ggufArray{ggufStr, toks}},
		{"tokenizer.ggml.scores", ggufArr, ggufArray{ggufF32, scores}},
		{"tokenizer.ggml.bos_token_id", ggufU32, uint32(1)},
		{"tokenizer.ggml.eos_token_id", ggufU32, uint32(2)},
	}
	if arch == "gpt-oss" {
		kvs = append(kvs, ggufKV{arch + ".attention.sliding_window", ggufU32, uint32(4)})
	}
	f32t := func(name string, rows, cols, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(cols), uint64(rows)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(rows*cols, seed))}
	}
	vec := func(name string, n, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(n)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(n, seed))}
	}
	expert := func(name string, input, output, count, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(input), uint64(output), uint64(count)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(input*output*count, seed))}
	}
	fusedExpert := func(name string, input, output, count, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(input), uint64(2 * output), uint64(count)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(input*2*output*count, seed))}
	}
	expertBias := func(name string, output, count, seed int) ggufTensor {
		return ggufTensor{name: name, dims: []uint64{uint64(output), uint64(count)}, dtype: GGMLTypeF32, data: f32Bytes(smallWeights(output*count, seed))}
	}
	tensors := []ggufTensor{
		f32t("token_embd.weight", vocab, dim, 1),
		vec("output_norm.weight", dim, 2),
		f32t("output.weight", vocab, dim, 3),
		vec("blk.0.attn_norm.weight", dim, 4),
		f32t("blk.0.attn_q.weight", heads*hdim, dim, 5),
		f32t("blk.0.attn_k.weight", kv*hdim, dim, 6),
		f32t("blk.0.attn_v.weight", kv*hdim, dim, 7),
		f32t("blk.0.attn_output.weight", dim, heads*hdim, 8),
		vec("blk.0.ffn_norm.weight", dim, 9),
		f32t("blk.0.ffn_gate_inp.weight", experts, dim, 10),
		expert("blk.0.ffn_down_exps.weight", expertHidden, dim, experts, 13),
	}
	if separateGateUp {
		tensors = append(tensors,
			expert("blk.0.ffn_gate_exps.weight", dim, expertHidden, experts, 11),
			expert("blk.0.ffn_up_exps.weight", dim, expertHidden, experts, 12),
		)
	}
	if fusedGateUp {
		tensors = append(tensors, fusedExpert("blk.0.ffn_gate_up_exps.weight", dim, expertHidden, experts, 26))
	}
	if arch == "qwen3moe" {
		tensors = append(tensors, vec("blk.0.attn_q_norm.weight", hdim, 14), vec("blk.0.attn_k_norm.weight", hdim, 15))
	}
	if shared {
		tensors = append(tensors,
			vec("blk.0.ffn_gate_inp_shexp.weight", dim, 16),
			f32t("blk.0.ffn_gate_shexp.weight", expertHidden, dim, 17),
			f32t("blk.0.ffn_up_shexp.weight", expertHidden, dim, 18),
			f32t("blk.0.ffn_down_shexp.weight", dim, expertHidden, 19),
		)
	}
	if arch == "gpt-oss" {
		tensors = append(tensors,
			vec("blk.0.attn_sinks.weight", heads, 20),
			vec("blk.0.attn_output.bias", dim, 21),
			vec("blk.0.ffn_gate_inp.bias", experts, 22),
			expertBias("blk.0.ffn_gate_exps.bias", expertHidden, experts, 23),
			expertBias("blk.0.ffn_up_exps.bias", expertHidden, experts, 24),
			expertBias("blk.0.ffn_down_exps.bias", dim, experts, 25),
		)
	}
	return buildGGUF(3, kvs, tensors)
}

func TestSparseMoEGGUFFamiliesLoadAndRun(t *testing.T) {
	tests := []struct {
		arch, name string
		shared     bool
		normalize  bool
	}{
		{"llama", "llama_declared_mixtral", false, true},
		{"mixtral", "mixtral", false, true},
		{"qwen2moe", "qwen2_shared", true, false},
		{"qwen3moe", "qwen3_qknorm", false, true},
		{"gpt-oss", "gpt_oss", false, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r, err := RunnerFromGGUFBytes(buildTinySparseMoEGGUF(tc.arch, tc.shared, 0))
			if err != nil {
				t.Fatal(err)
			}
			if tc.arch == "gpt-oss" {
				if r.kind != loadedGptOss {
					t.Fatalf("kind = %d, want GPT-OSS", r.kind)
				}
			} else if r.kind != loadedStandard {
				t.Fatalf("kind = %d, want standard", r.kind)
			}
			var layer LayerWeights
			if tc.arch == "gpt-oss" {
				layer = r.gptOss.Standard.Layers[0]
			} else {
				layer = r.standard.Layers[0]
			}
			if layer.MoE == nil {
				t.Fatal("sparse MoE weights were not loaded")
			}
			if layer.MoE.NormalizeTopK != tc.normalize {
				t.Fatalf("NormalizeTopK = %t, want %t", layer.MoE.NormalizeTopK, tc.normalize)
			}
			if tc.shared && (layer.MoE.SharedGateIn == nil || layer.MoE.SharedGate == nil || layer.MoE.SharedUp == nil || layer.MoE.SharedDown == nil) {
				t.Fatal("Qwen2 shared expert was not loaded")
			}
			if tc.arch == "qwen3moe" && (layer.AttnQNorm == nil || layer.AttnKNorm == nil) {
				t.Fatal("Qwen3-MoE QK norms were not loaded")
			}
			if tc.arch == "gpt-oss" {
				if !layer.MoE.OAIActivation || len(layer.AttnSinks) != r.config.NHeads || len(layer.BO) != r.config.Dim || len(layer.MoE.RouterBias) != r.config.ExpertCount || len(layer.MoE.GateBias.Values) == 0 {
					t.Fatal("GPT-OSS biases, sinks, or activation were not loaded")
				}
				if got := r.config.SWAPattern; len(got) != 1 || !got[0] {
					t.Fatalf("GPT-OSS SWA pattern = %v, want [true]", got)
				}
			}
			if r.canBatchPrefill() {
				t.Fatal("sparse MoE must use the sequential prefill path")
			}
			cache, buf := r.generationWorkspace(4)
			var logits []float32
			for pos, token := range []uint32{1, 3, 4} {
				r.forwardTokenInto(cache, buf, token, pos, &logits)
			}
			if len(logits) != r.config.VocabSize {
				t.Fatalf("logits len = %d, want %d", len(logits), r.config.VocabSize)
			}
			for i, v := range logits {
				if !finite32(v) {
					t.Fatalf("logit %d is non-finite: %v", i, v)
				}
			}
		})
	}
}

func TestSparseMoEGGUFFusedGateUpLoadsAndRunsOutOfCore(t *testing.T) {
	data := buildTinySparseMoEGGUFWithExpertLayout("llama", false, 0, false, true)
	r, err := RunnerFromGGUFBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	moe := r.standard.Layers[0].MoE
	if moe == nil {
		t.Fatal("fused sparse MoE weights were not loaded")
	}
	if moe.Gate.StorageOutput != 2*moe.Gate.Output || moe.Gate.RowOffset != 0 || moe.Up.RowOffset != moe.Gate.Output {
		t.Fatalf("fused expert views = gate(%+v) up(%+v), want contiguous gate/up halves", moe.Gate, moe.Up)
	}
	if moe.Gate.Weight.F32 == nil || moe.Up.Weight.F32 == nil || &moe.Gate.Weight.F32[0] != &moe.Up.Weight.F32[0] {
		t.Fatal("fused gate/up weights were copied instead of sharing their backing tensor")
	}

	// Exercise the out-of-core path as well: scalar F32 expert weights must
	// remain raw mmap-style bytes while each view addresses its own rows.
	gguf, err := ParseGGUFQuiet(data)
	if err != nil {
		t.Fatal(err)
	}
	cfg := ConfigFromGGUF(gguf)
	lazy, err := loadSparseMoEWeights(data, gguf.DataOffset, "blk.0.", cfg, indexTensors(gguf), inferTensorSizes(data, gguf), true, false, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if lazy.Gate.Weight.F32 != nil || lazy.Up.Weight.F32 != nil || len(lazy.Gate.Weight.Raw) == 0 || &lazy.Gate.Weight.Raw[0] != &lazy.Up.Weight.Raw[0] {
		t.Fatal("out-of-core fused gate/up weights did not retain one shared raw backing tensor")
	}
	buf := &DecodeBuffer{}
	sparseMoEForward(lazy, onesF32(cfg.Dim), buf)
	if len(buf.Proj) != cfg.Dim {
		t.Fatalf("out-of-core fused MoE output len = %d, want %d", len(buf.Proj), cfg.Dim)
	}
	for i, v := range buf.Proj {
		if !finite32(v) {
			t.Fatalf("out-of-core fused MoE output %d is non-finite: %v", i, v)
		}
	}
}

func TestSparseMoELoaderPrefersSeparateGateUpOverFused(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinySparseMoEGGUFWithExpertLayout("llama", false, 0, true, true))
	if err != nil {
		t.Fatal(err)
	}
	moe := r.standard.Layers[0].MoE
	if moe == nil {
		t.Fatal("sparse MoE weights were not loaded")
	}
	if moe.Gate.StorageOutput != 0 || moe.Up.StorageOutput != 0 || moe.Up.RowOffset != 0 {
		t.Fatalf("separate gate/up tensors were not preferred: gate=%+v up=%+v", moe.Gate, moe.Up)
	}
}

func fusedExpertReference(w ExpertWeight, expert int, x []float32) []float32 {
	out := make([]float32, w.Output)
	row := make([]float32, w.Input)
	for r := range w.Output {
		w.Weight.RowInto(expert*w.storageRows()+w.RowOffset+r, w.Input, &row)
		out[r] = DotF32(row, x)
	}
	return out
}

func TestFusedExpertWeightViewsAddressEachExpertPlane(t *testing.T) {
	const (
		input, output, experts = 2, 2, 2
	)
	// Each expert plane is [gate rows | up rows], and every row has a unique
	// dot product so a wrong plane stride or row offset is immediately visible.
	backing := []float32{
		1, 2, 3, 4, 5, 6, 7, 8, // expert 0
		9, 10, 11, 12, 13, 14, 15, 16, // expert 1
	}
	gate := ExpertWeight{
		Weight:        Weight{F32: backing},
		Input:         input,
		Output:        output,
		Experts:       experts,
		StorageOutput: 2 * output,
	}
	up := gate
	up.RowOffset = output
	x := []float32{2, -1}
	var gotGate, gotUp, sums []float32
	if !expertMatvec2Into(gate, up, 1, x, &sums, &gotGate, &gotUp) {
		t.Fatal("fused F32 expert pair path declined valid views")
	}
	for _, tc := range []struct {
		name string
		got  []float32
		want []float32
	}{
		{"gate", gotGate, []float32{8, 10}},
		{"up", gotUp, []float32{12, 14}},
	} {
		for i := range tc.want {
			closeMoEFloat(t, tc.name, tc.got[i], tc.want[i])
		}
	}

	// The same geometry must work for a raw scalar view, which is how F32/F16
	// out-of-core model mappings are represented.
	rawGate := gate
	rawGate.Weight = Weight{Raw: f32Bytes(backing), Type: GGMLTypeF32, Rows: 2 * output, Cols: input}
	rawUp := rawGate
	rawUp.RowOffset = output
	gotGate = nil
	gotUp = nil
	var row []float32
	expertMatvecInto(rawGate, 1, x, &gotGate, &row)
	expertMatvecInto(rawUp, 1, x, &gotUp, &row)
	for _, tc := range []struct {
		name string
		got  []float32
		want []float32
	}{
		{"raw gate", gotGate, []float32{8, 10}},
		{"raw up", gotUp, []float32{12, 14}},
	} {
		for i := range tc.want {
			closeMoEFloat(t, tc.name, tc.got[i], tc.want[i])
		}
	}
}

func TestFusedExpertWeightQuantizedViewsAddressEachExpertPlane(t *testing.T) {
	const (
		input, output, experts = 512, 8, 3
	)
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		t.Run(typ.String(), func(t *testing.T) {
			// Construct the physical [input, 2*output, expert] tensor directly;
			// gate and up then become zero-copy row views into each expert plane.
			fused := quantExpertWeightForTest(typ, input, 2*output, experts, int64(1000+typ))
			gate := fused
			gate.Output = output
			gate.StorageOutput = 2 * output
			up := gate
			up.RowOffset = output
			if len(gate.Weight.Raw) == 0 || &gate.Weight.Raw[0] != &up.Weight.Raw[0] {
				t.Fatal("fused quantized views do not share raw storage")
			}
			x := randomExpertInput(input, int64(2000+typ))
			wantGate := fusedExpertReference(gate, experts-1, x)
			wantUp := fusedExpertReference(up, experts-1, x)
			var gotGate, gotUp, sums []float32
			withQ8Activations(false, func() {
				if !expertMatvec2Into(gate, up, experts-1, x, &sums, &gotGate, &gotUp) {
					t.Fatalf("fused %s expert pair path declined valid views", typ)
				}
			})
			for _, tc := range []struct {
				name string
				got  []float32
				want []float32
			}{
				{"gate", gotGate, wantGate},
				{"up", gotUp, wantUp},
			} {
				for i := range tc.want {
					diff := math.Abs(float64(tc.got[i] - tc.want[i]))
					limit := 1e-2 * math.Max(1, math.Abs(float64(tc.want[i])))
					if diff > limit {
						t.Fatalf("%s fused %s row %d = %v, want %v (diff %g, limit %g)", typ, tc.name, i, tc.got[i], tc.want[i], diff, limit)
					}
				}
			}
		})
	}
}

// TestExpertMatvecGateUpBatchMatchesPerExpert checks that coalescing every
// selected expert's gate/up projection into one dispatch
// (expertMatvecGateUpBatchInto) produces bit-identical results to calling
// expertMatvec2Into once per expert, for both the float and int8-activation
// quantized paths. Rows/experts are sized well past the parallelRows
// oversubscription threshold so chunk boundaries actually land inside and
// across per-expert segments, not just at their edges.
func TestExpertMatvecGateUpBatchMatchesPerExpert(t *testing.T) {
	const (
		input, output, experts = 512, 96, 8
	)
	selected := []ExpertScore{{Index: 0}, {Index: experts - 1}, {Index: 3}, {Index: 1}, {Index: 5}}
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		t.Run(typ.String(), func(t *testing.T) {
			gate := quantExpertWeightForTest(typ, input, output, experts, int64(5000+typ))
			up := quantExpertWeightForTest(typ, input, output, experts, int64(6000+typ))
			x := randomExpertInput(input, int64(7000+typ))
			for _, q8 := range []bool{false, true} {
				t.Run(map[bool]string{true: "q8activations", false: "float"}[q8], func(t *testing.T) {
					withQ8Activations(q8, func() {
						var gotGate, gotUp, sums []float32
						if !expertMatvecGateUpBatchInto(gate, up, selected, x, &sums, &gotGate, &gotUp) {
							t.Fatalf("batched gate/up path declined %s", typ)
						}
						for j, choice := range selected {
							var wantGate, wantUp, refSums []float32
							if !expertMatvec2Into(gate, up, choice.Index, x, &refSums, &wantGate, &wantUp) {
								t.Fatalf("reference per-expert path declined %s", typ)
							}
							for i := range wantGate {
								if got, want := gotGate[j*output+i], wantGate[i]; got != want {
									t.Fatalf("expert %d gate row %d = %v, want %v (exact)", choice.Index, i, got, want)
								}
							}
							for i := range wantUp {
								if got, want := gotUp[j*output+i], wantUp[i]; got != want {
									t.Fatalf("expert %d up row %d = %v, want %v (exact)", choice.Index, i, got, want)
								}
							}
						}
					})
				})
			}
		})
	}
}

// TestSparseMoEForwardBatchedGateUpMatchesUnbatched drives the full
// sparseMoEForward path (router, top-k, SwiGLU, down projection, weighted
// sum) end to end on quantized Q4_K/Q6_K expert weights, comparing the
// batched-gate/up code path against a reference run that forces the original
// per-expert path. Confirms the optimization is invisible from the forward
// function's actual output, not just from the isolated matvec it changes.
func TestSparseMoEForwardBatchedGateUpMatchesUnbatched(t *testing.T) {
	const (
		dim, hidden, experts, used = 128, 96, 8, 4
	)
	rng := rand.New(rand.NewSource(4242))
	fillF32 := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = (rng.Float32()*2 - 1) * 0.05
		}
		return v
	}
	for _, typ := range []GGMLType{GGMLTypeQ4_K, GGMLTypeQ6_K} {
		t.Run(typ.String(), func(t *testing.T) {
			w := &SparseMoEWeights{
				Router:        Weight{F32: fillF32(experts * dim)},
				Gate:          quantExpertWeightForTest(typ, dim, hidden, experts, int64(8000+typ)),
				Up:            quantExpertWeightForTest(typ, dim, hidden, experts, int64(9000+typ)),
				Down:          quantExpertWeightForTest(typ, hidden, dim, experts, int64(10000+typ)),
				NormalizeTopK: true,
				Scale:         1,
				ExpertUsed:    used,
			}
			x := randomExpertInput(dim, int64(11000+typ))
			for _, q8 := range []bool{false, true} {
				t.Run(map[bool]string{true: "q8activations", false: "float"}[q8], func(t *testing.T) {
					withQ8Activations(q8, func() {
						batchedBuf := &DecodeBuffer{}
						sparseMoEForward(w, x, batchedBuf)

						refBuf := &DecodeBuffer{}
						restore := forceExpertGateUpBatchDisabledForTest(t)
						defer restore()
						sparseMoEForward(w, x, refBuf)

						if len(batchedBuf.Proj) != len(refBuf.Proj) {
							t.Fatalf("Proj length = %d, want %d", len(batchedBuf.Proj), len(refBuf.Proj))
						}
						for i := range refBuf.Proj {
							if got, want := batchedBuf.Proj[i], refBuf.Proj[i]; got != want {
								t.Fatalf("Proj[%d] = %v, want %v (exact)", i, got, want)
							}
						}
					})
				})
			}
		})
	}
}

func TestSparseMoELoaderRejectsMismatchedExpertGeometry(t *testing.T) {
	_, err := RunnerFromGGUFBytes(buildTinySparseMoEGGUF("llama", false, 4))
	if err == nil || !strings.Contains(err.Error(), "ffn_gate_inp.weight") {
		t.Fatalf("error = %v, want router geometry diagnostic", err)
	}
}

func TestQwen2MoELoaderRejectsMissingSharedExpert(t *testing.T) {
	_, err := RunnerFromGGUFBytes(buildTinySparseMoEGGUF("qwen2moe", false, 0))
	if err == nil || !strings.Contains(err.Error(), "qwen2moe requires") {
		t.Fatalf("error = %v, want missing shared-expert diagnostic", err)
	}
}

func TestAttentionSinkChangesSoftmaxDenominator(t *testing.T) {
	values := []float32{2}
	without := []float32{0}
	with := []float32{0}
	weightedVSumWithSink([]float32{0}, values, 1, 1, 0, 0, 0, false, without)
	weightedVSumWithSink([]float32{0}, values, 1, 1, 0, 0, 0, true, with)
	closeMoEFloat(t, "no sink", without[0], 2)
	closeMoEFloat(t, "zero-logit sink", with[0], 1)
	f16With := []float32{0}
	onlineAttentionF16WithSink([]float32{0}, []uint16{F32ToF16(0)}, []uint16{F32ToF16(2)}, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, true, f16With)
	closeMoEFloat(t, "f16 zero-logit sink", f16With[0], 1)

	pattern := swaPattern(&GGUFFile{}, "gpt-oss", "gpt-oss", 4)
	want := []bool{true, false, true, false}
	if len(pattern) != len(want) {
		t.Fatalf("GPT-OSS SWA pattern len = %d, want %d", len(pattern), len(want))
	}
	for i := range want {
		if pattern[i] != want[i] {
			t.Fatalf("GPT-OSS SWA pattern = %v, want %v", pattern, want)
		}
	}
}
