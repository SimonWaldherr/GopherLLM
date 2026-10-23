//go:build coreml

package gopherllm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	nativecoreml "github.com/SimonWaldherr/GopherLLM/internal/coreml"
)

type fakeCoreMLFFN struct {
	info    nativecoreml.FFNInfo
	calls   int
	closed  bool
	predict func([]float32, []float32) error
}

func (f *fakeCoreMLFFN) Info() nativecoreml.FFNInfo { return f.info }
func (f *fakeCoreMLFFN) Close() error               { f.closed = true; return nil }
func (f *fakeCoreMLFFN) Predict(input, out []float32) error {
	f.calls++
	if f.predict != nil {
		return f.predict(input, out)
	}
	copy(out, input)
	return nil
}

func coreMLFFNManifestFixture(t *testing.T) (*Runner, coreMLFFNManifest, string) {
	t.Helper()
	r := &Runner{arch: "ministral", kind: loadedStandard, config: Config{Dim: 4, HiddenDim: 8, NLayers: 1}}
	weight := func(rows, cols int, seed byte) Weight {
		raw := make([]byte, rows*cols*2)
		for i := range raw {
			raw[i] = byte(i) + seed
		}
		return Weight{Raw: raw, Rows: rows, Cols: cols, Type: GGMLTypeF16}
	}
	l := LayerWeights{W1: weight(8, 4, 1), W3: weight(8, 4, 2), W2: weight(4, 8, 3)}
	r.standard.Layers = []LayerWeights{l}
	m := coreMLFFNManifest{Version: 1, Architecture: r.arch, Layer: 0, Dimension: 4, HiddenDimension: 8,
		Tokens: 2, Input: "hidden", Output: "output", ComputePrecision: "float16", Compression: "none", Activation: "silu_exp", ActivationScale: 16, Model: "ffn.mlmodelc"}
	for i, w := range []Weight{l.W1, l.W3, l.W2} {
		digest := sha256.Sum256(w.Raw)
		entry := coreMLFFNWeightBinding{Name: "blk.0.ffn_" + []string{"gate", "up", "down"}[i] + ".weight", Shape: []int{w.Rows, w.Cols},
			TensorType: int(w.Type), SHA256: hex.EncodeToString(digest[:])}
		m.Weights = append(m.Weights, entry)
	}
	return r, m, filepath.Join(t.TempDir(), "ffn.json")
}

func writeCoreMLFFNManifest(t *testing.T, path string, m coreMLFFNManifest) {
	t.Helper()
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
}

func fakeCoreMLFFNForSource(t *testing.T, r *Runner, m coreMLFFNManifest) *fakeCoreMLFFN {
	t.Helper()
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	gate, up, down, err := r.coreMLFFNWeights(m.Layer)
	if err != nil {
		t.Fatal(err)
	}
	f := &fakeCoreMLFFN{info: nativecoreml.FFNInfo{Tokens: m.Tokens, Dimension: m.Dimension, Manifest: string(data)}}
	f.predict = func(input, out []float32) error {
		project := func(w Weight, x []float32, rows int) []float32 {
			values := make([]float32, rows)
			var row []float32
			for i := range values {
				w.RowInto(i, len(x), &row)
				values[i] = DotF32(row, x)
			}
			return values
		}
		for start := 0; start < len(input); start += m.Dimension {
			x := input[start : start+m.Dimension]
			a, b := project(gate, x, m.HiddenDimension), project(up, x, m.HiddenDimension)
			hidden := make([]float32, m.HiddenDimension)
			siluMulF32(a, b, hidden)
			copy(out[start:start+m.Dimension], project(down, hidden, m.Dimension))
		}
		return nil
	}
	return f
}

func TestCoreMLFFNManifestRejectsWrongSource(t *testing.T) {
	for _, change := range []string{"hash", "type", "shape", "duplicate", "architecture", "layer", "bias", "fused", "moe"} {
		t.Run(change, func(t *testing.T) {
			r, m, _ := coreMLFFNManifestFixture(t)
			if err := r.validateCoreMLFFNManifest(m); err != nil {
				t.Fatal(err)
			}
			switch change {
			case "hash":
				m.Weights[0].SHA256 = strings.Repeat("0", 64)
			case "type":
				m.Weights[0].TensorType++
			case "shape":
				m.Weights[0].Shape[0]++
			case "duplicate":
				m.Weights[1] = m.Weights[0]
			case "architecture":
				m.Architecture = "qwen35"
			case "layer":
				m.Layer++
			case "bias":
				r.standard.Layers[0].FFNUpBias = []float32{0}
			case "fused":
				r.standard.Layers[0].HasGateUp = true
			case "moe":
				r.standard.Layers[0].MoE = &SparseMoEWeights{}
			}
			if err := r.validateCoreMLFFNManifest(m); err == nil {
				t.Fatal("mismatched source accepted")
			}
		})
	}
}

func TestCoreMLFFNReplacementIsAtomicAndInvalidatesCaches(t *testing.T) {
	r, m, path := coreMLFFNManifestFixture(t)
	writeCoreMLFFNManifest(t, path, m)
	old := fakeCoreMLFFNForSource(t, r, m)
	r.coreMLFFN = coreMLFFNSet{0: {model: old, info: old.info}}
	r.prefixCache.tokens = []uint32{1}
	r.conversations.entries = []conversationSnapshot{{tokens: []uint32{1}}}
	opened := fakeCoreMLFFNForSource(t, r, m)
	open := func(string, nativecoreml.ComputeUnits) (coreMLFFNPredictor, error) { return opened, nil }
	// The first load succeeds, then a duplicated layer rejects the replacement.
	if err := r.setCoreMLFFN([]string{path, path}, CoreMLFFNOptions{}, open); err == nil {
		t.Fatal("duplicate accepted")
	}
	if !opened.closed || old.closed || r.coreMLFFN[0].model != old || len(r.prefixCache.tokens) == 0 {
		t.Fatal("failed replacement changed the active model/cache or leaked a new model")
	}
	opened = fakeCoreMLFFNForSource(t, r, m)
	if err := r.setCoreMLFFN([]string{path}, CoreMLFFNOptions{}, open); err != nil {
		t.Fatal(err)
	}
	if !old.closed || opened.closed || len(r.prefixCache.tokens) != 0 || len(r.conversations.entries) != 0 {
		t.Fatal("replacement did not release old weights and invalidate cached arithmetic")
	}
	if err := r.SetCoreMLFFN(nil, CoreMLFFNOptions{}); err != nil || !opened.closed || len(r.coreMLFFN) != 0 {
		t.Fatalf("disable did not release the model: %v", err)
	}
	_ = r.Close()
	if err := r.SetCoreMLFFN(nil, CoreMLFFNOptions{}); !errors.Is(err, ErrRunnerClosed) {
		t.Fatalf("closed runner error: %v", err)
	}
}

func TestCoreMLFFNRejectsMislabeledCompiledModelAndBadPrecision(t *testing.T) {
	for _, invalid := range []string{"missing", "different_graph", "trailing", "numerics"} {
		t.Run(invalid, func(t *testing.T) {
			r, m, path := coreMLFFNManifestFixture(t)
			writeCoreMLFFNManifest(t, path, m)
			fake := fakeCoreMLFFNForSource(t, r, m)
			switch invalid {
			case "missing":
				fake.info.Manifest = ""
			case "different_graph":
				m.HiddenChunk = 4
				data, _ := json.Marshal(m)
				fake.info.Manifest = string(data)
			case "trailing":
				fake.info.Manifest += " {}"
			case "numerics":
				fake.predict = func(_, out []float32) error { clear(out); return nil }
			}
			if err := r.setCoreMLFFN([]string{path}, CoreMLFFNOptions{}, func(string, nativecoreml.ComputeUnits) (coreMLFFNPredictor, error) {
				return fake, nil
			}); err == nil || !fake.closed || len(r.coreMLFFN) != 0 {
				t.Fatalf("invalid compiled model was accepted or leaked: %v", err)
			}
		})
	}
}

func TestCoreMLFFNOnlyRunsCompleteBlocksAndPropagatesFailures(t *testing.T) {
	fake := &fakeCoreMLFFN{info: nativecoreml.FFNInfo{Tokens: 2, Dimension: 4}}
	buf := &DecodeBuffer{coreMLFFN: coreMLFFNSet{0: {model: fake, info: fake.info}}}
	for _, tokens := range []int{0, 1, 3} {
		if handled, err := tryCoreMLFFN(context.Background(), buf, 0, make([]float32, tokens*4), make([]float32, tokens*4), tokens); err != nil || handled {
			t.Fatalf("ineligible chunk ran: %d, %v", tokens, err)
		}
	}
	in, out := make([]float32, 24), make([]float32, 24)
	for i := range in {
		in[i] = float32(i)
	}
	if ok, err := tryCoreMLFFN(context.Background(), buf, 0, in, out, 6); !ok || err != nil || fake.calls != 3 {
		t.Fatalf("block execution: ok=%v calls=%d err=%v", ok, fake.calls, err)
	}
	for i := range out {
		if out[i] != in[i] {
			t.Fatalf("row mixing at %d", i)
		}
	}
	want := errors.New("native failure")
	fake.predict = func([]float32, []float32) error { return want }
	if _, err := tryCoreMLFFN(context.Background(), buf, 0, in, out, 6); !errors.Is(err, want) {
		t.Fatalf("native error swallowed: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := tryCoreMLFFN(ctx, buf, 0, in, out, 6); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error swallowed: %v", err)
	}
}

func TestCoreMLFFNQwenPrefillScopeAndReference(t *testing.T) {
	cfg, weights := qwen35BatchFixture(false)
	r := &Runner{arch: "qwen35", kind: loadedQwen35, config: cfg, qwen35: weights}
	ffn := weights.Layers[0].FFN
	fake := &fakeCoreMLFFN{info: nativecoreml.FFNInfo{Tokens: 2, Dimension: cfg.Dim}}
	fake.predict = func(input, out []float32) error {
		for start := 0; start < len(input); start += cfg.Dim {
			gate, up := ffn.Gate.Matvec(input[start:start+cfg.Dim]), ffn.Up.Matvec(input[start:start+cfg.Dim])
			hidden := make([]float32, len(gate))
			siluMulF32(gate, up, hidden)
			row := out[start : start+cfg.Dim]
			ffn.Down.MatvecInto(hidden, &row)
		}
		return nil
	}
	tokens := []uint32{1, 2, 3, 4}
	wantCache, gotCache := qwen35BatchCache(cfg, weights, kvF32), qwen35BatchCache(cfg, weights, kvF32)
	wantBuf, gotBuf := NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim), NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim)
	var want, got []float32
	if err := r.prefillQwen35At(context.Background(), wantCache, wantBuf, tokens, 0, &want); err != nil {
		t.Fatal(err)
	}
	r.coreMLFFN = coreMLFFNSet{0: {model: fake, info: fake.info}}
	if err := r.prefillQwen35At(context.Background(), gotCache, gotBuf, tokens, 0, &got); err != nil {
		t.Fatal(err)
	}
	if fake.calls != 2 || gotBuf.coreMLFFN != nil {
		t.Fatalf("prefill scope/calls incorrect: %d", fake.calls)
	}
	requireQwen35BatchValuesWithin(t, "Core ML substituted FFN logits", got, want, 1e-6)
	ForwardQwen35Into(cfg, weights, gotCache, gotBuf, 5, len(tokens), &got)
	if fake.calls != 2 {
		t.Fatal("ordinary decode executed a prefill FFN")
	}
}

func TestCoreMLFFNStandardPrefillAndVerificationScope(t *testing.T) {
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	cfg := r.config
	m := coreMLFFNManifest{Layer: 0, Tokens: 2, Dimension: cfg.Dim, HiddenDimension: cfg.HiddenDim}
	fake := fakeCoreMLFFNForSource(t, r, m)
	tokens := []uint32{1, 2, 3, 4, 5, 6}
	cache := func() *KVCache {
		return NewKVCache(cfg.NLayers, cfg.NKVHeads*cfg.HeadDim, cfg.NKVHeads*cfg.ValueDim, 8)
	}
	buffer := func() *DecodeBuffer { return NewDecodeBuffer(cfg, cfg.HeadDim, cfg.NKVHeads, cfg.ValueDim) }
	wantBuf, gotBuf, wantCache, gotCache := buffer(), buffer(), cache(), cache()
	var want, got []float32
	if err := r.prefillBatchedAt(context.Background(), wantCache, wantBuf, tokens, 0, &want); err != nil {
		t.Fatal(err)
	}
	r.coreMLFFN = coreMLFFNSet{0: {model: fake, info: fake.info}}
	if err := r.prefillBatchedAt(context.Background(), gotCache, gotBuf, tokens, 0, &got); err != nil {
		t.Fatal(err)
	}
	if fake.calls != 3 || gotBuf.coreMLFFN != nil {
		t.Fatalf("standard prefill scope/calls: %d", fake.calls)
	}
	requireQwen35BatchValuesWithin(t, "standard FFN logits", got, want, 1e-6)
	if _, err := forwardBatchStatesIntoContext(context.Background(), cfg, r.standard, gotCache, gotBuf, []uint32{7, 8}, 6); err != nil {
		t.Fatal(err)
	}
	if fake.calls != 3 {
		t.Fatal("verification used approximate prefill arithmetic")
	}
	fake.predict = func(_, out []float32) error {
		for i := range out {
			out[i] = -999
		}
		return nativecoreml.ErrFFNNonFiniteOutput
	}
	if err := r.prefillBatchedAt(context.Background(), gotCache, gotBuf, tokens, 0, &got); err != nil {
		t.Fatal(err)
	}
	requireQwen35BatchValuesWithin(t, "FP16 range fallback logits", got, want, 1e-6)
	fake.predict = func([]float32, []float32) error { return errors.New("injected native failure") }
	if err := r.prefillBatchedAt(context.Background(), gotCache, gotBuf, tokens, 0, &got); err == nil || gotBuf.coreMLFFN != nil {
		t.Fatalf("failed prefill kept the scope or swallowed failure: %v", err)
	}
}

func TestCoreMLFFNMappedScalarSourceBinding(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, buildTinyLlamaGGUF(), 0600); err != nil {
		t.Fatal(err)
	}
	r, _, err := RunnerFromPathWithOptions(path, LoadOptions{Prefault: MmapPrefaultNone})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	// The binding uses tensor storage metadata, rather than the F32 Weight's
	// omitted Rows/Cols/Type. Architecture eligibility is independent of that.
	r.arch = "ministral"
	m := coreMLFFNManifest{Architecture: r.arch, Layer: 0, Tokens: 2, Dimension: r.config.Dim, HiddenDimension: r.config.HiddenDim}
	for _, tensor := range r.gguf.Tensors {
		if tensor.Name != "blk.0.ffn_gate.weight" && tensor.Name != "blk.0.ffn_up.weight" && tensor.Name != "blk.0.ffn_down.weight" {
			continue
		}
		n, ok := tensor.DType.DataSize(tensor.Numel())
		if !ok {
			t.Fatal("fixture has no scalar storage size")
		}
		data, offset := tensorContainer(r.mappedFile.Bytes(), r.gguf.DataOffset, tensor)
		digest := sha256.Sum256(data[offset : offset+n])
		m.Weights = append(m.Weights, coreMLFFNWeightBinding{tensor.Name, []int{int(tensor.Dims[1]), int(tensor.Dims[0])}, int(tensor.DType), hex.EncodeToString(digest[:])})
	}
	if len(m.Weights) != 3 || r.standard.Layers[0].W1.Rows != 0 || r.standard.Layers[0].W1.F32 == nil {
		t.Fatal("fixture does not exercise scalar loaded weights")
	}
	if err := r.validateCoreMLFFNManifest(m); err != nil {
		t.Fatalf("mapped scalar weights were rejected: %v", err)
	}
}
