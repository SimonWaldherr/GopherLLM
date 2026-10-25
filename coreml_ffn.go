//go:build coreml

package gopherllm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"

	nativecoreml "github.com/SimonWaldherr/GopherLLM/internal/coreml"
)

type coreMLFFNWeightBinding struct {
	Name       string `json:"name"`
	Shape      []int  `json:"shape"`
	TensorType int    `json:"tensor_type"`
	SHA256     string `json:"sha256"`
}

type coreMLFFNManifest struct {
	Version          int                      `json:"version"`
	Architecture     string                   `json:"architecture"`
	Layer            int                      `json:"layer"`
	Dimension        int                      `json:"dimension"`
	HiddenDimension  int                      `json:"hidden_dimension"`
	Tokens           int                      `json:"tokens"`
	Input            string                   `json:"input"`
	Output           string                   `json:"output"`
	ComputePrecision string                   `json:"compute_precision"`
	Compression      string                   `json:"weight_compression"`
	HiddenChunk      int                      `json:"hidden_chunk"`
	Activation       string                   `json:"activation"`
	ActivationScale  int                      `json:"activation_scale"`
	Model            string                   `json:"model"`
	Weights          []coreMLFFNWeightBinding `json:"weights"`
}

type coreMLFFNPredictor interface {
	Info() nativecoreml.FFNInfo
	Predict(input, out []float32) error
	Close() error
}

type coreMLFFNLayer struct {
	model coreMLFFNPredictor
	info  nativecoreml.FFNInfo
}

type coreMLFFNSet map[int]*coreMLFFNLayer

func (s coreMLFFNSet) close() {
	for _, layer := range s {
		_ = layer.model.Close()
	}
}

// SetCoreMLFFN replaces the optional prefill FFNs with exports described by
// manifests from scripts/export_coreml_ffn.py. An empty list restores the
// original path. Loading is atomic: an error leaves the old selection intact.
//
// This is an experimental, approximate FP16 path for dense Qwen and Ministral
// models. It retains the original GGUF weights for decode and small batches,
// so exported weights consume additional memory. Only complete static token
// blocks are eligible; decode and speculative verification keep their original
// arithmetic. A whole-chunk Metal prefill takes precedence when available.
// Qwen's MTP-prefill path also keeps its existing arithmetic. Exports must pass
// a small source-reference quality probe at load time; this does not establish
// generation quality on a corpus.
// Benchmark the actual model and prompt before enabling an export in service.
// Changing the selection invalidates cached conversational states.
func (r *Runner) SetCoreMLFFN(manifests []string, opts CoreMLFFNOptions) error {
	if r == nil {
		return ErrRunnerClosed
	}
	if err := r.acquireModelLease(); err != nil {
		return err
	}
	defer r.releaseModelLease()
	r.genLock.Lock()
	defer r.genLock.Unlock()
	return r.setCoreMLFFN(manifests, opts, func(path string, units nativecoreml.ComputeUnits) (coreMLFFNPredictor, error) {
		return nativecoreml.OpenFFN(path, units)
	})
}

func (r *Runner) setCoreMLFFN(manifests []string, opts CoreMLFFNOptions,
	open func(string, nativecoreml.ComputeUnits) (coreMLFFNPredictor, error)) (err error) {
	if len(manifests) != 0 && opts.AutoTune && opts.CPUOnly {
		return errors.New("coreml FFN: AutoTune and CPUOnly are mutually exclusive")
	}
	next := make(coreMLFFNSet, len(manifests))
	defer func() {
		if err != nil {
			next.close()
		}
	}()
	units := nativecoreml.CPUAndNeuralEngine
	if opts.CPUOnly {
		units = nativecoreml.CPUOnly
	}
	for _, path := range manifests {
		manifest, modelPath, err := readCoreMLFFNManifest(path)
		if err != nil {
			return err
		}
		if _, duplicate := next[manifest.Layer]; duplicate {
			return fmt.Errorf("coreml FFN: duplicate layer %d", manifest.Layer)
		}
		if err := r.validateCoreMLFFNManifest(manifest); err != nil {
			return err
		}
		gate, up, down, _ := r.coreMLFFNWeights(manifest.Layer)
		var input, output, reference []float32
		load := func(units nativecoreml.ComputeUnits) (layer *coreMLFFNLayer, err error) {
			model, err := open(modelPath, units)
			if err != nil {
				return nil, err
			}
			defer func() {
				if err != nil {
					_ = model.Close()
				}
			}()
			info := model.Info()
			if info.Tokens != manifest.Tokens || info.Dimension != manifest.Dimension {
				return nil, errors.New("coreml FFN: compiled model shape differs from its manifest")
			}
			if err := validateCoreMLFFNBinding(manifest, info.Manifest); err != nil {
				return nil, err
			}
			if input == nil {
				input = coreMLFFNProbeInput(manifest)
				output = make([]float32, len(input))
			}
			if err := model.Predict(input, output); err != nil {
				return nil, err
			}
			if reference == nil {
				reference = coreMLFFNReference(manifest, gate, up, down, input)
			}
			if err := checkCoreMLFFNQuality(manifest, output, reference); err != nil {
				return nil, err
			}
			return &coreMLFFNLayer{model: model, info: info}, nil
		}
		var layer *coreMLFFNLayer
		if opts.AutoTune {
			logw := opts.LogWriter
			if logw == nil {
				logw = io.Discard
			}
			fmt.Fprintf(logw, "Core ML FFN layer %d (%dx%d):\n", manifest.Layer, manifest.Tokens, manifest.Dimension)
			layer, err = tuneCoreMLFFN(load, func(layer *coreMLFFNLayer) (time.Duration, error) {
				return measureCoreMLFFN(layer.model, input, output)
			}, logw)
		} else {
			layer, err = load(units)
		}
		if err != nil {
			return fmt.Errorf("coreml FFN layer %d: %w", manifest.Layer, err)
		}
		next[manifest.Layer] = layer
	}
	r.coreMLFFN.close()
	r.coreMLFFN = next
	r.clearPrefixCache()
	r.conversations = conversationCache{limit: r.conversations.limit}
	r.ClearMistralKVPrefixCache()
	return nil
}

func readCoreMLFFNManifest(path string) (coreMLFFNManifest, string, error) {
	var m coreMLFFNManifest
	f, err := os.Open(path)
	if err != nil {
		return m, "", err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return m, "", err
	}
	if stat.Size() > 64<<10 {
		return m, "", errors.New("coreml FFN: manifest exceeds 64 KiB")
	}
	dec := json.NewDecoder(io.LimitReader(f, 64<<10))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return m, "", fmt.Errorf("coreml FFN manifest: %w", err)
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return m, "", errors.New("coreml FFN: trailing manifest data")
	}
	if m.Version != 1 || m.Tokens < 1 || m.Tokens > 128 || m.Dimension < 1 || m.Dimension > 16384 ||
		m.Tokens > (1<<20)/m.Dimension || m.HiddenDimension < 1 || m.HiddenDimension > 1<<20 ||
		m.Input != "hidden" || m.Output != "output" || m.ComputePrecision != "float16" ||
		m.Activation != "silu_exp" ||
		m.ActivationScale < 1 || m.ActivationScale > 4096 || m.ActivationScale&(m.ActivationScale-1) != 0 ||
		m.HiddenChunk < 0 || m.HiddenChunk > 16384 ||
		(m.Compression != "none" && m.Compression != "palette8") || len(m.Weights) != 3 {
		return m, "", errors.New("coreml FFN: unsupported manifest version, shape, precision or features")
	}
	if m.Model == "" || filepath.Base(m.Model) != m.Model || filepath.Ext(m.Model) != ".mlmodelc" {
		return m, "", errors.New("coreml FFN: manifest must name a compiled .mlmodelc beside it")
	}
	return m, filepath.Join(filepath.Dir(path), m.Model), nil
}

func (r *Runner) coreMLFFNWeights(layer int) (Weight, Weight, Weight, error) {
	if layer < 0 || layer >= r.config.NLayers {
		return Weight{}, Weight{}, Weight{}, errors.New("coreml FFN: invalid layer")
	}
	if r.kind == loadedQwen35 && layer < len(r.qwen35.Layers) {
		ffn := r.qwen35.Layers[layer].FFN
		if ffn.MoE != nil {
			return Weight{}, Weight{}, Weight{}, errors.New("coreml FFN: sparse FFNs are unsupported")
		}
		return ffn.Gate, ffn.Up, ffn.Down, nil
	} else if r.kind == loadedStandard && layer < len(r.standard.Layers) {
		l := r.standard.Layers[layer]
		if l.MoE != nil || l.HasGateUp || len(l.FFNUpBias) != 0 || len(l.FFNDownBias) != 0 {
			return Weight{}, Weight{}, Weight{}, errors.New("coreml FFN: requires split, bias-free dense SwiGLU")
		}
		return l.W1, l.W3, l.W2, nil
	}
	return Weight{}, Weight{}, Weight{}, errors.New("coreml FFN: incompatible loaded graph")
}

func (r *Runner) validateCoreMLFFNManifest(m coreMLFFNManifest) error {
	switch r.arch {
	case "qwen2", "qwen3", "qwen35", "mistral3", "ministral":
	default:
		return fmt.Errorf("coreml FFN: unsupported dense architecture %q", r.arch)
	}
	if r.outOfCore || m.Architecture != r.arch || m.Dimension != r.config.Dim || r.config.UseGELU || r.config.usesPlainMLP() {
		return errors.New("coreml FFN: export does not match the loaded dense model")
	}
	if r.kind == loadedQwen35 && m.Tokens > 32 {
		return errors.New("coreml FFN: hybrid Qwen prefill requires exports of at most 32 token rows")
	}
	gate, up, down, err := r.coreMLFFNWeights(m.Layer)
	if err != nil {
		return err
	}
	expected := map[string]struct {
		w          Weight
		rows, cols int
	}{
		fmt.Sprintf("blk.%d.ffn_gate.weight", m.Layer): {gate, m.HiddenDimension, m.Dimension},
		fmt.Sprintf("blk.%d.ffn_up.weight", m.Layer):   {up, m.HiddenDimension, m.Dimension},
		fmt.Sprintf("blk.%d.ffn_down.weight", m.Layer): {down, m.Dimension, m.HiddenDimension},
	}
	for _, entry := range m.Weights {
		x, ok := expected[entry.Name]
		if !ok || len(entry.Shape) != 2 || entry.Shape[0] != x.rows || entry.Shape[1] != x.cols {
			return fmt.Errorf("coreml FFN: mismatched tensor %q", entry.Name)
		}
		delete(expected, entry.Name)
		raw := x.w.Raw
		rows, cols, typ := x.w.Rows, x.w.Cols, x.w.Type
		if r.mappedFile != nil && r.gguf != nil {
			for _, tensor := range r.gguf.Tensors {
				if tensor.Name != entry.Name {
					continue
				}
				data, offset := tensorContainer(r.mappedFile.Bytes(), r.gguf.DataOffset, tensor)
				n, valid := tensor.DType.DataSize(tensor.Numel())
				if valid && offset >= 0 && offset <= len(data) && n <= len(data)-offset {
					raw = data[offset : offset+n]
				}
				if x.w.F32 != nil && len(tensor.Dims) == 2 && len(x.w.F32) == x.rows*x.cols {
					rows, cols, typ = int(tensor.Dims[1]), int(tensor.Dims[0]), tensor.DType
				}
				break
			}
		}
		if rows != x.rows || cols != x.cols || entry.TensorType != int(typ) {
			return fmt.Errorf("coreml FFN: mismatched loaded tensor %q", entry.Name)
		}
		if len(raw) == 0 {
			return fmt.Errorf("coreml FFN: original tensor bytes unavailable for %s; use a mapped GGUF", entry.Name)
		}
		digest := sha256.Sum256(raw)
		if hex.EncodeToString(digest[:]) != entry.SHA256 {
			return fmt.Errorf("coreml FFN: source hash mismatch for %s", entry.Name)
		}
	}
	return nil
}

func validateCoreMLFFNBinding(want coreMLFFNManifest, embedded string) error {
	var got coreMLFFNManifest
	decoder := json.NewDecoder(strings.NewReader(embedded))
	decoder.DisallowUnknownFields()
	if embedded == "" || decoder.Decode(&got) != nil {
		return errors.New("coreml FFN: compiled model lacks a valid source binding")
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		return errors.New("coreml FFN: trailing compiled source binding data")
	}
	// Offline package and compiled directory may have different basenames.
	want.Model, got.Model = "", ""
	want.Weights = append(want.Weights[:0:0], want.Weights...)
	sort.Slice(want.Weights, func(i, j int) bool { return want.Weights[i].Name < want.Weights[j].Name })
	sort.Slice(got.Weights, func(i, j int) bool { return got.Weights[i].Name < got.Weights[j].Name })
	if !reflect.DeepEqual(want, got) {
		return errors.New("coreml FFN: compiled model source binding differs from its manifest")
	}
	return nil
}

// A few deterministic, differently scaled rows catch bad placement/precision
// before any conversational KV is produced. Reference dots dequantize one row
// at a time, independent of global Q8-activation and accelerator settings.
// This is a numerical smoke check, not a substitute for corpus evaluation.
func validateCoreMLFFNQuality(model coreMLFFNPredictor, m coreMLFFNManifest, gate, up, down Weight) error {
	input := coreMLFFNProbeInput(m)
	output := make([]float32, len(input))
	if err := model.Predict(input, output); err != nil {
		return err
	}
	return checkCoreMLFFNQuality(m, output, coreMLFFNReference(m, gate, up, down, input))
}

func coreMLFFNProbeInput(m coreMLFFNManifest) []float32 {
	input := make([]float32, m.Tokens*m.Dimension)
	for i := range input {
		scale := []float64{.125, .75, 1.5}[(i/m.Dimension)%3]
		input[i] = float32(scale * (math.Sin(float64(i)*.271) + math.Cos(float64(i)*.037)/3))
	}
	return input
}

func checkCoreMLFFNQuality(m coreMLFFNManifest, output, reference []float32) error {
	for token := 0; token < min(m.Tokens, 3); token++ {
		want := reference[token*m.Dimension : (token+1)*m.Dimension]
		var dot, aa, bb, delta float64
		for i, reference := range want {
			value := float64(output[token*m.Dimension+i])
			if math.IsNaN(value) || math.IsInf(value, 0) || !finite32(reference) {
				return errors.New("non-finite quality probe")
			}
			a := float64(reference)
			dot += a * value
			aa += a * a
			bb += value * value
			delta += (a - value) * (a - value)
		}
		if aa == 0 && bb == 0 {
			continue
		}
		cosine, relative := dot/math.Sqrt(aa*bb), math.Sqrt(delta/math.Max(aa, 1e-30))
		if aa == 0 || bb == 0 || cosine < .9999 || relative > .01 {
			return fmt.Errorf("FP16 quality probe rejected: cosine=%.8f relative_l2=%.6f; keep the original FFN", cosine, relative)
		}
	}
	return nil
}

// Reuse each decoded weight row across the three probes. Blocking gate/up
// activations keeps scratch near the former three hidden-width buffers.
// A block is a multiple of both NEON and AVX2 lane counts, preserving the
// ordinary SwiGLU vector/tail arithmetic and each DotF32 reduction order.
func coreMLFFNReference(m coreMLFFNManifest, gate, up, down Weight, input []float32) []float32 {
	probes := min(m.Tokens, 3)
	const block = 128
	var gates, ups [3][block]float32
	hidden := make([]float32, probes*m.HiddenDimension)
	reference := make([]float32, probes*m.Dimension)
	row := make([]float32, max(m.Dimension, m.HiddenDimension))
	for begin := 0; begin < m.HiddenDimension; begin += block {
		end := min(begin+block, m.HiddenDimension)
		for i := begin; i < end; i++ {
			gate.RowInto(i, m.Dimension, &row)
			for token := 0; token < probes; token++ {
				x := input[token*m.Dimension : (token+1)*m.Dimension]
				gates[token][i-begin] = DotF32(row, x)
			}
			up.RowInto(i, m.Dimension, &row)
			for token := 0; token < probes; token++ {
				x := input[token*m.Dimension : (token+1)*m.Dimension]
				ups[token][i-begin] = DotF32(row, x)
			}
		}
		for token := 0; token < probes; token++ {
			siluMulF32(gates[token][:end-begin], ups[token][:end-begin],
				hidden[token*m.HiddenDimension+begin:token*m.HiddenDimension+end])
		}
	}
	for i := 0; i < m.Dimension; i++ {
		down.RowInto(i, m.HiddenDimension, &row)
		for token := 0; token < probes; token++ {
			x := hidden[token*m.HiddenDimension : (token+1)*m.HiddenDimension]
			reference[token*m.Dimension+i] = DotF32(row, x)
		}
	}
	return reference
}

// Scoped to ordinary prefill by the Runner. Never install this on decode or
// speculative scratch: those paths require their original arithmetic.
func tryCoreMLFFN(ctx context.Context, buf *DecodeBuffer, layer int, input, out []float32, tokens int) (bool, error) {
	ffn := buf.coreMLFFN[layer]
	if ffn == nil || tokens < ffn.info.Tokens || tokens%ffn.info.Tokens != 0 ||
		len(input) != tokens*ffn.info.Dimension || len(out) != len(input) {
		return false, nil
	}
	stride := ffn.info.Tokens * ffn.info.Dimension
	for start := 0; start < len(input); start += stride {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if err := ffn.model.Predict(input[start:start+stride], out[start:start+stride]); err != nil {
			if cancelErr := ctx.Err(); cancelErr != nil {
				return false, cancelErr
			}
			// FFNs have no state. Recompute the entire original projection if
			// a new input exceeds this export's FP16 range; earlier partial
			// output is scratch and will be overwritten by the caller.
			if errors.Is(err, nativecoreml.ErrFFNNonFiniteOutput) {
				return false, nil
			}
			return false, fmt.Errorf("coreml FFN layer %d: %w", layer, err)
		}
	}
	return true, ctx.Err()
}
