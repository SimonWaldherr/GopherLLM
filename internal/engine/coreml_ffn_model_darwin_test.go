//go:build darwin && cgo && metal && coreml

package engine

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	coremlbackend "github.com/SimonWaldherr/GopherLLM/internal/coreml"
)

// TestCoreMLConfiguredFFN compares one exported, real checkpoint FFN with
// GopherLLM's CPU and fused Metal paths. It maps the GGUF but loads only three
// matrices; no Runner, vocabulary bank, attention weights or KV cache is made.
// This measures a layer, including the host/device round trip, not generation.
// Set GOPHERLLM_COREML_FFN_GGUF and GOPHERLLM_COREML_FFN_PATH to opt in. The
// optional MANIFEST and UNITS names use that same prefix. UNITS defaults to
// cpu,ane; "ane" permits CPU fallback rather than proving ANE-only execution.
// TUNE_ONLY=1 instead checks attachment/calibration with only the selected
// layer's weights in a Runner; it does not load a complete generation graph.
func TestCoreMLConfiguredFFN(t *testing.T) {
	ggufPath, modelPath := os.Getenv("GOPHERLLM_COREML_FFN_GGUF"), os.Getenv("GOPHERLLM_COREML_FFN_PATH")
	if ggufPath == "" || modelPath == "" {
		t.Skip("set GOPHERLLM_COREML_FFN_GGUF and GOPHERLLM_COREML_FFN_PATH to compare a checkpoint FFN")
	}
	manifestPath := os.Getenv("GOPHERLLM_COREML_FFN_MANIFEST")
	if manifestPath == "" {
		manifestPath = strings.TrimSuffix(modelPath, filepath.Ext(modelPath)) + ".json"
	}
	var manifest struct {
		Version           int    `json:"version"`
		Architecture      string `json:"architecture"`
		Layer             int    `json:"layer"`
		Dimension         int    `json:"dimension"`
		HiddenDimension   int    `json:"hidden_dimension"`
		Tokens            int    `json:"tokens"`
		WeightCompression string `json:"weight_compression"`
		HiddenChunk       int    `json:"hidden_chunk"`
		Activation        string `json:"activation"`
		ActivationScale   int    `json:"activation_scale"`
		Weights           []struct {
			Name       string `json:"name"`
			Shape      []int  `json:"shape"`
			TensorType int    `json:"tensor_type"`
			SHA256     string `json:"sha256"`
		} `json:"weights"`
	}
	metadata, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(metadata, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Version != 1 || manifest.Layer < 0 || manifest.Tokens < 1 || manifest.Tokens > 128 ||
		manifest.Dimension < 1 || manifest.Dimension > 16384 || manifest.Dimension > (1<<20)/manifest.Tokens ||
		manifest.HiddenDimension < 1 || manifest.HiddenDimension > 65536 || len(manifest.Weights) != 3 ||
		manifest.HiddenChunk < 0 || manifest.HiddenChunk > 16384 || manifest.Activation != "silu_exp" ||
		manifest.ActivationScale < 1 || manifest.ActivationScale > 4096 || manifest.ActivationScale&(manifest.ActivationScale-1) != 0 {
		t.Fatal("invalid FFN benchmark manifest")
	}
	setPerformanceThreads(t, min(runtime.NumCPU(), 8))
	mapping, err := OpenMmap(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mapping.Close() })
	gguf, err := ParseGGUFQuiet(mapping.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	config := ConfigFromGGUF(gguf)
	if config.Arch != manifest.Architecture || config.Dim != manifest.Dimension {
		t.Fatal("FFN manifest does not match source architecture/dimension")
	}
	tensors, inferred := indexTensors(gguf), inferTensorSizes(mapping.Bytes(), gguf)
	var weights [3]Weight
	var packedBytes int
	for i, suffix := range []string{"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"} {
		name := fmt.Sprintf("blk.%d.%s", manifest.Layer, suffix)
		info, ok := tensors[name]
		if !ok || len(info.Dims) != 2 {
			t.Fatalf("missing or nonmatrix source tensor %s", name)
		}
		rows, cols := manifest.HiddenDimension, manifest.Dimension
		if i == 2 {
			rows, cols = cols, rows
		}
		if int(info.Dims[1]) != rows || int(info.Dims[0]) != cols {
			t.Fatalf("source tensor %s shape differs from export", name)
		}
		size, ok := info.DType.DataSize(info.Numel())
		container, offset := tensorContainer(mapping.Bytes(), gguf.DataOffset, info)
		if !ok || offset < 0 || offset > len(container) || size > len(container)-offset {
			t.Fatalf("invalid source tensor range %s", name)
		}
		digest := sha256.Sum256(container[offset : offset+size])
		matched := false
		for _, expected := range manifest.Weights {
			if expected.Name == name && expected.TensorType == int(info.DType) && len(expected.Shape) == 2 && expected.Shape[0] == rows &&
				expected.Shape[1] == cols && expected.SHA256 == hex.EncodeToString(digest[:]) {
				matched = true
			}
		}
		if !matched {
			t.Fatalf("source tensor %s differs from exported hash/shape", name)
		}
		weights[i], err = loadWeight(mapping.Bytes(), gguf.DataOffset, name, tensors, inferred, false, true, true, false, true)
		if err != nil {
			t.Fatal(err)
		}
		packedBytes += size
	}
	if os.Getenv("GOPHERLLM_COREML_FFN_TUNE_ONLY") == "1" {
		r := &Runner{arch: config.Arch, config: config, kind: loadedStandard}
		if config.Arch == "qwen35" {
			r.kind = loadedQwen35
			r.qwen35.Layers = make([]Qwen35LayerWeights, config.NLayers)
			r.qwen35.Layers[manifest.Layer].FFN = Qwen35FFNWeights{Gate: weights[0], Up: weights[1], Down: weights[2]}
		} else {
			r.standard.Layers = make([]LayerWeights, config.NLayers)
			r.standard.Layers[manifest.Layer] = LayerWeights{W1: weights[0], W3: weights[1], W2: weights[2]}
		}
		var log bytes.Buffer
		if err := r.setCoreMLFFN([]string{manifestPath}, CoreMLFFNOptions{AutoTune: true, LogWriter: &log}, func(path string, units coremlbackend.ComputeUnits) (coreMLFFNPredictor, error) {
			return coremlbackend.OpenFFN(path, units)
		}); err != nil {
			t.Fatalf("calibration failed: %v\n%s", err, &log)
		}
		defer r.coreMLFFN.close()
		raw, err := json.Marshal(map[string]any{"source": filepath.Base(ggufPath), "export": filepath.Base(modelPath), "calibration": log.String()})
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println("COREML_FFN_TUNING_JSON " + string(raw))
		return
	}
	tokens, dimension, hidden := manifest.Tokens, manifest.Dimension, manifest.HiddenDimension
	input, reference, output := make([]float32, tokens*dimension), make([]float32, tokens*dimension), make([]float32, tokens*dimension)
	for i := range input {
		input[i] = float32(.75*math.Sin(float64(i)*.271) + .25*math.Cos(float64(i)*.037))
	}
	views := func(values []float32, width int) [][]float32 {
		rows := make([][]float32, tokens)
		for i := range rows {
			rows[i] = values[i*width : (i+1)*width]
		}
		return rows
	}
	gate, up, activated := make([]float32, tokens*hidden), make([]float32, tokens*hidden), make([]float32, tokens*hidden)
	inputs, gates, ups, activations := views(input, dimension), views(gate, hidden), views(up, hidden), views(activated, hidden)
	referenceRows, outputRows := views(reference, dimension), views(output, dimension)
	cpu := func(destination []float32) error {
		matvecBatch2Slabs(weights[0], weights[1], inputs, gates, ups, input, gate, up)
		for i := range tokens {
			siluMulF32(gates[i], ups[i], activations[i])
		}
		rows := outputRows
		if &destination[0] == &reference[0] {
			rows = referenceRows
		}
		matvecBatchSlabs(weights[2], activations, rows, activated, destination)
		return nil
	}
	// Quantized CPU defaults can use Q8 activations. The quality reference
	// instead uses the unquantized activation path against the original packed
	// weights, separating FP16 export error from that existing CPU tradeoff.
	oldQ8 := useQ8Activations.Swap(false)
	err = cpu(reference)
	useQ8Activations.Store(oldQ8)
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		Source            string  `json:"source"`
		Model             string  `json:"model"`
		Architecture      string  `json:"architecture"`
		Layer             int     `json:"layer"`
		Tokens            int     `json:"tokens"`
		Dimension         int     `json:"dimension"`
		HiddenDimension   int     `json:"hidden_dimension"`
		WeightCompression string  `json:"weight_compression"`
		HiddenChunk       int     `json:"hidden_chunk"`
		Activation        string  `json:"activation"`
		ActivationScale   int     `json:"activation_scale"`
		Engine            string  `json:"engine"`
		PackedBytes       int     `json:"packed_bytes"`
		Threads           int     `json:"threads"`
		LoadMS            float64 `json:"load_ms"`
		FirstMS           float64 `json:"first_ms"`
		MedianMS          float64 `json:"median_ms"`
		P90MS             float64 `json:"p90_ms"`
		MinCosine         float64 `json:"min_cosine"`
		RelativeL2        float64 `json:"relative_l2"`
		OutputNormRatio   float64 `json:"output_norm_ratio"`
		WithinOnePercent  bool    `json:"within_one_percent_reference"`
		QualityProbeError string  `json:"quality_probe_error,omitempty"`
		QualityProbePass  *bool   `json:"quality_probe_accepted,omitempty"`
		Requests          int     `json:"requests"`
		Windows           int     `json:"windows"`
		MeasuredAt        string  `json:"measured_at"`
		GOARCH            string  `json:"goarch"`
		GoVersion         string  `json:"go_version"`
	}
	measure := func(name string, load time.Duration, run func([]float32) error, enforceQuality bool, probe func() error) {
		t.Helper()
		firstStart := time.Now()
		if err := run(output); err != nil {
			t.Fatal(err)
		}
		first := time.Since(firstStart)
		cosine, relative, normRatio := coreMLFFNBenchmarkQuality(t, output, reference, tokens, dimension)
		if enforceQuality && (cosine < .999 || relative > .01) {
			t.Fatalf("%s source parity failed: cosine=%g relative_l2=%g", name, cosine, relative)
		}
		if cosine < .999 || relative > .01 {
			t.Logf("%s differs from float-activation source reference: cosine=%g relative_l2=%g (report-only)", name, cosine, relative)
		}
		for range 3 {
			if err := run(output); err != nil {
				t.Fatal(err)
			}
		}
		var requests []time.Duration
		for range 9 {
			window := time.Now()
			for {
				start := time.Now()
				if err := run(output); err != nil {
					t.Fatal(err)
				}
				requests = append(requests, time.Since(start))
				if time.Since(window) >= 50*time.Millisecond {
					break
				}
			}
		}
		sort.Slice(requests, func(i, j int) bool { return requests[i] < requests[j] })
		ms := func(duration time.Duration) float64 { return float64(duration) / float64(time.Millisecond) }
		record := result{
			Source: filepath.Base(ggufPath), Model: filepath.Base(modelPath), Architecture: manifest.Architecture,
			Layer: manifest.Layer, Tokens: tokens, Dimension: dimension, HiddenDimension: hidden,
			WeightCompression: manifest.WeightCompression, HiddenChunk: manifest.HiddenChunk, Engine: name, PackedBytes: packedBytes,
			Activation: manifest.Activation, ActivationScale: manifest.ActivationScale,
			Threads: numThreads(), LoadMS: ms(load), FirstMS: ms(first), MedianMS: ms(requests[len(requests)/2]),
			P90MS: ms(requests[(len(requests)-1)*9/10]), MinCosine: cosine, RelativeL2: relative, OutputNormRatio: normRatio,
			WithinOnePercent: cosine >= .999 && relative <= .01,
			Requests:         len(requests), Windows: 9, MeasuredAt: time.Now().UTC().Format(time.RFC3339),
			GOARCH: runtime.GOARCH, GoVersion: runtime.Version(),
		}
		if probe != nil {
			probeErr := probe()
			accepted := probeErr == nil
			record.QualityProbePass = &accepted
			if probeErr != nil {
				record.QualityProbeError = probeErr.Error()
			}
		}
		raw, err := json.Marshal(record)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println("COREML_FFN_BENCHMARK_JSON " + string(raw))
	}
	measure("source_cpu", 0, cpu, false, nil)
	if MetalAvailable() {
		var metalWeights [3]Weight
		for i, weight := range weights {
			metalWeights[i] = weight
			metalWeights[i].Metal = prepareMetalWeight(weight.Raw, weight.Type, weight.Rows, weight.Cols, true)
		}
		defer func() {
			for i := range metalWeights {
				releaseMetalWeight(metalWeights[i].Metal)
			}
		}()
		metal := func(destination []float32) error {
			ok := false
			if tokens == 1 {
				ok = matvecMetalQwen35SwiGLUInto(metalWeights[0].Metal, metalWeights[1].Metal, metalWeights[2].Metal, input, &destination)
			} else {
				ok = matvecMetalSwiGLUBatchInto(metalWeights[0].Metal, metalWeights[1].Metal, metalWeights[2].Metal, input, tokens, &destination)
			}
			if !ok {
				return fmt.Errorf("fused Metal FFN declined")
			}
			return nil
		}
		if err := metal(output); err != nil {
			t.Logf("Metal comparison unavailable: %v", err)
		} else {
			measure("source_metal", 0, metal, true, nil)
		}
	}
	unitNames := os.Getenv("GOPHERLLM_COREML_FFN_UNITS")
	if unitNames == "" {
		unitNames = "cpu,ane"
	}
	for _, name := range strings.Split(unitNames, ",") {
		var units coremlbackend.ComputeUnits
		switch name {
		case "cpu":
			units = coremlbackend.CPUOnly
		case "ane":
			units = coremlbackend.CPUAndNeuralEngine
		case "gpu":
			units = coremlbackend.CPUAndGPU
		case "all":
			units = coremlbackend.All
		default:
			t.Fatalf("unknown FFN compute units %q (expected cpu, ane, gpu or all)", name)
		}
		start := time.Now()
		model, err := coremlbackend.OpenFFN(modelPath, units)
		load := time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
		func() {
			defer model.Close()
			info := model.Info()
			if info.Tokens != tokens || info.Dimension != dimension {
				t.Fatal("compiled FFN shape differs from manifest")
			}
			var embedded, external map[string]any
			if err := json.Unmarshal([]byte(info.Manifest), &embedded); err != nil {
				t.Fatalf("compiled FFN lacks a valid source binding: %v", err)
			}
			if err := json.Unmarshal(metadata, &external); err != nil {
				t.Fatal(err)
			}
			// The package and compiled artifact deliberately have different names;
			// every graph/numeric parameter and the source fingerprints must match.
			delete(embedded, "model")
			delete(external, "model")
			if !reflect.DeepEqual(embedded, external) {
				t.Fatal("compiled FFN source binding differs from the external manifest")
			}
			// Quantify export/reduction error rather than discarding the timing
			// of an approximate graph. Finite values, source hashes and shapes
			// remain hard requirements. This does not validate generation quality.
			measure("coreml_"+name, load, func(destination []float32) error { return model.Predict(input, destination) }, false, func() error {
				var sourceManifest coreMLFFNManifest
				if err := json.Unmarshal(metadata, &sourceManifest); err != nil {
					return err
				}
				return validateCoreMLFFNQuality(model, sourceManifest, weights[0], weights[1], weights[2])
			})
		}()
	}
}

func coreMLFFNBenchmarkQuality(t *testing.T, actual, reference []float32, tokens, dimension int) (float64, float64, float64) {
	t.Helper()
	minimum, squaredError, squaredReference, squaredActual := 1.0, 0.0, 0.0, 0.0
	for row := range tokens {
		dot, normActual, normReference := 0.0, 0.0, 0.0
		for column := range dimension {
			i := row*dimension + column
			a, r := float64(actual[i]), float64(reference[i])
			if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(r) || math.IsInf(r, 0) {
				t.Fatal("FFN returned non-finite values")
			}
			dot += a * r
			normActual += a * a
			normReference += r * r
			squaredError += (a - r) * (a - r)
			squaredReference += r * r
			squaredActual += a * a
		}
		if normActual == 0 || normReference == 0 {
			t.Fatal("FFN quality comparison requires nonzero source and output")
		}
		minimum = min(minimum, dot/math.Sqrt(normActual*normReference))
	}
	return minimum, math.Sqrt(squaredError / squaredReference), math.Sqrt(squaredActual / squaredReference)
}
