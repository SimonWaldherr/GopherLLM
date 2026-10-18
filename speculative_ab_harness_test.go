package gopherllm

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	workerthreads "github.com/SimonWaldherr/GopherLLM/internal/threads"
)

// Opt-in full-generation comparisons. Normal go test never loads a model.
// GOPHERLLM_SPEC_AB_MODEL selects a local GGUF; METAL=1 selects Metal.
// SAMPLES=3, MAX_TOKENS=192, MIN_TOKENS=64, WARMUPS=1, TIMEOUT=2m and
// THREADS=8 are the defaults. Prefix each variable with GOPHERLLM_SPEC_AB_.
// MODES=lookup7 or lookup1,lookup3,lookup7,mtp3,lookup7_mtp3 selects candidates.
// WORKLOAD=copy|code|all (default all). JSON=/private/tmp/result.json saves all
// samples and medians. LABEL can identify an immutable before/after binary.
//
// A separate TestSpeculationConfiguredFullMTPBounded uses RAW_MODEL, short raw
// prompt tokens, all original trunk layers, MAX_TOKENS=16|32 (default16),
// SAMPLES=1 (explicitly a smoke test), WARMUPS=0 and TIMEOUT=90s. It is not a
// replacement for a long-copy/code performance comparison.
type specABMode struct {
	Name        string
	Lookup, MTP int
}

type specABSample struct {
	Workload, Mode, Phase, Digest string
	Round, Order                  int
	Wall                          time.Duration
	Stats                         GenerationStats
}

type specABMedian struct {
	Workload, Mode              string
	Samples                     int
	Wall, Prefill, Decode, TTFT time.Duration
	WallSpeedup, DecodeSpeedup  float64
}

type specABReport struct {
	Model, Architecture, Backend, Label, GoVersion string
	Threads, Layers, MaxTokens, RawPromptTokens    int
	MTPAvailable, PrepareQuantized, Q8Activations  bool
	KVFormat                                       kvFormat
	Samples                                        []specABSample
	Medians                                        []specABMedian
}

type specABOutput struct {
	Result GenerationResult
	Tokens []uint32
}

func specABInt(t testing.TB, key string, fallback, lo, hi int) int {
	t.Helper()
	value := os.Getenv("GOPHERLLM_SPEC_AB_" + key)
	if value == "" {
		return fallback
	}
	n, err := strconv.Atoi(value)
	if err != nil || n < lo || n > hi {
		t.Fatalf("GOPHERLLM_SPEC_AB_%s must be in [%d,%d]", key, lo, hi)
	}
	return n
}

func specABModes(text string) ([]specABMode, error) {
	var modes []specABMode
	seen := map[string]bool{}
	for _, name := range strings.Split(text, ",") {
		name = strings.TrimSpace(name)
		mode := specABMode{Name: name}
		for _, part := range strings.Split(name, "_") {
			var field *int
			var digits string
			switch {
			case strings.HasPrefix(part, "lookup"):
				field, digits = &mode.Lookup, strings.TrimPrefix(part, "lookup")
			case strings.HasPrefix(part, "mtp"):
				field, digits = &mode.MTP, strings.TrimPrefix(part, "mtp")
			default:
				return nil, fmt.Errorf("unknown speculative mode %q", name)
			}
			n, err := strconv.Atoi(digits)
			if err != nil || n < 1 || n > 7 || *field != 0 {
				return nil, fmt.Errorf("invalid speculative mode %q (candidate caps 1..7)", name)
			}
			*field = n
		}
		if seen[name] {
			return nil, fmt.Errorf("duplicate speculative mode %q", name)
		}
		seen[name] = true
		modes = append(modes, mode)
	}
	return modes, nil
}

func specABDuration(values []time.Duration) time.Duration {
	values = slices.Clone(values)
	slices.Sort(values)
	middle := len(values) / 2
	if len(values)%2 != 0 {
		return values[middle]
	}
	return values[middle-1] + (values[middle]-values[middle-1])/2
}

func specABCheck(want, got specABOutput) error {
	if !slices.Equal(want.Tokens, got.Tokens) || want.Result.Text != got.Result.Text ||
		want.Result.ReasoningText != got.Result.ReasoningText || want.Result.FinishReason != got.Result.FinishReason ||
		want.Result.Stats.GeneratedTokens != got.Result.Stats.GeneratedTokens || !reflect.DeepEqual(want.Result.ToolCalls, got.Result.ToolCalls) {
		return fmt.Errorf("exact output parity failed: serial=%d tokens/%q, candidate=%d tokens/%q", len(want.Tokens), want.Result.FinishReason, len(got.Tokens), got.Result.FinishReason)
	}
	return nil
}

func specABDigest(tokens []uint32) string {
	h := fnv.New128a()
	var word [4]byte
	for _, token := range tokens {
		word[0], word[1], word[2], word[3] = byte(token), byte(token>>8), byte(token>>16), byte(token>>24)
		_, _ = h.Write(word[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func setPerformanceThreads(t testing.TB, threads int) {
	t.Helper()
	previous, procs := configuredThreads.Load(), runtime.GOMAXPROCS(threads)
	SetNumThreads(threads)
	t.Cleanup(func() { configuredThreads.Store(previous); workerthreads.Set(int(previous)); runtime.GOMAXPROCS(procs) })
}

func specABLoad(t testing.TB, path string, metal bool) (*Runner, int) {
	t.Helper()
	threads := specABInt(t, "THREADS", 8, 1, 64)
	setPerformanceThreads(t, threads)
	oldQ8 := useQ8Activations.Load()
	useQ8Activations.Store(q8ActivationsAvailable())
	t.Cleanup(func() { useQ8Activations.Store(oldQ8) })
	if metal && !MetalAvailable() {
		t.Fatal(MetalError())
	}
	// Packed weights stay mmap-backed. No prepared scale expansion or eager
	// whole-file page touching is hidden in this benchmark's load phase.
	r, info, err := RunnerFromPathWithOptions(path, LoadOptions{UseMetal: metal, PrepareQuantized: false, Prefault: MmapPrefaultNone, LogWriter: io.Discard})
	if err != nil {
		t.Fatalf("model unsupported or unreadable: %v", err)
	}
	t.Cleanup(func() { r.Close() })
	t.Logf("model=%s arch=%s layers=%d bytes=%d mapped=%t MTP=%t prepare=false prefault=none Q8=%t threads=%d load=%v", path, r.arch, r.config.NLayers, info.FileSizeBytes, r.IsMapped(), r.kind == loadedQwen35 && r.qwen35.MTP != nil, useQ8Activations.Load(), threads, info.LoadTime)
	return r, threads
}

func specABRun(t testing.TB, r *Runner, prompt string, raw []uint32, maxTokens int, timeout time.Duration, mode specABMode) (specABOutput, time.Duration) {
	t.Helper()
	r.ClearConversationCache()
	opts := DefaultGenerationOptions()
	opts.SystemPrompt, opts.MaxTokens, opts.Seed = "", maxTokens, 41
	opts.Sampler.Temperature, opts.Sampler.TopK, opts.Sampler.RepeatPenalty = 0, 1, 1
	opts.PromptLookupTokens, opts.MTPDraftTokens = mode.Lookup, mode.MTP
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	opts = opts.WithContext(ctx)
	started := time.Now()
	var result GenerationResult
	var err error
	if raw == nil {
		result, err = r.Generate(prompt, opts)
	} else {
		// Test-only raw token injection into the existing generation entry
		// point. Generate itself is a chat wrapper. No layers, logits, state
		// updates or decoding work are omitted here.
		prepared := &preparedChatData{runner: r, tokens: raw}
		result, err = r.generateChatStreamUntil([]ChatMessage{UserMessage(prompt)}, opts, nil, prepared)
	}
	wall := time.Since(started)
	if err != nil {
		t.Fatalf("%s failed after %v (deadline %v): %v; partial stats: %+v", mode.Name, wall, timeout, err, result.Stats)
	}
	if result.PromptCache != nil && result.PromptCache.ReusedTokens != 0 {
		t.Fatalf("unfair prompt cache reuse: %+v", result.PromptCache)
	}
	if r.workspaceBuf == nil || result.Stats.GeneratedTokens > cap(r.workspaceBuf.GeneratedTokens) {
		t.Fatal("missing emitted token buffer")
	}
	// Runtime retains emitted IDs at length zero after return. Capture only
	// the exact emitted prefix before the next request reuses that storage.
	tokens := slices.Clone(r.workspaceBuf.GeneratedTokens[:result.Stats.GeneratedTokens])
	return specABOutput{Result: result, Tokens: tokens}, wall
}

func specABCompare(t *testing.T, r *Runner, report *specABReport, workload, prompt string, raw []uint32, candidate specABMode, samples, warmups, maxTokens, minTokens int, timeout time.Duration) {
	t.Helper()
	serial := specABMode{Name: "serial"}
	var oracle *specABOutput
	measure := func(round, order int, phase string, mode specABMode) {
		t.Logf("start workload=%s mode=%s phase=%s round=%d order=%d", workload, mode.Name, phase, round, order)
		got, wall := specABRun(t, r, prompt, raw, maxTokens, timeout, mode)
		if got.Result.Stats.GeneratedTokens < minTokens {
			t.Fatalf("%s produced only %d tokens (minimum %d); do not infer a speedup from this short completion", mode.Name, got.Result.Stats.GeneratedTokens, minTokens)
		}
		if oracle == nil {
			copy := got
			oracle = &copy
		} else if err := specABCheck(*oracle, got); err != nil {
			t.Fatal(err)
		}
		record := specABSample{Workload: workload, Mode: mode.Name, Phase: phase, Round: round, Order: order, Wall: wall, Stats: got.Result.Stats, Digest: specABDigest(got.Tokens)}
		report.Samples = append(report.Samples, record)
		encoded, _ := json.Marshal(record)
		t.Logf("sample %s", encoded)
	}
	for round := range warmups {
		measure(round, 0, "warmup", serial)
		measure(round, 1, "warmup", candidate)
	}
	for round := range samples {
		order := []specABMode{serial, candidate}
		if round%2 != 0 {
			slices.Reverse(order)
		}
		for i, mode := range order {
			measure(round, i, "measured", mode)
		}
	}
	medians := map[string]specABMedian{}
	for _, mode := range []specABMode{serial, candidate} {
		var wall, prefill, decode, ttft []time.Duration
		for _, record := range report.Samples {
			if record.Workload == workload && record.Mode == mode.Name && record.Phase == "measured" {
				wall, prefill, decode, ttft = append(wall, record.Wall), append(prefill, record.Stats.PrefillTime), append(decode, record.Stats.DecodeTime), append(ttft, record.Stats.TTFT)
			}
		}
		medians[mode.Name] = specABMedian{Workload: workload, Mode: mode.Name, Samples: len(wall), Wall: specABDuration(wall), Prefill: specABDuration(prefill), Decode: specABDuration(decode), TTFT: specABDuration(ttft)}
	}
	for _, mode := range []specABMode{serial, candidate} {
		median := medians[mode.Name]
		median.WallSpeedup = float64(medians[serial.Name].Wall) / float64(median.Wall)
		median.DecodeSpeedup = float64(medians[serial.Name].Decode) / float64(median.Decode)
		report.Medians = append(report.Medians, median)
		encoded, _ := json.Marshal(median)
		t.Logf("median %s", encoded)
	}
}

func specABSave(t testing.TB, report *specABReport) {
	t.Helper()
	if path := os.Getenv("GOPHERLLM_SPEC_AB_JSON"); path != "" {
		data, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSpeculationConfiguredGenerationAB(t *testing.T) {
	path := os.Getenv("GOPHERLLM_SPEC_AB_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_SPEC_AB_MODEL to opt in to full generation A/B")
	}
	metal := os.Getenv("GOPHERLLM_SPEC_AB_METAL") == "1"
	r, threads := specABLoad(t, path, metal)
	report := &specABReport{Model: path, Architecture: r.arch, Backend: "cpu", Label: os.Getenv("GOPHERLLM_SPEC_AB_LABEL"), GoVersion: runtime.Version(), Threads: threads, Layers: r.config.NLayers, MTPAvailable: r.kind == loadedQwen35 && r.qwen35.MTP != nil, Q8Activations: useQ8Activations.Load(), KVFormat: desiredKVFormat(true)}
	if metal {
		report.Backend = "metal"
	}
	defer specABSave(t, report)
	modeText := os.Getenv("GOPHERLLM_SPEC_AB_MODES")
	if modeText == "" {
		modeText = "lookup7"
		if report.MTPAvailable {
			modeText += ",mtp3"
		}
	}
	modes, err := specABModes(modeText)
	if err != nil {
		t.Fatal(err)
	}
	if !report.MTPAvailable {
		t.Log("native MTP absent; only prompt lookup is available for this checkpoint")
	}
	maxTokens := specABInt(t, "MAX_TOKENS", 192, 16, 512)
	minTokens := specABInt(t, "MIN_TOKENS", 64, 1, maxTokens)
	samples, warmups := specABInt(t, "SAMPLES", 3, 1, 9), specABInt(t, "WARMUPS", 1, 0, 2)
	timeout := 2 * time.Minute
	if value := os.Getenv("GOPHERLLM_SPEC_AB_TIMEOUT"); value != "" {
		timeout, err = time.ParseDuration(value)
		if err != nil || timeout <= 0 {
			t.Fatal("invalid GOPHERLLM_SPEC_AB_TIMEOUT")
		}
	}
	report.MaxTokens = maxTokens
	var passage strings.Builder
	for i := 1; i <= 10; i++ {
		fmt.Fprintf(&passage, "%02d | item_%02d | enabled | retry=3 | timeout=30\n", i, i)
	}
	workloads := []struct{ name, prompt string }{
		{"copy", "Copy every record between BEGIN and END exactly. Preserve line breaks and punctuation. Output only the records.\nBEGIN\n" + passage.String() + "END"},
		{"code", "Return this complete Go function with only the parameter name changed from values to input. Preserve all other code and output code only.\n\nfunc SumPositive(values []int) int {\n    total := 0\n    for _, value := range values {\n        if value > 0 {\n            total += value\n        }\n    }\n    return total\n}\n\nAdd table-driven tests for positive, negative, empty and mixed input slices. Include complete test functions."},
	}
	selected := os.Getenv("GOPHERLLM_SPEC_AB_WORKLOAD")
	if selected != "" && selected != "all" && selected != "copy" && selected != "code" {
		t.Fatal("WORKLOAD must be copy, code or all")
	}
	for _, mode := range modes {
		if mode.MTP > 0 && !report.MTPAvailable {
			t.Fatalf("mode %s requires an MTP head; this checkpoint has none", mode.Name)
		}
		for _, workload := range workloads {
			if selected != "" && selected != "all" && selected != workload.name {
				continue
			}
			// Keep each candidate's serial samples separate for a paired A/B.
			name := workload.name + "/" + mode.Name
			specABCompare(t, r, report, name, workload.prompt, nil, mode, samples, warmups, maxTokens, minTokens, timeout)
		}
	}
}

func TestSpeculationConfiguredFullMTPBounded(t *testing.T) {
	path := os.Getenv("GOPHERLLM_SPEC_AB_RAW_MODEL")
	if path == "" {
		t.Skip("set GOPHERLLM_SPEC_AB_RAW_MODEL for bounded full-model MTP smoke/parity")
	}
	metal := os.Getenv("GOPHERLLM_SPEC_AB_METAL") == "1"
	r, threads := specABLoad(t, path, metal)
	if r.kind != loadedQwen35 || r.qwen35.MTP == nil {
		t.Fatal("raw MTP check requires a supported complete Qwen hybrid checkpoint with NextN weights")
	}
	maxTokens := specABInt(t, "MAX_TOKENS", 16, 16, 32)
	if maxTokens != 16 && maxTokens != 32 {
		t.Fatal("bounded raw MTP check permits exactly 16 or 32 output tokens")
	}
	prompt := "def add(a, b):\n    return a + b\n\n# Next function:\ndef subtract"
	tokens := r.tok.Encode(prompt)
	if len(tokens) < 8 {
		t.Fatal("raw prompt unexpectedly shorter than eight tokens")
	}
	rawLimit := specABInt(t, "RAW_PROMPT_TOKENS", 24, 8, 24)
	tokens = tokens[:min(rawLimit, len(tokens))]
	samples := specABInt(t, "SAMPLES", 1, 1, 5)
	warmups := specABInt(t, "WARMUPS", 0, 0, 1)
	timeout := 90 * time.Second
	var err error
	if value := os.Getenv("GOPHERLLM_SPEC_AB_TIMEOUT"); value != "" {
		timeout, err = time.ParseDuration(value)
		if err != nil || timeout <= 0 || timeout > 3*time.Minute {
			t.Fatal("bounded raw TIMEOUT must be positive and <=3m")
		}
	}
	report := &specABReport{Model: path, Architecture: r.arch, Backend: "cpu_raw_full_model", Label: os.Getenv("GOPHERLLM_SPEC_AB_LABEL"), GoVersion: runtime.Version(), Threads: threads, Layers: r.config.NLayers, MaxTokens: maxTokens, RawPromptTokens: len(tokens), MTPAvailable: true, Q8Activations: useQ8Activations.Load(), KVFormat: desiredKVFormat(true)}
	if metal {
		report.Backend = "metal_raw_full_model"
	}
	defer specABSave(t, report)
	t.Logf("bounded full-model raw check: %d prompt tokens, %d output cap, %d original trunk layers, %d samples; not a long-generation speed claim", len(tokens), maxTokens, r.config.NLayers, samples)
	modeText := os.Getenv("GOPHERLLM_SPEC_AB_MODES")
	if modeText == "" {
		modeText = "mtp3"
	}
	modes, err := specABModes(modeText)
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range modes {
		if mode.MTP == 0 {
			t.Fatal("bounded MTP check requires an MTP candidate mode")
		}
		specABCompare(t, r, report, "raw_mtp_full_model/"+mode.Name, prompt, tokens, mode, samples, warmups, maxTokens, 1, timeout)
	}
	// Metal pipeline warmup can leave a tiny workspace that the first request
	// grows by its ordinary doubling/chunk reserve. Keep that bounded reserve
	// without allowing a model-sized context cache in this raw-token test.
	required := len(tokens) + maxTokens + 1
	if r.workspaceCache.MaxLen > max(2*required, required+prefillChunkSize(r.config)) {
		t.Fatalf("unexpectedly large raw KV cache: %d positions", r.workspaceCache.MaxLen)
	}
}

func TestSpeculationABHarnessHelpers(t *testing.T) {
	modes, err := specABModes("lookup1,lookup7_mtp3,mtp7")
	if err != nil || len(modes) != 3 || modes[1].Lookup != 7 || modes[1].MTP != 3 {
		t.Fatalf("modes=%+v err=%v", modes, err)
	}
	for _, text := range []string{"", "serial", "lookup0", "mtp8", "lookup7_lookup2", "mtp3,mtp3"} {
		if _, err := specABModes(text); err == nil {
			t.Fatalf("invalid modes accepted: %q", text)
		}
	}
	if got := specABDuration([]time.Duration{9, 1, 4}); got != 4 {
		t.Fatalf("median=%v", got)
	}
	if got := specABDuration([]time.Duration{9, 1, 4, 6}); got != 5 {
		t.Fatalf("even median=%v", got)
	}
	base := specABOutput{Result: GenerationResult{Text: "same", ReasoningText: "reason", FinishReason: "length", Stats: GenerationStats{GeneratedTokens: 2}}, Tokens: []uint32{1, 2}}
	if err := specABCheck(base, base); err != nil {
		t.Fatal(err)
	}
	changed := base
	changed.Tokens = []uint32{1, 3}
	if err := specABCheck(base, changed); err == nil {
		t.Fatal("different token IDs with the same text passed parity")
	}
	changed = base
	changed.Result.ReasoningText = "different"
	if err := specABCheck(base, changed); err == nil {
		t.Fatal("different reasoning passed parity")
	}
}
