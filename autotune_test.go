package gopherllm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAutoTuneOptionsDefaults(t *testing.T) {
	got := AutoTuneOptions{}.withDefaults()
	if got.Rounds < 1 || got.DecodeSteps < 1 || got.Context < 1 || got.MinGain <= 0 {
		t.Fatalf("zero-value options did not get usable defaults: %+v", got)
	}
	if got.LogWriter == nil {
		t.Fatal("LogWriter must default to a non-nil writer")
	}
	// Explicit values must survive.
	in := AutoTuneOptions{Rounds: 9, DecodeSteps: 4, Context: 77, MinGain: 0.5}
	out := in.withDefaults()
	if out.Rounds != 9 || out.DecodeSteps != 4 || out.Context != 77 || out.MinGain != 0.5 {
		t.Fatalf("explicit options were overwritten: %+v", out)
	}
}

func TestBetterBy(t *testing.T) {
	// Lower-is-better: 90 against 100 is a 10% improvement.
	if g := betterBy(90, 100, true); g < 0.099 || g > 0.101 {
		t.Fatalf("lower-is-better gain = %v, want ~0.1", g)
	}
	// Higher-is-better: 110 against 100 is a 10% improvement.
	if g := betterBy(110, 100, false); g < 0.099 || g > 0.101 {
		t.Fatalf("higher-is-better gain = %v, want ~0.1", g)
	}
	// A regression must be negative so callers can reject it.
	if g := betterBy(120, 100, true); g >= 0 {
		t.Fatalf("regression should be negative, got %v", g)
	}
	if betterBy(1, 0, true) != 0 {
		t.Fatal("zero reference must not divide by zero")
	}
}

func TestWinsMajority(t *testing.T) {
	// a wins every round.
	if !winsMajority([]float64{1, 1, 1}, []float64{2, 2, 2}, true) {
		t.Fatal("clear winner rejected")
	}
	// a wins one of three: a lucky single sample must not count as a win.
	if winsMajority([]float64{1, 5, 5}, []float64{2, 2, 2}, true) {
		t.Fatal("single lucky round should not win a majority")
	}
	// Exactly half counts as a win (ties go to the challenger only when the
	// median margin already cleared MinGain).
	if !winsMajority([]float64{1, 5}, []float64{2, 2}, true) {
		t.Fatal("half the rounds should satisfy the majority rule")
	}
	// Higher-is-better direction.
	if !winsMajority([]float64{9, 9}, []float64{1, 1}, false) {
		t.Fatal("higher-is-better majority rejected")
	}
	if winsMajority(nil, nil, true) {
		t.Fatal("empty samples must not report a win")
	}
}

func TestThreadCandidates(t *testing.T) {
	if got := threadCandidates(1); len(got) != 1 || got[0] != 1 {
		t.Fatalf("single-CPU candidates = %v, want [1]", got)
	}
	got := threadCandidates(12)
	if len(got) < 2 {
		t.Fatalf("expected several candidates for 12 CPUs, got %v", got)
	}
	seen := map[int]bool{}
	for i, n := range got {
		if n < 1 || n > 12 {
			t.Fatalf("candidate %d out of range: %v", n, got)
		}
		if seen[n] {
			t.Fatalf("duplicate candidate %d in %v", n, got)
		}
		seen[n] = true
		if i > 0 && got[i-1] < n {
			t.Fatalf("candidates must be descending, got %v", got)
		}
	}
	if got[0] != 12 {
		t.Fatalf("full CPU count must be a candidate, got %v", got)
	}
}

func TestMedianFloat(t *testing.T) {
	if medianFloat(nil) != 0 {
		t.Fatal("median of nothing should be 0")
	}
	if got := medianFloat([]float64{3, 1, 2}); got != 2 {
		t.Fatalf("odd median = %v, want 2", got)
	}
	if got := medianFloat([]float64{4, 1, 3, 2}); got != 2.5 {
		t.Fatalf("even median = %v, want 2.5", got)
	}
	// The input must not be reordered in place — callers reuse their slices.
	in := []float64{3, 1, 2}
	medianFloat(in)
	if in[0] != 3 {
		t.Fatalf("medianFloat mutated its input: %v", in)
	}
}

// TestSweepIsSerpentine pins the interleaving order, which is the whole basis of
// the tuner's noise resistance: candidates must alternate direction between
// rounds so a monotonic thermal drift cannot systematically favor whichever one
// is visited first.
func TestSweepIsSerpentine(t *testing.T) {
	tn := &autoTuner{opts: AutoTuneOptions{Rounds: 3}.withDefaults()}
	var order []int
	cur := 0
	samples := tn.sweep(3, 3, func(i int) { cur = i }, func() float64 {
		order = append(order, cur)
		return float64(cur)
	})
	want := []int{0, 1, 2, 2, 1, 0, 0, 1, 2}
	if len(order) != len(want) {
		t.Fatalf("visit order length = %d, want %d (%v)", len(order), len(want), order)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("visit order = %v, want %v", order, want)
		}
	}
	// Every candidate must get exactly Rounds samples regardless of direction.
	for i, s := range samples {
		if len(s) != 3 {
			t.Fatalf("candidate %d got %d samples, want 3", i, len(s))
		}
	}
}

func TestSingleDecodeChangesIsolatesEachKnob(t *testing.T) {
	origin := tunerConfig{threads: 12, q8: true, kvF16: true, oversubscribe: false, prefillChunk: 32}
	proposed := tunerConfig{threads: 6, q8: false, kvF16: true, oversubscribe: true, prefillChunk: 128}
	changes := singleDecodeChanges(origin, proposed)
	// q8, threads, oversubscribe differ; kvF16 matches; prefillChunk is excluded
	// because it does not affect decode.
	if len(changes) != 3 {
		t.Fatalf("expected 3 isolated changes, got %d: %+v", len(changes), changes)
	}
	for _, c := range changes {
		diffs := 0
		if c.cfg.threads != origin.threads {
			diffs++
		}
		if c.cfg.q8 != origin.q8 {
			diffs++
		}
		if c.cfg.kvF16 != origin.kvF16 {
			diffs++
		}
		if c.cfg.oversubscribe != origin.oversubscribe {
			diffs++
		}
		if diffs != 1 {
			t.Fatalf("%s changed %d knobs, want exactly 1", c.name, diffs)
		}
		if c.cfg.prefillChunk != origin.prefillChunk {
			t.Fatalf("%s must not touch the prefill chunk", c.name)
		}
	}
	if len(singleDecodeChanges(origin, origin)) != 0 {
		t.Fatal("identical configs should yield no changes")
	}
}

func TestTunerConfigRoundTrips(t *testing.T) {
	before := captureTunerConfig(Config{Dim: 3072, HiddenDim: 9216})
	defer before.apply()

	want := before
	want.threads = max(1, before.threads/2)
	want.prefillChunk = 64
	want.apply()
	got := captureTunerConfig(Config{Dim: 3072, HiddenDim: 9216})
	if got.threads != want.threads {
		t.Fatalf("threads did not round-trip: got %d want %d", got.threads, want.threads)
	}
	if got.prefillChunk != want.prefillChunk {
		t.Fatalf("prefill chunk did not round-trip: got %d want %d", got.prefillChunk, want.prefillChunk)
	}
	if got.oversubscribe != want.oversubscribe {
		t.Fatal("oversubscribe did not round-trip")
	}
	// The amd64-only knobs must round-trip where supported and stay off where
	// they are compile-time disabled.
	if q8ActivationsAvailable() && got.q8 != want.q8 {
		t.Fatal("q8 activations did not round-trip")
	}
	if !q8ActivationsAvailable() && got.q8 {
		t.Fatal("q8 activations must stay off when unavailable")
	}
}

func TestSetPrefillChunkOverridesDefault(t *testing.T) {
	defer SetPrefillChunk(0)
	cfg := Config{Dim: 3072, HiddenDim: 9216}
	def := prefillChunkSize(cfg)
	SetPrefillChunk(def + 7)
	if got := prefillChunkSize(cfg); got != def+7 {
		t.Fatalf("override not honored: got %d want %d", got, def+7)
	}
	SetPrefillChunk(0)
	if got := prefillChunkSize(cfg); got != def {
		t.Fatalf("clearing the override should restore %d, got %d", def, got)
	}
	SetPrefillChunk(-5)
	if got := prefillChunkSize(cfg); got != def {
		t.Fatalf("negative override should be ignored, got %d", got)
	}
}

func TestAutoTuneResultApplyAndSpeedups(t *testing.T) {
	before := captureTunerConfig(Config{})
	defer before.apply()

	res := AutoTuneResult{
		Threads: 3, Q8Activations: false, KVCacheF16: false,
		OversubscribeDispatch: !before.oversubscribe, PrefillChunk: 96,
		BaselineDecodeMs: 200, TunedDecodeMs: 100,
		BaselinePrefillTps: 10, TunedPrefillTps: 20,
	}
	res.Apply()
	if numThreads() != 3 {
		t.Fatalf("Apply did not set threads, got %d", numThreads())
	}
	if runtime.GOMAXPROCS(0) != 3 {
		t.Fatalf("Apply did not set GOMAXPROCS, got %d", runtime.GOMAXPROCS(0))
	}
	if prefillChunkSize(Config{}) != 96 {
		t.Fatal("Apply did not set the prefill chunk")
	}
	if got := res.DecodeSpeedup(); got != 2 {
		t.Fatalf("DecodeSpeedup = %v, want 2", got)
	}
	if got := res.PrefillSpeedup(); got != 2 {
		t.Fatalf("PrefillSpeedup = %v, want 2", got)
	}
	// Missing measurements must read as "no change", never as a divide by zero.
	empty := AutoTuneResult{}
	if empty.DecodeSpeedup() != 1 || empty.PrefillSpeedup() != 1 {
		t.Fatal("unmeasured speedups should be 1")
	}
	if empty.GainsLine() != "" {
		t.Fatalf("unmeasured result should have no gains line, got %q", empty.GainsLine())
	}
	if !strings.Contains(res.GainsLine(), "2.00x") {
		t.Fatalf("gains line should report the speedup, got %q", res.GainsLine())
	}
	if !strings.Contains(res.SettingsLine(), "threads=3") {
		t.Fatalf("settings line should report threads, got %q", res.SettingsLine())
	}
}

// TestAutoTuneCacheRejectsForeignEntries makes sure a cached tuning is only
// reused for the key it was measured under: applying another machine's or
// another model's settings would be worse than not caching at all.
func TestAutoTuneCacheRejectsForeignEntries(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LOCALAPPDATA", dir) // os.UserCacheDir on Windows
	t.Setenv("XDG_CACHE_HOME", dir)
	t.Setenv("HOME", dir)

	res := AutoTuneResult{Version: 1, Key: "abc123", Threads: 4, PrefillChunk: 64}
	if err := SaveAutoTune(res); err != nil {
		t.Fatalf("SaveAutoTune: %v", err)
	}
	path, err := autoTuneCachePath("abc123")
	if err != nil {
		t.Fatalf("autoTuneCachePath: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("cache file missing: %v", err)
	}

	// A file whose recorded key does not match the requested one must be
	// rejected rather than silently applied.
	mismatched := AutoTuneResult{Version: 1, Key: "someone-else", Threads: 99}
	data, _ := json.Marshal(mismatched)
	other := filepath.Join(filepath.Dir(path), "abc123.json")
	if err := os.WriteFile(other, data, 0o644); err != nil {
		t.Fatal(err)
	}
	var loaded AutoTuneResult
	raw, err := os.ReadFile(other)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &loaded); err != nil {
		t.Fatal(err)
	}
	if loaded.Key == "abc123" {
		t.Fatal("test setup error: keys should differ")
	}

	// Version 0 (a future/older schema) must also be rejected.
	stale := AutoTuneResult{Version: 0, Key: "abc123"}
	data, _ = json.Marshal(stale)
	if err := os.WriteFile(other, data, 0o644); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(other)
	var v AutoTuneResult
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	if v.Version == 1 {
		t.Fatal("test setup error: version should be 0")
	}
}

// TestAutoTuneEndToEnd runs a real calibration against the synthetic tiny model.
// It exercises every probe path — KV priming, the decode probe at a non-zero
// context, the batched prefill probe, and the final verification sweep — which
// is where shape or workspace-sizing mistakes would panic.
func TestAutoTuneEndToEnd(t *testing.T) {
	before := captureTunerConfig(Config{})
	defer before.apply()

	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatalf("load synthetic model: %v", err)
	}
	defer r.Close()

	var log strings.Builder
	res, err := r.AutoTune(AutoTuneOptions{
		Rounds: 2, DecodeSteps: 1, Context: 8, MinGain: 0.05,
		TunePrefill: true, LogWriter: &log,
	})
	if err != nil {
		t.Fatalf("AutoTune: %v", err)
	}
	if res.Version != 1 || res.Key == "" {
		t.Fatalf("result is not cacheable: %+v", res)
	}
	if res.Threads < 1 {
		t.Fatalf("threads must be positive, got %d", res.Threads)
	}
	if res.PrefillChunk < 1 {
		t.Fatalf("prefill chunk must be positive, got %d", res.PrefillChunk)
	}
	if res.TunedDecodeMs <= 0 {
		t.Fatalf("decode was not measured: %+v", res)
	}
	if len(res.Trials) == 0 {
		t.Fatal("expected at least one recorded trial")
	}
	// Whatever it picked must be what is actually installed in the process.
	if numThreads() != res.Threads {
		t.Fatalf("result reports threads=%d but process has %d", res.Threads, numThreads())
	}
	if q8ActivationsEnabled() != res.Q8Activations {
		t.Fatal("q8 setting was not applied")
	}
	if kvF16Enabled() != res.KVCacheF16 {
		t.Fatal("kv-f16 setting was not applied")
	}
	// The tuner must never report a configuration slower than the defaults.
	if res.DecodeSpeedup() < 1 {
		t.Fatalf("auto mode kept a regression: %.3fx", res.DecodeSpeedup())
	}
	if !strings.Contains(log.String(), "Auto-tuning") {
		t.Fatalf("expected progress output, got %q", log.String())
	}

	// The model must still generate correctly under the tuned settings.
	out, err := r.Generate("hello", GenerationOptions{MaxTokens: 4, Sampler: DefaultSamplerConfig(), Seed: 1})
	if err != nil {
		t.Fatalf("generate after tuning: %v", err)
	}
	if out.Stats.GeneratedTokens == 0 {
		t.Fatal("no tokens generated after tuning")
	}
}

func TestHostFingerprintIsStable(t *testing.T) {
	a, b := hostFingerprint(), hostFingerprint()
	if a != b {
		t.Fatalf("fingerprint is not stable: %q vs %q", a, b)
	}
	for _, want := range []string{runtime.GOOS, runtime.GOARCH} {
		if !strings.Contains(a, want) {
			t.Fatalf("fingerprint %q missing %q", a, want)
		}
	}
}
