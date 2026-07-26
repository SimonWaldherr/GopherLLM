package gopherllm

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// Auto-tuning: measure this model on this machine and pick the fastest runtime
// settings, instead of shipping one static guess for every combination of
// quantization, model geometry, and CPU.
//
// The static defaults in this repo were each chosen from measurements on ONE
// machine, and several of them are genuinely machine-dependent: whether the
// int8-activation kernels beat the exact float ones depends on AVX2/F16C;
// whether oversubscribed dispatch helps depends on having heterogeneous cores
// (a win on Apple Silicon's P+E mix, pure channel-wakeup overhead on
// homogeneous x86); the best prefill chunk depends on cache size against the
// model's activation working set; the best thread count depends on how many
// cores it takes to saturate memory bandwidth, which differs by an order of
// magnitude between a laptop and a workstation. Auto mode replaces those
// guesses with measurement.
//
// METHODOLOGY, which is the part that matters. Naively timing candidate A then
// candidate B gives garbage on any thermally limited machine: a laptop that
// drops from ~4 GHz burst to ~1.2 GHz sustained will report a 2-3x swing
// between two runs of IDENTICAL code, so an ascending sweep systematically
// favors whatever ran first. This tuner therefore:
//
//   - interleaves candidates in SERPENTINE order (A B C, then C B A) rather
//     than measuring A to completion then B. Plain round-robin is not enough:
//     under a steady thermal ramp the candidate visited first in each round is
//     always sampled at that round's coolest moment and wins systematically;
//     alternating direction cancels the gradient (see sweep);
//   - reports the MEDIAN of each candidate's samples, not the mean or the min;
//   - requires two independent hurdles to change a setting — beat the
//     incumbent's median by MinGain AND win a majority of individual rounds —
//     so a single lucky sample during a clock spike cannot flip a knob;
//   - treats coordinate descent as a HYPOTHESIS. It explores one knob at a time
//     (cheap, and the knobs interact weakly in practice), then one final
//     interleaved sweep judges the starting config, the full proposed set, and
//     each proposed change in isolation. The starting config is always a
//     candidate there, so auto mode can never leave the model slower than it
//     found it (see finalizeDecode);
//   - repeats each probe until it has accumulated a measurable interval, since
//     one forward pass through a small model times as zero against a coarse
//     clock and would leave every candidate tied.
//
// Load-time choices (Metal offload, --prepare-quant) are NOT tuned here: they
// are fixed when the weights are loaded, so probing them would mean reloading
// the model per candidate. AutoTuneResult records what was active so the report
// is honest about it.

// AutoTuneOptions controls calibration cost. Zero values get sane defaults.
type AutoTuneOptions struct {
	// Rounds is how many interleaved passes each candidate gets. More rounds
	// means a more reliable median on a noisy machine. Default 3.
	Rounds int
	// DecodeSteps is how many forward passes make up one decode sample.
	// Default 2.
	DecodeSteps int
	// Context is the KV-cache position the decode probe runs at, so attention
	// cost is representative of real chat rather than an empty cache.
	// Default 1024.
	Context int
	// MinGain is the fractional improvement a candidate must show to displace
	// the incumbent setting. Default 0.03 (3%).
	MinGain float64
	// TunePrefill also calibrates the prompt-prefill chunk size. This is the
	// expensive half of calibration: one sample costs a whole chunk of prefill,
	// which on a multi-billion-parameter model is seconds, not milliseconds.
	TunePrefill bool
	// PrefillRounds is the round count for the prefill sweep, kept separate
	// from Rounds precisely because prefill samples are so much dearer than
	// decode samples. Defaults to 2 (or Rounds, whichever is lower).
	PrefillRounds int
	// MaxPrefillChunk caps the largest chunk size tried. Larger candidates cost
	// proportionally more to evaluate. Default 128.
	MaxPrefillChunk int
	// LogWriter receives human-readable progress. Nil discards.
	LogWriter io.Writer
}

func (o AutoTuneOptions) withDefaults() AutoTuneOptions {
	if o.Rounds <= 0 {
		o.Rounds = 3
	}
	if o.DecodeSteps <= 0 {
		o.DecodeSteps = 2
	}
	if o.Context <= 0 {
		o.Context = 1024
	}
	if o.MinGain <= 0 {
		o.MinGain = 0.03
	}
	if o.PrefillRounds <= 0 {
		o.PrefillRounds = min(2, o.Rounds)
	}
	if o.MaxPrefillChunk <= 0 {
		o.MaxPrefillChunk = 128
	}
	if o.LogWriter == nil {
		o.LogWriter = io.Discard
	}
	return o
}

// AutoTuneTrial is one knob's measured outcome.
type AutoTuneTrial struct {
	Knob       string           `json:"knob"`
	Chosen     string           `json:"chosen"`
	Incumbent  string           `json:"incumbent"`
	Candidates []AutoTuneSample `json:"candidates"`
	Metric     string           `json:"metric"`
}

// AutoTuneSample is the median measurement for one candidate value.
type AutoTuneSample struct {
	Value     string  `json:"value"`
	MedianMs  float64 `json:"median_ms"`
	TokPerSec float64 `json:"tokens_per_second"`
}

// AutoTuneResult is the chosen configuration plus the evidence for it. It is
// what gets cached, so a second run with the same model on the same machine
// starts instantly.
type AutoTuneResult struct {
	Version int    `json:"version"`
	Key     string `json:"key"`
	Model   string `json:"model"`
	Host    string `json:"host"`

	Threads               int  `json:"threads"`
	Q8Activations         bool `json:"q8_activations"`
	KVCacheF16            bool `json:"kv_cache_f16"`
	OversubscribeDispatch bool `json:"oversubscribe_dispatch"`
	PrefillChunk          int  `json:"prefill_chunk"`

	BaselineDecodeMs   float64 `json:"baseline_decode_ms"`
	TunedDecodeMs      float64 `json:"tuned_decode_ms"`
	BaselinePrefillTps float64 `json:"baseline_prefill_tokens_per_second"`
	TunedPrefillTps    float64 `json:"tuned_prefill_tokens_per_second"`

	// MetalActive and PreparedQuant record load-time state that auto mode
	// observed but did not tune (changing either requires reloading weights).
	MetalActive   bool `json:"metal_active"`
	PreparedQuant bool `json:"prepared_quant"`

	Trials     []AutoTuneTrial `json:"trials"`
	ElapsedMs  float64         `json:"elapsed_ms"`
	MeasuredAt time.Time       `json:"measured_at"`
}

// DecodeSpeedup reports the measured decode improvement over the settings that
// were active when tuning started (1.0 means the defaults were already best).
func (r AutoTuneResult) DecodeSpeedup() float64 {
	if r.TunedDecodeMs <= 0 || r.BaselineDecodeMs <= 0 {
		return 1
	}
	return r.BaselineDecodeMs / r.TunedDecodeMs
}

// PrefillSpeedup reports the measured prefill improvement over the starting
// settings.
func (r AutoTuneResult) PrefillSpeedup() float64 {
	if r.TunedPrefillTps <= 0 || r.BaselinePrefillTps <= 0 {
		return 1
	}
	return r.TunedPrefillTps / r.BaselinePrefillTps
}

// Apply installs a (possibly cached) tuning result into the process. Settings
// the running build cannot support are ignored rather than forced.
//
// This mutates process-global kernel selection, so it must not run concurrently
// with an in-flight generation — call it during startup, before serving
// requests. Runner.AutoTune and Runner.AutoTuneOrCached hold the generation lock
// and are safe to call at any time; prefer those.
func (r AutoTuneResult) Apply() {
	if r.Threads > 0 {
		SetNumThreads(r.Threads)
		runtime.GOMAXPROCS(r.Threads)
	}
	setQ8Activations(r.Q8Activations)
	setKVF16(r.KVCacheF16)
	oversubscribeDispatch = r.OversubscribeDispatch
	SetPrefillChunk(r.PrefillChunk)
}

// SettingsLine renders the chosen settings as one line.
func (r AutoTuneResult) SettingsLine() string {
	return fmt.Sprintf("threads=%d q8-activations=%v kv-f16=%v oversubscribe=%v prefill-chunk=%d",
		r.Threads, r.Q8Activations, r.KVCacheF16, r.OversubscribeDispatch, r.PrefillChunk)
}

// GainsLine renders the measured improvement as one line, or "" when nothing
// was measured (e.g. a result carried in from a cache written by an older
// version).
func (r AutoTuneResult) GainsLine() string {
	var parts []string
	if r.BaselineDecodeMs > 0 {
		parts = append(parts, fmt.Sprintf("decode %.1f -> %.1f ms/token (%.2fx)",
			r.BaselineDecodeMs, r.TunedDecodeMs, r.DecodeSpeedup()))
	}
	if r.BaselinePrefillTps > 0 {
		parts = append(parts, fmt.Sprintf("prefill %.1f -> %.1f tok/s (%.2fx)",
			r.BaselinePrefillTps, r.TunedPrefillTps, r.PrefillSpeedup()))
	}
	return strings.Join(parts, ", ")
}

// Summary renders the result as a few human-readable lines.
func (r AutoTuneResult) Summary() string {
	if g := r.GainsLine(); g != "" {
		return r.SettingsLine() + "\n" + g
	}
	return r.SettingsLine()
}

// tunerConfig is the full set of runtime knobs, so a candidate configuration
// can be captured, compared against, and restored as a unit.
type tunerConfig struct {
	threads       int
	q8            bool
	kvF16         bool
	oversubscribe bool
	// prefillChunk is the RAW setting: 0 means "leave the default resolution
	// alone". Storing the resolved number here instead would make restoring a
	// captured config pin that number as an explicit override, silently
	// disabling the env var and the model-aware default.
	prefillChunk int
}

// RuntimeTuning is an opaque snapshot of the process-wide runtime knobs that
// Auto Mode may change. Hosts that tune after initialization can capture it
// first and restore it when switching to an uncalibrated model.
//
// The snapshot deliberately includes the raw prefill override rather than its
// resolved value, preserving environment and model-aware defaults on restore.
type RuntimeTuning struct {
	config tunerConfig
	valid  bool
}

// CaptureRuntimeTuning captures the current process-wide runtime settings.
func CaptureRuntimeTuning() RuntimeTuning {
	return RuntimeTuning{config: captureTunerConfig(Config{}), valid: true}
}

// PrefillChunk reports the captured raw prefill-chunk setting, where 0 means
// "no override, resolve the default at use time". Exported so out-of-package
// callers can tell two captured tunings apart without reaching into internals.
func (t RuntimeTuning) PrefillChunk() int { return t.config.prefillChunk }

// Apply restores a captured runtime tuning. The zero value is a no-op so
// callers may safely leave an optional baseline unset.
func (t RuntimeTuning) Apply() {
	if t.valid {
		t.config.apply()
	}
}

func captureTunerConfig(Config) tunerConfig {
	return tunerConfig{
		threads:       numThreads(),
		q8:            q8ActivationsEnabled(),
		kvF16:         kvF16Enabled(),
		oversubscribe: oversubscribeDispatch,
		prefillChunk:  prefillChunkOverrideValue(),
	}
}

// resolvedPrefillChunk is the chunk size this config actually runs with.
func (c tunerConfig) resolvedPrefillChunk(config Config) int {
	if c.prefillChunk > 0 {
		return c.prefillChunk
	}
	return prefillChunkDefault(config)
}

func (c tunerConfig) apply() {
	SetNumThreads(c.threads)
	runtime.GOMAXPROCS(c.threads)
	setQ8Activations(c.q8)
	setKVF16(c.kvF16)
	oversubscribeDispatch = c.oversubscribe
	SetPrefillChunk(c.prefillChunk)
}

func (c tunerConfig) String() string {
	return fmt.Sprintf("threads=%d q8=%v kv-f16=%v oversubscribe=%v chunk=%d",
		c.threads, c.q8, c.kvF16, c.oversubscribe, c.prefillChunk) // raw chunk: 0 = default
}

// autoTuner holds the probe workspace and current knob state during a run.
type autoTuner struct {
	r    *Runner
	opts AutoTuneOptions
	// probe token ids kept in vocabulary range.
	tokens []uint32
	logits []float32
	trials []AutoTuneTrial
	// primedCache tracks which KV cache already holds probe entries. Priming
	// writes tens of MB, so it must happen once per cache — not once per
	// sample. Toggling the f16 KV setting rebuilds the cache, which shows up
	// here as a new pointer and triggers a re-prime.
	primedCache *KVCache
	// decodeProbeFn overrides the decode measurement. Nil means the real probe;
	// tests substitute scripted timings so the accept/reject logic can be
	// verified deterministically instead of through noisy hardware.
	decodeProbeFn func() float64
}

// measureDecode is the decode metric every knob is judged on.
func (t *autoTuner) measureDecode() float64 {
	if t.decodeProbeFn != nil {
		return t.decodeProbeFn()
	}
	return t.decodeProbe()
}

// workspace returns the probe workspace at a fixed size. Every probe asks for
// the same length so that toggling knobs does not also churn the KV cache
// allocation between samples.
func (t *autoTuner) workspace() (*KVCache, *DecodeBuffer) {
	return t.r.generationWorkspace(t.opts.Context + t.opts.DecodeSteps + maxProbePrefillChunk + 4)
}

// maxProbePrefillChunk is the largest prefill chunk the tuner tries.
const maxProbePrefillChunk = 256

// AutoTune measures the model on the current hardware and applies the fastest
// settings it finds. It holds the generation lock, so no request runs
// concurrently. The returned result is already applied.
func (r *Runner) AutoTune(opts AutoTuneOptions) (AutoTuneResult, error) {
	opts = opts.withDefaults()
	r.genLock.Lock()
	defer r.genLock.Unlock()
	// Calibration writes probe tokens into the shared KV workspace and can also
	// change its f16 representation. A chat prefix must be rebuilt afterward.
	r.clearPrefixCache()

	start := time.Now()
	t := &autoTuner{r: r, opts: opts}
	vocab := max(2, r.config.VocabSize)
	// A short repeating token pattern: contents do not affect timing, only
	// shapes do, and staying in range keeps the embedding lookup honest.
	t.tokens = make([]uint32, max(256, opts.Context/4))
	for i := range t.tokens {
		t.tokens[i] = uint32((i * 7919) % vocab)
	}

	fmt.Fprintf(opts.LogWriter, "Auto-tuning %s on %s (%d CPUs)...\n",
		r.arch, cpuFeatureString(), runtime.NumCPU())

	origin := captureTunerConfig(r.config)

	// Coordinate descent, most-impactful knob first. Each tuneX applies the
	// winner before the next knob is measured, so later knobs are tuned against
	// the configuration that is actually going to run.
	t.tuneQ8Activations()
	t.tuneThreads()
	t.tuneDispatch()
	t.tuneKVF16()
	tuned := captureTunerConfig(r.config)

	// Coordinate descent explores cheaply but accepts each knob against a noisy
	// measurement, so its output is a HYPOTHESIS, not the answer. The decision
	// is made here, in one interleaved sweep over the starting config, the full
	// tuned set, and each individual change on its own. That way a set which
	// only looked good because two knobs each got a lucky sample is rejected,
	// while a single genuine win inside a losing set is still kept — and the
	// starting config is always in the running, so auto mode can never leave
	// the model slower than it found it.
	tuned, baselineDecode, tunedDecode := t.finalizeDecode(origin, tuned)
	tuned.apply()

	// Prefill last, and separately: it is the only knob measured on prompt
	// throughput rather than decode latency, it cannot affect decode, and its
	// samples are by far the most expensive (one sample = one whole chunk of
	// prefill). Because it is a single knob, its interleaved sweep is already
	// the authoritative comparison — the incumbent is one of the candidates —
	// so it needs no separate verification pass.
	var basePrefill, tunedPrefill float64
	if opts.TunePrefill && r.canBatchPrefill() {
		fmt.Fprintf(opts.LogWriter, "  (%.0fs) measuring prefill chunk sizes, this is the slow part...\n",
			time.Since(start).Seconds())
		basePrefill, tunedPrefill = t.tunePrefillChunk()
		tuned = captureTunerConfig(r.config)
	}

	res := AutoTuneResult{
		Version:               1,
		Key:                   r.autoTuneKey(),
		Model:                 r.arch,
		Host:                  hostFingerprint(),
		Threads:               tuned.threads,
		Q8Activations:         tuned.q8,
		KVCacheF16:            tuned.kvF16,
		OversubscribeDispatch: tuned.oversubscribe,
		PrefillChunk:          tuned.resolvedPrefillChunk(r.config),
		BaselineDecodeMs:      baselineDecode,
		TunedDecodeMs:         tunedDecode,
		BaselinePrefillTps:    basePrefill,
		TunedPrefillTps:       tunedPrefill,
		MetalActive:           r.standard.Output.Metal != nil,
		PreparedQuant:         r.standard.Output.Prepared != nil,
		Trials:                t.trials,
		ElapsedMs:             float64(time.Since(start).Microseconds()) / 1000,
		MeasuredAt:            time.Now().UTC(),
	}
	// Pin the winners so nothing later re-derives a default.
	res.Apply()
	return res, nil
}

// finalizeDecode makes the authoritative decode decision. It measures, in ONE
// interleaved sweep so every candidate sees the same thermal conditions: the
// starting config, the full set of changes coordinate descent proposed, and each
// proposed change applied on its own. The winner must beat the starting config
// by MinGain and in a majority of rounds; otherwise the starting config stands.
//
// Measuring the baseline before tuning and the winner after — the obvious
// structure — would instead compare two different thermal states, and on a
// laptop that drops from ~4 GHz to ~1.2 GHz under load that difference dwarfs
// any real tuning gain.
func (t *autoTuner) finalizeDecode(origin, proposed tunerConfig) (tunerConfig, float64, float64) {
	changes := singleDecodeChanges(origin, proposed)
	// Candidate 0 is always the starting config, so "keep the defaults" is a
	// possible outcome and the tuner can never end up slower than it began.
	configs := []decodeChange{{"defaults", origin}}
	if len(changes) > 1 {
		// Only worth measuring the combination when there is more than one
		// change; with exactly one it is the same config as that change alone.
		configs = append(configs, decodeChange{"all", proposed})
	}
	configs = append(configs, changes...)

	if len(configs) == 1 {
		// Nothing changed; still measure once so the report has a real number.
		m := medianFloat(t.sweep(1, 1, func(int) { origin.apply() }, t.measureDecode)[0])
		fmt.Fprintf(t.opts.LogWriter, "  verify: defaults already best (%.1f ms/token)\n", m)
		return origin, m, m
	}

	cands := make([]candidate, len(configs))
	for i, c := range configs {
		cfg := c.cfg
		cands[i] = candidate{label: c.name, apply: func() { cfg.apply() }}
	}
	best, meds := t.evaluate("verify", "ms/token", cands, 0, t.opts.Rounds, true, t.measureDecode)

	if best == 0 {
		fmt.Fprintf(t.opts.LogWriter,
			"  verify: no change beat the defaults (%.1f ms/token) — keeping them\n", meds[0])
	} else {
		fmt.Fprintf(t.opts.LogWriter, "  verify: %s wins, %.1f -> %.1f ms/token\n",
			configs[best].name, meds[0], meds[best])
	}
	// Carry over the prefill chunk either way: it is judged separately on prefill
	// throughput and has no effect on decode.
	out := configs[best].cfg
	out.prefillChunk = proposed.prefillChunk
	return out, meds[0], meds[best]
}

// decodeChange is one candidate configuration for the verification sweep: a
// label for the report and the config it stands for.
type decodeChange struct {
	name string
	cfg  tunerConfig
}

// singleDecodeChanges enumerates the proposed changes that affect decode, each
// applied to the starting config in isolation. The prefill chunk is excluded: it
// does not affect decode and is judged on prefill throughput instead.
func singleDecodeChanges(origin, proposed tunerConfig) []decodeChange {
	var out []decodeChange
	add := func(name string, mutate func(*tunerConfig)) {
		c := origin
		mutate(&c)
		out = append(out, decodeChange{name, c})
	}
	if proposed.q8 != origin.q8 {
		add("q8-activations", func(c *tunerConfig) { c.q8 = proposed.q8 })
	}
	if proposed.threads != origin.threads {
		add(fmt.Sprintf("threads=%d", proposed.threads), func(c *tunerConfig) { c.threads = proposed.threads })
	}
	if proposed.oversubscribe != origin.oversubscribe {
		add("oversubscribe", func(c *tunerConfig) { c.oversubscribe = proposed.oversubscribe })
	}
	if proposed.kvF16 != origin.kvF16 {
		add("kv-cache-f16", func(c *tunerConfig) { c.kvF16 = proposed.kvF16 })
	}
	return out
}

// minProbeSeconds is the shortest interval a probe is allowed to report. The
// Windows clock's granularity is coarse enough (~1 ms, sometimes ~15 ms) that a
// single forward pass through a small model measures as exactly zero, which
// would leave every candidate tied and the tuner blind. Probes therefore repeat
// until they have accumulated a measurable interval.
const minProbeSeconds = 0.02

// maxProbeSteps bounds that repetition so a tiny model cannot turn calibration
// into a long spin.
const maxProbeSteps = 4096

// decodeProbe times real forward passes at opts.Context and returns ms per
// token. The KV cache is pre-populated so attention reads a realistic span
// instead of an empty cache.
//
// Every step writes the SAME cache position: decode cost depends on the length
// of the attended span, not on which slot is being written, so holding the
// position fixed keeps the attended span identical across all candidates and
// lets the step count grow without bounding it by the cache size.
func (t *autoTuner) decodeProbe() float64 {
	r := t.r
	pos := t.opts.Context
	cache, buf := t.workspace()
	if cache != t.primedCache {
		primeKVCache(cache, r.config, pos)
		t.primedCache = cache
	}
	// One untimed pass so scratch buffers and page faults do not land inside
	// the measurement.
	r.forwardTokenInto(cache, buf, t.tokens[0], pos, &t.logits)
	start := time.Now()
	steps := 0
	for {
		r.forwardTokenInto(cache, buf, t.tokens[steps%len(t.tokens)], pos, &t.logits)
		steps++
		el := time.Since(start).Seconds()
		if steps >= maxProbeSteps || (steps >= t.opts.DecodeSteps && el >= minProbeSeconds) {
			return el * 1000 / float64(steps)
		}
	}
}

func (t *autoTuner) prefillProbe(chunk int) float64 {
	r := t.r
	chunk = min(chunk, len(t.tokens))
	if chunk <= 0 {
		return 0
	}
	cache, buf := t.workspace()
	toks := t.tokens[:chunk]
	// No per-sample warm-up here: at seconds per batch it would double the cost
	// of the tuner's most expensive probe. tunePrefillChunk warms up once, with
	// the largest candidate, which sizes every batch buffer the smaller
	// candidates then reuse without growing.
	start := time.Now()
	reps := 0
	for {
		ForwardBatchInto(r.config, r.standard, cache, buf, toks, 0, true, &t.logits)
		reps++
		el := time.Since(start).Seconds()
		if reps >= maxProbeSteps || el >= minProbeSeconds {
			if el <= 0 {
				return 0
			}
			return float64(chunk*reps) / el
		}
	}
}

// sweep measures every candidate opts.Rounds times, interleaved, and returns
// each candidate's per-round samples.
//
// The visit order alternates direction between rounds (A B C, then C B A). Plain
// round-robin is not enough: if the machine is steadily heating, the candidate
// visited first in each round is always measured at the coolest moment of that
// round and wins systematically. Serpentine order cancels that gradient.
func (t *autoTuner) sweep(rounds, n int, apply func(int), measure func() float64) [][]float64 {
	samples := make([][]float64, n)
	for round := 0; round < rounds; round++ {
		for k := 0; k < n; k++ {
			i := k
			if round%2 == 1 {
				i = n - 1 - k
			}
			apply(i)
			samples[i] = append(samples[i], measure())
		}
	}
	return samples
}

// candidate is one setting under test: a label for the report, and the closure
// that installs it.
type candidate struct {
	label string
	apply func()
}

// evaluate measures every candidate in one interleaved sweep, records the trial,
// installs the winner, and returns its index alongside every candidate's median.
//
// This is the ONLY place the acceptance rule lives, so every knob — and the
// final verification pass — is judged by identical criteria. A candidate must
// clear two independent hurdles to displace the incumbent:
//
//  1. beat the incumbent's median by more than MinGain, and
//  2. win in at least half the individual rounds.
//
// A genuinely faster setting clears both; one that got a single lucky sample
// during a clock spike clears neither reliably, and on a machine whose clock
// swings 2-3x under load that distinction is the difference between tuning and
// rolling dice.
func (t *autoTuner) evaluate(knob, metric string, cands []candidate, incumbent, rounds int,
	lowerIsBetter bool, measure func() float64) (int, []float64) {
	samples := t.sweep(rounds, len(cands), func(i int) { cands[i].apply() }, measure)
	meds := make([]float64, len(cands))
	trial := AutoTuneTrial{Knob: knob, Metric: metric, Incumbent: cands[incumbent].label}
	for i := range cands {
		meds[i] = medianFloat(samples[i])
		s := AutoTuneSample{Value: cands[i].label}
		if lowerIsBetter {
			s.MedianMs = meds[i]
		} else {
			s.TokPerSec = meds[i]
		}
		trial.Candidates = append(trial.Candidates, s)
	}
	best := incumbent
	for i := range cands {
		if i == incumbent || meds[i] <= 0 {
			continue
		}
		if betterBy(meds[i], meds[best], lowerIsBetter) > t.opts.MinGain &&
			winsMajority(samples[i], samples[best], lowerIsBetter) {
			best = i
		}
	}
	trial.Chosen = cands[best].label
	t.trials = append(t.trials, trial)
	cands[best].apply()
	return best, meds
}

// choose is evaluate for a simple decode knob: string-labelled values, the
// decode probe, and a one-line report.
func (t *autoTuner) choose(knob string, values []string, incumbent int, apply func(int)) int {
	if len(values) < 2 {
		return incumbent
	}
	cands := make([]candidate, len(values))
	for i, v := range values {
		i := i
		cands[i] = candidate{label: v, apply: func() { apply(i) }}
	}
	best, _ := t.evaluate(knob, "ms/token", cands, incumbent, t.opts.Rounds, true, t.measureDecode)
	fmt.Fprintf(t.opts.LogWriter, "  %-14s %s", knob, values[best])
	if best != incumbent {
		fmt.Fprintf(t.opts.LogWriter, " (was %s)", values[incumbent])
	}
	fmt.Fprintf(t.opts.LogWriter, "\n")
	return best
}

// betterBy returns the fractional improvement of candidate over reference.
func betterBy(candidate, reference float64, lowerIsBetter bool) float64 {
	if reference <= 0 {
		return 0
	}
	if lowerIsBetter {
		return (reference - candidate) / reference
	}
	return (candidate - reference) / reference
}

// winsMajority reports whether a beat b in at least half of the paired rounds.
// Rounds are paired by index, and both series were measured in the same sweep,
// so a pair shares roughly the same thermal state.
func winsMajority(a, b []float64, lowerIsBetter bool) bool {
	n := min(len(a), len(b))
	if n == 0 {
		return false
	}
	wins := 0
	for i := 0; i < n; i++ {
		if lowerIsBetter == (a[i] < b[i]) {
			wins++
		}
	}
	return wins*2 >= n
}

// chooseToggle compares a boolean knob as "on" versus "off". Candidate 0 is
// always "on", so the on/off ordering and the incumbent index are decided in one
// place rather than re-derived (and previously inverted) per knob.
func (t *autoTuner) chooseToggle(knob string, current bool, set func(bool)) {
	incumbent := 0
	if !current {
		incumbent = 1
	}
	t.choose(knob, []string{"on", "off"}, incumbent, func(i int) { set(i == 0) })
}

func (t *autoTuner) tuneQ8Activations() {
	if !q8ActivationsAvailable() {
		return
	}
	t.chooseToggle("q8-activations", q8ActivationsEnabled(), setQ8Activations)
}

func (t *autoTuner) tuneThreads() {
	cands := threadCandidates(runtime.NumCPU())
	if len(cands) < 2 {
		return
	}
	cur := numThreads()
	incumbent := 0
	vals := make([]string, len(cands))
	for i, n := range cands {
		vals[i] = fmt.Sprint(n)
		if n == cur {
			incumbent = i
		}
	}
	t.choose("threads", vals, incumbent, func(i int) { SetNumThreads(cands[i]); runtime.GOMAXPROCS(cands[i]) })
}

// threadCandidates covers the range where memory-bandwidth saturation
// plausibly sits: all logical CPUs, the physical-core count (hyperthread
// siblings often add contention, not bandwidth), and half of that.
func threadCandidates(nproc int) []int {
	if nproc <= 1 {
		return []int{1}
	}
	seen := map[int]bool{}
	var out []int
	for _, n := range []int{nproc, nproc / 2, nproc * 3 / 4, max(1, nproc/4)} {
		if n >= 1 && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(out)))
	return out
}

func (t *autoTuner) tuneDispatch() {
	t.chooseToggle("oversubscribe", oversubscribeDispatch,
		func(on bool) { oversubscribeDispatch = on })
}

func (t *autoTuner) tuneKVF16() {
	if !kvF16Available() {
		return
	}
	t.chooseToggle("kv-cache-f16", kvF16Enabled(), setKVF16)
}

// tunePrefillChunk picks the prompt-prefill chunk size and returns the
// incumbent's and the winner's throughput in tokens/sec.
func (t *autoTuner) tunePrefillChunk() (float64, float64) {
	cands := []int{}
	for _, n := range []int{32, 64, 128, 256} {
		if n <= t.opts.MaxPrefillChunk {
			cands = append(cands, n)
		}
	}
	cur := prefillChunkSize(t.r.config)
	incumbent := -1
	for i, n := range cands {
		if n == cur {
			incumbent = i
		}
	}
	if incumbent < 0 {
		// Whatever is configured is not among the candidates; measure it too so
		// the sweep can never pick something slower than the status quo.
		cands = append(cands, cur)
		incumbent = len(cands) - 1
	}
	if len(cands) < 2 {
		return 0, 0
	}

	// A single warm-up with the LARGEST candidate: it faults in the pages and
	// grows every batch scratch buffer to its high-water mark, so no later
	// sample pays an allocation the others did not.
	largest := cands[0]
	for _, n := range cands {
		largest = max(largest, n)
	}
	SetPrefillChunk(largest)
	t.prefillProbe(largest)

	chunk := cands[incumbent]
	probes := make([]candidate, len(cands))
	for i, n := range cands {
		n := n
		probes[i] = candidate{label: fmt.Sprint(n), apply: func() { chunk = n; SetPrefillChunk(n) }}
	}
	// Throughput, so higher is better — the one knob measured in tok/s rather
	// than ms/token.
	best, meds := t.evaluate("prefill-chunk", "tok/s", probes, incumbent, t.opts.PrefillRounds, false,
		func() float64 { return t.prefillProbe(chunk) })

	fmt.Fprintf(t.opts.LogWriter, "  %-14s %d", "prefill-chunk", cands[best])
	if best != incumbent {
		fmt.Fprintf(t.opts.LogWriter, " (was %d)", cands[incumbent])
	}
	fmt.Fprintf(t.opts.LogWriter, ", %.1f -> %.1f tok/s\n", meds[incumbent], meds[best])
	return meds[incumbent], meds[best]
}

// primeKVCache fills cache positions [0, pos) with small non-zero values so the
// decode probe's attention reads a realistic span. Timing depends on the span
// length, not the values.
func primeKVCache(cache *KVCache, config Config, pos int) {
	if cache == nil || pos <= 0 {
		return
	}
	kDim := config.NKVHeads * config.HeadDim
	vDim := config.KVDim
	if kDim <= 0 || vDim <= 0 {
		return
	}
	k := make([]float32, kDim)
	v := make([]float32, vDim)
	for i := range k {
		k[i] = float32((i%17)-8) * 0.05
	}
	for i := range v {
		v[i] = float32((i%13)-6) * 0.05
	}
	layers := cache.layerCount()
	for l := 0; l < layers; l++ {
		for p := 0; p < pos; p++ {
			cache.storeKV(l, p, k, v)
		}
	}
}

func medianFloat(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	if len(s)%2 == 1 {
		return s[len(s)/2]
	}
	return (s[len(s)/2-1] + s[len(s)/2]) / 2
}

// --- result cache -----------------------------------------------------------

// autoTuneKey identifies "this model on this machine": tuning transfers between
// runs only when both the model geometry/quantization and the hardware match.
func (r *Runner) autoTuneKey() string {
	h := sha256.New()
	fmt.Fprintf(h, "v1|%s|%s|dim=%d|layers=%d|heads=%d/%d|hidden=%d|vocab=%d|out=%v|embd=%v|metal=%t|prepared=%t|%s",
		r.arch, hostFingerprint(),
		r.config.Dim, r.config.NLayers, r.config.NHeads, r.config.NKVHeads,
		r.config.HiddenDim, r.config.VocabSize,
		r.standard.Output.Type, r.standard.TokenEmbd.Type,
		r.standard.Output.Metal != nil, r.standard.Output.Prepared != nil,
		quantMixFingerprint(r))
	return hex.EncodeToString(h.Sum(nil))[:24]
}

func quantMixFingerprint(r *Runner) string {
	if len(r.standard.Layers) == 0 {
		return "none"
	}
	l := r.standard.Layers[0]
	return fmt.Sprintf("%v/%v/%v/%v", l.WQ.Type, l.WO.Type, l.W1.Type, l.W2.Type)
}

func hostFingerprint() string {
	return fmt.Sprintf("%s/%s/%s/cpu%d", runtime.GOOS, runtime.GOARCH, cpuFeatureString(), runtime.NumCPU())
}

func autoTuneCachePath(key string) (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "gopherllm", "autotune", key+".json"), nil
}

// LoadAutoTune returns a previously measured result for this model+machine, or
// ok=false when none is cached.
func (r *Runner) LoadAutoTune() (AutoTuneResult, bool) {
	key := r.autoTuneKey()
	path, err := autoTuneCachePath(key)
	if err != nil {
		return AutoTuneResult{}, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return AutoTuneResult{}, false
	}
	var res AutoTuneResult
	if err := json.Unmarshal(data, &res); err != nil || res.Version != 1 || res.Key != key {
		return AutoTuneResult{}, false
	}
	return res, true
}

// SaveAutoTune persists a result so later runs can skip calibration. Cache
// failures are reported but never fatal — a missing cache only costs time.
func SaveAutoTune(res AutoTuneResult) error {
	path, err := autoTuneCachePath(res.Key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// ValidAutoTuneEffort reports whether effort is an accepted public calibration
// level. An empty value means the default balanced profile.
func ValidAutoTuneEffort(effort string) bool {
	switch effort {
	case "", "quick", "balanced", "thorough":
		return true
	default:
		return false
	}
}

// AutoTuneOptionsForEffort maps a calibration effort level onto AutoTuneOptions.
// It is the single source of truth for what "quick", "balanced", and
// "thorough" mean, shared by the CLI's --auto-effort flag and the HTTP API's
// POST /autotune/run so the two surfaces can never silently drift apart.
// Unrecognized values (including "") fall back to "balanced".
//
// Measured on a 3B Q4_K_M model: quick ~10-20s, balanced ~75s, thorough
// several minutes. The extra time buys interleaved rounds, which is exactly
// what helps on a machine whose clock swings under sustained load.
func AutoTuneOptionsForEffort(effort string) AutoTuneOptions {
	switch effort {
	case "quick":
		// Decode knobs only: a prefill sample costs a whole chunk of prompt
		// processing, which is seconds on a multi-billion-parameter model.
		return AutoTuneOptions{Rounds: 2, DecodeSteps: 1, Context: 512, MinGain: 0.06}
	case "thorough":
		return AutoTuneOptions{Rounds: 5, DecodeSteps: 3, Context: 2048, MinGain: 0.02,
			TunePrefill: true, PrefillRounds: 3, MaxPrefillChunk: 256}
	default:
		return AutoTuneOptions{Rounds: 3, DecodeSteps: 2, Context: 512, MinGain: 0.03,
			TunePrefill: true, PrefillRounds: 2, MaxPrefillChunk: 128}
	}
}

// AutoTuneOrCached applies a cached result when one exists, otherwise measures
// and caches a new one. refresh forces re-measurement.
func (r *Runner) AutoTuneOrCached(opts AutoTuneOptions, refresh bool) (AutoTuneResult, bool, error) {
	if !refresh {
		if res, ok := r.LoadAutoTune(); ok {
			// Under the generation lock, like AutoTune, so installing a cached
			// result cannot race a request that is already running.
			r.genLock.Lock()
			r.clearPrefixCache()
			res.Apply()
			r.genLock.Unlock()
			return res, true, nil
		}
	}
	res, err := r.AutoTune(opts)
	if err != nil {
		return res, false, err
	}
	if err := SaveAutoTune(res); err != nil {
		fmt.Fprintf(opts.withDefaults().LogWriter, "  (could not cache tuning: %v)\n", err)
	}
	return res, false, nil
}
