package gopherllm

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"time"
)

// Metal is chosen when the weights are loaded, not while they are resident, so
// AutoTune — which works on an already-loaded Runner — structurally cannot
// decide it. That left a large speedup (measured ~1.8x on a 3B Q4_K_M) behind
// a flag most people never pass. ProbeMetal closes that gap by loading the
// model both ways and timing each, then caching the verdict per model+machine
// so only the first run pays for it.

// MetalProbe is the measured verdict on whether to enable Metal.
type MetalProbe struct {
	Version int    `json:"version"`
	Key     string `json:"key"`
	// Available reports whether this build and machine can use Metal at all.
	Available bool `json:"available"`
	// UseMetal is the recommendation.
	UseMetal bool `json:"use_metal"`
	// Reason is a short human-readable explanation, shown by the CLI.
	Reason string `json:"reason"`
	// MetalMsPerToken and CPUMsPerToken are 0 when that side was not measured
	// (for example on a build without Metal).
	MetalMsPerToken float64   `json:"metal_ms_per_token"`
	CPUMsPerToken   float64   `json:"cpu_ms_per_token"`
	Speedup         float64   `json:"speedup"`
	MeasuredAt      time.Time `json:"measured_at"`
}

// metalProbeMinGain is how much faster Metal must be before the probe switches
// to it. Two loads of the same weights are never timed under identical thermal
// conditions, so a near-tie is noise, and the conservative choice is the
// configuration the caller already asked for.
const metalProbeMinGain = 1.05

const metalProbeVersion = 1

// SummaryLine renders the verdict as one line for the CLI.
func (p MetalProbe) SummaryLine() string {
	if !p.Available {
		return "Metal: unavailable (" + p.Reason + ")"
	}
	if p.MetalMsPerToken <= 0 || p.CPUMsPerToken <= 0 {
		return fmt.Sprintf("Metal: %v (%s)", p.UseMetal, p.Reason)
	}
	return fmt.Sprintf("Metal: %v — %.1f ms/token with, %.1f without (%.2fx)",
		p.UseMetal, p.MetalMsPerToken, p.CPUMsPerToken, p.Speedup)
}

// metalProbeKey identifies a model+machine pairing. The file's size and
// modification time stand in for its contents, which is enough to notice a
// swapped or re-quantized model without hashing gigabytes. FNV-1a for the same
// reason as autoTuneKey: this only names a file in the caller's cache
// directory, so there is no adversary to defend against, and a cryptographic
// hash would pull the whole crypto tree into the dependency closure of every
// program embedding this library.
func metalProbeKey(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	h := fnv.New128a()
	fmt.Fprintf(h, "metal-v%d|%s|%d|%d|%s",
		metalProbeVersion, filepath.Base(path), info.Size(), info.ModTime().UnixNano(), hostFingerprint())
	return hex.EncodeToString(h.Sum(nil))[:24], nil
}

func metalProbeCachePath(key string) (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "gopherllm", "autotune", "metal-"+key+".json"), nil
}

// LoadMetalProbe returns a cached verdict for this model+machine.
func LoadMetalProbe(path string) (MetalProbe, bool) {
	key, err := metalProbeKey(path)
	if err != nil {
		return MetalProbe{}, false
	}
	cachePath, err := metalProbeCachePath(key)
	if err != nil {
		return MetalProbe{}, false
	}
	data, err := os.ReadFile(cachePath)
	if err != nil {
		return MetalProbe{}, false
	}
	var probe MetalProbe
	if err := json.Unmarshal(data, &probe); err != nil || probe.Version != metalProbeVersion || probe.Key != key {
		return MetalProbe{}, false
	}
	return probe, true
}

// SaveMetalProbe persists a verdict. A cache failure only costs time, so it is
// reported rather than fatal.
func SaveMetalProbe(probe MetalProbe) error {
	cachePath, err := metalProbeCachePath(probe.Key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(probe, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(cachePath, append(data, '\n'), 0o644)
}

// measureDecodeMsPerToken loads the model with the given options and times a
// short greedy generation, returning milliseconds per generated token.
func measureDecodeMsPerToken(path string, options LoadOptions, tokens int) (float64, error) {
	options.LogWriter = io.Discard
	runner, _, err := RunnerFromPathWithOptions(path, options)
	if err != nil {
		return 0, err
	}
	defer runner.Close()

	genOptions := DefaultGenerationOptions()
	genOptions.MaxTokens = tokens
	genOptions.Sampler.Temperature = 0
	genOptions.Sampler.TopK = 1
	genOptions.Seed = 1
	genOptions.SystemPrompt = ""

	// One short warm-up so the comparison is not dominated by first-touch page
	// faults on the memory-mapped weights.
	if _, err := runner.Generate("Warm up.", withMaxTokens(genOptions, 2)); err != nil {
		return 0, err
	}
	result, err := runner.Generate("Count from one to twenty.", genOptions)
	if err != nil {
		return 0, err
	}
	if result.Stats.GeneratedTokens <= 0 {
		return 0, fmt.Errorf("probe generated no tokens")
	}
	return float64(result.Stats.DecodeTime.Microseconds()) / 1000 / float64(result.Stats.GeneratedTokens), nil
}

func withMaxTokens(o GenerationOptions, n int) GenerationOptions {
	o.MaxTokens = n
	return o
}

// ProbeMetal measures this model on this machine with and without Metal and
// reports which is faster. tokens is the generation length per side; 0 picks a
// sensible default.
//
// It loads the weights twice and discards both runners, so callers should use
// the cached path (ProbeMetalOrCached) rather than calling this per run.
func ProbeMetal(path string, base LoadOptions, tokens int) (MetalProbe, error) {
	key, err := metalProbeKey(path)
	if err != nil {
		return MetalProbe{}, err
	}
	probe := MetalProbe{Version: metalProbeVersion, Key: key, MeasuredAt: time.Now().UTC()}

	if !MetalAvailable() {
		probe.Available = false
		probe.UseMetal = false
		probe.Reason = MetalError()
		if probe.Reason == "" {
			probe.Reason = "not available on this build or machine"
		}
		return probe, nil
	}
	probe.Available = true
	if tokens <= 0 {
		tokens = 24
	}

	cpuOptions := base
	cpuOptions.UseMetal = false
	cpuMs, err := measureDecodeMsPerToken(path, cpuOptions, tokens)
	if err != nil {
		return probe, fmt.Errorf("pure-Go probe: %w", err)
	}

	metalOptions := base
	metalOptions.UseMetal = true
	metalMs, err := measureDecodeMsPerToken(path, metalOptions, tokens)
	if err != nil {
		// A Metal load that fails is a definitive answer, not an error: run
		// without it.
		probe.CPUMsPerToken = cpuMs
		probe.UseMetal = false
		probe.Reason = "Metal load failed (" + err.Error() + "); using the pure-Go kernels"
		return probe, nil
	}

	probe.CPUMsPerToken = cpuMs
	probe.MetalMsPerToken = metalMs
	if metalMs > 0 {
		probe.Speedup = cpuMs / metalMs
	}
	switch {
	case probe.Speedup >= metalProbeMinGain:
		probe.UseMetal = true
		probe.Reason = fmt.Sprintf("Metal decoded %.2fx faster", probe.Speedup)
	default:
		probe.UseMetal = false
		probe.Reason = fmt.Sprintf("Metal was not faster here (%.2fx); staying on the pure-Go kernels", probe.Speedup)
	}
	return probe, nil
}

// ProbeMetalOrCached returns a cached verdict when one exists for this
// model+machine, otherwise measures and caches a new one. refresh forces a
// re-measurement. The bool reports whether the result came from the cache.
func ProbeMetalOrCached(path string, base LoadOptions, tokens int, refresh bool) (MetalProbe, bool, error) {
	if !refresh {
		if probe, ok := LoadMetalProbe(path); ok {
			return probe, true, nil
		}
	}
	probe, err := ProbeMetal(path, base, tokens)
	if err != nil {
		return probe, false, err
	}
	// Only a real measurement is worth caching; an "unavailable" verdict is
	// cheap to recompute and would go stale the moment the build changes.
	if probe.Available {
		if saveErr := SaveMetalProbe(probe); saveErr != nil {
			return probe, false, nil
		}
	}
	return probe, false, nil
}
