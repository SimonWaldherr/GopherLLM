//go:build !amd64

package gopherllm

import "runtime"

// The int8-activation matvecs and the f16 KV cache are amd64-only (VPMADDUBSW
// and F16C respectively), so on every other target these knobs are fixed off
// and the autotuner simply skips them.

func q8ActivationsAvailable() bool { return false }

func q8ActivationsEnabled() bool { return false }

func setQ8Activations(bool) {}

func kvF16Available() bool { return false }

func kvF16Enabled() bool { return false }

func setKVF16(bool) {}

func cpuFeatureString() string { return runtime.GOARCH }
