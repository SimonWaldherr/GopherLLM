//go:build !amd64

package gopherllm

import "runtime"

// The int8-activation matvecs are amd64-only (VPMADDUBSW). Non-amd64 builds
// do have a correct f16 KV-cache implementation (NEON on Apple Silicon,
// scalar elsewhere); it is opt-in by default and available to the autotuner.

func q8ActivationsAvailable() bool { return false }

func q8ActivationsEnabled() bool { return false }

func setQ8Activations(bool) {}

func kvF16Available() bool { return true }

func kvF16Enabled() bool { return useF16KVCache }

func setKVF16(on bool) { useF16KVCache = on }

func cpuFeatureString() string { return runtime.GOARCH }
