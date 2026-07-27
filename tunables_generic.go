//go:build !amd64

package gopherllm

import "runtime"

// The int8-activation matvecs are amd64-only (VPMADDUBSW). Non-amd64 builds
// do have a correct scalar f16 KV-cache implementation: it is opt-in by
// default because it trades decode speed for half the cache footprint.

func q8ActivationsAvailable() bool { return false }

func q8ActivationsEnabled() bool { return false }

func setQ8Activations(bool) {}

func kvF16Available() bool { return true }

func kvF16Enabled() bool { return useF16KVCache }

func setKVF16(on bool) { useF16KVCache = on }

func cpuFeatureString() string { return runtime.GOARCH }
