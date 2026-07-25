//go:build amd64

package gopherllm

// Runtime-settable views of the amd64 kernel toggles, so the autotuner can A/B
// them on the live model instead of trusting a compile-time or env-var guess.
// On other targets these are compile-time false (tunables_generic.go).

func q8ActivationsAvailable() bool { return hasAVX2 && hasF16C }

func q8ActivationsEnabled() bool { return useQ8Activations }

func setQ8Activations(on bool) { useQ8Activations = on && q8ActivationsAvailable() }

func kvF16Available() bool { return hasAVX2 && hasF16C }

func kvF16Enabled() bool { return useF16KVCache }

func setKVF16(on bool) { useF16KVCache = on && kvF16Available() }

func cpuFeatureString() string {
	s := "amd64"
	if hasAVX2 {
		s += "+avx2"
	}
	if hasF16C {
		s += "+f16c"
	}
	return s
}
