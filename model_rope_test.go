package main

import (
	"math"
	"testing"
)

func TestBuildRopeInvFreqYarn(t *testing.T) {
	cfg := Config{
		Arch:                      "mistral3",
		RopeTheta:                 1e6,
		RopeDimensionCount:        128,
		RopeScalingType:           "yarn",
		RopeScalingFactor:         16,
		RopeOriginalContextLength: 16384,
		RopeYarnBetaFast:          32,
		RopeYarnBetaSlow:          1,
		RopeYarnLogMultiplier:     1,
		RopeAttentionFactor:       1,
		MaxSeqLen:                 262144,
	}
	inv, mscale := buildRopeInvFreq(cfg, 128)
	if len(inv) != 64 {
		t.Fatalf("pairs = %d, want 64", len(inv))
	}
	// The YaRN attention-magnitude scale is intentionally left at 1 (see
	// buildRopeInvFreqYarn) because enabling it degraded output quality.
	if mscale != 1 {
		t.Fatalf("mscale = %v, want 1", mscale)
	}
	// Highest-frequency pair is extrapolated (unchanged): 1/theta^0 = 1.
	if math.Abs(float64(inv[0]-1)) > 1e-5 {
		t.Fatalf("inv[0] = %v, want ~1 (extrapolated)", inv[0])
	}
	// Lowest-frequency pair is fully interpolated: extrapolated / factor.
	extrapLast := 1 / math.Pow(1e6, 126.0/128.0)
	wantLast := extrapLast / 16
	if math.Abs(float64(inv[63])-wantLast) > wantLast*0.02 {
		t.Fatalf("inv[63] = %v, want ~%v (interpolated)", inv[63], wantLast)
	}
	// Interpolation only ever lowers a frequency, never below extrapolated/factor.
	for i, f := range inv {
		extrap := 1 / math.Pow(1e6, float64(2*i)/128.0)
		if float64(f) > extrap+1e-6 || float64(f) < extrap/16-1e-9 {
			t.Fatalf("inv[%d] = %v out of [extrap/16, extrap] = [%v, %v]", i, f, extrap/16, extrap)
		}
	}
}

func TestBuildRopeInvFreqNonYarnHasUnitMscale(t *testing.T) {
	cfg := Config{RopeTheta: 1e4, RopeDimensionCount: 128}
	inv, mscale := buildRopeInvFreq(cfg, 128)
	if mscale != 1 {
		t.Fatalf("non-yarn mscale = %v, want 1", mscale)
	}
	if math.Abs(float64(inv[0]-1)) > 1e-6 {
		t.Fatalf("inv[0] = %v, want 1", inv[0])
	}
}
