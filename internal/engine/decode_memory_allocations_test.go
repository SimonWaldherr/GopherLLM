//go:build !race

package engine

import "testing"

// Race instrumentation deliberately drops sync.Pool entries, so warmed matvec
// allocations are measured only in ordinary builds. Decoder parity is also
// tested under the race detector.
func TestVoxtralTeacherForcedDecodeReusesScratch(t *testing.T) {
	cfg, weights := buildTinyVoxtralRealtimeWeights()
	state, err := NewVoxtralRealtimeDecoderState(cfg, weights, 2)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, cfg.Decoder.HiddenSize)
	step := func() {
		out, err := forwardVoxtralRealtimeDecoderStep(cfg, weights, state, input, state.nextPos, false)
		if err != nil || out != nil {
			t.Fatalf("teacher-forced step: %v, %v", out, err)
		}
	}
	for range cfg.Decoder.SlidingWindow {
		step()
	}
	if got := testing.AllocsPerRun(100, step); got != 0 {
		t.Fatalf("warmed teacher-forced step allocates %g times", got)
	}
}
