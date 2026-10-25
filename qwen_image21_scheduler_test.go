package gopherllm

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
)

func qi21BaseScheduler() qi21SchedulerConfig {
	terminal := .02
	return qi21SchedulerConfig{ClassName: "FlowMatchEulerDiscreteScheduler", TrainSteps: 1000, BaseSequence: 256, MaxSequence: 8192, BaseShift: .5, MaxShift: .9, Shift: 1, ShiftTerminal: &terminal, Dynamic: true, ShiftType: "exponential"}
}

func TestQwenImage21SchedulerReference(t *testing.T) {
	data, err := os.ReadFile("testdata/qwen_image21_transformer/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref qi21TransformerReference
	if err = json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	for _, tc := range ref.Schedules {
		got, err := qi21Schedule(qi21BaseScheduler(), tc.Steps, tc.Tokens, nil)
		if err != nil {
			t.Fatal(err)
		}
		qi21AssertNear(t, got, tc.Expected, 3e-7)
	}
	saved := []float32{1, .978453, .95418, .926626, .89508, .845148, .704534, .414568}
	cfg := qi21BaseScheduler()
	cfg.Dynamic = false
	cfg.ShiftTerminal = nil
	got, err := qi21Schedule(cfg, 40, 16384, saved)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 9 || !slices.Equal(got[:8], saved) || got[8] != 0 {
		t.Fatalf("distilled schedule changed: %v", got)
	}
	latent := []float32{1, -1, .5}
	prediction := []float32{2, 3, -4}
	if err = qi21EulerStep(latent, prediction, 1, .5); err != nil {
		t.Fatal(err)
	}
	qi21AssertNear(t, latent, []float32{0, -2.5, 2.5}, 0)
}

func TestQwenImage21SchedulerRejectsInvalidSchedules(t *testing.T) {
	for _, saved := range [][]float32{{1, 1}, {1, .5, .7}, {1, 0}, {1, -1}, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := qi21Schedule(qi21BaseScheduler(), 40, 256, saved); err == nil {
			t.Fatalf("accepted %v", saved)
		}
	}
	if _, err := qi21Schedule(qi21BaseScheduler(), 1, 256, nil); err == nil {
		t.Fatal("accepted singular terminal stretch")
	}
	for _, bad := range []int{0, -1, 1001} {
		if _, err := qi21Schedule(qi21BaseScheduler(), bad, 256, nil); err == nil {
			t.Fatal("accepted invalid step count")
		}
	}
	x := []float32{1, 2}
	before := slices.Clone(x)
	if err := qi21EulerStep(x, []float32{1, float32(math.NaN())}, 1, .5); err == nil || !slices.Equal(x, before) {
		t.Fatal("invalid prediction changed latent state")
	}
	x = []float32{1, math.MaxFloat32}
	before = slices.Clone(x)
	if err := qi21EulerStep(x, []float32{1, -math.MaxFloat32}, 1, 0); err == nil || !slices.Equal(x, before) {
		t.Fatal("overflowing step changed latent state")
	}
}
