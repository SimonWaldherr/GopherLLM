//go:build coreml

package gopherllm

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	nativecoreml "github.com/SimonWaldherr/GopherLLM/internal/coreml"
)

func TestCoreMLFFNTuningSelectionAndLifetime(t *testing.T) {
	const cpu, ane = nativecoreml.CPUOnly, nativecoreml.CPUAndNeuralEngine
	failure := errors.New("candidate rejected")
	for _, tc := range []struct {
		name          string
		durations     map[int]time.Duration
		loadErrors    map[int]error
		measureErrors map[int]error
		want          nativecoreml.ComputeUnits
		loads         int
		fails         bool
	}{
		{name: "clear CPU gain", durations: map[int]time.Duration{1: 5, 3: 6}, want: cpu, loads: 3},
		{name: "ANE faster", durations: map[int]time.Duration{1: 12}, want: ane, loads: 2},
		{name: "small CPU gain", durations: map[int]time.Duration{1: 95, 2: 100}, want: ane, loads: 2},
		{name: "threshold inclusive", durations: map[int]time.Duration{1: 9, 3: 9}, want: cpu, loads: 3},
		{name: "transient gain", durations: map[int]time.Duration{1: 5, 3: 12}, want: ane, loads: 4},
		{name: "CPU unavailable", loadErrors: map[int]error{1: failure}, want: ane, loads: 2},
		{name: "ANE unavailable", loadErrors: map[int]error{2: failure}, want: cpu, loads: 3},
		{name: "recheck failure", durations: map[int]time.Duration{1: 5}, measureErrors: map[int]error{3: failure}, want: ane, loads: 4},
		{name: "both invalid", loadErrors: map[int]error{1: failure, 2: failure}, loads: 2, fails: true},
		{name: "both predictions fail", measureErrors: map[int]error{1: failure, 2: failure}, loads: 2, fails: true},
		{name: "only CPU fails recheck", loadErrors: map[int]error{2: failure}, measureErrors: map[int]error{3: failure}, loads: 3, fails: true},
		{name: "clock resolution", durations: map[int]time.Duration{1: 0}, want: ane, loads: 2},
		{name: "final reopen fails", durations: map[int]time.Duration{1: 5, 3: 12}, loadErrors: map[int]error{4: failure}, loads: 4, fails: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var models []*fakeCoreMLFFN
			calls := 0
			units := make(map[*coreMLFFNLayer]nativecoreml.ComputeUnits)
			attempts := make(map[*coreMLFFNLayer]int)
			load := func(unit nativecoreml.ComputeUnits) (*coreMLFFNLayer, error) {
				for _, model := range models {
					if !model.closed {
						t.Fatal("loading a candidate while the previous model is still live")
					}
				}
				calls++
				if err := tc.loadErrors[calls]; err != nil {
					return nil, err
				}
				model := &fakeCoreMLFFN{}
				models = append(models, model)
				layer := &coreMLFFNLayer{model: model}
				units[layer], attempts[layer] = unit, calls
				return layer, nil
			}
			measure := func(layer *coreMLFFNLayer) (time.Duration, error) {
				if err := tc.measureErrors[attempts[layer]]; err != nil {
					return 0, err
				}
				if duration, ok := tc.durations[attempts[layer]]; ok {
					return duration * time.Millisecond, nil
				}
				return 10 * time.Millisecond, nil
			}
			selected, err := tuneCoreMLFFN(load, measure, io.Discard)
			if (err != nil) != tc.fails || calls != tc.loads {
				t.Fatalf("calls=%d, err=%v; want calls=%d, failure=%v", calls, err, tc.loads, tc.fails)
			}
			if !tc.fails && (selected == nil || units[selected] != tc.want) {
				t.Fatalf("wrong selected device: %v", units[selected])
			}
			for _, model := range models {
				chosen := selected != nil && selected.model == model
				if model.closed == chosen {
					t.Fatal("selected model closed or rejected candidate leaked")
				}
			}
		})
	}
}

func TestCoreMLFFNTuningUsesSourceValidation(t *testing.T) {
	r, manifest, path := coreMLFFNManifestFixture(t)
	writeCoreMLFFNManifest(t, path, manifest)
	var rejected, selected *fakeCoreMLFFN
	var log bytes.Buffer
	err := r.setCoreMLFFN([]string{path}, CoreMLFFNOptions{AutoTune: true, LogWriter: &log}, func(_ string, units nativecoreml.ComputeUnits) (coreMLFFNPredictor, error) {
		model := fakeCoreMLFFNForSource(t, r, manifest)
		if units == nativecoreml.CPUOnly {
			model.predict = func(_, out []float32) error { clear(out); return nil }
			rejected = model
		} else {
			predict := model.predict
			model.predict = func(in, out []float32) error {
				// Simulate native latency so coarse platform clocks cannot
				// reject a valid fake predictor as a zero-duration trial.
				time.Sleep(time.Millisecond)
				return predict(in, out)
			}
			selected = model
		}
		return model, nil
	})
	if err != nil || rejected == nil || !rejected.closed || selected == nil || selected.closed || r.coreMLFFN[manifest.Layer].model != selected {
		t.Fatalf("quality rejection or selection failed: %v", err)
	}
	defer r.coreMLFFN.close()
	if !strings.Contains(log.String(), "CPU rejected:") || !strings.Contains(log.String(), "selected CPU/Neural Engine") {
		t.Fatalf("missing calibration diagnostics: %s", &log)
	}
	if err := r.setCoreMLFFN([]string{path}, CoreMLFFNOptions{CPUOnly: true, AutoTune: true}, nil); err == nil || selected.closed || r.coreMLFFN[manifest.Layer].model != selected {
		t.Fatal("conflicting options changed the attached model")
	}
	var failed []*fakeCoreMLFFN
	err = r.setCoreMLFFN([]string{path}, CoreMLFFNOptions{AutoTune: true}, func(_ string, _ nativecoreml.ComputeUnits) (coreMLFFNPredictor, error) {
		model := fakeCoreMLFFNForSource(t, r, manifest)
		model.predict = func(_, _ []float32) error { return errors.New("prediction failed") }
		failed = append(failed, model)
		return model, nil
	})
	if err == nil || len(failed) != 2 || selected.closed || r.coreMLFFN[manifest.Layer].model != selected {
		t.Fatalf("failed calibration replaced the attached model: %v", err)
	}
	for _, model := range failed {
		if !model.closed {
			t.Fatal("failed calibration leaked a candidate")
		}
	}
}
