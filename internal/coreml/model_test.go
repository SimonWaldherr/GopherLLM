package coreml

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

type testBackend struct {
	predictFn func([]float32) error
	closeFn   func()
}

func (b *testBackend) predict(_, _, _ []int32, out []float32) error {
	if b.predictFn != nil {
		return b.predictFn(out)
	}
	clear(out)
	return nil
}

func (b *testBackend) close() {
	if b.closeFn != nil {
		b.closeFn()
	}
}

func TestModelInputValidationAndClose(t *testing.T) {
	calls, closes := 0, 0
	m := &Model{info: Info{BatchSize: 1, SequenceLength: 4, Dimension: 3}, backend: &testBackend{
		predictFn: func(out []float32) error { calls++; clear(out); return nil },
		closeFn:   func() { closes++ },
	}}
	ids, mask, out := make([]int32, 4), make([]int32, 4), make([]float32, 3)
	for _, tc := range []struct{ ids, mask, types, output int }{{3, 4, 0, 3}, {4, 3, 0, 3}, {4, 4, 3, 3}, {4, 4, 0, 2}} {
		if err := m.Predict(make([]int32, tc.ids), make([]int32, tc.mask), make([]int32, tc.types), make([]float32, tc.output)); err == nil {
			t.Fatal("accepted a buffer whose length differs from the model interface")
		}
	}
	if calls != 0 {
		t.Fatal("invalid buffers reached the native backend")
	}
	if err := m.Predict(ids, mask, nil, out); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("predictions=%d, want 1", calls)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil || closes != 1 {
		t.Fatalf("second close=%v, backend releases=%d", err, closes)
	}
	if err := m.Predict(ids, mask, nil, out); err == nil || calls != 1 {
		t.Fatal("closed model reached the native backend")
	}
	if m.Info().BatchSize != 1 || m.Info().SequenceLength != 4 || (*Model)(nil).Info() != (Info{}) {
		t.Fatal("model metadata changed after close")
	}
}

func TestModelBatchLengthValidation(t *testing.T) {
	for _, batch := range []int{1, 2, 4, 8} {
		calls := 0
		m := &Model{info: Info{BatchSize: batch, SequenceLength: 3, Dimension: 2}, backend: &testBackend{
			predictFn: func(out []float32) error {
				calls++
				for i := range out {
					out[i] = float32(i + 1)
				}
				return nil
			},
		}}
		inputCount, outputCount := batch*3, batch*2
		for _, tc := range []struct{ ids, mask, types, output int }{
			{inputCount - 1, inputCount, 0, outputCount},
			{inputCount, inputCount - 1, 0, outputCount},
			{inputCount, inputCount, inputCount - 1, outputCount},
			{inputCount, inputCount, 0, outputCount - 1},
		} {
			if err := m.Predict(make([]int32, tc.ids), make([]int32, tc.mask), make([]int32, tc.types), make([]float32, tc.output)); err == nil {
				t.Fatalf("batch %d accepted incomplete input/output buffers: %+v", batch, tc)
			}
		}
		if calls != 0 {
			t.Fatalf("batch %d sent invalid buffers to the backend", batch)
		}
		for _, types := range [][]int32{nil, make([]int32, inputCount)} {
			out := make([]float32, outputCount)
			if err := m.Predict(make([]int32, inputCount), make([]int32, inputCount), types, out); err != nil {
				t.Fatalf("batch %d: %v", batch, err)
			}
			if out[outputCount-1] != float32(outputCount) {
				t.Fatalf("batch %d lost the final row: %v", batch, out)
			}
		}
		if calls != 2 {
			t.Fatalf("batch %d predictions=%d, want 2", batch, calls)
		}
	}
}

func TestModelRejectsInvalidFixedShape(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, info := range []Info{
		{BatchSize: 0, SequenceLength: 3, Dimension: 2},
		{BatchSize: -1, SequenceLength: 3, Dimension: 2},
		{BatchSize: maxBatchSize + 1, SequenceLength: 3, Dimension: 2},
		{BatchSize: 1, SequenceLength: 0, Dimension: 2},
		{BatchSize: 1, SequenceLength: 3, Dimension: -1},
		{BatchSize: 8, SequenceLength: maxBufferElements/8 + 1, Dimension: 2},
		{BatchSize: 8, SequenceLength: 3, Dimension: maxBufferElements/8 + 1},
		{BatchSize: 8, SequenceLength: maxInt, Dimension: 2},
		{BatchSize: 8, SequenceLength: 3, Dimension: maxInt},
	} {
		m := &Model{info: info, backend: &testBackend{predictFn: func([]float32) error {
			t.Fatal("invalid fixed shape reached the backend")
			return nil
		}}}
		if err := m.Predict(nil, nil, nil, nil); err == nil {
			t.Fatalf("accepted invalid fixed shape: %+v", info)
		}
	}
}

func TestModelRejectsNonFiniteOutput(t *testing.T) {
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		m := &Model{info: Info{BatchSize: 4, SequenceLength: 1, Dimension: 1}, backend: &testBackend{
			predictFn: func(out []float32) error { clear(out); out[3] = value; return nil },
		}}
		if err := m.Predict(make([]int32, 4), []int32{1, 1, 1, 1}, nil, make([]float32, 4)); err == nil {
			t.Fatal("accepted a non-finite embedding")
		}
	}
	want := errors.New("prediction failed")
	m := &Model{info: Info{BatchSize: 1, SequenceLength: 1, Dimension: 1}, backend: &testBackend{predictFn: func([]float32) error { return want }}}
	if err := m.Predict([]int32{0}, []int32{1}, nil, make([]float32, 1)); !errors.Is(err, want) {
		t.Fatalf("prediction error=%v, want %v", err, want)
	}
}

func TestCloseWaitsForPrediction(t *testing.T) {
	started, release, predicted, closed := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var active, releasedWhileActive atomic.Bool
	m := &Model{info: Info{BatchSize: 1, SequenceLength: 1, Dimension: 1}, backend: &testBackend{
		predictFn: func(out []float32) error {
			active.Store(true)
			close(started)
			<-release
			out[0] = 1
			active.Store(false)
			return nil
		},
		closeFn: func() { releasedWhileActive.Store(active.Load()) },
	}}
	go func() {
		defer close(predicted)
		if err := m.Predict([]int32{0}, []int32{1}, nil, make([]float32, 1)); err != nil {
			t.Errorf("prediction: %v", err)
		}
	}()
	<-started
	go func() { _ = m.Close(); close(closed) }()
	close(release)
	<-predicted
	<-closed
	if releasedWhileActive.Load() {
		t.Fatal("released the native model during prediction")
	}
}

func TestConfigAndCompiledPathValidation(t *testing.T) {
	valid := Config{InputIDs: "input_ids", AttentionMask: "attention_mask", Output: "embedding"}
	if err := validateConfig(valid); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Config){
		func(c *Config) { c.ComputeUnits = All + 1 },
		func(c *Config) { c.InputIDs = "" },
		func(c *Config) { c.AttentionMask = c.InputIDs },
		func(c *Config) { c.TokenTypeIDs = c.InputIDs },
		func(c *Config) { c.Output = "bad\x00name" },
		func(c *Config) { c.InputIDs = "\xff" },
	} {
		c := valid
		change(&c)
		if err := validateConfig(c); err == nil {
			t.Fatal("accepted invalid feature configuration")
		}
	}
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.mlmodelc")
	if err := os.Mkdir(modelPath, 0700); err != nil {
		t.Fatal(err)
	}
	got, err := compiledModelPath(modelPath)
	if err != nil || !filepath.IsAbs(got) {
		t.Fatalf("compiled path=%q, err=%v", got, err)
	}
	file := filepath.Join(dir, "file.mlmodelc")
	if err := os.WriteFile(file, nil, 0600); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{file, filepath.Join(dir, "model.mlpackage"), filepath.Join(dir, "missing.mlmodelc"), "bad\x00.mlmodelc"} {
		if _, err := compiledModelPath(path); err == nil {
			t.Fatalf("accepted invalid compiled model path %q", path)
		}
	}
	t.Run("canonical_alias", func(t *testing.T) {
		alias := filepath.Join(dir, "alias.mlmodelc")
		if err := os.Symlink(modelPath, alias); err != nil {
			t.Skip(err)
		}
		if canonical, err := compiledModelPath(alias); err != nil || canonical != got {
			t.Fatalf("alias=%q, err=%v; want %q", canonical, err, got)
		}
	})
}
