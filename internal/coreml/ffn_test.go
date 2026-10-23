package coreml

import (
	"errors"
	"math"
	"sync/atomic"
	"testing"
)

type testFFNBackend struct {
	predictFn func([]float32, []float32) error
	closeFn   func()
}

func (b *testFFNBackend) predict(input, output []float32) error {
	if b.predictFn != nil {
		return b.predictFn(input, output)
	}
	copy(output, input)
	return nil
}

func (b *testFFNBackend) close() {
	if b.closeFn != nil {
		b.closeFn()
	}
}

func TestFFNValidatesShapeAndBuffers(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, info := range []FFNInfo{{Tokens: 0, Dimension: 2}, {Tokens: -1, Dimension: 2}, {Tokens: 129, Dimension: 2}, {Tokens: 1, Dimension: 0}, {Tokens: 1, Dimension: -1}, {Tokens: 1, Dimension: 16385}, {Tokens: 128, Dimension: 8193}, {Tokens: maxInt, Dimension: 1}, {Tokens: 1, Dimension: maxInt}} {
		m := &FFN{info: info, backend: &testFFNBackend{predictFn: func(_, _ []float32) error {
			t.Fatal("invalid fixed shape reached the backend")
			return nil
		}}}
		if err := m.Predict(nil, nil); err == nil {
			t.Fatalf("accepted invalid fixed shape %+v", info)
		}
	}
	for _, shape := range []FFNInfo{{Tokens: 1, Dimension: 1}, {Tokens: 16, Dimension: 5120}, {Tokens: 128, Dimension: 8192}, {Tokens: 64, Dimension: 16384}} {
		if !validFFNShape(shape) {
			t.Fatalf("rejected supported shape %+v", shape)
		}
	}
	calls := 0
	m := &FFN{info: FFNInfo{Tokens: 2, Dimension: 3}, backend: &testFFNBackend{predictFn: func(_, output []float32) error {
		calls++
		clear(output)
		return nil
	}}}
	for _, lengths := range [][2]int{{5, 6}, {6, 5}, {0, 0}, {7, 6}, {6, 7}} {
		if err := m.Predict(make([]float32, lengths[0]), make([]float32, lengths[1])); err == nil {
			t.Fatalf("accepted incomplete buffers %v", lengths)
		}
	}
	if calls != 0 {
		t.Fatal("invalid buffers reached the backend")
	}
	if err := m.Predict(make([]float32, 6), make([]float32, 6)); err != nil || calls != 1 {
		t.Fatalf("valid prediction err=%v, calls=%d", err, calls)
	}
}

func TestFFNFiniteValuesAndBackendErrors(t *testing.T) {
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		m := &FFN{info: FFNInfo{Tokens: 2, Dimension: 2}, backend: &testFFNBackend{predictFn: func(_, _ []float32) error {
			t.Fatal("non-finite input reached backend")
			return nil
		}}}
		if err := m.Predict([]float32{1, 2, 3, value}, make([]float32, 4)); err == nil {
			t.Fatal("accepted non-finite input")
		}
		m.backend = &testFFNBackend{predictFn: func(_, output []float32) error {
			clear(output)
			output[3] = value
			return nil
		}}
		if err := m.Predict(make([]float32, 4), make([]float32, 4)); err == nil {
			t.Fatal("accepted non-finite output")
		}
	}
	want := errors.New("prediction failed")
	m := &FFN{info: FFNInfo{Tokens: 1, Dimension: 1}, backend: &testFFNBackend{predictFn: func(_, _ []float32) error { return want }}}
	if err := m.Predict([]float32{1}, make([]float32, 1)); !errors.Is(err, want) {
		t.Fatalf("prediction error=%v, want %v", err, want)
	}
}

func TestFFNCloseWaitsForPrediction(t *testing.T) {
	started, release, predicted, closed := make(chan struct{}), make(chan struct{}), make(chan struct{}), make(chan struct{})
	var active, releasedWhileActive atomic.Bool
	closes := 0
	m := &FFN{info: FFNInfo{Tokens: 2, Dimension: 1}, backend: &testFFNBackend{
		predictFn: func(input, output []float32) error {
			active.Store(true)
			close(started)
			<-release
			copy(output, input)
			active.Store(false)
			return nil
		},
		closeFn: func() { closes++; releasedWhileActive.Store(active.Load()) },
	}}
	go func() {
		defer close(predicted)
		if err := m.Predict([]float32{1, 2}, make([]float32, 2)); err != nil {
			t.Errorf("prediction: %v", err)
		}
	}()
	<-started
	go func() { _ = m.Close(); close(closed) }()
	close(release)
	<-predicted
	<-closed
	if releasedWhileActive.Load() {
		t.Fatal("released FFN during prediction")
	}
	if err := m.Close(); err != nil || closes != 1 {
		t.Fatalf("repeated Close=%v, releases=%d", err, closes)
	}
	if err := m.Predict([]float32{1, 2}, make([]float32, 2)); err == nil {
		t.Fatal("accepted prediction after Close")
	}
	if m.Info() != (FFNInfo{Tokens: 2, Dimension: 1}) || (*FFN)(nil).Info() != (FFNInfo{}) {
		t.Fatal("Close changed immutable metadata")
	}
	if err := (*FFN)(nil).Predict(nil, nil); err == nil {
		t.Fatal("nil FFN accepted prediction")
	}
	if err := (*FFN)(nil).Close(); err != nil {
		t.Fatal(err)
	}
}
