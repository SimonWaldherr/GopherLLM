//go:build !darwin || !cgo || !coreml

package coreml

import (
	"errors"
	"testing"
)

func TestFFNUnavailableWithoutBackend(t *testing.T) {
	if model, err := OpenFFN("missing.mlmodelc", CPUAndNeuralEngine); model != nil || !errors.Is(err, ErrUnavailable) {
		t.Fatalf("OpenFFN=%v, %v", model, err)
	}
}
