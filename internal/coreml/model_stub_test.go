//go:build !darwin || !cgo || !coreml

package coreml

import (
	"errors"
	"testing"
)

func TestUnavailableWithoutBackend(t *testing.T) {
	if model, err := Open("missing.mlmodelc", Config{}); model != nil || !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Open=%v, %v", model, err)
	}
	if plan, err := Inspect("missing.mlmodelc", Config{}); plan != (Plan{}) || !errors.Is(err, ErrUnavailable) {
		t.Fatalf("Inspect=%v, %v", plan, err)
	}
}
