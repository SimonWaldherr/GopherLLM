//go:build !darwin || !cgo || !metal

package main

type MetalWeight struct{}

func MetalAvailable() bool { return false }

func prepareMetalWeight(_ []byte, _ GGMLType, _, _ int) *MetalWeight {
	return nil
}

func matvecMetalQ6KInto(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool {
	return false
}

func releaseMetalWeight(_ *MetalWeight) {}
