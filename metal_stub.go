//go:build !darwin || !cgo || !metal

package gopherllm

type MetalWeight struct{}

func MetalAvailable() bool { return false }

func MetalError() string { return "not built with CGO_ENABLED=1 -tags metal on macOS" }

func prepareMetalWeight(_ []byte, _ GGMLType, _, _ int) *MetalWeight {
	return nil
}

func matvecMetalQ6KInto(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool {
	return false
}

func releaseMetalWeight(_ *MetalWeight) {}
