//go:build !darwin || !cgo || !metal

package gopherllm

type MetalWeight struct{}

func MetalAvailable() bool { return false }

func MetalError() string { return "not built with CGO_ENABLED=1 -tags metal on macOS" }

func prepareMetalWeight(_ []byte, _ GGMLType, _, _ int, _ bool) *MetalWeight {
	return nil
}

func metalWeightUsesDirect(_ *MetalWeight) bool { return false }

func matvecMetalQ4KInto(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool {
	return false
}

func matvecMetalQ6KInto(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool {
	return false
}

func argmaxMetalQ6K(_ *MetalWeight, _ []float32) (uint32, bool) { return 0, false }

func matvecMetalQ4K2Into(_, _ *MetalWeight, _ []float32, _, _, _ int, _, _ *[]float32) bool {
	return false
}

func matvecMetalQ4K2Q6KInto(_, _, _ *MetalWeight, _ []float32, _, _, _, _ int, _, _, _ *[]float32) bool {
	return false
}

func matvecMetalSwiGLUInto(_, _, _ *MetalWeight, _ []float32, _ *[]float32) bool {
	return false
}

func releaseMetalWeight(_ *MetalWeight) {}
