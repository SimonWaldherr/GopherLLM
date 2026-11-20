//go:build !darwin || !cgo || !metal

package engine

func prepareMetalBF16Head(Weight) *MetalWeight { return nil }

func matvecMetalBF16HeadInto(*MetalWeight, []float32, *[]float32) bool { return false }

func argmaxMetalBF16Head(*MetalWeight, []float32, []uint32, float32) (uint32, bool) {
	return 0, false
}
