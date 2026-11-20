//go:build !darwin || !cgo || !metal

package engine

type MetalLittleBitWeight struct{}

func newMetalLittleBitWeight(*LittleBitWeight, int, int) *MetalLittleBitWeight { return nil }

func matvecMetalLittleBitInto(*MetalLittleBitWeight, []float32, int, int, *[]float32) bool {
	return false
}

func matvecMetalLittleBitBatch(*MetalLittleBitWeight, []float32, []float32, int) bool {
	return false
}

func matvecMetalLittleBitQKV(Weight, Weight, Weight, []float32, []float32, []float32, []float32, int) bool {
	return false
}

func matvecMetalLittleBitQKVInto(Weight, Weight, Weight, []float32, *[]float32, *[]float32, *[]float32) bool {
	return false
}

func matvecMetalLittleBitSwiGLUInto(Weight, Weight, Weight, []float32, int, *[]float32) bool {
	return false
}

func releaseMetalLittleBitWeight(*MetalLittleBitWeight) {}

func (*Runner) LittleBitMetalMemory() (factors, scratch uint64) { return 0, 0 }
