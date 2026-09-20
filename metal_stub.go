//go:build !darwin || !cgo || !metal

package gopherllm

type MetalWeight struct{}

func MetalAvailable() bool { return false }

func matvecMetalQ8_0Into(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool { return false }

func argmaxMetalQ8_0Penalized(_ *MetalWeight, _ []float32, _ []uint32, _ float32) (uint32, bool) {
	return 0, false
}

func MetalError() string { return "not built with CGO_ENABLED=1 -tags metal on macOS" }

func prepareMetalWeight(_ []byte, _ GGMLType, _, _ int, _ bool) *MetalWeight {
	return nil
}

func metalWeightUsesDirect(_ *MetalWeight) bool { return false }

func matvecMetalQ4KInto(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool {
	return false
}

func matvecMetalQ5KInto(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool {
	return false
}

func matvecMetalQ6KInto(_ *MetalWeight, _ []float32, _, _ int, _ *[]float32) bool {
	return false
}

func argmaxMetalQ6K(_ *MetalWeight, _ []float32) (uint32, bool) { return 0, false }

func argmaxMetalQ6KPenalized(_ *MetalWeight, _ []float32, _ []uint32, _ float32) (uint32, bool) {
	return 0, false
}

func argmaxMetalQ6KBatch(_ *MetalWeight, _ []float32, _ []uint32, _ int) bool { return false }

func matvecMetalQ4K2Into(_, _ *MetalWeight, _ []float32, _, _, _ int, _, _ *[]float32) bool {
	return false
}

func matvecMetalQ4K2Q6KInto(_, _, _ *MetalWeight, _ []float32, _, _, _, _ int, _, _, _ *[]float32) bool {
	return false
}

func matvecMetalSwiGLUInto(_, _, _ *MetalWeight, _ []float32, _ *[]float32) bool {
	return false
}

func matvecMetalSwiGLUBatchInto(_, _, _ *MetalWeight, _ []float32, _ int, _ *[]float32) bool {
	return false
}

func (r *Runner) metalBatchFFNPrefillChunk() int { return 0 }

func releaseMetalWeight(_ *MetalWeight) {}

type metalDenseState struct{}

func tryMetalDenseDecode(Config, ModelWeights, *KVCache, *DecodeBuffer, uint32, int) bool {
	return false
}

func tryMetalDenseDecodeOutput(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer, token uint32, pos int, logits *[]float32) bool {
	return false
}

func (*metalDenseState) close() {}

func tryMetalDenseGreedy(c Config, w ModelWeights, k *KVCache, b *DecodeBuffer, token uint32, pos int, recent []uint32, penalty float32) (uint32, bool) {
	return 0, false
}

func makeKVF32(n int) []float32 { return make([]float32, n) }

func metalGemmaEnabled() bool                                                          { return false }
func prepareMetalGeluWeight(*Weight, bool)                                             {}
func matvecMetalGeGLUInto(g, u, d Weight, x []float32, batch int, out *[]float32) bool { return false }

func matvecMetalGemmaOutputInto(w Weight, x []float32, out *[]float32) bool { return false }
func argmaxMetalGemmaOutput(w Weight, x []float32, recent []uint32, penalty float32) (uint32, bool) {
	return 0, false
}

func prepareMetalDenseBatch(Config, ModelWeights, *KVCache, *DecodeBuffer, int) bool    { return false }
func metalDenseBatchProjection(*DecodeBuffer, int, int, []float32, []float32, int) bool { return false }
func metalDenseBatchProjectionQKV(*DecodeBuffer, int, []float32, []float32, []float32, []float32, int) bool {
	return false
}
