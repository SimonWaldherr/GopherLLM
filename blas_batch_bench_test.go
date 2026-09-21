package gopherllm

import "testing"

// buildBlasBenchShape builds a plain F32 weight and a batch of n input rows
// at the given matvec shape (rows x cols), so blasMatvecBatch (Accelerate's
// GEMM on Darwin+cgo, matvecBatch's per-row dot otherwise) and matvecBatch
// (the portable, always-available path) can be compared head-to-head on
// identical data.
func buildBlasBenchShape(n, rows, cols int) (Weight, [][]float32, [][]float32) {
	w := Weight{F32: make([]float32, rows*cols)}
	for i := range w.F32 {
		w.F32[i] = float32(i%97) * 0.01
	}
	xs := make([][]float32, n)
	outs := make([][]float32, n)
	for i := range xs {
		xs[i] = make([]float32, cols)
		for j := range xs[i] {
			xs[i][j] = float32((i+j)%53) * 0.02
		}
		outs[i] = make([]float32, rows)
	}
	return w, xs, outs
}

// BenchmarkMatvecBatch_VoxtralFFNShape_BLAS and its _Portable counterpart
// below quantify why the encoder's blasMatvecBatch prefers Accelerate's GEMM
// kernel over the portable path at Voxtral's own FFN gate/up projection
// shape (n=372 timesteps for a 7-second clip, DModel=1280 -> FFNDim=5120).
// On this machine (Apple M2 Max) BLAS runs roughly 7x faster here -- see
// TranscribeVoxtralRealtime's prepareVoxtralStreamWeights fix, which exists
// specifically so production traffic reaches this fast path instead of
// silently falling back to the slow one.
func BenchmarkMatvecBatch_VoxtralFFNShape_BLAS(b *testing.B) {
	w, xs, outs := buildBlasBenchShape(372, 5120, 1280)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blasMatvecBatch(w, xs, outs)
	}
}

func BenchmarkMatvecBatch_VoxtralFFNShape_Portable(b *testing.B) {
	w, xs, outs := buildBlasBenchShape(372, 5120, 1280)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matvecBatch(w, xs, outs)
	}
}

// BenchmarkMatvecBatch_ParakeetFFNShape_{BLAS,Portable} do the same at
// Parakeet's own FFN shape (n=44 timesteps for a 7-second clip after 8x
// subsampling, DModel=1024 -> DFF=4096) -- a much smaller batch, so this
// also documents how the BLAS/portable gap narrows (or doesn't) at the
// other model's scale.
func BenchmarkMatvecBatch_ParakeetFFNShape_BLAS(b *testing.B) {
	w, xs, outs := buildBlasBenchShape(44, 4096, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blasMatvecBatch(w, xs, outs)
	}
}

func BenchmarkMatvecBatch_ParakeetFFNShape_Portable(b *testing.B) {
	w, xs, outs := buildBlasBenchShape(44, 4096, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matvecBatch(w, xs, outs)
	}
}
