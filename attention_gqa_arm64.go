//go:build arm64

package gopherllm

// dotF32x4 computes four dot products against one shared input row. Keeping
// the shared K row in NEON registers is the important GQA optimization.
//
//go:noescape
func dotF32x4(q0, q1, q2, q3, x *float32, n int) (s0, s1, s2, s3 float32)

// axpyF32x4 applies one shared V row to four independent attention outputs.
//
//go:noescape
func axpyF32x4(out0, out1, out2, out3 *float32, a0, a1, a2, a3 float32, x *float32, n int)

// hasFastDotF32x4 reports whether dotF32x4 is a genuine 4-wide kernel rather
// than a composition of four separate dot products. arm64 has one, so batched
// matvec tiles its token loop through it (see dotRowIntoBatch); on targets
// where dotF32x4 just wraps DotF32 four times, tiling only adds call overhead
// and measurably slows the batch down.
const hasFastDotF32x4 = true
