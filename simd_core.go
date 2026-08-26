package gopherllm

// CPU compute kernels: dot products and matrix-vector products over the
// quantized block formats (layouts documented on GGMLType.DataSize),
// dequantization, and the worker pool that parallelizes matvecs across rows.
//
// Every kernel exists in up to three tiers, chosen at runtime:
//
//	portable Go scalar  (always present; the correctness reference the
//	                     differential tests compare against)
//	ARM64 NEON          (hasQuantSIMD const true on arm64; *_arm64.s)
//	x86-64 AVX2+FMA     (hasAVX2/hasQuantSIMD via CPUID; *_amd64.s,
//	                     GOPHERLLM_DISABLE_SIMD=1 forces scalar)
//
// The "xsums" trick used by the Q4_K/Q6_K fast paths: both formats apply a
// per-sub-block affine dequant (val = d*sc*q - dmin*m for Q4_K; a -32 offset
// for Q6_K), so dot(row, x) splits into a quant-dependent term and a term
// that only needs the SUM of x over each sub-block. Those sums are computed
// once per matvec (fillQ4KXSums / fillQ6KXSums16) and shared by every row,
// removing the offset handling from the inner loop.

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

// oversubscribeDispatch issues more matvec chunks than workers so faster
// cores absorb stragglers — a win on heterogeneous big.LITTLE parts (Apple
// Silicon), pure channel-wakeup overhead on homogeneous x86 cores. Auto Mode
// changes it process-wide while other Runners may be generating, so reads on
// the hot path must be atomic.
var oversubscribeDispatch = newAtomicBool(runtime.GOARCH == "arm64")

var configuredThreads atomic.Int64

// newAtomicBool builds an already-initialized atomic.Bool for a package-level
// var declaration (atomic.Bool's zero value is unset, and it has no literal
// form, so a one-line `var x = newAtomicBool(cond)` needs this instead of an
// init() func). Returns a pointer rather than a value: atomic.Bool carries a
// noCopy guard, so returning it by value trips `go vet`'s copylocks check
// even though nothing has touched it concurrently yet at package-init time.
// Callers keep using x.Load()/x.Store() unchanged — those methods have
// pointer receivers either way.
func newAtomicBool(v bool) *atomic.Bool {
	b := &atomic.Bool{}
	b.Store(v)
	return b
}

// SetNumThreads overrides the worker count used by the parallel matvec
// dispatch (default GOMAXPROCS). The CLI's --threads flag calls this and sets
// GOMAXPROCS to the same value.
func SetNumThreads(n int) {
	if n < 1 {
		n = 1
	}
	configuredThreads.Store(int64(n))
}

func numThreads() int {
	if n := int(configuredThreads.Load()); n > 0 {
		return n
	}
	return max(1, runtime.GOMAXPROCS(0))
}

// f16LUT maps every possible f16 bit pattern to its float32 value (256 KB,
// built once at startup). Block scales in every quant format are f16, so this
// lookup sits on the innermost dequant loops; a table beats bit manipulation
// there.
var f16LUT []float32
var int8ToFloat32LUT [256]float32
var byteToFloat32LUT [256]float32

func init() {
	f16LUT = make([]float32, 65536)
	for i := 0; i < 65536; i++ {
		f16LUT[i] = f16ToF32Soft(uint16(i))
	}
	for i := 0; i < 256; i++ {
		int8ToFloat32LUT[i] = float32(int8(byte(i)))
		byteToFloat32LUT[i] = float32(i)
	}
}

func f16ToF32Soft(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		var e uint32
		m := mant
		for (m & 0x400) == 0 {
			m <<= 1
			e++
		}
		m &= 0x3ff
		return math.Float32frombits((sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13))
	}
	if exp == 31 {
		return math.Float32frombits((sign << 31) | (0xff << 23) | (mant << 13))
	}
	return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}

func F16ToF32(h uint16) float32 {
	return f16LUT[h]
}

func DotF32(a, b []float32) float32 {
	return dotF32(a, b)
}

func dotF32Scalar(a, b []float32) float32 {
	n := min(len(a), len(b))
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func AxpyF32(out []float32, alpha float32, x []float32) {
	axpyF32(out, alpha, x)
}

func axpyF32Scalar(out []float32, alpha float32, x []float32) {
	for i := 0; i < min(len(out), len(x)); i++ {
		out[i] += alpha * x[i]
	}
}

func ScaleF32(out []float32, alpha float32) {
	scaleF32(out, alpha)
}

func scaleF32Scalar(out []float32, alpha float32) {
	for i := range out {
		out[i] *= alpha
	}
}

func ScaleAddF32(out []float32, alpha float32, x []float32) {
	scaleAddF32(out, alpha, x)
}

func scaleAddF32Scalar(out []float32, alpha float32, x []float32) {
	for i := 0; i < min(len(out), len(x)); i++ {
		out[i] = out[i]*alpha + x[i]
	}
}

func mulScaleF32Scalar(x []float32, weight []float32, scale float32, out []float32) {
	for i := 0; i < min(len(x), len(weight), len(out)); i++ {
		out[i] = x[i] * weight[i] * scale
	}
}

func siluMulF32Scalar(gate, up, out []float32, start, end int) {
	// fastSigmoidF32 rather than math.Exp(float64(-g)): this is the SwiGLU path
	// on every target without an AVX2 kernel, so on arm64 it runs for every FFN
	// element of every token. See fastmath.go for the accuracy measurements.
	for i := start; i < end; i++ {
		g := gate[i]
		out[i] = g * fastSigmoidF32(g) * up[i]
	}
}

func MatvecF32(data, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	MatvecF32Into(data, x, rows, cols, &out)
	return out
}

// matvecF32Task keeps the hot F32 path out of an escaping callback. F32
// checkpoints invoke this for every projection, so even one closure allocation
// per matvec becomes visible in small/full-precision models and benchmarks.
type matvecF32Task struct {
	data []float32
	x    []float32
	out  []float32
	cols int
}

func (t *matvecF32Task) runRows(start, end int) {
	for r := start; r < end; r++ {
		row := t.data[r*t.cols : min((r+1)*t.cols, len(t.data))]
		t.out[r] = DotF32(row, t.x)
	}
}

var matvecF32TaskPool = sync.Pool{New: func() any { return new(matvecF32Task) }}

func MatvecF32Into(data, x []float32, rows, cols int, out *[]float32) {
	ensureLenNoClear(out, rows)
	task := matvecF32TaskPool.Get().(*matvecF32Task)
	task.data, task.x, task.out, task.cols = data, x, *out, cols
	parallelRowsTask(rows, task)
	*task = matvecF32Task{}
	matvecF32TaskPool.Put(task)
}

var xsumsScratchPool = sync.Pool{New: func() any {
	s := make([]float32, 0, 1024)
	return &s
}}

// ensureLen resizes *s to length n, reusing capacity when possible, and
// zeroes the contents. ensureLenNoClear is the same without the zeroing, for
// buffers that are fully overwritten anyway — it is the standard idiom for
// every scratch buffer on the decode path.
func ensureLen[T any](s *[]T, n int) {
	if cap(*s) < n {
		*s = make([]T, n)
		return
	}
	*s = (*s)[:n]
	var zero T
	for i := range *s {
		(*s)[i] = zero
	}
}

func ensureLenNoClear[T any](s *[]T, n int) {
	if cap(*s) < n {
		*s = make([]T, n)
		return
	}
	*s = (*s)[:n]
}

func binaryLE16(b []byte) uint16 {
	if len(b) < 2 {
		return 0
	}
	return uint16(b[0]) | uint16(b[1])<<8
}

func getScaleMinK4(j int, q []byte) (byte, byte) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}
