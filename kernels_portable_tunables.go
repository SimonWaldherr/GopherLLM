//go:build !amd64

package gopherllm

import (
	"os"
	"runtime"
	"sync"
)

// The int8-activation ("Q8K") matvec path on every non-amd64 target.
//
// This used to be a wall of panics: the path was amd64-only because it was
// written against VPMADDUBSW, so `useQ8Activations` was a compile-time false
// and all of this was dead code. That left Apple Silicon — where the whole
// point is that decode has compute headroom under a much wider memory bus —
// dequantizing every weight to f32 and doing float FMAs.
//
// The arithmetic now lives in quant_q8k_portable.go, shared by all
// architectures and differentially tested against the amd64 assembly. Where a
// target also has hand-written int8 SIMD (hasQ8KDotAsm), the row dots go
// through it; otherwise they go through the portable scalar kernels.
//
// Default-on is deliberately gated on hasQ8KDotAsm rather than on mere
// availability. On a target whose FLOAT path is already vectorised — arm64 has
// NEON q4kDotPrepared/q4kQDots8 — scalar integer dots can lose to vectorised
// float ones, and turning this on by default would be a silent regression. So
// where there is no int8 assembly the path is merely *offered*: the autotuner
// measures it against the float path on the real machine and model
// (tuneQ8Activations, gated on q8ActivationsAvailable) and GOPHERLLM_Q8_ACTIVATIONS
// forces the decision either way.

// useQ8Activations mirrors the amd64 variable of the same name. It is a var
// rather than a const now, so the branches in the matvec entry points are live
// and the autotuner can flip them.
//
// atomic.Bool, not a plain bool: chooseToggle (autotune.go) flips this
// repeatedly mid-calibration while holding only the calibrating Runner's own
// genLock, but every matvec on every live Runner in the process (e.g. a
// server's chat model plus a separately loaded RAG embedding model, see
// server/server.go) reads it with no lock of its own — a plain bool here was
// a data race across concurrent Runners.
var useQ8Activations = newAtomicBool(defaultQ8Activations())

func defaultQ8Activations() bool {
	switch os.Getenv("GOPHERLLM_Q8_ACTIVATIONS") {
	case "0":
		return false
	case "1":
		return true
	}
	return hasQ8KDotAsm
}

var (
	q8ScratchPool     = sync.Pool{New: func() any { s := make([]int8, 0, 16384); return &s }}
	xscaleScratchPool = sync.Pool{New: func() any { s := make([]float32, 0, 64); return &s }}
)

type q8ScratchLease struct {
	q8     *[]int8
	xscale *[]float32
}

// acquireQ8 quantizes x once for the whole matvec and hands back the int8
// activations, per-256-block scales, and their pool lease. Keeping release as
// an ordinary function rather than a returned closure avoids one allocation
// on every quantized projection. cols must be a multiple of 256.
func acquireQ8(x []float32, cols int) (q8 []int8, xscale []float32, lease q8ScratchLease) {
	blocks := cols / 256
	q8s := q8ScratchPool.Get().(*[]int8)
	scs := xscaleScratchPool.Get().(*[]float32)
	ensureLenNoClear(q8s, cols)
	ensureLenNoClear(scs, blocks)
	q8 = *q8s
	xscale = *scs
	q8kQuantize(x, q8, xscale, blocks)
	return q8, xscale, q8ScratchLease{q8: q8s, xscale: scs}
}

func releaseQ8(q8 []int8, xscale []float32, lease q8ScratchLease) {
	*lease.q8 = q8
	q8ScratchPool.Put(lease.q8)
	*lease.xscale = xscale
	xscaleScratchPool.Put(lease.xscale)
}

func dotQ4KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q4kDotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ5KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q5kDotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ6KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q6kDotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ8_0RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q8_0DotQ8KRow(data[r*rowBytes:], q8, xscale, blocks)
	}
}

func dotQ4_0RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q4_0DotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ4_1RowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q4_1DotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotMXFP4RowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = mxfp4DotQ8KRow(data[r*rowBytes:], q8, xscale, blocks)
	}
}

func dotQ2KRowsQ8(data []byte, q8 []int8, xscale, xsums []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q2kDotQ8KRow(data[r*rowBytes:], q8, xscale, xsums, blocks)
	}
}

func dotQ3KRowsQ8(data []byte, q8 []int8, xscale []float32, cols, rowBytes, start, end int, out []float32) {
	blocks := cols / 256
	for r := start; r < end; r++ {
		out[r] = q3kDotQ8KRow(data[r*rowBytes:], q8, xscale, blocks)
	}
}

// q8kRowLayout describes how one weight type lays out a 256-element
// superchunk and how many activation-sum slots it needs per token, so the
// batched path and the single matvec agree on both.
type q8kRowLayout struct {
	rowBytes   int
	sumsPerTok int
	dot        func(row []byte, q8 []int8, xsc, xsums []float32, blocks int) float32
}

func q8kLayoutFor(t GGMLType, blocks int) (q8kRowLayout, bool) {
	switch t {
	case GGMLTypeQ4_K:
		return q8kRowLayout{blocks * 144, blocks * 8, q4kDotQ8KRow}, true
	case GGMLTypeQ5_K:
		return q8kRowLayout{blocks * 176, blocks * 8, q5kDotQ8KRow}, true
	case GGMLTypeQ6_K:
		return q8kRowLayout{blocks * 210, blocks * 16, q6kDotQ8KRow}, true
	case GGMLTypeQ8_0:
		// Symmetric: no offset term, so one dummy sums slot keeps the shared
		// per-token indexing valid.
		return q8kRowLayout{blocks * 272, 1, func(row []byte, q8 []int8, xsc, _ []float32, blocks int) float32 {
			return q8_0DotQ8KRow(row, q8, xsc, blocks)
		}}, true
	case GGMLTypeQ4_0:
		return q8kRowLayout{blocks * 144, blocks * 8, q4_0DotQ8KRow}, true
	case GGMLTypeQ4_1:
		return q8kRowLayout{blocks * 160, blocks * 8, q4_1DotQ8KRow}, true
	case GGMLTypeMXFP4:
		return q8kRowLayout{blocks * 136, 1, func(row []byte, q8 []int8, xsc, _ []float32, blocks int) float32 {
			return mxfp4DotQ8KRow(row, q8, xsc, blocks)
		}}, true
	case GGMLTypeQ2_K:
		return q8kRowLayout{blocks * 84, blocks * 16, q2kDotQ8KRow}, true
	case GGMLTypeQ3_K:
		// Symmetric like Q8_0/MXFP4: no offset term, one dummy sums slot.
		return q8kRowLayout{blocks * 110, 1, func(row []byte, q8 []int8, xsc, _ []float32, blocks int) float32 {
			return q3kDotQ8KRow(row, q8, xsc, blocks)
		}}, true
	}
	return q8kRowLayout{}, false
}

// fillQ8KXSums writes the activation-sum term this weight type needs.
func fillQ8KXSums(t GGMLType, x []float32, cols int, sub *[]float32) {
	switch t {
	case GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ4_0, GGMLTypeQ4_1:
		fillQ4KXSums(x, cols, sub)
	case GGMLTypeQ6_K:
		fillQ6KXSums16(x, cols, sub)
		ScaleF32(*sub, 32)
	case GGMLTypeQ2_K:
		fillQ6KXSums16(x, cols, sub)
	}
}

type batchQ8Scratch struct {
	q8                    []int8
	xsc                   []float32
	xsums, xsums2, xsums3 []float32
}

var batchQ8Pool = sync.Pool{New: func() any { return &batchQ8Scratch{} }}

type batchQ8MatvecTask struct {
	w                Weight
	outs             [][]float32
	layout           q8kRowLayout
	q8All            []int8
	xscAll, xsumsAll []float32
	p, cols, blocks  int
}

func (t *batchQ8MatvecTask) runRows(start, end int) {
	if batchQ4KRows4(t.w, t.outs[:t.p], t.q8All, t.xscAll, t.xsumsAll, start, end) {
		return
	}
	const rowTile = 16
	for tileStart := start; tileStart < end; tileStart += rowTile {
		tileEnd := min(tileStart+rowTile, end)
		for token := 0; token < t.p; token++ {
			q8 := t.q8All[token*t.cols:]
			xsc := t.xscAll[token*t.blocks:]
			xsum := t.xsumsAll[token*t.layout.sumsPerTok:]
			out := t.outs[token]
			for row := tileStart; row < tileEnd; row++ {
				out[row] = t.layout.dot(t.w.Raw[row*t.layout.rowBytes:], q8, xsc, xsum, t.blocks)
			}
		}
	}
}

var batchQ8MatvecTaskPool = sync.Pool{New: func() any { return new(batchQ8MatvecTask) }}

// matvecBatchQ8 is the batched-prefill analogue of the int8-activation matvec:
// each prompt token's activations are quantized once, then every weight row is
// streamed from memory exactly once and dotted against all tokens in row tiles
// small enough that both the tile and one token's int8 activations stay
// cache-resident.
//
// This used to be a hardcoded `return false` on non-amd64, which meant prefill
// on arm64 had no batched path at all: it fell back to re-reading and
// re-decoding every weight row once per prompt token. Returns false (writing
// nothing) for shapes and types it does not handle.
func matvecBatchQ8(w Weight, xs, outs [][]float32) bool {
	if !useQ8Activations.Load() {
		return false
	}
	p := len(xs)
	if p == 0 {
		return false
	}
	cols := len(xs[0])
	if cols <= 0 || cols%256 != 0 || w.Rows <= 0 || w.Cols != cols {
		return false
	}
	blocks := cols / 256
	layout, ok := q8kLayoutFor(w.Type, blocks)
	if !ok {
		return false
	}
	if len(w.Raw) < w.Rows*layout.rowBytes {
		return false
	}
	for t := range p {
		if len(xs[t]) < cols || len(outs[t]) < w.Rows {
			return false
		}
	}

	scratch := batchQ8Pool.Get().(*batchQ8Scratch)
	ensureLenNoClear(&scratch.q8, p*cols)
	ensureLenNoClear(&scratch.xsc, p*blocks)
	ensureLenNoClear(&scratch.xsums, p*layout.sumsPerTok)
	q8All, xscAll, xsumsAll := scratch.q8, scratch.xsc, scratch.xsums
	for t := range p {
		q8kQuantize(xs[t], q8All[t*cols:], xscAll[t*blocks:], blocks)
		sub := xsumsAll[t*layout.sumsPerTok : (t+1)*layout.sumsPerTok : (t+1)*layout.sumsPerTok]
		fillQ8KXSums(w.Type, xs[t], cols, &sub)
	}

	// Each row performs the complete prompt batch. Keep one coarse range per
	// worker: ARM's decode-oriented over-dispatch creates many wakeups here
	// without exposing additional independent work. The reusable task also
	// avoids retaining this projection's large slice graph in an escaping
	// worker closure.
	task := batchQ8MatvecTaskPool.Get().(*batchQ8MatvecTask)
	task.w, task.outs, task.layout = w, outs, layout
	task.q8All, task.xscAll, task.xsumsAll = q8All, xscAll, xsumsAll
	task.p, task.cols, task.blocks = p, cols, blocks
	parallelRowsBatchedTask(w.Rows, task)
	*task = batchQ8MatvecTask{}
	batchQ8MatvecTaskPool.Put(task)
	batchQ8Pool.Put(scratch)
	return true
}

// matvecBatchQ8Fused{2,3} execute independent Q8K batch projections that
// share their activation rows. Standard transformer prefill invokes this for
// Q/K/V and SwiGLU gate/up. Quantizing the same p inputs for every projection
// was pure duplicated work, and each separate projection also introduced a
// worker-pool barrier. Dot arithmetic and the per-weight row order below are
// identical to matvecBatchQ8, so this is a scheduling/data-reuse optimization
// rather than a numerics change.
func matvecBatchQ8Fused2(a, b Weight, xs, aOut, bOut [][]float32) bool {
	// Keep the small-prompt fallback allocation-free. matvecBatchQ8Fused makes
	// the same decision, but constructing the two slice-backed argument arrays
	// first makes them escape even when the fused path immediately declines.
	if !useQ8Activations.Load() || len(xs) < 32 {
		return false
	}
	task := batchQ8FusedTaskPool.Get().(*batchQ8FusedTask)
	task.count = 2
	task.weights[0], task.weights[1] = a, b
	task.outs[0], task.outs[1] = aOut, bOut
	ok := matvecBatchQ8Fused(task, xs)
	*task = batchQ8FusedTask{}
	batchQ8FusedTaskPool.Put(task)
	return ok
}

func matvecBatchQ8Fused3(a, b, c Weight, xs, aOut, bOut, cOut [][]float32) bool {
	if !useQ8Activations.Load() || len(xs) < 32 {
		return false
	}
	task := batchQ8FusedTaskPool.Get().(*batchQ8FusedTask)
	task.count = 3
	task.weights[0], task.weights[1], task.weights[2] = a, b, c
	task.outs[0], task.outs[1], task.outs[2] = aOut, bOut, cOut
	ok := matvecBatchQ8Fused(task, xs)
	*task = batchQ8FusedTask{}
	batchQ8FusedTaskPool.Put(task)
	return ok
}

// batchQ8FusedTask carries the fixed-arity projection metadata through the
// worker pool. Keeping both the wrapper arrays and the row loop in this pooled
// task removes the last three heap allocations from long Q8 prompt batches:
// two escaping slice-backed argument arrays plus the worker closure.
type batchQ8FusedTask struct {
	weights [3]Weight
	outs    [3][][]float32
	layouts [3]q8kRowLayout
	offsets [4]int
	sums    [3][]float32
	q8All   []int8
	xscAll  []float32
	count   int
	p       int
	cols    int
	blocks  int
}

func (t *batchQ8FusedTask) runRows(start, end int) {
	const rowTile = 16
	for wi := 0; wi < t.count; wi++ {
		w := t.weights[wi]
		localStart := max(start, t.offsets[wi]) - t.offsets[wi]
		localEnd := min(end, t.offsets[wi+1]) - t.offsets[wi]
		if localStart >= localEnd {
			continue
		}
		if batchQ4KRows4(w, t.outs[wi][:t.p], t.q8All, t.xscAll, t.sums[wi], localStart, localEnd) {
			continue
		}
		layout := t.layouts[wi]
		for tileStart := localStart; tileStart < localEnd; tileStart += rowTile {
			tileEnd := min(tileStart+rowTile, localEnd)
			for token := 0; token < t.p; token++ {
				q8 := t.q8All[token*t.cols:]
				xsc := t.xscAll[token*t.blocks:]
				xsum := t.sums[wi][token*layout.sumsPerTok:]
				out := t.outs[wi][token]
				for row := tileStart; row < tileEnd; row++ {
					out[row] = layout.dot(w.Raw[row*layout.rowBytes:], q8, xsc, xsum, t.blocks)
				}
			}
		}
	}
}

var batchQ8FusedTaskPool = sync.Pool{New: func() any { return new(batchQ8FusedTask) }}

func matvecBatchQ8Fused(task *batchQ8FusedTask, xs [][]float32) bool {
	if !useQ8Activations.Load() || task.count < 2 || task.count > 3 {
		return false
	}
	p := len(xs)
	// The saved activation preparation and worker barrier become measurable
	// only once a prompt chunk has enough token rows. Small chat prompts keep
	// the established per-projection dispatch, which avoids trading their low
	// latency for a larger combined work queue.
	if p < 32 {
		return false
	}
	cols := len(xs[0])
	if cols <= 0 || cols%256 != 0 {
		return false
	}
	blocks := cols / 256
	for wi := 0; wi < task.count; wi++ {
		w := task.weights[wi]
		if w.Rows <= 0 || w.Cols != cols || len(task.outs[wi]) != p {
			return false
		}
		layout, ok := q8kLayoutFor(w.Type, blocks)
		if !ok || len(w.Raw) < w.Rows*layout.rowBytes {
			return false
		}
		for t := range p {
			if len(xs[t]) < cols || len(task.outs[wi][t]) < w.Rows {
				return false
			}
		}
		task.layouts[wi] = layout
		task.offsets[wi+1] = task.offsets[wi] + w.Rows
	}

	scratch := batchQ8Pool.Get().(*batchQ8Scratch)
	ensureLenNoClear(&scratch.q8, p*cols)
	ensureLenNoClear(&scratch.xsc, p*blocks)
	q8All, xscAll := scratch.q8, scratch.xsc
	sumStorage := [3]*[]float32{&scratch.xsums, &scratch.xsums2, &scratch.xsums3}
	var owner [3]int
	uniqueSums := 0
	for wi := 0; wi < task.count; wi++ {
		owner[wi] = wi
		for prior := 0; prior < wi; prior++ {
			if task.weights[prior].Type == task.weights[wi].Type {
				owner[wi] = owner[prior]
				task.sums[wi] = task.sums[prior]
				break
			}
		}
		if owner[wi] != wi {
			continue
		}
		ensureLenNoClear(sumStorage[uniqueSums], p*task.layouts[wi].sumsPerTok)
		task.sums[wi] = *sumStorage[uniqueSums]
		uniqueSums++
	}
	for t := range p {
		q8kQuantize(xs[t], q8All[t*cols:], xscAll[t*blocks:], blocks)
		for wi := 0; wi < task.count; wi++ {
			if owner[wi] != wi {
				continue
			}
			sumsPerTok := task.layouts[wi].sumsPerTok
			sub := task.sums[wi][t*sumsPerTok : (t+1)*sumsPerTok : (t+1)*sumsPerTok]
			fillQ8KXSums(task.weights[wi].Type, xs[t], cols, &sub)
		}
	}

	// The fused projection has the same long, weight-stationary row work as
	// matvecBatchQ8 above. One chunk per worker avoids dispatch overhead on
	// heterogeneous ARM CPUs while retaining the shared activation preparation.
	task.p, task.cols, task.blocks = p, cols, blocks
	task.q8All, task.xscAll = q8All, xscAll
	parallelRowsBatchedTask(task.offsets[task.count], task)
	batchQ8Pool.Put(scratch)
	return true
}

type argmaxQ8Task struct {
	data       []byte
	q8         []int8
	xsc, xsums []float32
	kind       GGMLType
	blocks     int
	rowBytes   int
	mu         sync.Mutex
	bestToken  int
	bestValue  float32
	found      bool
}

func (t *argmaxQ8Task) runRows(start, end int) {
	localToken := start
	localValue := negInf32
	localFound := false
	// Select the concrete kernel once per worker range. Keeping the switch out
	// of the row loop avoids the indirect function-value call that is material
	// at vocabulary-output sizes.
	switch t.kind {
	case GGMLTypeQ6_K:
		for row := start; row < end; row++ {
			v := q6kDotQ8KRow(t.data[row*t.rowBytes:], t.q8, t.xsc, t.xsums, t.blocks)
			if finiteLogit(v) && (!localFound || v > localValue) {
				localToken, localValue, localFound = row, v, true
			}
		}
	case GGMLTypeQ4_K:
		for row := start; row < end; row++ {
			v := q4kDotQ8KRow(t.data[row*t.rowBytes:], t.q8, t.xsc, t.xsums, t.blocks)
			if finiteLogit(v) && (!localFound || v > localValue) {
				localToken, localValue, localFound = row, v, true
			}
		}
	case GGMLTypeQ5_K:
		for row := start; row < end; row++ {
			v := q5kDotQ8KRow(t.data[row*t.rowBytes:], t.q8, t.xsc, t.xsums, t.blocks)
			if finiteLogit(v) && (!localFound || v > localValue) {
				localToken, localValue, localFound = row, v, true
			}
		}
	case GGMLTypeQ8_0:
		for row := start; row < end; row++ {
			v := q8_0DotQ8KRow(t.data[row*t.rowBytes:], t.q8, t.xsc, t.blocks)
			if finiteLogit(v) && (!localFound || v > localValue) {
				localToken, localValue, localFound = row, v, true
			}
		}
	}
	if !localFound {
		return
	}
	t.mu.Lock()
	if !t.found || localValue > t.bestValue || (localValue == t.bestValue && localToken < t.bestToken) {
		t.bestToken, t.bestValue, t.found = localToken, localValue, true
	}
	t.mu.Unlock()
}

var argmaxQ8TaskPool = sync.Pool{New: func() any { return new(argmaxQ8Task) }}

func argmaxRowsQ8(kind GGMLType, raw []byte, x, xsums []float32, rows, cols, rowBytes int) (uint32, bool) {
	if !useQ8Activations.Load() || rows <= 0 || cols <= 0 || cols%256 != 0 {
		return 0, false
	}
	blocks := cols / 256
	q8, xsc, lease := acquireQ8(x, cols)
	task := argmaxQ8TaskPool.Get().(*argmaxQ8Task)
	task.data, task.q8, task.xsc, task.xsums = raw, q8, xsc, xsums
	task.kind, task.blocks, task.rowBytes = kind, blocks, rowBytes
	task.bestValue = negInf32
	parallelRowsTask(rows, task)
	bestToken := task.bestToken
	*task = argmaxQ8Task{}
	argmaxQ8TaskPool.Put(task)
	releaseQ8(q8, xsc, lease)
	return uint32(bestToken), true
}

// argmaxQ6KRowsQ8 finds argmax(W·x) over a Q6_K matrix with the same int8
// kernel as the materializing matvec, skipping the logits writeback. Returns
// false when the int8 path is off so callers keep the exact float kernel.
func argmaxQ6KRowsQ8(data []byte, x, xsums []float32, rows, cols, rowBytes int) (uint32, bool) {
	return argmaxRowsQ8(GGMLTypeQ6_K, data, x, xsums, rows, cols, rowBytes)
}

// argmaxQ4KRowsQ8 is the Q4_K analogue of argmaxQ6KRowsQ8. xsums must be the
// per-32-element sums of the original activations (fillQ4KXSums).
func argmaxQ4KRowsQ8(data []byte, x, xsums []float32, rows, cols, rowBytes int) (uint32, bool) {
	return argmaxRowsQ8(GGMLTypeQ4_K, data, x, xsums, rows, cols, rowBytes)
}

// argmaxQ5KRowsQ8 is the Q5_K analogue of argmaxQ4KRowsQ8. xsums are the same
// per-32-element sums Q5_K shares with Q4_K (fillQ4KXSums).
func argmaxQ5KRowsQ8(data []byte, x, xsums []float32, rows, cols, rowBytes int) (uint32, bool) {
	return argmaxRowsQ8(GGMLTypeQ5_K, data, x, xsums, rows, cols, rowBytes)
}

// argmaxQ8_0RowsQ8 is the Q8_0 analogue of argmaxQ4KRowsQ8. Q8_0 is symmetric
// (no dmin term), so unlike the K-quant variants it needs no xsums.
func argmaxQ8_0RowsQ8(data []byte, x []float32, rows, cols, rowBytes int) (uint32, bool) {
	return argmaxRowsQ8(GGMLTypeQ8_0, data, x, nil, rows, cols, rowBytes)
}

// The int8-activation path is now implemented everywhere, so the autotuner is
// allowed to consider it. It reports available even without int8 assembly:
// "available" means correct and selectable, and letting --auto measure the
// scalar kernels against the vectorised float path on the real machine is
// strictly better than guessing which wins.

func q8ActivationsAvailable() bool { return true }

func q8ActivationsEnabled() bool { return useQ8Activations.Load() }

func setQ8Activations(on bool) { useQ8Activations.Store(on) }

func kvF16Available() bool { return true }

func kvF16Enabled() bool { return useF16KVCache.Load() }

func setKVF16(on bool) { useF16KVCache.Store(on) }

func cpuFeatureString() string {
	if hasQ8KDotAsm {
		return runtime.GOARCH + "+dotprod"
	}
	return runtime.GOARCH
}
