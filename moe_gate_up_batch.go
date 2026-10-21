package gopherllm

import "sync"

// Tests can compare the complete mixture with independent expert projections.
var disableExpertGateUpBatchForTest = newAtomicBool(false)

type expertGateUpTask struct {
	gate, up       ExpertWeight
	selected       []ExpertScore
	x, sums        []float32
	gateOut, upOut []float32
	q8             []int8
	scales         []float32
	rowBytes       int
	useQ8          bool
}

func (t *expertGateUpTask) runRows(start, end int) {
	rows, cols := t.gate.Output, t.gate.Input
	// Each expert owns adjacent gate/up segments. Start at the first segment
	// touched by this worker instead of scanning the entire active set.
	for segment := start / rows; segment*rows < end; segment++ {
		j := segment / 2
		w, dst := t.gate, t.gateOut[j*rows:(j+1)*rows]
		if segment&1 != 0 {
			w, dst = t.up, t.upOut[j*rows:(j+1)*rows]
		}
		lo, hi := max(0, start-segment*rows), min(rows, end-segment*rows)
		if w.Weight.F32 != nil {
			plane, _ := w.f32Plane(t.selected[j].Index)
			for row := lo; row < hi; row++ {
				dst[row] = DotF32(plane[row*cols:(row+1)*cols], t.x)
			}
			continue
		}
		plane, _ := expertPlaneWeight(w, t.selected[j].Index)
		switch w.Weight.Type {
		case GGMLTypeQ4_K:
			if t.useQ8 {
				dotQ4KRowsQ8(plane.Raw, t.q8, t.scales, t.sums, cols, t.rowBytes, lo, hi, dst)
			} else {
				dotQ4KRowsWithXSums(plane.Raw, t.x, t.sums, cols, t.rowBytes, lo, hi, dst)
			}
		case GGMLTypeQ6_K:
			if t.useQ8 {
				dotQ6KRowsQ8(plane.Raw, t.q8, t.scales, t.sums, cols, t.rowBytes, lo, hi, dst)
			} else {
				dotQ6KRowsWithXSums(plane.Raw, t.x, t.sums, cols, t.rowBytes, lo, hi, dst)
			}
		}
	}
}

var expertGateUpTaskPool = sync.Pool{New: func() any { return new(expertGateUpTask) }}

// All selected gate/up projections share one input and worker dispatch.
// Keep the original row kernels and router order; no expert bank is copied.
// Validate every plane before changing outputs so callers can safely fall back.
func expertMatvecGateUpBatchInto(gate, up ExpertWeight, selected []ExpertScore, x []float32, q4Sums *[]float32, gateOut, upOut *[]float32) bool {
	n, cols, rows := len(selected), gate.Input, gate.Output
	if disableExpertGateUpBatchForTest.Load() || n == 0 || cols <= 0 || len(x) != cols ||
		cols != up.Input || rows != up.Output || rows <= 0 || n > int(^uint(0)>>1)/2/rows {
		return false
	}
	f32 := gate.Weight.F32 != nil && up.Weight.F32 != nil
	if !f32 && (gate.Weight.F32 != nil || up.Weight.F32 != nil || gate.Weight.Type != up.Weight.Type ||
		cols%256 != 0 || gate.Weight.Type != GGMLTypeQ4_K && gate.Weight.Type != GGMLTypeQ6_K) {
		return false
	}
	for _, w := range []ExpertWeight{gate, up} {
		for _, choice := range selected {
			if f32 {
				if _, ok := w.f32Plane(choice.Index); !ok {
					return false
				}
			} else if _, ok := expertPlaneWeight(w, choice.Index); !ok {
				return false
			}
		}
	}
	ensureLenNoClear(gateOut, n*rows)
	ensureLenNoClear(upOut, n*rows)
	task := expertGateUpTaskPool.Get().(*expertGateUpTask)
	task.gate, task.up, task.selected, task.x = gate, up, selected, x
	task.gateOut, task.upOut = *gateOut, *upOut
	var scratch *[]float32
	var lease q8ScratchLease
	if !f32 {
		task.rowBytes, _ = gate.Weight.Type.DataSize(cols)
		if gate.Weight.Type == GGMLTypeQ4_K && q4Sums != nil {
			task.sums = fillQ4KXSums(x, cols, q4Sums)
		} else {
			scratch = xsumsScratchPool.Get().(*[]float32)
			if gate.Weight.Type == GGMLTypeQ4_K {
				task.sums = fillQ4KXSums(x, cols, scratch)
			} else {
				task.sums = fillQ6KXSums16(x, cols, scratch)
				ScaleF32(task.sums, 32)
			}
		}
		task.useQ8 = useQ8Activations.Load()
		if task.useQ8 {
			task.q8, task.scales, lease = acquireQ8(x, cols)
		}
	}
	parallelRowsTask(n*2*rows, task)
	if task.useQ8 {
		releaseQ8(task.q8, task.scales, lease)
	}
	if scratch != nil {
		*scratch = task.sums
		xsumsScratchPool.Put(scratch)
	}
	// The pool must not keep mapped weights or request buffers alive.
	*task = expertGateUpTask{}
	expertGateUpTaskPool.Put(task)
	return true
}
