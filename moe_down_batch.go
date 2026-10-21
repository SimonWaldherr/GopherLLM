package gopherllm

import (
	"os"
	"sync"
)

// Active experts have different hidden inputs, but their independent down
// rows can share one worker dispatch. Keep the mixture reduction in router
// order: parallelizing that sum would change floating-point rounding.
const moeDownBatchMaxExperts = 32
const moeDownBatchScratchBytes = 8 << 20

var disableExpertDownBatch = newAtomicBool(os.Getenv("GOPHERLLM_NO_MOE_DOWN_BATCH") == "1")

type moeDownBatchScratch struct {
	hidden, down, sums []float32
}

func (s *moeDownBatchScratch) prepare(w ExpertWeight, n int) {
	hidden, down, sums := n*w.Input, n*w.Output, n*w.Input/16
	// Different layers can trade expert width for model width. Bound retained
	// capacities together, rather than keeping each layer's largest slab.
	if max(cap(s.hidden), hidden)+max(cap(s.down), down)+max(cap(s.sums), sums) > moeDownBatchScratchBytes/4 {
		*s = moeDownBatchScratch{}
	}
	ensureLenNoClear(&s.hidden, hidden)
}

func expertDownBatchReady(w ExpertWeight, selected []ExpertScore) bool {
	n := len(selected)
	if disableExpertDownBatch.Load() || n < 2 || n > moeDownBatchMaxExperts || w.Input <= 0 || w.Output <= 0 {
		return false
	}
	limit := moeDownBatchScratchBytes / (4 * n)
	if w.Input > limit || w.Output > limit || w.Input+w.Output+w.Input/16 > limit {
		return false
	}
	if w.Weight.F32 == nil {
		if w.Input%256 != 0 {
			return false
		}
		switch w.Weight.Type {
		case GGMLTypeQ4_K, GGMLTypeQ5_K, GGMLTypeQ6_K, GGMLTypeQ8_0, GGMLTypeMXFP4:
		default:
			return false
		}
	}
	for _, choice := range selected {
		if w.Weight.F32 != nil {
			if _, ok := w.f32Plane(choice.Index); !ok {
				return false
			}
		} else if _, ok := expertPlaneWeight(w, choice.Index); !ok {
			return false
		}
	}
	return true
}

type expertDownTask struct {
	w        ExpertWeight
	selected []ExpertScore
	x, out   []float32
	sums     []float32
	rowBytes int
	sumWidth int
	useQ8    bool
	q8       []int8
	scales   []float32
}

func (t *expertDownTask) runRows(start, end int) {
	cols, rows := t.w.Input, t.w.Output
	for j := start / rows; j < len(t.selected) && j*rows < end; j++ {
		lo, hi := max(start-j*rows, 0), min(end-j*rows, rows)
		x := t.x[j*cols : (j+1)*cols]
		out := t.out[j*rows : (j+1)*rows]
		if t.w.Weight.F32 != nil {
			plane, _ := t.w.f32Plane(t.selected[j].Index)
			for row := lo; row < hi; row++ {
				out[row] = DotF32(plane[row*cols:(row+1)*cols], x)
			}
			continue
		}
		plane, _ := expertPlaneWeight(t.w, t.selected[j].Index)
		sums := t.sums[j*t.sumWidth : (j+1)*t.sumWidth]
		if t.useQ8 {
			q8, scales := t.q8[j*cols:(j+1)*cols], t.scales[j*(cols/256):(j+1)*(cols/256)]
			switch t.w.Weight.Type {
			case GGMLTypeQ4_K:
				dotQ4KRowsQ8(plane.Raw, q8, scales, sums, cols, t.rowBytes, lo, hi, out)
			case GGMLTypeQ5_K:
				dotQ5KRowsQ8(plane.Raw, q8, scales, sums, cols, t.rowBytes, lo, hi, out)
			case GGMLTypeQ6_K:
				dotQ6KRowsQ8(plane.Raw, q8, scales, sums, cols, t.rowBytes, lo, hi, out)
			case GGMLTypeQ8_0:
				dotQ8_0RowsQ8(plane.Raw, q8, scales, cols, t.rowBytes, lo, hi, out)
			case GGMLTypeMXFP4:
				dotMXFP4RowsQ8(plane.Raw, q8, scales, cols, t.rowBytes, lo, hi, out)
			}
			continue
		}
		switch t.w.Weight.Type {
		case GGMLTypeQ4_K:
			dotQ4KRowsWithXSums(plane.Raw, x, sums, cols, t.rowBytes, lo, hi, out)
		case GGMLTypeQ6_K:
			dotQ6KRowsWithXSums(plane.Raw, x, sums, cols, t.rowBytes, lo, hi, out)
		default:
			for row := lo; row < hi; row++ {
				raw := plane.Raw[row*t.rowBytes : (row+1)*t.rowBytes]
				switch t.w.Weight.Type {
				case GGMLTypeQ5_K:
					out[row] = DotQ5KF32(raw, x, cols)
				case GGMLTypeQ8_0:
					out[row] = DotQ8_0F32(raw, x, cols)
				case GGMLTypeMXFP4:
					out[row] = DotMXFP4F32(raw, x, cols)
				}
			}
		}
	}
}

var expertDownTaskPool = sync.Pool{New: func() any { return new(expertDownTask) }}

// Caller has checked expertDownBatchReady and filled scratch.hidden in selected
// order. Each row uses the existing scalar/SIMD dot, including the same Q8
// quantization and offset sums as expertMatvecInto.
func expertDownBatchInto(w ExpertWeight, selected []ExpertScore, scratch *moeDownBatchScratch) {
	n, cols := len(selected), w.Input
	ensureLenNoClear(&scratch.down, n*w.Output)
	task := expertDownTaskPool.Get().(*expertDownTask)
	task.w, task.selected = w, selected
	task.x, task.out = scratch.hidden[:n*cols], scratch.down
	var lease q8ScratchLease
	if w.Weight.F32 == nil {
		task.rowBytes, _ = w.Weight.Type.DataSize(cols)
		task.useQ8 = useQ8Activations.Load()
		lease = task.prepareInputs(&scratch.sums)
	}
	parallelRowsTask(n*w.Output, task)
	if task.useQ8 {
		releaseQ8(task.q8, task.scales, lease)
	}
	// Never retain mapped model weights or request activation views in the pool.
	*task = expertDownTask{}
	expertDownTaskPool.Put(task)
}

// Expert inputs occupy complete 256-element blocks in one slab. Quantize
// that slab with a single lease, keeping each expert's block scales and sums
// independent. This also avoids one SIMD call per expert for offset sums.
func (t *expertDownTask) prepareInputs(sums *[]float32) q8ScratchLease {
	cols := len(t.x)
	switch t.w.Weight.Type {
	case GGMLTypeQ4_K, GGMLTypeQ5_K:
		t.sumWidth = t.w.Input / 32
		t.sums = fillQ4KXSums(t.x, cols, sums)
	case GGMLTypeQ6_K:
		t.sumWidth = t.w.Input / 16
		t.sums = fillQ6KXSums16(t.x, cols, sums)
		ScaleF32(t.sums, 32)
	}
	var lease q8ScratchLease
	if t.useQ8 {
		t.q8, t.scales, lease = acquireQ8(t.x, cols)
	}
	return lease
}
