package gopherllm

import "sync"

type q8_0Projection struct {
	data []byte
	out  []float32
	rows int
}

// Carry projections through the worker pool without an escaping callback.
// The small projection descriptors are reused; model and activation buffers
// are released after all workers finish, so a pool never retains a model.
type q8_0RowsTask struct {
	projections    []q8_0Projection
	q8             []int8
	scales         []float32
	cols, rowBytes int
}

func (t *q8_0RowsTask) runRows(start, end int) {
	base := 0
	for _, projection := range t.projections {
		if s, e := clippedRange(start, end, base, base+projection.rows); s < e {
			dotQ8_0RowsQ8(projection.data, t.q8, t.scales, t.cols, t.rowBytes, s-base, e-base, projection.out)
		}
		base += projection.rows
	}
}

var q8_0RowsTaskPool = sync.Pool{New: func() any { return new(q8_0RowsTask) }}

func acquireQ8_0RowsTask(projections, cols int, q8 []int8, scales []float32) *q8_0RowsTask {
	task := q8_0RowsTaskPool.Get().(*q8_0RowsTask)
	ensureLenNoClear(&task.projections, projections)
	task.q8, task.scales = q8, scales
	task.cols, task.rowBytes = cols, cols/32*34
	return task
}

func releaseQ8_0RowsTask(task *q8_0RowsTask) {
	clear(task.projections)
	task.projections = task.projections[:0]
	task.q8, task.scales = nil, nil
	task.cols, task.rowBytes = 0, 0
	q8_0RowsTaskPool.Put(task)
}
