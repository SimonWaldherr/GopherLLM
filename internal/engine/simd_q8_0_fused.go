package engine

// matvecQ8_0GroupInto shares activation quantization and one worker dispatch
// across Q/K/V or gate/up. Generic float fusion otherwise bypasses the Q8
// activation backend used by individual Q8_0 matvecs (notably Qwen Q8_0).
// The caller checks the activation mode; unsupported shapes decline before
// touching outputs. Exact float fusion remains available with Q8 disabled.
func matvecQ8_0GroupInto(weights []Weight, x []float32, outputs []*[]float32) bool {
	cols := len(x)
	if cols == 0 || cols%256 != 0 || len(weights) != len(outputs) {
		return false
	}
	rowBytes := cols / 32 * 34
	total := 0
	for _, w := range weights {
		if w.Type != GGMLTypeQ8_0 || w.Cols != cols || w.Rows < 0 || w.F32 != nil || w.GPU != nil || w.Metal != nil || len(w.Raw)/rowBytes < w.Rows {
			return false
		}
		total += w.Rows
	}
	for i, w := range weights {
		ensureLenNoClear(outputs[i], w.Rows)
	}
	q8, scales, lease := acquireQ8(x, cols)
	task := acquireQ8_0RowsTask(len(weights), cols, q8, scales)
	for i, w := range weights {
		task.projections[i] = q8_0Projection{data: w.Raw, out: *outputs[i], rows: w.Rows}
	}
	parallelRowsTask(total, task)
	releaseQ8_0RowsTask(task)
	releaseQ8(q8, scales, lease)
	return true
}
