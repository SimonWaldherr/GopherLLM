package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type KernelBenchRow struct {
	Name    string  `json:"name"`
	DType   string  `json:"dtype"`
	Rows    int     `json:"rows"`
	Cols    int     `json:"cols"`
	Runs    int     `json:"runs"`
	AvgMS   float64 `json:"avg_ms"`
	TotalMS float64 `json:"total_ms"`
}

func RunKernelBench(r *Runner, modelPath string, runs, requestedLayer int, jsonOut bool) error {
	if runs <= 0 {
		return fmt.Errorf("--kernel-bench-runs must be greater than 0")
	}
	if r.kind != loadedStandard {
		return fmt.Errorf("--kernel-bench currently supports standard transformer weights only")
	}
	layer := requestedLayer
	if layer < 0 {
		layer = 0
	}
	if n := len(r.standard.Layers); n > 0 && layer >= n {
		layer = n - 1
	}
	rows := kernelBenchRows(r, runs, layer)
	if jsonOut {
		name, _ := r.ModelName()
		payload := map[string]any{
			"type":    "gopherllm.kernel_benchmark",
			"format":  "llm-kernel-bench.v1",
			"runtime": "GopherLLM",
			"model": map[string]any{
				"path":       modelPath,
				"name":       name,
				"arch":       r.Architecture(),
				"dim":        r.config.Dim,
				"hidden_dim": r.config.HiddenDim,
				"layers":     r.config.NLayers,
			},
			"layer":   layer,
			"runs":    runs,
			"kernels": rows,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(payload)
	}
	fmt.Printf("Kernel benchmark format=llm-kernel-bench.v1 runtime=GopherLLM layer=%d runs=%d\n", layer, runs)
	for _, row := range rows {
		fmt.Printf("%s dtype=%s rows=%d cols=%d avg=%.3fms total=%.3fms\n", row.Name, row.DType, row.Rows, row.Cols, row.AvgMS, row.TotalMS)
	}
	return nil
}

func kernelBenchRows(r *Runner, runs, layerIndex int) []KernelBenchRow {
	rows := []KernelBenchRow{}
	rowOut := []float32{}
	rows = append(rows, measureKernel("token_embd.row", r.standard.TokenEmbd, r.standard.TokenEmbd.Rows, r.config.Dim, runs, func() {
		r.standard.TokenEmbd.RowInto(0, r.config.Dim, &rowOut)
	}))
	if len(r.standard.Layers) == 0 {
		return rows
	}
	layer := r.standard.Layers[layerIndex]
	dimInput := deterministicBenchVector(r.config.Dim)
	attnOutInput := deterministicBenchVector(r.config.NHeads * r.config.ValueDim)
	hiddenInput := deterministicBenchVector(r.config.HiddenDim)
	out := []float32{}
	if layer.HasQKV {
		rows = append(rows, measureMatvec("attn_qkv", layer.WQKV, dimInput, runs, &out))
	} else {
		rows = append(rows, measureMatvec("attn_q", layer.WQ, dimInput, runs, &out))
		rows = append(rows, measureMatvec("attn_k", layer.WK, dimInput, runs, &out))
		rows = append(rows, measureMatvec("attn_v", layer.WV, dimInput, runs, &out))
	}
	rows = append(rows, measureMatvec("attn_output", layer.WO, attnOutInput, runs, &out))
	if layer.HasGateUp {
		rows = append(rows, measureMatvec("ffn_gate_up", layer.WGateUp, dimInput, runs, &out))
	} else {
		rows = append(rows, measureMatvec("ffn_gate", layer.W1, dimInput, runs, &out))
		rows = append(rows, measureMatvec("ffn_up", layer.W3, dimInput, runs, &out))
	}
	rows = append(rows, measureMatvec("ffn_down", layer.W2, hiddenInput, runs, &out))
	rows = append(rows, measureMatvec("output", r.standard.Output, dimInput, runs, &out))
	return rows
}

func measureMatvec(name string, weight Weight, x []float32, runs int, out *[]float32) KernelBenchRow {
	return measureKernel(name, weight, weight.Rows, weight.Cols, runs, func() {
		weight.MatvecInto(x, out)
	})
}

func measureKernel(name string, weight Weight, rows, cols, runs int, body func()) KernelBenchRow {
	body()
	start := time.Now()
	for range runs {
		body()
	}
	totalMS := float64(time.Since(start).Microseconds()) / 1000
	return KernelBenchRow{Name: name, DType: weight.Type.String(), Rows: rows, Cols: cols, Runs: runs, AvgMS: totalMS / float64(runs), TotalMS: totalMS}
}

func deterministicBenchVector(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%251)-125) / 125
	}
	return out
}
