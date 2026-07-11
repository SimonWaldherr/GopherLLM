package gopherllm

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// KernelBenchRow is one kernel's timing in the --kernel-bench report: a named
// matvec (attn_q, ffn_down, output, ...) from a real layer of the loaded
// model, so the numbers reflect the model's true shapes and quant types
// rather than synthetic sizes.
type KernelBenchRow struct {
	Name    string  `json:"name"`
	DType   string  `json:"dtype"`
	Rows    int     `json:"rows"`
	Cols    int     `json:"cols"`
	Runs    int     `json:"runs"`
	AvgMS   float64 `json:"avg_ms"`
	TotalMS float64 `json:"total_ms"`
}

// RunKernelBench times each weight matvec of one transformer layer (plus the
// embedding row lookup and the output projection) in isolation, printing a
// table or, with jsonOut, the machine-readable llm-kernel-bench.v1 document
// that `make kernel-bench` emits for cross-runtime comparisons.
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
		q, k, v, xs := []float32{}, []float32{}, []float32{}, []float32{}
		rows = append(rows, measureFunction(
			"attn_qkv_path",
			fmt.Sprintf("%s/%s/%s", layer.WQ.Type, layer.WK.Type, layer.WV.Type),
			layer.WQ.Rows+layer.WK.Rows+layer.WV.Rows,
			layer.WQ.Cols,
			runs,
			func() { tryMatvecAttentionInto(layer.WQ, layer.WK, layer.WV, dimInput, &xs, &q, &k, &v) },
		))
	}
	rows = append(rows, measureMatvec("attn_output", layer.WO, attnOutInput, runs, &out))
	if layer.HasGateUp {
		rows = append(rows, measureMatvec("ffn_gate_up", layer.WGateUp, dimInput, runs, &out))
	} else {
		rows = append(rows, measureMatvec("ffn_gate", layer.W1, dimInput, runs, &out))
		rows = append(rows, measureMatvec("ffn_up", layer.W3, dimInput, runs, &out))
		gate, up, xs := []float32{}, []float32{}, []float32{}
		rows = append(rows, measureFunction(
			"ffn_gate_up_path",
			fmt.Sprintf("%s/%s", layer.W1.Type, layer.W3.Type),
			layer.W1.Rows+layer.W3.Rows,
			layer.W1.Cols,
			runs,
			func() {
				if !tryMatvec2Into(layer.W1, layer.W3, dimInput, &xs, &gate, &up) {
					layer.W1.MatvecInto(dimInput, &gate)
					layer.W3.MatvecInto(dimInput, &up)
				}
			},
		))
	}
	rows = append(rows, measureMatvec("ffn_down", layer.W2, hiddenInput, runs, &out))
	if !layer.HasGateUp && !r.config.UseGELU {
		gate, up, hidden, projection, xs := []float32{}, []float32{}, []float32{}, []float32{}, []float32{}
		oldPath := func() {
			if !tryMatvec2Into(layer.W1, layer.W3, dimInput, &xs, &gate, &up) {
				layer.W1.MatvecInto(dimInput, &gate)
				layer.W3.MatvecInto(dimInput, &up)
			}
			ensureLenNoClear(&hidden, r.config.HiddenDim)
			siluMulF32(gate, up, hidden)
			layer.W2.MatvecInto(hidden, &projection)
		}
		rows = append(rows, measureFunction(
			"ffn_swiglu_down_reference",
			fmt.Sprintf("%s/%s/SiLU/%s", layer.W1.Type, layer.W3.Type, layer.W2.Type),
			layer.W1.Rows+layer.W3.Rows+layer.W2.Rows,
			layer.W1.Cols,
			runs,
			oldPath,
		))
		rows = append(rows, measureFunction(
			"ffn_swiglu_down_path",
			fmt.Sprintf("%s/%s/SiLU/%s", layer.W1.Type, layer.W3.Type, layer.W2.Type),
			layer.W1.Rows+layer.W3.Rows+layer.W2.Rows,
			layer.W1.Cols,
			runs,
			func() {
				if !matvecMetalSwiGLUInto(layer.W1.Metal, layer.W3.Metal, layer.W2.Metal, dimInput, &projection) {
					oldPath()
				}
			},
		))
	}
	rows = append(rows, measureMatvec("output", r.standard.Output, dimInput, runs, &out))
	rows = append(rows, measureFunction(
		"output_argmax_path",
		fmt.Sprintf("%s/argmax", r.standard.Output.Type),
		r.standard.Output.Rows,
		r.standard.Output.Cols,
		runs,
		func() {
			if _, ok := argmaxMetalQ6K(r.standard.Output.Metal, dimInput); !ok {
				_, _ = r.standard.Output.ArgmaxMatvec(dimInput)
			}
		},
	))
	return rows
}

func measureMatvec(name string, weight Weight, x []float32, runs int, out *[]float32) KernelBenchRow {
	return measureKernel(name, weight, weight.Rows, weight.Cols, runs, func() {
		weight.MatvecInto(x, out)
	})
}

func measureKernel(name string, weight Weight, rows, cols, runs int, body func()) KernelBenchRow {
	return measureFunction(name, weight.Type.String(), rows, cols, runs, body)
}

func measureFunction(name, dtype string, rows, cols, runs int, body func()) KernelBenchRow {
	body()
	start := time.Now()
	for range runs {
		body()
	}
	totalMS := float64(time.Since(start).Microseconds()) / 1000
	return KernelBenchRow{Name: name, DType: dtype, Rows: rows, Cols: cols, Runs: runs, AvgMS: totalMS / float64(runs), TotalMS: totalMS}
}

func deterministicBenchVector(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%251)-125) / 125
	}
	return out
}
