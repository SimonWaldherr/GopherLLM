package gopherllm

import "io"

// CoreMLFFNOptions selects devices for explicitly exported prefill FFNs.
// The default permits CPU and Neural Engine execution, leaving the GPU to
// Metal. Core ML decides where each operation executes.
type CoreMLFFNOptions struct {
	CPUOnly bool
	// AutoTune compares warm CPU-only and CPU/Neural Engine predictions at
	// attachment time. If both device modes are valid, CPU-only must be at
	// least 10% faster in two trials. Candidates are loaded sequentially per
	// layer and must pass the source quality probe. This adds startup work;
	// calibrate on an otherwise idle machine.
	// AutoTune and CPUOnly are mutually exclusive.
	AutoTune bool
	// LogWriter receives optional calibration timings and the chosen device
	// permission. These timings do not measure whole-model generation.
	LogWriter io.Writer
}
