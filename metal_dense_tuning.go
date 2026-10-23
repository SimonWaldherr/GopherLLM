package gopherllm

import (
	"os"
	"sync"
	"sync/atomic"
)

// Zero preserves the uncalibrated schedule. Keep the raw value so restoring a
// RuntimeTuning also restores environment/default resolution, rather than
// pinning the currently resolved default.
var metalDenseRowsOverride atomic.Int32
var metalDenseRowsApplyMu sync.Mutex

func metalDenseRowsOverrideValue() int { return int(metalDenseRowsOverride.Load()) }

func setMetalDenseRows(rows int) {
	if rows != 2 && rows != 4 && rows != 8 {
		rows = 0
	}
	// Keep the Go snapshot and native atomic in the same writer order. This
	// lock is only on calibration/restore, never on the decode hot path.
	metalDenseRowsApplyMu.Lock()
	defer metalDenseRowsApplyMu.Unlock()
	metalDenseRowsOverride.Store(int32(rows))
	applyMetalDenseRows(rows)
}

// Explicit environment settings take precedence over calibration, including
// disabling the vector kernels. Invalid row counts retain default resolution.
func metalDenseRowsEnvPinned() bool {
	return metalDenseScheduleEnvironment() != "vector/default"
}

// Cache calibration separately when an environment setting changes the code
// under measurement. The calibrated runtime value itself is deliberately not
// part of the key: it is the cached result, rather than an input to tuning.
func metalDenseScheduleEnvironment() string {
	if os.Getenv("GOPHERLLM_METAL_DENSE_VECTOR") == "0" {
		return "legacy"
	}
	rows := os.Getenv("GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP")
	switch rows {
	case "2", "4", "8":
		return "vector/" + rows
	default:
		return "vector/default"
	}
}
