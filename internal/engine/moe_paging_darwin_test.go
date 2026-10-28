package engine

import (
	"fmt"
	"syscall"
	"testing"
	"time"
)

// Real full-model A/B on macOS, including process-local VM fault counters.
// Each backend has its own serial oracle: CPU and Metal need not round alike.
func TestMoEConfiguredPrefillParity(t *testing.T) {
	r := loadMoEPerformanceModel(t)
	if r.config.Arch != "qwen35moe" {
		t.Skip("grouped prefill requires qwen35moe")
	}
	t.Setenv("GOPHERLLM_MOE_PREFILL", "1")
	t.Setenv("GOPHERLLM_NO_MOE_PREFILL", "")
	cache, _ := r.generationWorkspace(256)
	if !r.canBatchQwen35Metal(cache) {
		t.Fatal("configured MoE did not qualify for grouped prefill")
	}
	const prompt = "Write a Go function that removes duplicate strings while preserving their original order. Include a small example, handle an empty input, and briefly explain the time and memory complexity."
	var reference *specABOutput
	for round := -1; round < 3; round++ {
		for order := range 2 {
			serial := (round+order)&1 == 0
			if serial {
				t.Setenv("GOPHERLLM_NO_MOE_PREFILL", "1")
			} else {
				t.Setenv("GOPHERLLM_NO_MOE_PREFILL", "")
			}
			var before, after syscall.Rusage
			if err := syscall.Getrusage(syscall.RUSAGE_SELF, &before); err != nil {
				t.Fatal(err)
			}
			got, wall := specABRun(t, r, prompt, nil, 64, 3*time.Minute, specABMode{Name: fmt.Sprintf("serial_prefill=%t", serial)})
			if err := syscall.Getrusage(syscall.RUSAGE_SELF, &after); err != nil {
				t.Fatal(err)
			}
			if reference == nil {
				reference = &got
			} else if err := specABCheck(*reference, got); err != nil {
				t.Fatal(err)
			}
			t.Logf("round=%d order=%d serial_prefill=%t prompt=%d generated=%d digest=%s wall=%v prefill=%v decode=%v major_faults=%d minor_faults=%d scratch_bytes=%d", round, order, serial, got.Result.Stats.PromptTokens, len(got.Tokens), specABDigest(got.Tokens), wall, got.Result.Stats.PrefillTime, got.Result.Stats.DecodeTime, after.Majflt-before.Majflt, after.Minflt-before.Minflt, r.workspaceBuf.moePrefill.bytes())
		}
	}
}
