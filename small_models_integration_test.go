package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

const smallModelLimitBytes int64 = 5 * 1024 * 1024 * 1024

func TestSmallLocalModelsAnswerEinsteinPrompt(t *testing.T) {
	if os.Getenv("GOPHERLLM_RUN_MODEL_SWEEP") != "1" {
		t.Skip("set GOPHERLLM_RUN_MODEL_SWEEP=1 to run the local <5GB GGUF model sweep")
	}
	modelDir := os.Getenv("GOPHERLLM_MODEL_DIR")
	if modelDir == "" {
		modelDir = DefaultModelDir()
	}
	entries, err := DiscoverModels(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	small := smallUsableModels(entries)
	if len(small) == 0 {
		t.Fatalf("no supported non-projector GGUF models below 5GB found in %s", modelDir)
	}
	binary := os.Getenv("GOPHERLLM_SWEEP_BINARY")
	if binary == "" {
		t.Fatal("GOPHERLLM_SWEEP_BINARY must point to a built gopherllm binary for timeout-safe model sweeps")
	}
	timeout := modelSweepTimeout(t)

	prompt := "Antworte in einem vollstaendigen deutschen Satz: Wer war Albert Einstein?"
	for _, entry := range small {
		entry := entry
		t.Run(safeTestName(entry.ID), func(t *testing.T) {
			text, err := runModelPromptWithTimeout(binary, entry.Path, prompt, timeout)
			t.Logf("%s size=%.2fGB timeout=%s output=%q", entry.ID, float64(entry.SizeBytes)/(1024*1024*1024), timeout, text)
			if err != nil {
				t.Fatal(err)
			}
			if !looksLikeEinsteinAnswer(text) {
				t.Fatalf("model output does not look like an Einstein answer")
			}
		})
	}
}

func runModelPromptWithTimeout(binary, modelPath, prompt string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(
		ctx,
		binary,
		modelPath,
		"--prompt", prompt,
		"--max-tokens", "24",
		"--temp", "0",
		"--system-prompt", "",
		"--timeout", timeout.String(),
	)
	var combined bytes.Buffer
	cmd.Stdout = &combined
	cmd.Stderr = &combined
	err := cmd.Run()
	text := strings.TrimSpace(combined.String())
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return text, errors.New("model prompt timed out after " + timeout.String())
	}
	if err != nil {
		return text, err
	}
	return text, nil
}

func modelSweepTimeout(t *testing.T) time.Duration {
	t.Helper()
	raw := os.Getenv("GOPHERLLM_MODEL_SWEEP_TIMEOUT")
	if raw == "" {
		return 2 * time.Minute
	}
	timeout, err := time.ParseDuration(raw)
	if err != nil {
		t.Fatalf("invalid GOPHERLLM_MODEL_SWEEP_TIMEOUT %q: %v", raw, err)
	}
	if timeout <= 0 {
		t.Fatalf("GOPHERLLM_MODEL_SWEEP_TIMEOUT must be greater than zero")
	}
	return timeout
}

func smallUsableModels(entries []ModelEntry) []ModelEntry {
	out := []ModelEntry{}
	for _, entry := range entries {
		if entry.IsSupported && !entry.IsProjector && entry.SizeBytes < smallModelLimitBytes {
			out = append(out, entry)
		}
	}
	return out
}

func looksLikeEinsteinAnswer(text string) bool {
	lower := strings.ToLower(text)
	if len([]rune(strings.TrimSpace(text))) < 12 {
		return false
	}
	if strings.ContainsRune(text, '\uFFFD') {
		return false
	}
	keywords := []string{
		"einstein",
		"physiker",
		"physicist",
		"wissenschaftler",
		"scientist",
		"relativ",
		"nobel",
		"theoret",
	}
	for _, keyword := range keywords {
		if strings.Contains(lower, keyword) {
			return true
		}
	}
	return false
}

func safeTestName(name string) string {
	name = strings.ReplaceAll(name, "/", "_")
	name = strings.ReplaceAll(name, " ", "_")
	return name
}
