package integration_test

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

const smallModelLimitBytes int64 = 5 * 1024 * 1024 * 1024
const modelSweepOutOfCoreThreshold int64 = 4 * 1024 * 1024 * 1024

func TestSmallLocalModelsAnswerEinsteinPrompt(t *testing.T) {
	if os.Getenv("GOPHERLLM_RUN_MODEL_SWEEP") != "1" {
		t.Skip("set GOPHERLLM_RUN_MODEL_SWEEP=1 to run the local <5GB GGUF model sweep")
	}
	modelDir := os.Getenv("GOPHERLLM_MODEL_DIR")
	if modelDir == "" {
		modelDir = gopherllm.DefaultModelDir()
	}
	entries, err := gopherllm.DiscoverModels(modelDir, os.Stderr)
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

	prompt := "Wer war Albert Einstein?"
	for _, entry := range small {
		entry := entry
		t.Run(safeTestName(entry.ID), func(t *testing.T) {
			maxTokens := 16
			if entry.Architecture == "nemotron_h" || entry.Architecture == "nemotron_h_moe" {
				// Hybrid recurrent decode is deliberately per-token; a shorter
				// answer keeps the complete opt-in sweep practical on CPU.
				maxTokens = 12
			}
			text, err := runModelPromptWithTimeout(binary, entry.Path, prompt, maxTokens, timeout, entry.SizeBytes >= modelSweepOutOfCoreThreshold)
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

func runModelPromptWithTimeout(binary, modelPath, prompt string, maxTokens int, timeout time.Duration, outOfCore bool) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	args := []string{
		modelPath,
		"--prompt", prompt,
		"--max-tokens", strconv.Itoa(maxTokens),
		"--temp", "0",
		"--system-prompt", "",
		"--timeout", timeout.String(),
	}
	if outOfCore {
		args = append(args, "--out-of-core")
	}
	cmd := exec.CommandContext(ctx, binary, args...)
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

func smallUsableModels(entries []gopherllm.ModelEntry) []gopherllm.ModelEntry {
	out := []gopherllm.ModelEntry{}
	for _, entry := range entries {
		if entry.IsSupported && !entry.IsProjector && !entry.IsEmbedding && entry.SizeBytes < smallModelLimitBytes {
			out = append(out, entry)
		}
	}
	return out
}

func TestSmallUsableModelsExcludeProjectorsAndEmbeddings(t *testing.T) {
	entries := []gopherllm.ModelEntry{
		{ID: "chat", IsSupported: true, SizeBytes: 1},
		{ID: "embedding", IsSupported: true, IsEmbedding: true, SizeBytes: 1},
		{ID: "projector", IsSupported: true, IsProjector: true, SizeBytes: 1},
		{ID: "large", IsSupported: true, SizeBytes: smallModelLimitBytes},
		{ID: "unsupported", SizeBytes: 1},
	}
	got := smallUsableModels(entries)
	if len(got) != 1 || got[0].ID != "chat" {
		t.Fatalf("small usable models = %+v, want only chat", got)
	}
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
