package gopherllm

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
)

func TestOpenBytesGenerateAndTokenizeRoundTrip(t *testing.T) {
	ctx := context.Background()
	var logs bytes.Buffer
	m, err := OpenBytes(ctx, buildTinyLlamaGGUF(), WithLogWriter(&logs))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !strings.Contains(logs.String(), "GGUF v3") {
		t.Fatalf("expected load diagnostics in the provided writer, got %q", logs.String())
	}
	if m.Config().Dim != 8 {
		t.Fatalf("config dim = %d", m.Config().Dim)
	}

	res, err := m.Generate(ctx, "a b c",
		WithMaxTokens(4), WithTemperature(0), WithTopK(1), WithSystemPrompt(""), WithSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	if res.Stats.GeneratedTokens < 1 {
		t.Fatalf("generated = %d", res.Stats.GeneratedTokens)
	}

	ids := m.Tokenize("abc")
	if len(ids) < 2 || ids[0] != m.Tokenizer().BOSID {
		t.Fatalf("tokenize = %v", ids)
	}
	if got := m.Detokenize(ids[1:]); strings.TrimSpace(got) != "abc" {
		t.Fatalf("detokenize = %q", got)
	}
}

func TestOpenBytesIsSilentByDefault(t *testing.T) {
	// The library must not write to stderr/stdout on its own: OpenBytes with
	// no WithLogWriter goes through io.Discard. (Nothing to assert directly
	// on process stderr here; instead assert the plumbing default.)
	settings := applyLoadOptions(nil)
	if settings.logw == nil {
		t.Fatal("default log writer must be non-nil (io.Discard)")
	}
	if _, err := settings.logw.Write([]byte("x")); err != nil {
		t.Fatalf("default writer must accept writes: %v", err)
	}
}

func TestLoadOptionsAndModelAccessors(t *testing.T) {
	var logs bytes.Buffer
	settings := applyLoadOptions([]Option{
		WithLogWriter(&logs),
		WithThreads(1),
		WithPrepareQuantized(true),
		WithMetal(true),
	})
	if settings.logw != &logs || settings.threads != 1 || !settings.prepareQuantized || !settings.useMetal {
		t.Fatalf("settings = %+v", settings)
	}
	load := settings.loadOptions()
	if load.LogWriter != &logs || !load.PrepareQuantized || !load.UseMetal {
		t.Fatalf("load options = %+v", load)
	}

	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Runner() == nil || m.GGUF() == nil {
		t.Fatal("model accessors returned nil")
	}
	if m.Info().FileSizeBytes != len(buildTinyLlamaGGUF()) || m.Name() != "tiny" {
		t.Fatalf("info=%+v name=%q", m.Info(), m.Name())
	}
}

func TestOpenRejectsCanceledContextBeforeReadingFile(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Open(ctx, "does-not-exist.gguf"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Open error = %v, want context.Canceled", err)
	}
}

func TestBuildGenOptionsAppliesEveryOption(t *testing.T) {
	ctx := context.WithValue(context.Background(), "key", "value")
	tools := []ToolDefinition{{Type: "function", Function: ToolFunctionDef{Name: "weather"}}}
	got := buildGenOptions(ctx, []GenOption{
		WithMaxTokens(12), WithTemperature(0.4), WithTopP(0.8), WithTopK(5), WithMinP(0.2),
		WithRepeatPenalty(1.3), WithSeed(9), WithSystemPrompt("system"), WithStop("END", "STOP"),
		WithTools(tools...), WithToolChoice("required"),
	})
	if got.MaxTokens != 12 || got.Sampler.Temperature != 0.4 || got.Sampler.TopP != 0.8 || got.Sampler.TopK != 5 || got.Sampler.MinP != 0.2 || got.Sampler.RepeatPenalty != 1.3 || got.Seed != 9 || got.SystemPrompt != "system" || got.ToolChoice != "required" {
		t.Fatalf("options = %+v", got)
	}
	if len(got.StopSequences) != 2 || len(got.Tools) != 1 || got.ctx != ctx {
		t.Fatalf("stops=%v tools=%v ctx=%v", got.StopSequences, got.Tools, got.ctx)
	}
}

func TestGenerateContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	// Pre-canceled context: must fail fast with the context error.
	cancel()
	if _, err := m.Generate(ctx, "a b c", WithMaxTokens(4), WithSystemPrompt("")); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}

	// Cancel mid-stream: first delta cancels; generation stops with ctx error
	// and a partial result.
	ctx2, cancel2 := context.WithCancel(context.Background())
	deltas := 0
	_, err = m.Stream(ctx2, []ChatMessage{UserMessage("a b c d e f")}, func(string) error {
		deltas++
		cancel2()
		return nil
	}, WithMaxTokens(64), WithTemperature(0), WithTopK(1), WithSystemPrompt(""))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-stream err = %v, want context.Canceled", err)
	}
	if deltas == 0 {
		t.Fatal("expected at least one delta before cancellation")
	}
}

func TestStreamCallbackErrorStopsGeneration(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	boom := errors.New("boom")
	_, err = m.Stream(context.Background(), []ChatMessage{UserMessage("a b c")}, func(string) error {
		return boom
	}, WithMaxTokens(16), WithTemperature(0), WithTopK(1), WithSystemPrompt(""))
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want wrapped boom", err)
	}
}

func TestWithGenerationOptionsBasePlusOverride(t *testing.T) {
	base := DefaultGenerationOptions()
	base.MaxTokens = 99
	base.Sampler.TopK = 3
	got := buildGenOptions(context.Background(), []GenOption{
		WithGenerationOptions(base),
		WithMaxTokens(7), // later options apply on top of the base
	})
	if got.MaxTokens != 7 || got.Sampler.TopK != 3 {
		t.Fatalf("options = %+v", got)
	}
	if got.ctx == nil {
		t.Fatal("ctx must survive WithGenerationOptions")
	}
}

func TestEmbedViaModelAPI(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	emb, err := m.Embed(context.Background(), "a b c")
	if err != nil {
		t.Fatal(err)
	}
	if len(emb.Embedding) != m.Config().Dim {
		t.Fatalf("embedding dim = %d", len(emb.Embedding))
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Embed(ctx, "a"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled embed err = %v", err)
	}
}
