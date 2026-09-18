package gopherllm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"
)

func jsonFixtureRunner(t *testing.T) *Runner {
	t.Helper()
	r, err := RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	for i := range r.tok.Vocab {
		r.tok.Vocab[i] = "x"
	}
	r.tok.Vocab[3] = "{"
	r.tok.Vocab[4] = "}"
	return r
}
func TestConstrainedJSONRuntimeAndIsolation(t *testing.T) {
	r := jsonFixtureRunner(t)
	o := DefaultGenerationOptions()
	o.JSONObject = true
	o.SystemPrompt = ""
	o.MaxTokens = 8
	o.Sampler.Temperature = 0
	for i := 0; i < 3; i++ {
		res, err := r.Generate("x", o)
		if err != nil || res.Text != "{}" || !json.Valid([]byte(res.Text)) || res.FinishReason != "stop" {
			t.Fatalf("%+v %v", res, err)
		}
	}
	o.MaxTokens = 1
	res, err := r.Generate("x", o)
	if !errors.Is(err, ErrStructuredOutputIncomplete) || res.FinishReason != "length" || res.Text != "{" {
		t.Fatalf("%+v %v", res, err)
	}
	o.MaxTokens = 8
	_, err = r.GenerateChatStreamUntil([]ChatMessage{UserMessage("x")}, o, func(string) bool { return false })
	if !errors.Is(err, ErrGenerationCanceled) {
		t.Fatal(err)
	}
	if _, err := r.Generate("x", o); err != nil {
		t.Fatal(err)
	}
}
func TestRunnerAdmissionCancellationAndBound(t *testing.T) {
	r := jsonFixtureRunner(t)
	r.genLock.Lock()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	go func() { _, err := r.Generate("x", DefaultGenerationOptions().WithContext(ctx)); done <- err }()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("uncancellable generation wait")
	}
	if r.inferencePending.Load() != 0 {
		t.Fatal("admission leaked")
	}
	r.genLock.Unlock()
	r.inferencePending.Store(64)
	if _, err := r.EmbedContext(context.Background(), "x"); !errors.Is(err, ErrRunnerBusy) {
		t.Fatal(err)
	}
	r.inferencePending.Store(0)
}

func BenchmarkJSONLogitMask8192(b *testing.B) {
	tok := &Tokenizer{Vocab: make([]string, 8192)}
	for i := range tok.Vocab {
		tok.Vocab[i] = "example"
	}
	c := newJSONTokenConstraint(tok)
	c.state, _ = c.state.Advance(`{"value":"`)
	logits := make([]float32, len(tok.Vocab))
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := c.mask(ctx, logits, func(uint32) bool { return false }); err != nil {
			b.Fatal(err)
		}
	}
}
