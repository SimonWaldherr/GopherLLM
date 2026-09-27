package mobile

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/SimonWaldherr/GopherLLM/internal/testmodel"
)

// expectedText is what internal/testmodel's fixed weights generate for
// "Hallo" with deterministicOptions; the Rust and Python binding tests
// assert the same string.
const (
	expectedText         = "$(F=bBz\".C\\[}3WeO`>i[\"cZ"
	deterministicOptions = `{"max_tokens":24,"seed":1}`
)

func loadedEngine(t *testing.T) (*Engine, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := testmodel.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	e := NewEngine()
	t.Cleanup(func() { e.Close() })
	if err := e.Load(path, `{"threads":1}`); err != nil {
		t.Fatal(err)
	}
	return e, path
}

type recordingSink struct {
	mu       sync.Mutex
	deltas   []string
	complete string
	errors   []string
	onDelta  func(string)
}

func (s *recordingSink) OnDelta(d string) {
	s.mu.Lock()
	s.deltas = append(s.deltas, d)
	hook := s.onDelta
	s.mu.Unlock()
	if hook != nil {
		hook(d)
	}
}
func (s *recordingSink) OnComplete(r string) { s.mu.Lock(); s.complete = r; s.mu.Unlock() }
func (s *recordingSink) OnError(m string)    { s.mu.Lock(); s.errors = append(s.errors, m); s.mu.Unlock() }

type chatResult struct {
	Text            string  `json:"text"`
	FinishReason    string  `json:"finish_reason"`
	GeneratedTokens int     `json:"generated_tokens"`
	PromptTokens    int     `json:"prompt_tokens"`
	TotalMS         float64 `json:"total_ms"`
}

func decode[T any](t *testing.T, raw string) T {
	t.Helper()
	var v T
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		t.Fatalf("decode %q: %v", raw, err)
	}
	return v
}

func TestEngineGeneratesDeterministically(t *testing.T) {
	e, _ := loadedEngine(t)
	if !e.IsLoaded() || e.ModelName() != testmodel.Name {
		t.Fatalf("loaded=%v name=%q", e.IsLoaded(), e.ModelName())
	}
	text, err := e.Generate("Hallo", deterministicOptions)
	if err != nil || text != expectedText {
		t.Fatalf("Generate = %q, %v; want %q", text, err, expectedText)
	}

	raw, err := e.Chat(`[{"role":"user","content":"Hallo"}]`, deterministicOptions)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(raw, "`>i") {
		t.Errorf("Chat result is HTML-escaped: %s", raw)
	}
	r := decode[chatResult](t, raw)
	if r.Text != expectedText || r.FinishReason != "length" || r.GeneratedTokens != 24 || r.PromptTokens == 0 || r.TotalMS < 0 { // Windows timers can round a tiny run down to 0
		t.Fatalf("Chat = %+v", r)
	}

	sink := &recordingSink{}
	if err := e.ChatStream(`[{"role":"user","content":"Hallo"}]`, deterministicOptions, sink); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(sink.deltas, ""); got != expectedText || len(sink.errors) != 0 {
		t.Fatalf("stream deltas = %q, errors = %v", got, sink.errors)
	}
	if decode[chatResult](t, sink.complete).Text != expectedText {
		t.Fatalf("OnComplete = %s", sink.complete)
	}
}

func TestEngineChatUsesHistory(t *testing.T) {
	e, _ := loadedEngine(t)
	single, err := e.Chat(`[{"role":"user","content":"Hallo"}]`, deterministicOptions)
	if err != nil {
		t.Fatal(err)
	}
	multi, err := e.Chat(`[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Hallo"}]`, deterministicOptions)
	if err != nil {
		t.Fatal(err)
	}
	if decode[chatResult](t, multi).PromptTokens <= decode[chatResult](t, single).PromptTokens {
		t.Fatalf("history not rendered into the prompt: single=%s multi=%s", single, multi)
	}
	for _, bad := range []string{``, `[]`, `{}`, `[{"role":"tool","content":"x"}]`, `[{"role":"user","content":"x","extra":1}]`} {
		if _, err := e.Chat(bad, ""); err == nil {
			t.Errorf("Chat(%q) succeeded, want an error", bad)
		}
	}
	sink := &recordingSink{}
	if err := e.ChatStream(`[]`, "", sink); err == nil || len(sink.errors) != 1 {
		t.Fatalf("ChatStream(invalid) = %v, sink errors %v", err, sink.errors)
	}
}

func TestEngineInfoAndTokens(t *testing.T) {
	e, path := loadedEngine(t)
	info := decode[map[string]any](t, e.InfoJSON())
	for key, want := range map[string]any{"name": testmodel.Name, "architecture": "llama", "vocab_size": 98.0, "context_length": 256.0, "supported": true, "quantization": "F32"} {
		if info[key] != want {
			t.Errorf("InfoJSON[%s] = %v, want %v (%v)", key, info[key], want, info)
		}
	}
	raw, err := InspectModel(path)
	if err != nil {
		t.Fatal(err)
	}
	inspected := decode[map[string]any](t, raw)
	for _, key := range []string{"name", "architecture", "supported", "parameters", "tensor_bytes", "kv_cache_bytes_per_token"} {
		if inspected[key] != info[key] {
			t.Errorf("InspectModel[%s] = %v, InfoJSON has %v", key, inspected[key], info[key])
		}
	}
	if _, err := InspectModel(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Error("InspectModel(missing) succeeded")
	}

	n, err := e.CountTokens("Hallo")
	if err != nil || n != 6 { // BOS + one token per character
		t.Fatalf("CountTokens = %d, %v", n, err)
	}
	if rt := decode[map[string]any](t, RuntimeInfoJSON()); rt["version"] != Version() || rt["cpus"].(float64) < 1 {
		t.Fatalf("RuntimeInfoJSON = %v", rt)
	}
}

func TestEngineCancelStopsGeneration(t *testing.T) {
	e, _ := loadedEngine(t)
	sink := &recordingSink{}
	sink.onDelta = func(string) { e.Cancel() }
	err := e.GenerateStream("Hallo", `{"max_tokens":200}`, sink)
	if !errors.Is(err, context.Canceled) || len(sink.errors) != 1 || sink.complete != "" {
		t.Fatalf("err = %v, sink errors = %v, complete = %q", err, sink.errors, sink.complete)
	}
	if len(sink.deltas) >= 200 {
		t.Fatalf("generation ran to completion: %d deltas", len(sink.deltas))
	}
	// The engine is usable again afterwards.
	if text, err := e.Generate("Hallo", deterministicOptions); err != nil || text != expectedText {
		t.Fatalf("Generate after cancel = %q, %v", text, err)
	}
}

// Metadata calls must not wait for a running generation (a UI thread reads
// them while a reply streams), and Unload must cancel it rather than wait.
func TestEngineStaysResponsiveWhileGenerating(t *testing.T) {
	e, _ := loadedEngine(t)
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	sink := &recordingSink{}
	sink.onDelta = func(string) {
		once.Do(func() { close(started) })
		<-release
	}
	done := make(chan error, 1)
	go func() { done <- e.GenerateStream("Hallo", `{"max_tokens":200}`, sink) }()
	<-started

	within(t, "metadata during generation", func() {
		if e.ModelName() != testmodel.Name || !e.IsLoaded() || e.InfoJSON() == "{}" {
			t.Error("metadata unavailable during generation")
		}
		if _, err := e.CountTokens("Hallo"); err != nil {
			t.Error(err)
		}
	})
	unloaded := make(chan error, 1)
	go func() { unloaded <- e.Unload() }()
	// Unload cancels before it waits for the model; once it waits, readers
	// are refused, so the generation is sure to see the cancellation.
	within(t, "unload to start waiting", func() {
		for e.life.TryRLock() {
			e.life.RUnlock()
			time.Sleep(time.Millisecond)
		}
	})
	close(release)
	within(t, "unload during generation", func() {
		if err := <-unloaded; err != nil {
			t.Error(err)
		}
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Errorf("generation ended with %v, want context.Canceled", err)
		}
	})
	if e.IsLoaded() || e.InfoJSON() != "{}" {
		t.Fatal("model still loaded after Unload")
	}
	if _, err := e.Generate("Hallo", ""); err != ErrNoModel {
		t.Fatalf("Generate after Unload = %v, want %v", err, ErrNoModel)
	}
}

func TestEngineRejectsSecondLoad(t *testing.T) {
	e, path := loadedEngine(t)
	if err := e.Load(path, ""); err != ErrLoaded {
		t.Fatalf("second Load = %v, want %v", err, ErrLoaded)
	}
	if err := e.Load(path, `{"threads":-1}`); err == nil {
		t.Fatal("invalid load options accepted")
	}
}

func within(t *testing.T, what string, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() { defer close(done); f() }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("%s blocked", what)
	}
}
