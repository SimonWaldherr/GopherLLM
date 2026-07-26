package gopherllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestNewHandlerServesEndToEnd mounts the handler on an httptest server (no
// real network listener management, no Serve) and exercises the health, chat,
// and skills endpoints against the tiny synthetic model — proving the HTTP
// surface works as a plain mountable http.Handler.
func TestNewHandlerServesEndToEnd(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	defaults := DefaultGenerationOptions()
	defaults.MaxTokens = 4
	defaults.SystemPrompt = ""
	defaults.Sampler.Temperature = 0
	defaults.Sampler.TopK = 1

	srv := httptest.NewServer(m.HTTPHandler(HandlerOptions{Defaults: defaults}))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	var health map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if health["ok"] != true {
		t.Fatalf("health = %v", health)
	}

	body := strings.NewReader(`{"messages":[{"role":"user","content":"a b c"}],"max_tokens":4}`)
	resp, err = http.Post(srv.URL+"/v1/chat/completions", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status = %d", resp.StatusCode)
	}
	var chat struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chat); err != nil {
		t.Fatal(err)
	}
	if len(chat.Choices) != 1 || chat.Choices[0].FinishReason == "" {
		t.Fatalf("chat response = %+v", chat)
	}

	resp, err = http.Get(srv.URL + "/v1/skills")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("skills status = %d", resp.StatusCode)
	}
}

// TestNewHandlerMountsUnderPrefix proves the handler composes with a host
// application's mux via StripPrefix, the pattern the docs recommend.
func TestNewHandlerMountsUnderPrefix(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	hostMux := http.NewServeMux()
	hostMux.Handle("/llm/", http.StripPrefix("/llm", m.HTTPHandler(HandlerOptions{})))
	srv := httptest.NewServer(hostMux)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/llm/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("prefixed health status = %d", resp.StatusCode)
	}
}

func TestHandlerModelHotSwapUsesConfiguredCatalog(t *testing.T) {
	modelDir := t.TempDir()
	allowedPath := filepath.Join(modelDir, "catalog", "allowed.gguf")
	if err := os.MkdirAll(filepath.Dir(allowedPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(allowedPath, buildTinyLlamaGGUF(), 0o600); err != nil {
		t.Fatal(err)
	}
	outsidePath := filepath.Join(t.TempDir(), "outside.gguf")
	if err := os.WriteFile(outsidePath, buildTinyLlamaGGUF(), 0o600); err != nil {
		t.Fatal(err)
	}

	runner, _, err := RunnerFromPath(allowedPath)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	srv := httptest.NewServer(NewHandler(runner, HandlerOptions{
		ModelDir:  " " + modelDir + " ", // whitespace is normalized at the boundary.
		ModelPath: allowedPath,
	}))
	defer srv.Close()

	post := func(payload map[string]string) *http.Response {
		t.Helper()
		body, err := json.Marshal(payload)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := http.Post(srv.URL+"/models/load", "application/json", strings.NewReader(string(body)))
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	resp := post(map[string]string{"path": outsidePath})
	if resp.StatusCode != http.StatusBadRequest {
		body := readAll(t, resp)
		t.Fatalf("outside model status = %d body=%s", resp.StatusCode, body)
	}
	resp.Body.Close()

	resp, err = http.Get(srv.URL + "/models")
	if err != nil {
		t.Fatal(err)
	}
	var listed struct {
		Models []struct {
			ID            string `json:"id"`
			Loaded        bool   `json:"loaded"`
			ContextLength int    `json:"context_length"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listed); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if len(listed.Models) != 1 || listed.Models[0].ID != filepath.Join("catalog", "allowed") || !listed.Models[0].Loaded || listed.Models[0].ContextLength != 1024 {
		t.Fatalf("catalog after rejected load = %+v", listed.Models)
	}

	resp = post(map[string]string{"model": filepath.Join("catalog", "allowed")})
	if resp.StatusCode != http.StatusOK {
		body := readAll(t, resp)
		t.Fatalf("catalog id load status = %d body=%s", resp.StatusCode, body)
	}
	var loaded struct {
		ID            string `json:"id"`
		ContextLength int    `json:"context_length"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&loaded); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if loaded.ID != filepath.Join("catalog", "allowed") || loaded.ContextLength != 1024 {
		t.Fatalf("loaded response = %+v", loaded)
	}

	// Older clients send a path. It remains supported only when it resolves
	// to an entry in the configured catalog.
	resp = post(map[string]string{"path": allowedPath})
	if resp.StatusCode != http.StatusOK {
		body := readAll(t, resp)
		t.Fatalf("catalog path load status = %d body=%s", resp.StatusCode, body)
	}
	resp.Body.Close()
}

func TestWithLimitStopsWaitingWhenRequestIsCanceled(t *testing.T) {
	sem := make(chan struct{}, 1)
	sem <- struct{}{}
	called := make(chan struct{})
	handler := withLimit(sem, func(http.ResponseWriter, *http.Request) {
		close(called)
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodGet, "/", nil).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(httptest.NewRecorder(), req)
		close(done)
	}()

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("limited handler kept waiting after request cancellation")
	}
	select {
	case <-called:
		t.Fatal("limited handler ran after request cancellation")
	default:
	}
}

func TestHandlerLogsInferenceMetricsWithRequestID(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	defaults := DefaultGenerationOptions()
	defaults.MaxTokens = 2
	defaults.SystemPrompt = ""
	defaults.Sampler.Temperature = 0
	defaults.Sampler.TopK = 1

	var logs strings.Builder
	handler := m.HTTPHandler(HandlerOptions{Defaults: defaults, LogWriter: &logs})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"a b c"}],"max_tokens":2}`))
	req.Header.Set("X-Request-ID", "req-test-123")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("X-Request-ID"); got != "req-test-123" {
		t.Fatalf("response request id = %q", got)
	}

	var row map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(logs.String())), &row); err != nil {
		t.Fatalf("log json: %v in %q", err, logs.String())
	}
	if row["event"] != "inference" || row["request_id"] != "req-test-123" || row["provider"] != "local" {
		t.Fatalf("log identity fields = %#v", row)
	}
	if row["endpoint"] != "/v1/chat/completions" || row["model"] == "" {
		t.Fatalf("log route/model fields = %#v", row)
	}
	if row["prompt_tokens"].(float64) <= 0 || row["completion_tokens"].(float64) <= 0 {
		t.Fatalf("token fields = %#v", row)
	}
	if _, ok := row["ttft_ms"].(float64); !ok {
		t.Fatalf("missing ttft_ms in %#v", row)
	}
	if row["cache"] != "prefix" || row["cache_hit"] != false || row["cached_prompt_tokens"].(float64) != 0 || row["retry_count"].(float64) != 0 {
		t.Fatalf("cache/retry fields = %#v", row)
	}
}

func TestHandlerLogsInferenceErrors(t *testing.T) {
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	var logs strings.Builder
	handler := m.HTTPHandler(HandlerOptions{LogWriter: &logs})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"a"}],"max_tokens":0}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}

	var row map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(logs.String())), &row); err != nil {
		t.Fatalf("log json: %v in %q", err, logs.String())
	}
	if row["event"] != "inference" || row["error_type"] == "" || row["error"] == "" {
		t.Fatalf("error log fields = %#v", row)
	}
}
