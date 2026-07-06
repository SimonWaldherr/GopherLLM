package gopherllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
	if row["cache"] != "none" || row["cache_hit"] != false || row["retry_count"].(float64) != 0 {
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
