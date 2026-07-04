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
