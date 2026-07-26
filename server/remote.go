package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// remoteState is a small proxy for OpenAI-compatible chat APIs. Providers
// such as Ollama, llama.cpp, LM Studio and OpenAI expose this common shape;
// keeping the key here (rather than in browser storage) avoids sending it to
// arbitrary browser extensions or requiring the remote service to allow CORS.
type remoteState struct {
	mu      sync.RWMutex
	baseURL string
	apiKey  string
	model   string
	client  *http.Client
}

type remoteConfigRequest struct {
	BaseURL string `json:"base_url"`
	APIKey  string `json:"api_key"`
	Model   string `json:"model"`
}

func (s *remoteState) configure(request remoteConfigRequest) error {
	baseURL, err := normalizeRemoteBaseURL(request.BaseURL)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.baseURL = baseURL
	s.apiKey = strings.TrimSpace(request.APIKey)
	s.model = strings.TrimSpace(request.Model)
	s.mu.Unlock()
	return nil
}

func (s *remoteState) clear() {
	s.mu.Lock()
	s.baseURL, s.apiKey, s.model = "", "", ""
	s.mu.Unlock()
}

func (s *remoteState) snapshot() (baseURL, apiKey, model string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.baseURL, s.apiKey, s.model
}

func (s *remoteState) enabled() bool {
	baseURL, _, _ := s.snapshot()
	return baseURL != ""
}

func (s *remoteState) publicConfig() map[string]any {
	baseURL, _, model := s.snapshot()
	return map[string]any{"enabled": baseURL != "", "base_url": baseURL, "model": model}
}

func normalizeRemoteBaseURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", nil
	}
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("base_url must be an absolute http(s) URL without credentials, query, or fragment")
	}
	u.Path = strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(u.Path, "/v1") {
		u.Path += "/v1"
	}
	return strings.TrimRight(u.String(), "/"), nil
}

func (s *remoteState) proxyChat(w http.ResponseWriter, req *http.Request) {
	baseURL, apiKey, configuredModel := s.snapshot()
	if baseURL == "" {
		http.Error(w, "no remote API is configured", http.StatusServiceUnavailable)
		return
	}
	body, err := io.ReadAll(io.LimitReader(req.Body, 16<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if configuredModel != "" {
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(w, "invalid chat request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if model, _ := payload["model"].(string); strings.TrimSpace(model) == "" {
			payload["model"] = configuredModel
			body, err = json.Marshal(payload)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
	}
	upstream, err := http.NewRequestWithContext(req.Context(), http.MethodPost, baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	upstream.Header.Set("Accept", req.Header.Get("Accept"))
	if apiKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+apiKey)
	}
	response, err := s.client.Do(upstream)
	if err != nil {
		http.Error(w, "remote API request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	for _, name := range []string{"Content-Type", "Cache-Control"} {
		if value := response.Header.Get(name); value != "" {
			w.Header().Set(name, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	_, _ = io.Copy(w, response.Body)
}

func (s *remoteState) listModels(req *http.Request) (any, error) {
	baseURL, apiKey, _ := s.snapshot()
	if baseURL == "" {
		return nil, fmt.Errorf("no remote API is configured")
	}
	upstream, err := http.NewRequestWithContext(req.Context(), http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	if apiKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+apiKey)
	}
	response, err := s.client.Do(upstream)
	if err != nil {
		return nil, fmt.Errorf("remote API request failed: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		return nil, fmt.Errorf("remote API returned %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	var payload any
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func newRemoteState() *remoteState {
	return &remoteState{client: &http.Client{Timeout: 10 * time.Minute}}
}
