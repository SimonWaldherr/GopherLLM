package server

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"

	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newTestServer(t *testing.T, opts HandlerOptions) *httptest.Server {
	t.Helper()
	m, err := gopherllm.OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	if opts.Defaults.MaxTokens == 0 {
		opts.Defaults = gopherllm.DefaultGenerationOptions()
		opts.Defaults.MaxTokens = 4
		opts.Defaults.SystemPrompt = ""
		opts.Defaults.Sampler.Temperature = 0
		opts.Defaults.Sampler.TopK = 1
	}
	srv := httptest.NewServer(HandlerForModel(m, opts))
	t.Cleanup(srv.Close)
	return srv
}

func TestOllamaTagsShowPsVersionEndpoints(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{})

	resp, err := http.Get(srv.URL + "/api/tags")
	if err != nil {
		t.Fatal(err)
	}
	var tags struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if len(tags.Models) != 1 {
		t.Fatalf("tags models = %+v", tags.Models)
	}
	details, ok := tags.Models[0]["details"].(map[string]any)
	if !ok || details["quantization_level"] == "" {
		t.Fatalf("tags details = %+v", tags.Models[0])
	}

	resp, err = http.Post(srv.URL+"/api/show", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	var show map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("show status = %d body=%+v", resp.StatusCode, show)
	}
	if _, ok := show["model_info"]; !ok {
		t.Fatalf("show missing model_info: %+v", show)
	}

	resp, err = http.Get(srv.URL + "/api/ps")
	if err != nil {
		t.Fatal(err)
	}
	var ps struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ps); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if len(ps.Models) != 1 || ps.Models[0]["digest"] == "" {
		t.Fatalf("ps models = %+v", ps.Models)
	}

	resp, err = http.Get(srv.URL + "/api/version")
	if err != nil {
		t.Fatal(err)
	}
	var version map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if version["version"] == "" {
		t.Fatalf("version = %+v", version)
	}
}

func TestOllamaEmbedBatches(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{})

	resp, err := http.Post(srv.URL+"/api/embed", "application/json", strings.NewReader(`{"input":["a b","c d"]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got.Embeddings) != 2 {
		t.Fatalf("embeddings = %d, want 2", len(got.Embeddings))
	}
}

func TestOllamaGenerateStreamsNDJSONByDefault(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{})

	resp, err := http.Post(srv.URL+"/api/generate", "application/json", strings.NewReader(`{"prompt":"a b c"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var lines []map[string]any
	for scanner.Scan() {
		var line map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &line); err != nil {
			t.Fatalf("ndjson line: %v (%s)", err, scanner.Text())
		}
		lines = append(lines, line)
	}
	if len(lines) < 2 {
		t.Fatalf("expected multiple NDJSON lines, got %d", len(lines))
	}
	last := lines[len(lines)-1]
	if last["done"] != true {
		t.Fatalf("last line not done: %+v", last)
	}
	if _, ok := last["total_duration"]; !ok {
		t.Fatalf("last line missing total_duration: %+v", last)
	}
	first := lines[0]
	if first["done"] != false {
		t.Fatalf("first line should be done=false: %+v", first)
	}
}

func TestOllamaGenerateNonStreamingWhenStreamFalse(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{})

	resp, err := http.Post(srv.URL+"/api/generate", "application/json", strings.NewReader(`{"prompt":"a b c","stream":false}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got["done"] != true {
		t.Fatalf("expected single done=true response, got %+v", got)
	}
	if _, ok := got["eval_duration"]; !ok {
		t.Fatalf("missing eval_duration: %+v", got)
	}
}

func TestOpenAIDeveloperRoleMapsToSystem(t *testing.T) {
	msgs := apiMessages([]APIMessage{{Role: "developer", Content: "be terse"}})
	if len(msgs) != 1 || msgs[0].Role != gopherllm.ChatRoleSystem {
		t.Fatalf("developer role mapping = %+v", msgs)
	}
}

func TestNormalizeToolChoiceForcesNamedFunction(t *testing.T) {
	got := normalizeToolChoice(map[string]any{"type": "function", "function": map[string]any{"name": "get_weather"}})
	if got != "function:get_weather" {
		t.Fatalf("normalizeToolChoice(named) = %q", got)
	}
}

func TestActiveToolsNarrowsToForcedFunction(t *testing.T) {
	tools := []gopherllm.ToolDefinition{
		{Type: "function", Function: gopherllm.ToolFunctionDef{Name: "get_weather"}},
		{Type: "function", Function: gopherllm.ToolFunctionDef{Name: "get_time"}},
	}
	opts := gopherllm.GenerationOptions{Tools: tools, ToolChoice: "function:get_time"}
	active := opts.ActiveTools()
	if len(active) != 1 || active[0].Function.Name != "get_time" {
		t.Fatalf("ActiveTools = %+v", active)
	}

	// Unknown forced name degrades to offering everything rather than nothing.
	opts.ToolChoice = "function:does_not_exist"
	active = opts.ActiveTools()
	if len(active) != 2 {
		t.Fatalf("ActiveTools (unknown forced name) = %+v", active)
	}
}

func TestOpenAIStreamOptionsGatesUsage(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{})

	// Without stream_options.include_usage, the final chunk must not carry usage.
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"messages":[{"role":"user","content":"a b c"}],"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	body := readAll(t, resp)
	if strings.Contains(body, `"usage"`) {
		t.Fatalf("usage should be absent by default: %s", body)
	}
	if !strings.Contains(body, systemFingerprint) {
		t.Fatalf("missing system_fingerprint: %s", body)
	}

	// With it set, the final chunk must carry usage.
	resp, err = http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"messages":[{"role":"user","content":"a b c"}],"stream":true,"stream_options":{"include_usage":true}}`))
	if err != nil {
		t.Fatal(err)
	}
	body = readAll(t, resp)
	if !strings.Contains(body, `"usage"`) {
		t.Fatalf("usage should be present when include_usage=true: %s", body)
	}
}

func TestOpenAIUsageReportsCachedInputTokens(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{})
	const request = `{"messages":[{"role":"user","content":"a b c"}]}`

	post := func(payload string) *http.Response {
		t.Helper()
		resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(payload))
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}
	decodeUsage := func(resp *http.Response) struct {
		PromptTokens       int `json:"prompt_tokens"`
		PromptTokenDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} {
		t.Helper()
		defer resp.Body.Close()
		var body struct {
			Usage struct {
				PromptTokens       int `json:"prompt_tokens"`
				PromptTokenDetails struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		return body.Usage
	}

	cold := decodeUsage(post(request))
	if cold.PromptTokenDetails.CachedTokens != 0 {
		t.Fatalf("cold cached_tokens = %d, want 0", cold.PromptTokenDetails.CachedTokens)
	}
	warm := decodeUsage(post(request))
	if warm.PromptTokenDetails.CachedTokens <= 0 {
		t.Fatalf("warm cached_tokens = %d, want > 0", warm.PromptTokenDetails.CachedTokens)
	}
	if warm.PromptTokenDetails.CachedTokens > warm.PromptTokens {
		t.Fatalf("warm cached_tokens = %d, prompt_tokens = %d", warm.PromptTokenDetails.CachedTokens, warm.PromptTokens)
	}

	stream := readAll(t, post(`{"messages":[{"role":"user","content":"a b c"}],"stream":true,"stream_options":{"include_usage":true}}`))
	streamUsage, choices := usageFromSSE(t, stream)
	if streamUsage.PromptTokenDetails.CachedTokens <= 0 {
		t.Fatalf("stream cached_tokens = %d, want > 0; body=%s", streamUsage.PromptTokenDetails.CachedTokens, stream)
	}
	if len(choices) != 0 {
		t.Fatalf("stream usage chunk choices = %d, want 0; body=%s", len(choices), stream)
	}
}

func usageFromSSE(t *testing.T, body string) (struct {
	PromptTokens       int `json:"prompt_tokens"`
	PromptTokenDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}, []any) {
	t.Helper()
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		data := strings.TrimPrefix(scanner.Text(), "data: ")
		if data == scanner.Text() || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []any `json:"choices"`
			Usage   *struct {
				PromptTokens       int `json:"prompt_tokens"`
				PromptTokenDetails struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("decode SSE chunk: %v (%s)", err, data)
		}
		if chunk.Usage != nil {
			return *chunk.Usage, chunk.Choices
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	t.Fatalf("stream contained no usage chunk: %s", body)
	panic("unreachable")
}

func readAll(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer resp.Body.Close()
	var sb strings.Builder
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		sb.Write(buf[:n])
		if err != nil {
			break
		}
	}
	return sb.String()
}
