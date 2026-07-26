package gopherllm

import (
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
	m, err := OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	if opts.Defaults.MaxTokens == 0 {
		opts.Defaults = DefaultGenerationOptions()
		opts.Defaults.MaxTokens = 4
		opts.Defaults.SystemPrompt = ""
		opts.Defaults.Sampler.Temperature = 0
		opts.Defaults.Sampler.TopK = 1
	}
	srv := httptest.NewServer(m.HTTPHandler(opts))
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
	if len(msgs) != 1 || msgs[0].Role != ChatRoleSystem {
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
	tools := []ToolDefinition{
		{Type: "function", Function: ToolFunctionDef{Name: "get_weather"}},
		{Type: "function", Function: ToolFunctionDef{Name: "get_time"}},
	}
	opts := GenerationOptions{Tools: tools, ToolChoice: "function:get_time"}
	active := opts.activeTools()
	if len(active) != 1 || active[0].Function.Name != "get_time" {
		t.Fatalf("activeTools = %+v", active)
	}

	// Unknown forced name degrades to offering everything rather than nothing.
	opts.ToolChoice = "function:does_not_exist"
	active = opts.activeTools()
	if len(active) != 2 {
		t.Fatalf("activeTools (unknown forced name) = %+v", active)
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
