package server

import (
	"context"
	"encoding/json"
	"errors"
	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// /v1/completions' "suffix" field (OpenAI's legacy fill-in-the-middle shape)
// must reach GenerationOptions.FIMSuffix and fail with a clear, client-facing
// error on a checkpoint whose vocabulary has no Codestral [PREFIX]/[SUFFIX]/
// [MIDDLE] control tokens, rather than silently completing as if "suffix" had
// been ignored.
func TestCompletionsSuffixRequestsFIMAndFailsClearlyWithoutInfillTokens(t *testing.T) {
	r, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	h := NewHandler(r, HandlerOptions{Defaults: gopherllm.DefaultGenerationOptions()})
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"prompt":"def add(a, b):\n    ","suffix":"\n    return result\n"}`)))
	if w.Code < 400 {
		t.Fatalf("expected a client error, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "PREFIX") && !strings.Contains(w.Body.String(), "Codestral") {
		t.Fatalf("expected the error to explain the missing FIM tokens, got %s", w.Body.String())
	}
}

func TestOpenAIRejectsUnknownModelAndFormat(t *testing.T) {
	r, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	h := NewHandler(r, HandlerOptions{Defaults: gopherllm.DefaultGenerationOptions()})
	for _, body := range []string{`{"model":"missing","messages":[{"role":"user","content":"hi"}]}`, `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"xml"}}`, `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object","$ref":"x"}}}}`, `{"messages":[{"role":"unknown","content":"hi"}]}`, `{"messages":[{"role":"user","content":"hi"}],"n":2}`} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if w.Code < 400 || !strings.Contains(w.Body.String(), `"error":`) {
			t.Fatalf("%d %s", w.Code, w.Body.String())
		}
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/v1/models", nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), modelID(r)) {
		t.Fatal(w.Body.String())
	}
}
func TestToolRoundtripContract(t *testing.T) {
	calls := []gopherllm.ToolCall{{ID: "call_a", Type: "function", Function: gopherllm.ToolCallFunction{Name: "lookup", Arguments: `{"key":"a"}`}}, {ID: "call_b", Type: "function", Function: gopherllm.ToolCallFunction{Name: "lookup", Arguments: `{"key":"b"}`}}}
	history := []APIMessage{{Role: "assistant", ToolCalls: calls}, {Role: "tool", ToolCallID: "call_b", Content: "b"}, {Role: "tool", ToolCallID: "call_a", Content: "a"}}
	if err := validateMessages(history); err != nil {
		t.Fatal(err)
	}
	if err := validateMessages(history[:2]); err == nil {
		t.Fatal("missing result accepted")
	}
	options := gopherllm.DefaultGenerationOptions()
	options.Tools = []gopherllm.ToolDefinition{{Type: "function", Function: gopherllm.ToolFunctionDef{Name: "lookup"}}}
	options.ToolChoice = "required"
	result := gopherllm.GenerationResult{ToolCalls: calls, FinishReason: "tool_calls"}
	if err := validateChatResult(result, options); err != nil {
		t.Fatal(err)
	}
	if err := validateChatResult(gopherllm.GenerationResult{Text: "no"}, options); err == nil {
		t.Fatal("required choice ignored")
	}
	w := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	streamChatWithGenerator(w, req, io.Discard, "test", "llama", "test", false, options, true, func(func(string) bool, func(gopherllm.AgentEvent)) (gopherllm.GenerationResult, error) {
		return result, nil
	})
	var reconstructed []gopherllm.ToolCall
	for _, line := range strings.Split(w.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Calls []struct {
						Index int `json:"index"`
						gopherllm.ToolCall
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatal(err)
		}
		for _, choice := range chunk.Choices {
			for i, c := range choice.Delta.Calls {
				if c.Index != i {
					t.Fatal(c.Index)
				}
				reconstructed = append(reconstructed, c.ToolCall)
			}
		}
	}
	if len(reconstructed) != 2 || reconstructed[1].ID != "call_b" || !strings.Contains(w.Body.String(), `"finish_reason":"tool_calls"`) || !strings.Contains(w.Body.String(), `"choices":[],`) || !strings.HasSuffix(w.Body.String(), "data: [DONE]\n\n") {
		t.Fatal(w.Body.String())
	}
}
func TestSSEUnexpectedEmptyAndFailure(t *testing.T) {
	for _, inferenceErr := range []error{nil, errors.New("failed")} {
		w := httptest.NewRecorder()
		streamChatWithGenerator(w, httptest.NewRequest("POST", "/", nil), io.Discard, "id", "llama", "model", false, gopherllm.DefaultGenerationOptions(), false, func(func(string) bool, func(gopherllm.AgentEvent)) (gopherllm.GenerationResult, error) {
			return gopherllm.GenerationResult{}, inferenceErr
		})
		if !strings.Contains(w.Body.String(), `"error":`) || strings.Contains(w.Body.String(), `"finish_reason":"stop"`) {
			t.Fatal(w.Body.String())
		}
	}
}

// The native [THINK]/[/THINK] streaming splitter must engage from the
// mistralThink flag alone, not from the raw architecture string: GGUFs across
// the mistral/mistral3/ministral/mixtral labels declare general.architecture
// inconsistently, so a model resolved to plain "mistral" (not "mistral3")
// that still emits the native protocol must not fall through to the generic
// <think> splitter and leak "[THINK]"/"[/THINK]" as literal content.
func TestStreamMistralThinkProtocolKeyedOnFlagNotArchString(t *testing.T) {
	w := httptest.NewRecorder()
	streamChatWithGenerator(w, httptest.NewRequest("POST", "/", nil), io.Discard, "id", "mistral", "model", true, gopherllm.DefaultGenerationOptions(), false,
		func(onToken func(string) bool, observe func(gopherllm.AgentEvent)) (gopherllm.GenerationResult, error) {
			onToken("[THINK]reasoning[/THINK]answer")
			return gopherllm.GenerationResult{FinishReason: "stop"}, nil
		})
	body := w.Body.String()
	if !strings.Contains(body, `"reasoning_content":"reasoning"`) {
		t.Fatalf("expected native [THINK] protocol to produce a reasoning_content delta, got %s", body)
	}
	if !strings.Contains(body, `"content":"answer"`) {
		t.Fatalf("expected the post-[/THINK] text as a content delta, got %s", body)
	}
	if strings.Contains(body, "[THINK]") || strings.Contains(body, "[/THINK]") {
		t.Fatalf("[THINK]/[/THINK] markers must not leak into any delta, got %s", body)
	}
}
func TestAdmissionRejectsWithoutQueueAndReleases(t *testing.T) {
	sem := make(chan struct{}, 1)
	sem <- struct{}{}
	called := false
	h := withLimit(sem, func(http.ResponseWriter, *http.Request) { called = true })
	w := httptest.NewRecorder()
	h(w, httptest.NewRequest("POST", "/", nil))
	if w.Code != 429 || called || len(sem) != 1 {
		t.Fatalf("%d %v %d", w.Code, called, len(sem))
	}
	<-sem
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w = httptest.NewRecorder()
	h(w, httptest.NewRequest("POST", "/", nil).WithContext(ctx))
	if called || len(sem) != 0 {
		t.Fatal("cancelled admitted")
	}
	w = httptest.NewRecorder()
	h(w, httptest.NewRequest("POST", "/", nil))
	if !called || len(sem) != 0 {
		t.Fatal("slot leak")
	}
}
func TestEmbeddingContract(t *testing.T) {
	for _, input := range []any{nil, []any{}, []any{"a", 123}, 123, " "} {
		if _, err := validatedEmbeddingInputs(EmbeddingsRequest{Input: input}); err == nil {
			t.Fatalf("accepted %#v", input)
		}
	}
	got, err := validatedEmbeddingInputs(EmbeddingsRequest{Input: []any{"b", "a"}})
	if err != nil || strings.Join(got, ",") != "b,a" {
		t.Fatal(got, err)
	}
	if got := encodeEmbedding([]float32{1, -1}, "base64"); got != "AACAPwAAgL8=" {
		t.Fatal(got)
	}
}

func TestStructuredHTTPGuaranteesAndBufferedStream(t *testing.T) {
	r, err := gopherllm.RunnerFromGGUFBytes(buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	tok := r.Tokenizer()
	for i := range tok.Vocab {
		tok.Vocab[i] = "x"
	}
	tok.Vocab[3] = "{"
	tok.Vocab[4] = "}"
	options := gopherllm.DefaultGenerationOptions()
	options.SystemPrompt = ""
	options.MaxTokens = 8
	options.Sampler.Temperature = 0
	h := NewHandler(r, HandlerOptions{Defaults: options})
	for _, stream := range []bool{false, true} {
		body := `{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}`
		if stream {
			body = `{"messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true},"response_format":{"type":"json_object"}}`
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if w.Code != 200 || !strings.Contains(w.Body.String(), `"content":"{}"`) {
			t.Fatalf("%d %s", w.Code, w.Body.String())
		}
		if stream && !strings.HasSuffix(w.Body.String(), "data: [DONE]\n\n") {
			t.Fatal(w.Body.String())
		}
	}
	for _, body := range []string{`{"messages":[{"role":"user","content":"x"}],"max_tokens":1,"stream":true,"response_format":{"type":"json_object"}}`, `{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"test","strict":true,"schema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}}}`} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body)))
		if w.Code != 422 || strings.Contains(w.Body.String(), "data: [DONE]") {
			t.Fatalf("%d %s", w.Code, w.Body.String())
		}
	}
}
func TestObserverCapturesOverloadWithoutPayload(t *testing.T) {
	var observation RequestObservation
	next := observeRequests(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { writeAPIError(w, 429, "overloaded", "", "busy") }), func(o RequestObservation) { observation = o })
	next.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("secret prompt")))
	if observation.Status != 429 || observation.Bytes == 0 || observation.Endpoint != "/v1/chat/completions" {
		t.Fatalf("%+v", observation)
	}
}
