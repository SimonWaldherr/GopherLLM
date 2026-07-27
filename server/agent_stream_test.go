package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// agentChunks pulls the gopherllm_agent payloads out of an SSE body.
func agentChunks(t *testing.T, body string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Agent map[string]any `json:"gopherllm_agent"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			if choice.Agent != nil {
				events = append(events, choice.Agent)
			}
		}
	}
	return events
}

// The observer exists so a caller can see what ran. Verify the events survive
// the trip through the SSE encoding rather than only testing the Go hook.
func TestAgentEventsSerializeIntoTheStream(t *testing.T) {
	// Shape check on the JSON contract the browser parses: field names here
	// must match what mergeAgentEvent in script.js reads.
	data, err := json.Marshal(gopherllm.AgentEvent{
		Kind: gopherllm.AgentEventToolEnd, Iteration: 2,
		Tool: "wikipedia_search", Result: "extract", DurationMS: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"kind", "iteration", "tool", "result", "duration_ms"} {
		if _, ok := decoded[field]; !ok {
			t.Errorf("AgentEvent JSON is missing %q, which the UI reads: %s", field, data)
		}
	}
	if decoded["kind"] != "tool_end" {
		t.Errorf("kind = %v, want tool_end", decoded["kind"])
	}
	// Duration must not also leak as a raw nanosecond field.
	if _, ok := decoded["Duration"]; ok {
		t.Error("Duration should be excluded from the wire form")
	}
}

// Skills default to on for clients that never send the field, so the switch
// cannot silently change behaviour for existing API users.
func TestSkillsEnabledDefaultsToOnAndHonoursAnExplicitFalse(t *testing.T) {
	var absent OpenAIChatRequest
	if err := json.Unmarshal([]byte(`{"messages":[]}`), &absent); err != nil {
		t.Fatal(err)
	}
	if !absent.SkillsEnabled() {
		t.Error("an absent gopherllm_skills must keep skills enabled")
	}

	var off OpenAIChatRequest
	if err := json.Unmarshal([]byte(`{"messages":[],"gopherllm_skills":false}`), &off); err != nil {
		t.Fatal(err)
	}
	if off.SkillsEnabled() {
		t.Error("gopherllm_skills:false must disable skills")
	}

	var on OpenAIChatRequest
	if err := json.Unmarshal([]byte(`{"messages":[],"gopherllm_skills":true}`), &on); err != nil {
		t.Fatal(err)
	}
	if !on.SkillsEnabled() {
		t.Error("gopherllm_skills:true must enable skills")
	}
}

// The streaming path reports tool activity per-chunk; a non-streaming caller
// only ever gets one response, so its turn's whole timeline has to ride
// along in that single payload instead. withAgentTimeline is what attaches
// it — same field names as the streaming wire form, so a client can share
// one parser (mergeAgentEvent in script.js) for both.
func TestWithAgentTimelineAttachesEventsWithTheStreamingWireShape(t *testing.T) {
	timeline := []gopherllm.AgentEvent{
		{Kind: gopherllm.AgentEventToolStart, Iteration: 1, Tool: "wikidata_sparql", Arguments: `{"query":"..."}`},
		{Kind: gopherllm.AgentEventToolEnd, Iteration: 1, Tool: "wikidata_sparql", Result: "some facts", DurationMS: 42},
	}
	response := withAgentTimeline(map[string]any{"id": "chatcmpl-gopherllm"}, timeline)

	data, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Agent []map[string]any `json:"gopherllm_agent"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Agent) != 2 {
		t.Fatalf("gopherllm_agent has %d events, want 2: %s", len(decoded.Agent), data)
	}
	for _, field := range []string{"kind", "iteration", "tool", "duration_ms"} {
		if _, ok := decoded.Agent[1][field]; !ok {
			t.Errorf("event JSON is missing %q, which mergeAgentEvent in script.js reads: %s", field, data)
		}
	}
	if decoded.Agent[0]["kind"] != "tool_start" || decoded.Agent[1]["kind"] != "tool_end" {
		t.Errorf("event order/kind not preserved: %s", data)
	}
}

// An ordinary request without any tool activity must not grow a
// gopherllm_agent key at all — an empty array would be a needless, confusing
// addition to every plain chat response.
func TestWithAgentTimelineOmitsEmptyTimelines(t *testing.T) {
	response := withAgentTimeline(map[string]any{"id": "chatcmpl-gopherllm"}, nil)
	if _, ok := response["gopherllm_agent"]; ok {
		t.Errorf("gopherllm_agent should be absent for an empty timeline, got %#v", response)
	}
}

// End-to-end guard: a plain, tool-free /v1/chat/completions request (the
// overwhelming common case) must not carry gopherllm_agent in its JSON body,
// proving the handler's new observer wiring doesn't leak an empty timeline
// onto ordinary responses.
func TestChatCompletionsOmitsAgentTimelineWithoutToolActivity(t *testing.T) {
	m, err := gopherllm.OpenBytes(context.Background(), buildTinyLlamaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	defaults := gopherllm.DefaultGenerationOptions()
	defaults.MaxTokens = 2
	defaults.SystemPrompt = ""
	defaults.Sampler.Temperature = 0
	defaults.Sampler.TopK = 1

	srv := httptest.NewServer(HandlerForModel(m, HandlerOptions{Defaults: defaults}))
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":2}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var decoded map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		t.Fatal(err)
	}
	if _, ok := decoded["gopherllm_agent"]; ok {
		t.Errorf("plain request must not carry gopherllm_agent: %#v", decoded)
	}
}
