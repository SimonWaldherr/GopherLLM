// gopherllm-eval evaluates configurable Chat Completions endpoints. It owns no
// model and never downloads weights. Fixtures and predicates are replaceable.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"
	"time"
)

type fixture struct {
	ID             string           `json:"id"`
	Task           string           `json:"task"`
	Messages       []map[string]any `json:"messages"`
	Tools          []map[string]any `json:"tools,omitempty"`
	ResponseFormat map[string]any   `json:"response_format,omitempty"`
	Contains       []string         `json:"contains,omitempty"`
	JSONEquals     map[string]any   `json:"json_equals,omitempty"`
	ToolArguments  map[string]any   `json:"tool_arguments,omitempty"`
	ToolName       string           `json:"tool_name,omitempty"`
}
type measurement struct {
	Fixture            string   `json:"fixture"`
	Task               string   `json:"task"`
	Repetition         int      `json:"repetition"`
	Conformant         bool     `json:"api_conformant"`
	OutputValid        bool     `json:"output_valid"`
	Semantic           *bool    `json:"semantic_predicate_pass"`
	TTFTMS             *float64 `json:"ttft_ms"`
	EndToEndMS         float64  `json:"end_to_end_ms"`
	CompletionTokens   int      `json:"completion_tokens"`
	DecodeTPS          *float64 `json:"observed_decode_tokens_per_second"`
	ClientHeapBytes    uint64   `json:"client_heap_bytes_after"`
	ServerPeakRSSBytes *uint64  `json:"server_peak_rss_bytes"`
	Error              string   `json:"error,omitempty"`
}
type runConfig struct {
	endpoint, model, token string
	maxTokens              int
	seed                   int64
	timeout                time.Duration
	serverPID              int
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	endpoint := flag.String("url", "http://127.0.0.1:8080/v1/chat/completions", "compatible Chat Completions endpoint")
	model := flag.String("model", "", "exact served model ID (required)")
	fixturesPath := flag.String("fixtures", "", "JSON fixture array; default synthetic suite")
	n := flag.Int("n", 3, "measured repetitions per fixture")
	warmup := flag.Int("warmup", 1, "unmeasured full-suite repetitions")
	maxTokens := flag.Int("max-tokens", 128, "completion token budget")
	seed := flag.Int64("seed", 1, "sampling seed; temperature is fixed at zero")
	timeout := flag.Duration("timeout", time.Minute, "per-request deadline")
	hardware := flag.String("hardware", "unspecified", "server hardware description")
	backend := flag.String("backend", "unspecified", "server backend")
	tags := flag.String("build-tags", "unspecified", "server build tags")
	quant := flag.String("quantization", "unspecified", "server model quantization")
	serverPID := flag.Int("server-pid", 0, "optional local server PID for 100ms RSS sampling; loopback endpoints only")
	revision := flag.String("server-revision", "unspecified", "server build revision")
	flag.Parse()
	u, err := url.Parse(*endpoint)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.User != nil || u.RawQuery != "" {
		return fmt.Errorf("url must be an HTTP(S) endpoint without credentials/query")
	}
	if *serverPID < 0 || *serverPID > 0 && u.Hostname() != "127.0.0.1" && u.Hostname() != "localhost" && u.Hostname() != "::1" {
		return fmt.Errorf("server-pid requires a local loopback endpoint")
	}
	if *model == "" || *n < 1 || *warmup < 0 || *maxTokens < 1 || *timeout <= 0 {
		return fmt.Errorf("model, n, max-tokens and timeout must be valid")
	}
	fixtures := syntheticFixtures()
	if *fixturesPath != "" {
		data, err := os.ReadFile(*fixturesPath)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(data, &fixtures); err != nil {
			return err
		}
	}
	if len(fixtures) == 0 {
		return fmt.Errorf("fixture suite is empty")
	}
	enc := json.NewEncoder(os.Stdout)
	_ = enc.Encode(map[string]any{"kind": "environment", "endpoint": *endpoint, "model": *model, "hardware": *hardware, "backend": *backend, "build_tags": *tags, "quantization": *quant, "server_revision": *revision, "temperature": 0, "seed": *seed, "max_tokens": *maxTokens, "warmup_repetitions": *warmup, "measured_repetitions": *n, "fixtures": len(fixtures), "client_go": runtime.Version(), "client_os": runtime.GOOS, "client_arch": runtime.GOARCH, "server_memory": "optional local ps RSS, 100ms sampling; null when unavailable", "server_pid": *serverPID, "semantic_method": "fixture predicates, not an LLM judge"})
	cfg := runConfig{*endpoint, *model, os.Getenv("GOPHERLLM_EVAL_TOKEN"), *maxTokens, *seed, *timeout, *serverPID}
	client := &http.Client{Timeout: *timeout}
	failures := 0
	for repeat := -*warmup; repeat < *n; repeat++ {
		for _, f := range fixtures {
			m := evaluate(context.Background(), client, cfg, f)
			m.Repetition = repeat
			if repeat >= 0 {
				if !m.Conformant || !m.OutputValid || m.Semantic != nil && !*m.Semantic {
					failures++
				}
				if err := enc.Encode(m); err != nil {
					return err
				}
			}
		}
	}
	if failures > 0 {
		return fmt.Errorf("%d measured evaluations failed; see JSONL results", failures)
	}
	return nil
}
func evaluate(parent context.Context, client *http.Client, cfg runConfig, f fixture) measurement {
	m := measurement{Fixture: f.ID, Task: f.Task}
	stopMemory := sampleRSS(parent, cfg.serverPID)
	start := time.Now()
	finish := func() measurement {
		m.EndToEndMS = float64(time.Since(start)) / float64(time.Millisecond)
		m.ServerPeakRSSBytes = stopMemory()
		var stats runtime.MemStats
		runtime.ReadMemStats(&stats)
		m.ClientHeapBytes = stats.HeapAlloc
		return m
	}
	payload := map[string]any{"model": cfg.model, "messages": f.Messages, "max_tokens": cfg.maxTokens, "temperature": 0, "seed": cfg.seed, "stream": true, "stream_options": map[string]any{"include_usage": true}}
	if len(f.Tools) > 0 {
		payload["tools"] = f.Tools
		payload["tool_choice"] = "required"
	}
	if f.ResponseFormat != nil {
		payload["response_format"] = f.ResponseFormat
	}
	body, err := json.Marshal(payload)
	if err != nil {
		m.Error = "invalid_fixture"
		return finish()
	}
	ctx, cancel := context.WithTimeout(parent, cfg.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "POST", cfg.endpoint, bytes.NewReader(body))
	if err != nil {
		m.Error = "invalid_endpoint"
		return finish()
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.token)
	}
	resp, err := client.Do(req)
	if err != nil {
		m.Error = "transport_error"
		return finish()
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		m.Error = fmt.Sprintf("http_%d", resp.StatusCode)
		return finish()
	}
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		m.Error = "not_sse"
		return finish()
	}
	scanner := bufio.NewScanner(io.LimitReader(resp.Body, 16<<20))
	scanner.Buffer(make([]byte, 8192), 1<<20)
	scanner.Split(splitSSEEvents)
	text := ""
	calls := map[int]struct{ Name, Args, ID string }{}
	done, terminal, usageSeen := false, false, false
	valid := true
	reason := ""
	var first, last time.Time
	for scanner.Scan() {
		var lines []string
		for _, line := range strings.Split(scanner.Text(), "\n") {
			if strings.HasPrefix(line, "data:") {
				lines = append(lines, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			}
		}
		if len(lines) == 0 {
			continue
		}
		data := strings.Join(lines, "\n")
		if data == "[DONE]" {
			done = true
			break
		}
		var chunk struct {
			Object string          `json:"object"`
			Model  string          `json:"model"`
			Error  json.RawMessage `json:"error"`
			Usage  *struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
			Choices []struct {
				Index  int     `json:"index"`
				Finish *string `json:"finish_reason"`
				Delta  struct {
					Content string `json:"content"`
					Calls   []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil || chunk.Object != "chat.completion.chunk" || chunk.Model != cfg.model || len(chunk.Error) > 0 {
			valid = false
			continue
		}
		if chunk.Usage != nil {
			usageSeen = true
			m.CompletionTokens = chunk.Usage.CompletionTokens
		}
		for _, choice := range chunk.Choices {
			if choice.Index != 0 {
				valid = false
			}
			if choice.Finish != nil {
				if terminal {
					valid = false
				}
				terminal = true
				reason = *choice.Finish
			}
			if choice.Delta.Content != "" || len(choice.Delta.Calls) > 0 {
				now := time.Now()
				if first.IsZero() {
					first = now
					v := float64(now.Sub(start)) / float64(time.Millisecond)
					m.TTFTMS = &v
				}
				last = now
			}
			text += choice.Delta.Content
			for _, call := range choice.Delta.Calls {
				x := calls[call.Index]
				x.Name += call.Function.Name
				x.Args += call.Function.Arguments
				x.ID += call.ID
				calls[call.Index] = x
			}
		}
	}
	m.Conformant = valid && done && terminal && usageSeen && scanner.Err() == nil
	m.OutputValid = (reason == "stop" && strings.TrimSpace(text) != "") || (reason == "tool_calls" && len(calls) > 0)
	if f.ResponseFormat != nil {
		var obj map[string]any
		m.OutputValid = m.OutputValid && json.Unmarshal([]byte(text), &obj) == nil && obj != nil
	}
	ids := map[string]bool{}
	for index, call := range calls {
		if index < 0 || index >= len(calls) || ids[call.ID] {
			m.OutputValid = false
		}
		ids[call.ID] = true
		var obj map[string]any
		if call.ID == "" || call.Name == "" || json.Unmarshal([]byte(call.Args), &obj) != nil || obj == nil {
			m.OutputValid = false
		}
	}
	if !m.Conformant {
		m.Error = "invalid_or_incomplete_stream"
	}
	if !first.IsZero() && last.After(first) && m.CompletionTokens > 1 && len(calls) == 0 {
		v := float64(m.CompletionTokens-1) / last.Sub(first).Seconds()
		m.DecodeTPS = &v
	}
	if len(f.Contains) > 0 || len(f.JSONEquals) > 0 || f.ToolName != "" {
		pass := m.Conformant && m.OutputValid
		for _, word := range f.Contains {
			pass = pass && strings.Contains(strings.ToLower(text), strings.ToLower(word))
		}
		if len(f.JSONEquals) > 0 {
			var obj map[string]any
			if json.Unmarshal([]byte(text), &obj) != nil {
				pass = false
			}
			for k, want := range f.JSONEquals {
				a, _ := json.Marshal(obj[k])
				b, _ := json.Marshal(want)
				pass = pass && bytes.Equal(a, b)
			}
		}
		if f.ToolName != "" {
			found := false
			for _, c := range calls {
				matches := c.Name == f.ToolName
				var args map[string]any
				if json.Unmarshal([]byte(c.Args), &args) != nil {
					matches = false
				}
				for k, want := range f.ToolArguments {
					a, _ := json.Marshal(args[k])
					b, _ := json.Marshal(want)
					matches = matches && bytes.Equal(a, b)
				}
				found = found || matches
			}
			pass = pass && found
		}
		m.Semantic = &pass
	}
	return finish()
}
func syntheticFixtures() []fixture {
	msg := func(text string) []map[string]any { return []map[string]any{{"role": "user", "content": text}} }
	return []fixture{
		{ID: "extract", Task: "extraction", Messages: msg("Extract the integer count from: There are 7 blue boxes. Return JSON with key count."), ResponseFormat: map[string]any{"type": "json_object"}, JSONEquals: map[string]any{"count": 7}},
		{ID: "summary", Task: "summary", Messages: msg("Summarize in one sentence: The launch was scheduled Monday. Testing found a defect. The team moved the launch to Friday. Include Friday."), Contains: []string{"Friday"}},
		{ID: "tool", Task: "tool-use", Messages: msg("Call lookup with key alpha."), Tools: []map[string]any{{"type": "function", "function": map[string]any{"name": "lookup", "parameters": map[string]any{"type": "object", "properties": map[string]any{"key": map[string]any{"type": "string"}}, "required": []string{"key"}}}}}, ToolName: "lookup", ToolArguments: map[string]any{"key": "alpha"}},
		{ID: "german", Task: "multilingual", Messages: msg("Übersetze ins Englische, antworte nur mit dem Wort: Katze"), Contains: []string{"cat"}},
		{ID: "long", Task: "long-context", Messages: msg("Remember the code: ORBIT-732.\n" + strings.Repeat("This is irrelevant synthetic filler.\n", 120) + "Return only the remembered code."), Contains: []string{"ORBIT-732"}},
	}
}

// splitSSEEvents retains data lines until the empty line terminating an event.
// CRLF and LF framing are both accepted; incomplete EOF events are rejected.
func splitSSEEvents(data []byte, atEOF bool) (int, []byte, error) {
	end := bytes.Index(data, []byte("\n\n"))
	delimiter := 2
	if crlf := bytes.Index(data, []byte("\r\n\r\n")); crlf >= 0 && (end < 0 || crlf < end) {
		end, delimiter = crlf, 4
	}
	if end >= 0 {
		return end + delimiter, bytes.ReplaceAll(data[:end], []byte("\r\n"), []byte("\n")), nil
	}
	if atEOF && len(bytes.TrimSpace(data)) > 0 {
		return 0, nil, fmt.Errorf("unterminated SSE event")
	}
	return 0, nil, nil
}
