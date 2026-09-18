package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/jsonconstraint"
)

var errSchemaOutput = errors.New("schema validation failed")
var errOutputContract = errors.New("invalid model output")

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   any    `json:"param"`
	Code    string `json:"code"`
}

func writeAPIError(w http.ResponseWriter, status int, code, param, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	errorType := "invalid_request_error"
	if status >= 500 || status == 422 {
		errorType = "server_error"
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"error": apiError{message, errorType, param, code}})
}
func decodeAPIRequest(w http.ResponseWriter, r *http.Request, v any) bool {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		writeAPIError(w, 405, "method_not_allowed", "", "POST required")
		return false
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(v); err != nil {
		writeAPIError(w, 400, "invalid_json", "", err.Error())
		return false
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		writeAPIError(w, 400, "invalid_json", "", "exactly one JSON object required")
		return false
	}
	return true
}
func validateModel(w http.ResponseWriter, requested, actual string) bool {
	if actual == "" {
		writeAPIError(w, 503, "model_unavailable", "model", "no model is loaded")
		return false
	}
	if requested != "" && requested != actual {
		writeAPIError(w, 404, "model_not_found", "model", "unknown model ID; use /v1/models")
		return false
	}
	return true
}
func inferenceAPIError(w http.ResponseWriter, err error) {
	status, code := 400, "generation_failed"
	if errors.Is(err, errOutputContract) {
		status, code = 502, "invalid_model_output"
	}
	if errors.Is(err, errSchemaOutput) {
		status, code = 422, "schema_validation_failed"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		status, code = 504, "deadline_exceeded"
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, gopherllm.ErrGenerationCanceled) {
		status, code = 499, "cancelled"
	}
	if errors.Is(err, gopherllm.ErrRunnerBusy) {
		status, code = 429, "overloaded"
	}
	if errors.Is(err, gopherllm.ErrStructuredOutputIncomplete) {
		status, code = 422, "incomplete_output"
	}
	writeAPIError(w, status, code, "", err.Error())
}

// ResponseFormat selects text or validated structured Chat Completions output.
type ResponseFormat struct {
	Type       string            `json:"type"`
	JSONSchema *JSONSchemaFormat `json:"json_schema,omitempty"`
}

// JSONSchemaFormat supports the subset documented in docs/inference-contract.md.
// Strict must be true. Schema validation happens after JSON-constrained decoding.
type JSONSchemaFormat struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Strict      bool            `json:"strict"`
	Schema      json.RawMessage `json:"schema"`
}

func (o OpenAIChatRequest) validateContract() (*jsonconstraint.Schema, error) {
	if o.ParallelToolCalls != nil && !*o.ParallelToolCalls {
		return nil, errors.New("parallel_tool_calls:false is not supported")
	}
	if len(o.Messages) == 0 {
		return nil, errors.New("messages must not be empty")
	}
	if o.N != nil && *o.N != 1 {
		return nil, errors.New("only n=1 is supported")
	}
	if o.MaxTokens != nil && o.MaxCompletionTokens != nil {
		return nil, errors.New("use only one max token field")
	}
	if err := validateStop(o.Stop); err != nil {
		return nil, err
	}
	if o.StreamOptions != nil && !o.Stream {
		return nil, errors.New("stream_options requires stream:true")
	}
	if err := validateMessages(o.Messages); err != nil {
		return nil, err
	}
	if len(o.Tools) > 0 && (o.Wikimedia || o.OpenStreetMap || o.RAG || o.Skills != nil && *o.Skills) {
		return nil, errors.New("client tools cannot be combined with server orchestration")
	}
	definitions := map[string]bool{}
	for _, t := range o.Tools {
		if t.Type != "function" || t.Function.Name == "" || definitions[t.Function.Name] {
			return nil, errors.New("tools require unique named functions")
		}
		definitions[t.Function.Name] = true
		if len(t.Function.Parameters) > 0 {
			var obj map[string]any
			if json.Unmarshal(t.Function.Parameters, &obj) != nil || obj == nil {
				return nil, errors.New("tool parameters must be a JSON object")
			}
		}
	}
	choice := normalizeToolChoice(o.ToolChoice)
	if o.ToolChoice != nil && choice != "none" && choice != "auto" && choice != "required" && !strings.HasPrefix(choice, "function:") {
		return nil, errors.New("unsupported tool_choice")
	}
	if (choice == "required" || strings.HasPrefix(choice, "function:")) && len(o.Tools) == 0 {
		return nil, errors.New("tool_choice requires tools")
	}
	if name, ok := strings.CutPrefix(choice, "function:"); ok && !definitions[name] {
		return nil, errors.New("tool_choice names an undeclared function")
	}
	if o.ResponseFormat == nil {
		return nil, nil
	}
	f := o.ResponseFormat
	if f.Type == "text" && f.JSONSchema == nil {
		return nil, nil
	}
	if f.Type != "json_object" && f.Type != "json_schema" {
		return nil, errors.New("unsupported response_format")
	}
	if len(o.Tools) > 0 || o.Wikimedia || o.OpenStreetMap || o.RAG || o.Skills != nil && *o.Skills || o.Stop != nil {
		return nil, errors.New("structured output cannot be combined with tools, skills, retrieval or stop")
	}
	if f.Type == "json_object" {
		if f.JSONSchema != nil {
			return nil, errors.New("json_schema is not valid with json_object")
		}
		return nil, nil
	}
	if f.JSONSchema == nil || !f.JSONSchema.Strict || f.JSONSchema.Name == "" {
		return nil, errors.New("json_schema requires name, strict:true and schema")
	}
	return jsonconstraint.CompileSchema(f.JSONSchema.Schema)
}
func validateMessages(messages []APIMessage) error {
	pending := map[string]bool{}
	names := map[string]string{}
	seen := map[string]bool{}
	for _, m := range messages {
		switch m.Role {
		case "system", "developer", "user", "assistant", "tool":
		default:
			return fmt.Errorf("unsupported message role %q", m.Role)
		}
		if m.Role == "tool" {
			if !pending[m.ToolCallID] {
				return errors.New("tool result has no matching pending call")
			}
			if m.Name != "" && m.Name != names[m.ToolCallID] {
				return errors.New("tool result name does not match call ID")
			}
			delete(pending, m.ToolCallID)
		} else if len(pending) > 0 {
			return errors.New("all tool calls require results before the next message")
		}
		if m.Role != "assistant" && m.Content == nil {
			return errors.New("message content is required")
		}
		if m.Content != nil {
			switch c := m.Content.(type) {
			case string:
			case []any:
				for _, raw := range c {
					p, ok := raw.(map[string]any)
					if !ok {
						return errors.New("invalid content part")
					}
					switch p["type"] {
					case "text":
						if _, ok := p["text"].(string); !ok {
							return errors.New("invalid text part")
						}
					case "image_url":
						if len(contentImages([]any{raw})) != 1 {
							return errors.New("image_url requires a valid base64 data URL")
						}
					default:
						return errors.New("unsupported content part")
					}
				}
			default:
				return errors.New("content must be string, parts or null")
			}
		}
		if len(m.ToolCalls) > 0 && m.Role != "assistant" {
			return errors.New("tool_calls require assistant role")
		}
		for _, call := range m.ToolCalls {
			var args map[string]any
			if call.ID == "" || seen[call.ID] || call.Type != "function" || call.Function.Name == "" || json.Unmarshal([]byte(call.Function.Arguments), &args) != nil || args == nil {
				return errors.New("invalid or duplicate tool call; arguments must be a JSON object string")
			}
			seen[call.ID] = true
			names[call.ID] = call.Function.Name
			pending[call.ID] = true
		}
	}
	if len(pending) > 0 {
		return errors.New("missing tool results")
	}
	return nil
}
func validateChatResult(result gopherllm.GenerationResult, options gopherllm.GenerationOptions) error {
	if len(result.ToolCalls) > 0 {
		if result.FinishReason == "length" {
			return errors.New("truncated tool call")
		}
		seen := map[string]bool{}
		for _, c := range result.ToolCalls {
			var args map[string]any
			declared := false
			for _, t := range options.ActiveTools() {
				if t.Function.Name == c.Function.Name {
					declared = true
				}
			}
			if c.ID == "" || seen[c.ID] || c.Type != "function" || !declared || json.Unmarshal([]byte(c.Function.Arguments), &args) != nil || args == nil {
				return errors.New("model emitted invalid tool call")
			}
			seen[c.ID] = true
		}
		return nil
	}
	if options.ToolChoice == "required" || strings.HasPrefix(options.ToolChoice, "function:") {
		return errors.New("model did not produce the required tool call")
	}
	if strings.TrimSpace(result.Text) == "" {
		return errors.New("model produced no answer or tool calls")
	}
	return nil
}

var errResponseWritten = errors.New("response already written")

func validatedEmbeddingInputs(e EmbeddingsRequest) ([]string, error) {
	if e.EncodingFormat != "" && e.EncodingFormat != "float" && e.EncodingFormat != "base64" {
		return nil, errors.New("encoding_format must be float or base64")
	}
	var inputs []string
	switch v := e.Input.(type) {
	case string:
		inputs = []string{v}
	case []any:
		for _, x := range v {
			text, ok := x.(string)
			if !ok {
				return nil, errors.New("input batch must contain only strings")
			}
			inputs = append(inputs, text)
		}
	default:
		return nil, errors.New("input must be string or array of strings")
	}
	if len(inputs) == 0 || len(inputs) > 128 {
		return nil, errors.New("embedding batch must contain 1..128 inputs")
	}
	for _, text := range inputs {
		if strings.TrimSpace(text) == "" || len(text) > 1<<20 {
			return nil, errors.New("each embedding input must contain 1..1048576 bytes")
		}
	}
	return inputs, nil
}

func validateStop(v any) error {
	if v == nil {
		return nil
	}
	values := []string{}
	switch x := v.(type) {
	case string:
		values = []string{x}
	case []any:
		for _, entry := range x {
			text, ok := entry.(string)
			if !ok {
				return errors.New("stop entries must be strings")
			}
			values = append(values, text)
		}
	default:
		return errors.New("stop must be string or string array")
	}
	if len(values) == 0 || len(values) > 4 {
		return errors.New("stop must contain 1..4 strings")
	}
	for _, text := range values {
		if text == "" {
			return errors.New("stop must not be empty")
		}
	}
	return nil
}
