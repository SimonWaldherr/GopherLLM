package gopherllm

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestNewToolDerivesRequiredFromOmitempty(t *testing.T) {
	type args struct {
		City string `json:"city" desc:"City name"`
		Unit string `json:"unit,omitempty" desc:"Temperature unit" enum:"c,f"`
	}
	schema := SchemaOf[args]()
	var decoded struct {
		Type       string                    `json:"type"`
		Required   []string                  `json:"required"`
		Properties map[string]map[string]any `json:"properties"`
	}
	if err := json.Unmarshal(schema, &decoded); err != nil {
		t.Fatalf("schema is not valid JSON: %v\n%s", err, schema)
	}
	if decoded.Type != "object" {
		t.Fatalf("type = %q, want object", decoded.Type)
	}
	if len(decoded.Required) != 1 || decoded.Required[0] != "city" {
		t.Fatalf("required = %v, want exactly [\"city\"]", decoded.Required)
	}
	if decoded.Properties["city"]["type"] != "string" || decoded.Properties["city"]["description"] != "City name" {
		t.Fatalf("city property = %+v", decoded.Properties["city"])
	}
	enum, _ := decoded.Properties["unit"]["enum"].([]any)
	if len(enum) != 2 || enum[0] != "c" || enum[1] != "f" {
		t.Fatalf("unit enum = %v, want [c f]", decoded.Properties["unit"]["enum"])
	}
}

func TestNewToolNoArgsProducesEmptyObjectSchema(t *testing.T) {
	schema := SchemaOf[struct{}]()
	var decoded struct {
		Type     string   `json:"type"`
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(schema, &decoded); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
	if decoded.Type != "object" || len(decoded.Required) != 0 {
		t.Fatalf("decoded = %+v, want an object with no required fields", decoded)
	}
}

func TestNewToolPanicsOnUnsupportedFieldKind(t *testing.T) {
	type bad struct {
		C chan int `json:"c"`
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic for a channel-typed field")
		}
	}()
	SchemaOf[bad]()
}

func TestNewToolReturnsRepairMessageOnBadJSON(t *testing.T) {
	type args struct {
		City string `json:"city"`
	}
	tool := NewTool("get_weather", "Weather lookup", func(context.Context, args) (string, error) {
		return "should not run", nil
	})
	result, err := tool.Execute(context.Background(), ToolCall{Function: ToolCallFunction{Arguments: "{not json"}})
	if err != nil {
		t.Fatalf("a decode failure must be reported to the model, not returned as an error: %v", err)
	}
	if !strings.Contains(result, "Error:") || !strings.Contains(result, "invalid arguments") {
		t.Fatalf("result = %q, want a repair message", result)
	}
}

func TestNewToolReturnsRepairMessageOnMissingRequiredField(t *testing.T) {
	type args struct {
		City string `json:"city"`
	}
	tool := NewTool("get_weather", "Weather lookup", func(context.Context, args) (string, error) {
		return "should not run", nil
	})
	result, err := tool.Execute(context.Background(), ToolCall{Function: ToolCallFunction{Arguments: `{}`}})
	if err != nil {
		t.Fatalf("a missing required field must be reported to the model, not returned as an error: %v", err)
	}
	if !strings.Contains(result, "Error:") || !strings.Contains(result, `"city"`) {
		t.Fatalf("result = %q, want a repair message naming city", result)
	}
}

func TestNewToolCallsThroughWithDecodedArguments(t *testing.T) {
	type args struct {
		City string `json:"city"`
	}
	var got args
	tool := NewTool("get_weather", "Weather lookup", func(_ context.Context, a args) (string, error) {
		got = a
		return "42C in " + a.City, nil
	})
	result, err := tool.Execute(context.Background(), ToolCall{Function: ToolCallFunction{Arguments: `{"city":"Hamburg"}`}})
	if err != nil {
		t.Fatal(err)
	}
	if got.City != "Hamburg" || result != "42C in Hamburg" {
		t.Fatalf("got = %+v, result = %q", got, result)
	}
}

func TestNewToolSchemaBytesAreStableAcrossCalls(t *testing.T) {
	type args struct {
		City string `json:"city" desc:"City name"`
		Unit string `json:"unit,omitempty" enum:"c,f"`
	}
	a := SchemaOf[args]()
	b := SchemaOf[args]()
	if string(a) != string(b) {
		t.Fatalf("schema bytes differ across calls, which would defeat the Runner's KV prefix cache:\na=%s\nb=%s", a, b)
	}
}

func TestNewToolOptionalPointerFieldIsNotRequired(t *testing.T) {
	type args struct {
		City  string `json:"city"`
		Limit *int   `json:"limit"`
	}
	schema := SchemaOf[args]()
	var decoded struct {
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(schema, &decoded); err != nil {
		t.Fatal(err)
	}
	for _, name := range decoded.Required {
		if name == "limit" {
			t.Fatalf("a pointer field must not be required: %v", decoded.Required)
		}
	}
}

func TestNewToolNestedStructAndSlice(t *testing.T) {
	type inner struct {
		Zip string `json:"zip"`
	}
	type args struct {
		Tags    []string `json:"tags"`
		Address inner    `json:"address"`
	}
	schema := SchemaOf[args]()
	var decoded struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(schema, &decoded); err != nil {
		t.Fatal(err)
	}
	var tags struct {
		Type  string         `json:"type"`
		Items map[string]any `json:"items"`
	}
	if err := json.Unmarshal(decoded.Properties["tags"], &tags); err != nil {
		t.Fatal(err)
	}
	if tags.Type != "array" || tags.Items["type"] != "string" {
		t.Fatalf("tags schema = %+v", tags)
	}
	var address struct {
		Type       string                    `json:"type"`
		Properties map[string]map[string]any `json:"properties"`
	}
	if err := json.Unmarshal(decoded.Properties["address"], &address); err != nil {
		t.Fatal(err)
	}
	if address.Type != "object" || address.Properties["zip"]["type"] != "string" {
		t.Fatalf("address schema = %+v", address)
	}
}
