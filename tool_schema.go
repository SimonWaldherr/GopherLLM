package gopherllm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

// NewTool builds an AgenticTool from a Go function, deriving the tool's JSON
// Schema from Args by reflection and unmarshalling the model's arguments back
// into it.
//
//	type args struct {
//	    City string `json:"city" desc:"City name, e.g. Hamburg"`
//	    Unit string `json:"unit,omitempty" desc:"Temperature unit" enum:"c,f"`
//	}
//	tool := gopherllm.NewTool("get_weather", "Current weather for a city.",
//	    func(ctx context.Context, a args) (string, error) { ... })
//
// Schema rules, all derivable from tags a Go developer already writes:
//   - the property name is the json tag, or the lowercased field name;
//   - desc:"…" becomes the property description — write it for a small
//     model, since this text is most of what decides whether the tool gets
//     called correctly, and it is paid for on every request;
//   - enum:"a,b,c" becomes a JSON Schema enum, which small models honor far
//     more reliably than the same constraint stated in prose;
//   - a field is required unless it is a pointer or its json tag carries
//     omitempty — chosen so the zero value of Args is always a valid,
//     all-optional tool with no extra ceremony.
//
// Supported field kinds: string, the integer and float kinds, bool, slices of
// those, and one level of nested struct. Use struct{} for a tool that takes
// no arguments.
//
// NewTool panics on a type it cannot describe (a channel, a func, an
// interface, a map with non-string keys). This is deliberate and is the one
// panic in this package's public API: the failure depends only on a
// compile-time type, so it cannot be triggered by a model, a request, or a
// file on disk — it fires on the first line of your program or never.
// Returning an error here would cost every caller a check for a condition
// their own first test run already catches, and would stop tools composing
// inside a slice literal.
//
// A decode failure or a missing required field is returned to the MODEL as
// the tool's result (as ordinary content, with a nil error), not to the
// caller as an error, so the model can repair its own call on the next round
// instead of the turn simply failing.
//
// The generated schema bytes are fixed at construction time. Keep the
// returned AgenticTool value and reuse it across requests: the rendered tools
// block must be byte-identical across the loop's rounds for the Runner's KV
// prefix cache to hit on it, which on local CPU decode is real prefill time.
func NewTool[Args any](name, description string, run func(context.Context, Args) (string, error)) AgenticTool {
	schema := SchemaOf[Args]() // panics on an unsupported Args type
	required := requiredFieldsOf[Args]()
	execute := func(ctx context.Context, call ToolCall) (string, error) {
		var args Args
		if strings.TrimSpace(call.Function.Arguments) != "" {
			if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
				return fmt.Sprintf("Error: invalid arguments for %s: %v. Expected an object matching this schema: %s", name, err, schema), nil
			}
		}
		if missing := firstMissingRequired(args, required); missing != "" {
			return fmt.Sprintf("Error: %s: missing required argument %q.", name, missing), nil
		}
		return run(ctx, args)
	}
	return AgenticTool{
		Definition: ToolDefinition{Type: "function", Function: ToolFunctionDef{
			Name: name, Description: description, Parameters: json.RawMessage(schema),
		}},
		Execute: execute,
	}
}

// SchemaOf derives a JSON Schema object (as raw bytes) from the exported
// fields of Args, using the same field-tag rules as NewTool. It is exported
// so a caller who needs a construct this subset does not cover (oneOf, a
// regex pattern, a tuned enum) can start from the derived bytes, edit them,
// and build an AgenticTool by hand — the escape hatch that keeps this subset
// allowed to stay small. Panics on an unsupported field kind; see NewTool.
func SchemaOf[Args any]() json.RawMessage {
	t := reflect.TypeFor[Args]()
	props := map[string]any{}
	var required []string
	if t.Kind() == reflect.Struct {
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !f.IsExported() {
				continue
			}
			name, omitempty := jsonPropName(f)
			if name == "-" {
				continue
			}
			props[name] = schemaForField(f)
			if !omitempty && f.Type.Kind() != reflect.Pointer {
				required = append(required, name)
			}
		}
	}
	out := map[string]any{"type": "object", "properties": props}
	if len(required) > 0 {
		out["required"] = required
	}
	b, err := json.Marshal(out)
	if err != nil {
		// json.Marshal only fails here on a channel/func/complex value, all of
		// which schemaForField already rejects by panicking earlier — this is
		// an assertion that that coverage is complete, not a reachable path.
		panic(fmt.Sprintf("gopherllm: SchemaOf[%s]: %v", t, err))
	}
	return b
}

// requiredFieldsOf mirrors SchemaOf's required-field selection, kept as a
// separate pass over Args (rather than parsed back out of the JSON schema
// bytes) so NewTool's post-decode validation cannot drift from what the
// schema actually declared required.
func requiredFieldsOf[Args any]() map[string]bool {
	t := reflect.TypeFor[Args]()
	required := map[string]bool{}
	if t.Kind() != reflect.Struct {
		return required
	}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		name, omitempty := jsonPropName(f)
		if name == "-" || omitempty || f.Type.Kind() == reflect.Pointer {
			continue
		}
		required[name] = true
	}
	return required
}

// firstMissingRequired reports the first required property (in struct
// declaration order, via requiredFieldsOf's iteration) whose decoded value is
// still its Go zero value — the observable signature of a required field the
// model's JSON simply omitted. This cannot distinguish an omitted field from
// one explicitly sent as the zero value (empty string, 0, false); that
// ambiguity is inherent to matching a dynamic JSON payload back against a
// static Go struct without also carrying a presence map, and is judged
// cheaper than the alternative of decoding into a map first.
func firstMissingRequired(args any, required map[string]bool) string {
	if len(required) == 0 {
		return ""
	}
	v := reflect.ValueOf(args)
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		name, _ := jsonPropName(f)
		if !required[name] {
			continue
		}
		if v.Field(i).IsZero() {
			return name
		}
	}
	return ""
}

// jsonPropName reads a field's `json` tag the same way encoding/json does:
// the name before a comma, "-" to skip the field entirely, and the
// "omitempty" option. A field with no json tag falls back to its lowercased
// Go name, matching what a reader of the generated schema would expect from
// the struct.
func jsonPropName(f reflect.StructField) (name string, omitempty bool) {
	tag := f.Tag.Get("json")
	if tag == "" {
		return strings.ToLower(f.Name), false
	}
	parts := strings.Split(tag, ",")
	name = parts[0]
	if name == "" {
		name = strings.ToLower(f.Name)
	}
	for _, opt := range parts[1:] {
		if opt == "omitempty" {
			omitempty = true
		}
	}
	return name, omitempty
}

// schemaForField derives one JSON Schema property object from a struct field,
// honoring `desc` and `enum` tags. It panics on a field kind this subset does
// not support — see NewTool's doc comment for the panic's justification.
func schemaForField(f reflect.StructField) map[string]any {
	prop := jsonSchemaType(f.Type, f)
	if desc := f.Tag.Get("desc"); desc != "" {
		prop["description"] = desc
	}
	if enum := f.Tag.Get("enum"); enum != "" {
		values := make([]string, 0)
		for _, v := range strings.Split(enum, ",") {
			if v = strings.TrimSpace(v); v != "" {
				values = append(values, v)
			}
		}
		if len(values) > 0 {
			prop["enum"] = values
		}
	}
	return prop
}

// jsonSchemaType maps a Go type to its JSON Schema {"type": ...} object, one
// level of nested struct and one level of slice deep. field carries the
// original struct field only for the panic message; nested/element calls pass
// a zero reflect.StructField and rely on their own recursion to name the type.
func jsonSchemaType(t reflect.Type, field reflect.StructField) map[string]any {
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": jsonSchemaType(t.Elem(), reflect.StructField{})}
	case reflect.Struct:
		props := map[string]any{}
		var required []string
		for i := 0; i < t.NumField(); i++ {
			nf := t.Field(i)
			if !nf.IsExported() {
				continue
			}
			name, omitempty := jsonPropName(nf)
			if name == "-" {
				continue
			}
			props[name] = schemaForField(nf)
			if !omitempty && nf.Type.Kind() != reflect.Pointer {
				required = append(required, name)
			}
		}
		out := map[string]any{"type": "object", "properties": props}
		if len(required) > 0 {
			out["required"] = required
		}
		return out
	default:
		where := t.String()
		if field.Name != "" {
			where = field.Name + " (" + t.String() + ")"
		}
		panic("gopherllm: NewTool: unsupported field " + strconv.Quote(where) + " of kind " + t.Kind().String())
	}
}
