package jsonconstraint

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Schema supports type, properties, required, additionalProperties:false,
// items, enum, description and title. No references or composition keywords.
// Unsupported keywords are rejected, never silently ignored.
type Schema struct {
	Type                 string                     `json:"type"`
	Properties           map[string]json.RawMessage `json:"properties,omitempty"`
	Required             []string                   `json:"required,omitempty"`
	AdditionalProperties *bool                      `json:"additionalProperties,omitempty"`
	Items                json.RawMessage            `json:"items,omitempty"`
	Enum                 []any                      `json:"enum,omitempty"`
	Description          string                     `json:"description,omitempty"`
	Title                string                     `json:"title,omitempty"`
	children             map[string]*Schema
	item                 *Schema
}

func CompileSchema(raw []byte) (*Schema, error) {
	if len(raw) > 65536 {
		return nil, fmt.Errorf("schema exceeds 64 KiB")
	}
	nodes := 0
	s, err := compileSchema(raw, 0, &nodes)
	if err == nil && s.Type != "object" {
		return nil, fmt.Errorf("schema root must be an object")
	}
	return s, err
}
func compileSchema(raw []byte, depth int, nodes *int) (*Schema, error) {
	*nodes++
	if depth > 16 || *nodes > 256 {
		return nil, fmt.Errorf("schema complexity limit exceeded")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	d.UseNumber()
	var s Schema
	if err := d.Decode(&s); err != nil {
		return nil, err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return nil, fmt.Errorf("trailing schema data")
	}
	switch s.Type {
	case "object", "array", "string", "number", "integer", "boolean", "null":
	default:
		return nil, fmt.Errorf("unsupported schema type %q", s.Type)
	}
	if s.Type != "object" && (s.Properties != nil || s.Required != nil || s.AdditionalProperties != nil) {
		return nil, fmt.Errorf("object keywords require object type")
	}
	if s.Type != "array" && len(s.Items) > 0 {
		return nil, fmt.Errorf("items requires array type")
	}
	if s.Type == "object" {
		if s.AdditionalProperties == nil || *s.AdditionalProperties {
			return nil, fmt.Errorf("objects require additionalProperties:false")
		}
		s.children = map[string]*Schema{}
		for k, v := range s.Properties {
			child, err := compileSchema(v, depth+1, nodes)
			if err != nil {
				return nil, fmt.Errorf("property %s: %w", k, err)
			}
			s.children[k] = child
		}
		seen := map[string]bool{}
		for _, k := range s.Required {
			if s.children[k] == nil || seen[k] {
				return nil, fmt.Errorf("invalid required property %q", k)
			}
			seen[k] = true
		}
	}
	if s.Type == "array" {
		var err error
		s.item, err = compileSchema(s.Items, depth+1, nodes)
		if err != nil {
			return nil, err
		}
	}
	if s.Enum != nil {
		if len(s.Enum) == 0 || len(s.Enum) > 256 {
			return nil, fmt.Errorf("enum must have 1..256 entries")
		}
		values := s.Enum
		s.Enum = nil
		for _, v := range values {
			if err := s.validate(v); err != nil {
				return nil, fmt.Errorf("invalid enum: %w", err)
			}
		}
		s.Enum = values
	}
	return &s, nil
}
func (s *Schema) Validate(raw []byte) error {
	var v any
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := d.Decode(&v); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("trailing output data")
	}
	return s.validate(v)
}
func (s *Schema) validate(v any) error {
	if s.Enum != nil {
		ok := false
		for _, entry := range s.Enum {
			if equalJSON(v, entry) {
				ok = true
				break
			}
		}
		if !ok {
			return fmt.Errorf("value not in enum")
		}
	}
	switch s.Type {
	case "object":
		obj, ok := v.(map[string]any)
		if !ok {
			return fmt.Errorf("expected object")
		}
		for _, k := range s.Required {
			if _, ok := obj[k]; !ok {
				return fmt.Errorf("missing required property %q", k)
			}
		}
		for k, v := range obj {
			child := s.children[k]
			if child == nil {
				return fmt.Errorf("unexpected property %q", k)
			}
			if err := child.validate(v); err != nil {
				return fmt.Errorf("property %s: %w", k, err)
			}
		}
	case "array":
		a, ok := v.([]any)
		if !ok {
			return fmt.Errorf("expected array")
		}
		for _, x := range a {
			if err := s.item.validate(x); err != nil {
				return err
			}
		}
	case "string":
		if _, ok := v.(string); !ok {
			return fmt.Errorf("expected string")
		}
	case "number", "integer":
		n, ok := v.(json.Number)
		_, exponent, valid := canonicalNumber(n)
		if !ok || !valid || s.Type == "integer" && exponent < 0 {
			return fmt.Errorf("expected %s", s.Type)
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return fmt.Errorf("expected boolean")
		}
	case "null":
		if v != nil {
			return fmt.Errorf("expected null")
		}
	}
	return nil
}

// canonicalNumber compares decimal JSON numbers without floating point rounding.
// Bounds apply to the encoded number and exponent, not the represented magnitude.
func canonicalNumber(n json.Number) (string, int, bool) {
	text := string(n)
	if len(text) == 0 || len(text) > 1024 {
		return "", 0, false
	}
	exponent := 0
	if i := strings.IndexAny(text, "eE"); i >= 0 {
		var err error
		exponent, err = strconv.Atoi(text[i+1:])
		if err != nil || exponent < -1000000 || exponent > 1000000 {
			return "", 0, false
		}
		text = text[:i]
	}
	negative := strings.HasPrefix(text, "-")
	text = strings.TrimPrefix(text, "-")
	if i := strings.IndexByte(text, '.'); i >= 0 {
		exponent -= len(text) - i - 1
		text = text[:i] + text[i+1:]
	}
	text = strings.TrimLeft(text, "0")
	if text == "" {
		return "0", 0, true
	}
	trimmed := strings.TrimRight(text, "0")
	exponent += len(text) - len(trimmed)
	if negative {
		trimmed = "-" + trimmed
	}
	return trimmed, exponent, true
}
func equalJSON(a, b any) bool {
	switch x := a.(type) {
	case json.Number:
		y, ok := b.(json.Number)
		if !ok {
			return false
		}
		xd, xe, xv := canonicalNumber(x)
		yd, ye, yv := canonicalNumber(y)
		return xv && yv && xd == yd && xe == ye
	case map[string]any:
		y, ok := b.(map[string]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for k, v := range x {
			w, exists := y[k]
			if !exists || !equalJSON(v, w) {
				return false
			}
		}
		return true
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !equalJSON(x[i], y[i]) {
				return false
			}
		}
		return true
	case string:
		y, ok := b.(string)
		return ok && x == y
	case bool:
		y, ok := b.(bool)
		return ok && x == y
	case nil:
		return b == nil
	}
	return false
}
