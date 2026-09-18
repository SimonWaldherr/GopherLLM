package jsonconstraint

import (
	"encoding/json"
	"testing"
)

func TestJSONGrammarAcrossByteBoundaries(t *testing.T) {
	for _, text := range []string{`{}`, ` {"text":"Grüße 🌍","a":[true,false,null,-12.5e+2,{},[]],"escape":"\u1234\\\""} `} {
		for split := 0; split <= len(text); split++ {
			s, ok := (State{}).Advance(text[:split])
			if !ok {
				t.Fatalf("prefix %q", text[:split])
			}
			s, ok = s.Advance(text[split:])
			if !ok || !s.Complete() || !json.Valid([]byte(text)) {
				t.Fatalf("%q split %d", text, split)
			}
		}
	}
	for _, text := range []string{`[]`, `null`, `{"a":01}`, `{"a":1.}`, `{"a":true,}`, `{"a":[1,]}`, `{"a":"\x"}`, "{\"a\":\"\xff\"}", `{}x`, `{"a":+1}`} {
		s, ok := (State{}).Advance(text)
		if ok && s.Complete() {
			t.Fatalf("accepted %q", text)
		}
	}
}
func FuzzJSONGrammar(f *testing.F) {
	for _, s := range []string{`{}`, `{"x":[1,2]}`, `{"a":"x"}`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, text string) {
		s, ok := (State{}).Advance(text)
		if ok && s.Complete() && !json.Valid([]byte(text)) {
			t.Fatalf("invalid accepted %q", text)
		}
	})
}
func TestSchemaSubset(t *testing.T) {
	s, err := CompileSchema([]byte(`{"type":"object","properties":{"x":{"type":"integer"},"label":{"type":"string","enum":["a","b"]}},"required":["x"],"additionalProperties":false}`))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Validate([]byte(`{"x":3,"label":"a"}`)); err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{`{}`, `{"x":1.2}`, `{"x":1,"other":true}`, `{"x":1,"label":"c"}`} {
		if s.Validate([]byte(v)) == nil {
			t.Fatal(v)
		}
	}
	for _, v := range []string{`{"type":"object","$ref":"x"}`, `{"type":"object"}`, `{"type":"object","additionalProperties":false,"properties":{"x":{"type":"string","pattern":"x"}}}`} {
		if _, err := CompileSchema([]byte(v)); err == nil {
			t.Fatal(v)
		}
	}
}
func BenchmarkGrammarCandidate(b *testing.B) {
	s, _ := (State{}).Advance(`{"value":"`)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = s.Advance("example text")
	}
}

func TestSchemaExactNumbers(t *testing.T) {
	s, err := CompileSchema([]byte(`{"type":"object","additionalProperties":false,"properties":{"n":{"type":"integer","enum":[9007199254740993]}}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{`{"n":9007199254740993}`, `{"n":9007199254740993.0}`, `{"n":90071992547409930e-1}`} {
		if err := s.Validate([]byte(v)); err != nil {
			t.Fatalf("%s: %v", v, err)
		}
	}
	for _, v := range []string{`{"n":9007199254740992}`, `{"n":9007199254740993.1}`, `{"n":9e9999999}`, `{} {}`} {
		if s.Validate([]byte(v)) == nil {
			t.Fatalf("accepted %s", v)
		}
	}
}
