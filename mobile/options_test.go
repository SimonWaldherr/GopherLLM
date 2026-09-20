package mobile

import "testing"

func TestLoadOptions(t *testing.T) {
	o, err := parseLoadOptions("")
	if err != nil || o.Prefault != "none" || o.Threads != 0 {
		t.Fatalf("defaults = %+v, %v", o, err)
	}
	if _, err := parseLoadOptions(`{"threads":2,"prefault":"core"}`); err != nil {
		t.Fatal(err)
	}
	if o, err := parseLoadOptions(`{"metal":true,"prefault":"none"}`); err != nil || !o.Metal {
		t.Fatalf("metal option = %+v, %v", o, err)
	}
	for _, raw := range []string{`{`, `{"unknown":true}`, `{"threads":-1}`, `{"prefault":"everything"}`} {
		if _, err := parseLoadOptions(raw); err == nil {
			t.Fatalf("%s: expected error", raw)
		}
	}
}

func TestGenerationOptions(t *testing.T) {
	o, err := parseGenerationOptions("")
	if err != nil || o.MaxTokens != 256 {
		t.Fatalf("defaults = %+v, %v", o, err)
	}
	if _, err := parseGenerationOptions(`{"max_tokens":12,"temperature":0,"top_p":1,"top_k":0,"min_p":0,"repeat_penalty":1}`); err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{`{"unknown":1}`, `{"max_tokens":0}`, `{"temperature":-1}`, `{"top_p":2}`, `{"repeat_penalty":0}`} {
		if _, err := parseGenerationOptions(raw); err == nil {
			t.Fatalf("%s: expected error", raw)
		}
	}
}
