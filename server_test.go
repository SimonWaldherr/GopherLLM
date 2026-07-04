package main

import (
	"strings"
	"testing"
)

func TestDisplayServerURLNormalizesWildcardAddress(t *testing.T) {
	got := displayServerURL(":8081", true)
	if got != "http://localhost:8081/chat" {
		t.Fatalf("url = %q", got)
	}
}

func TestDisplayServerURLKeepsExplicitHost(t *testing.T) {
	got := displayServerURL("127.0.0.1:9090", false)
	if got != "http://127.0.0.1:9090" {
		t.Fatalf("url = %q", got)
	}
}

func TestChatTemplateUsesServerDefaults(t *testing.T) {
	var out strings.Builder
	err := chatTemplate.Execute(&out, chatTemplateData{
		Title:       "GopherLLM Chat",
		Model:       "test-model",
		MaxTokens:   123,
		Temperature: 0.25,
	})
	if err != nil {
		t.Fatal(err)
	}
	html := out.String()
	if !strings.Contains(html, `value="123"`) {
		t.Fatalf("max token default missing from template")
	}
	if !strings.Contains(html, `value="0.25"`) {
		t.Fatalf("temperature default missing from template")
	}
}

func TestApplyRequestOptionsPreservesDefaultStopsWhenStopOmitted(t *testing.T) {
	def := DefaultGenerationOptions()
	def.StopSequences = []string{"</s>"}

	got := applyRequestOptions(def, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, "")
	if len(got.StopSequences) != 1 || got.StopSequences[0] != "</s>" {
		t.Fatalf("stop sequences = %#v", got.StopSequences)
	}
}

func TestApplyRequestOptionsOverridesStopsWhenProvided(t *testing.T) {
	def := DefaultGenerationOptions()
	def.StopSequences = []string{"</s>"}

	got := applyRequestOptions(def, nil, nil, nil, nil, nil, nil, nil, nil, []any{"END", "STOP"}, nil, "")
	if len(got.StopSequences) != 2 || got.StopSequences[0] != "END" || got.StopSequences[1] != "STOP" {
		t.Fatalf("stop sequences = %#v", got.StopSequences)
	}
}
