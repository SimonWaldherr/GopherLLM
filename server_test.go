package gopherllm

import (
	"io"
	"net/http"
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
		Title:         "GopherLLM Chat",
		Model:         "test-model",
		MaxTokens:     123,
		Temperature:   0.25,
		TopP:          0.85,
		TopK:          10,
		RepeatPenalty: 1.2,
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
	if !strings.Contains(html, `value="0.85"`) {
		t.Fatalf("top-p default missing from template")
	}
	if !strings.Contains(html, `value="10"`) {
		t.Fatalf("top-k default missing from template")
	}
	if !strings.Contains(html, `value="1.2"`) {
		t.Fatalf("repeat penalty default missing from template")
	}
}

func TestChatUIAssetsArePrivateAndSelfContained(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{ChatUI: true})
	for _, want := range []struct {
		path        string
		contentType string
		marker      string
	}{
		{path: "/chat", contentType: "text/html", marker: `id="chatList"`},
		{path: "/style.css", contentType: "text/css", marker: ".sidebar"},
		{path: "/script.js", contentType: "text/javascript", marker: "IndexedDB"},
	} {
		resp, err := http.Get(srv.URL + want.path)
		if err != nil {
			t.Fatalf("GET %s: %v", want.path, err)
		}
		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			t.Fatalf("read %s: %v", want.path, readErr)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s status = %d", want.path, resp.StatusCode)
		}
		if !strings.HasPrefix(resp.Header.Get("Content-Type"), want.contentType) {
			t.Fatalf("GET %s content-type = %q", want.path, resp.Header.Get("Content-Type"))
		}
		if resp.Header.Get("Cache-Control") != "no-store" {
			t.Fatalf("GET %s cache-control = %q", want.path, resp.Header.Get("Cache-Control"))
		}
		if !strings.Contains(resp.Header.Get("Content-Security-Policy"), "default-src 'self'") {
			t.Fatalf("GET %s missing CSP: %q", want.path, resp.Header.Get("Content-Security-Policy"))
		}
		if !strings.Contains(string(body), want.marker) {
			t.Fatalf("GET %s missing %q", want.path, want.marker)
		}
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
