package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func TestWikipediaSearchUsesRESTSearch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/w/rest.php/v1/search/page" || r.URL.Query().Get("q") != "Berlin" {
			t.Fatalf("request = %s", r.URL.String())
		}
		if !strings.Contains(r.Header.Get("User-Agent"), "GopherLLM") {
			t.Fatalf("user-agent = %q", r.Header.Get("User-Agent"))
		}
		_, _ = w.Write([]byte(`{"pages":[{"title":"Berlin","description":"capital","excerpt":"<span>Berlin</span>"}]}`))
	}))
	defer server.Close()
	client := newWikimediaClient(server.Client())
	client.wikipediaBase = server.URL
	got, err := client.search(context.Background(), gopherllm.ToolCall{Function: gopherllm.ToolCallFunction{Arguments: `{"query":"Berlin"}`}})
	if err != nil || !strings.Contains(got, `"title":"Berlin"`) || !strings.Contains(got, `"source":"Wikipedia search"`) {
		t.Fatalf("result = %q, err = %v", got, err)
	}
}

func TestWikidataSPARQLRejectsUnsafeAndBoundsRows(t *testing.T) {
	client := newWikimediaClient(nil)
	if _, err := client.sparql(context.Background(), gopherllm.ToolCall{Function: gopherllm.ToolCallFunction{Arguments: `{"query":"SELECT * WHERE { SERVICE <https://example.test/> {} }"}`}}); err == nil {
		t.Fatal("SERVICE query was accepted")
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("query") != "SELECT ?item WHERE {}" {
			t.Fatalf("query = %q", r.URL.Query().Get("query"))
		}
		_, _ = w.Write([]byte(`{"results":{"bindings":[{"item":{"type":"uri","value":"https://www.wikidata.org/entity/Q64"}}]}}`))
	}))
	defer server.Close()
	client = newWikimediaClient(server.Client())
	client.sparqlBase = server.URL
	got, err := client.sparql(context.Background(), gopherllm.ToolCall{Function: gopherllm.ToolCallFunction{Arguments: `{"query":"SELECT ?item WHERE {}"}`}})
	if err != nil || !strings.Contains(got, "Q64") {
		t.Fatalf("result = %q, err = %v", got, err)
	}
}
