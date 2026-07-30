package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func TestNewResearchToolsAreExplicitlyOptIn(t *testing.T) {
	if tools := NewResearchTools(ResearchOptions{}); tools != nil {
		t.Fatalf("default research tools = %#v, want nil", tools)
	}
	tools := NewResearchTools(ResearchOptions{Wikimedia: true, OpenStreetMap: true})
	if len(tools) != 6 { // five Wikimedia tools plus one OSM tool
		t.Fatalf("tool count = %d, want 6", len(tools))
	}
	var found bool
	for _, tool := range tools {
		if tool.Definition.Function.Name == "openstreetmap_search" {
			found = true
		}
	}
	if !found {
		t.Fatal("openstreetmap_search was not exposed to Go callers")
	}
}

func TestOpenStreetMapSearchIsBoundedAttributedAndIdentified(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("q") != "Brandenburg Gate" || r.URL.Query().Get("countrycodes") != "de" || r.URL.Query().Get("limit") != "5" {
			t.Fatalf("request = %s", r.URL.String())
		}
		if !strings.Contains(r.Header.Get("User-Agent"), "GopherLLM") {
			t.Fatalf("user-agent = %q", r.Header.Get("User-Agent"))
		}
		_, _ = io.WriteString(w, `[{"display_name":"Brandenburg Gate, Berlin","lat":"52.516","lon":"13.377","category":"historic","type":"memorial","osm_type":"way","osm_id":123}]`)
	}))
	defer server.Close()
	client := newOSMClient(server.Client(), server.URL)
	got, err := client.search(context.Background(), gopherllm.ToolCall{Function: gopherllm.ToolCallFunction{Arguments: `{"query":"Brandenburg Gate","country_code":"DE"}`}})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"OpenStreetMap Nominatim", "OpenStreetMap contributors", "https://www.openstreetmap.org/way/123", "nominatim"} {
		if !strings.Contains(got, want) {
			t.Fatalf("result missing %q: %s", want, got)
		}
	}
	if _, err := client.search(context.Background(), gopherllm.ToolCall{Function: gopherllm.ToolCallFunction{Arguments: `{"query":"Berlin","country_code":"Germany"}`}}); err == nil {
		t.Fatal("invalid country code was accepted")
	}
}

func TestPrivacyEndpointShowsOptInResearchPolicy(t *testing.T) {
	handler := NewHandler(nil, HandlerOptions{})
	request := httptest.NewRequest(http.MethodGet, "/privacy", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", response.Code, response.Body.String())
	}
	for _, want := range []string{`"telemetry":false`, "gopherllm_openstreetmap", "OpenStreetMap place search", "Nominatim"} {
		if !strings.Contains(response.Body.String(), want) {
			t.Fatalf("privacy response missing %q: %s", want, response.Body.String())
		}
	}
}
