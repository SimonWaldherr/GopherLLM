package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The whole point of choosing a CDN in Settings is that the page may load a
// script from exactly one extra origin. If an unrecognised value widened the
// policy — or if the default did — the choice would be security theatre.
func TestChatCSPOnlyOpensForARecognisedCDN(t *testing.T) {
	for _, tc := range []struct {
		query      string
		wantOrigin string
	}{
		{query: "", wantOrigin: ""},
		{query: "?mermaid=jsdelivr", wantOrigin: "https://cdn.jsdelivr.net"},
		{query: "?mermaid=unpkg", wantOrigin: "https://unpkg.com"},
		{query: "?mermaid=cdnjs", wantOrigin: "https://cdnjs.cloudflare.com"},
		// Anything unrecognised must fall back to same-origin only.
		{query: "?mermaid=evil.example.com", wantOrigin: ""},
		{query: "?mermaid=https://evil.example.com", wantOrigin: ""},
		{query: "?mermaid=jsdelivr%20https://evil.example.com", wantOrigin: ""},
		{query: "?mermaid=", wantOrigin: ""},
	} {
		srv := newTestServer(t, HandlerOptions{ChatUI: true})
		resp, err := http.Get(srv.URL + "/chat" + tc.query)
		if err != nil {
			t.Fatal(err)
		}
		csp := resp.Header.Get("Content-Security-Policy")
		resp.Body.Close()

		scriptSrc := ""
		for _, directive := range strings.Split(csp, ";") {
			if strings.HasPrefix(strings.TrimSpace(directive), "script-src") {
				scriptSrc = strings.TrimSpace(directive)
			}
		}
		if scriptSrc == "" {
			t.Fatalf("%q: no script-src in %q", tc.query, csp)
		}
		if !strings.Contains(scriptSrc, "'self'") {
			t.Errorf("%q: script-src lost 'self': %q", tc.query, scriptSrc)
		}
		if tc.wantOrigin == "" {
			if scriptSrc != "script-src 'self'" {
				t.Errorf("%q: expected same-origin-only script-src, got %q", tc.query, scriptSrc)
			}
			continue
		}
		if !strings.Contains(scriptSrc, tc.wantOrigin) {
			t.Errorf("%q: script-src %q missing %q", tc.query, scriptSrc, tc.wantOrigin)
		}
		// One CDN at a time: choosing jsDelivr must not also permit unpkg.
		for _, other := range []string{"https://cdn.jsdelivr.net", "https://unpkg.com", "https://cdnjs.cloudflare.com"} {
			if other != tc.wantOrigin && strings.Contains(scriptSrc, other) {
				t.Errorf("%q: script-src also allowed %q", tc.query, other)
			}
		}
	}
}

// The page must only carry a CDN script tag when a CDN was actually chosen.
func TestChatPageLoadsTheRendererOnlyWhenChosen(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{ChatUI: true})

	body := getBody(t, srv.URL+"/chat")
	if strings.Contains(body, "mermaid.min.js") {
		t.Error("default page pulls in a CDN script")
	}
	if !strings.Contains(body, `data-mermaid-cdn=""`) {
		t.Error("default page did not report an empty CDN choice to the UI")
	}

	body = getBody(t, srv.URL+"/chat?mermaid=unpkg")
	if !strings.Contains(body, "https://unpkg.com/mermaid@11/dist/mermaid.min.js") {
		t.Error("chosen CDN script tag missing")
	}
	if !strings.Contains(body, `data-mermaid-cdn="unpkg"`) {
		t.Error("chosen CDN not reported to the UI")
	}

	// A rejected choice must not leave a script tag behind either.
	body = getBody(t, srv.URL+"/chat?mermaid=evil")
	if strings.Contains(body, "mermaid.min.js") {
		t.Error("unrecognised choice still produced a script tag")
	}
}

// Assets are same-origin regardless of the page's choice; only /chat needs the
// relaxation, and widening it for the script and stylesheet would be pointless.
func TestAssetCSPIsAlwaysSameOrigin(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{ChatUI: true})
	for _, path := range []string{"/script.js?mermaid=jsdelivr", "/style.css?mermaid=jsdelivr"} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		csp := resp.Header.Get("Content-Security-Policy")
		resp.Body.Close()
		if strings.Contains(csp, "jsdelivr") {
			t.Errorf("%s CSP widened for a CDN: %q", path, csp)
		}
	}
}

func TestMermaidChoiceValidation(t *testing.T) {
	for _, good := range []string{"jsdelivr", "unpkg", "cdnjs", "JSDELIVR", " unpkg "} {
		if mermaidChoice(good) == "" {
			t.Errorf("mermaidChoice(%q) rejected a known CDN", good)
		}
	}
	for _, bad := range []string{"", "evil", "https://cdn.jsdelivr.net", "jsdelivr evil", "../jsdelivr"} {
		if got := mermaidChoice(bad); got != "" {
			t.Errorf("mermaidChoice(%q) = %q, want \"\"", bad, got)
		}
	}
}

func getBody(t *testing.T, url string) string {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	rec := httptest.NewRecorder()
	if _, err := rec.Body.ReadFrom(resp.Body); err != nil {
		t.Fatal(err)
	}
	return rec.Body.String()
}
