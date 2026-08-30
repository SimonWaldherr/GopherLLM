package server

import (
	"net"
	"net/url"
	"strings"
	"testing"
)

func TestIsDisallowedRAGFetchIP(t *testing.T) {
	disallowed := []string{
		"127.0.0.1",       // loopback
		"::1",             // loopback (IPv6)
		"10.0.0.5",        // private
		"172.16.0.1",      // private
		"192.168.1.1",     // private
		"169.254.169.254", // link-local / cloud metadata
		"fe80::1",         // link-local (IPv6)
		"0.0.0.0",         // unspecified
		"::",              // unspecified (IPv6)
		"224.0.0.1",       // multicast
	}
	for _, s := range disallowed {
		ip := net.ParseIP(s)
		if ip == nil {
			t.Fatalf("test bug: %q did not parse as an IP", s)
		}
		if !isDisallowedRAGFetchIP(ip) {
			t.Errorf("isDisallowedRAGFetchIP(%s) = false, want true", s)
		}
	}

	allowed := []string{"93.184.216.34", "2001:4860:4860::8888"}
	for _, s := range allowed {
		ip := net.ParseIP(s)
		if ip == nil {
			t.Fatalf("test bug: %q did not parse as an IP", s)
		}
		if isDisallowedRAGFetchIP(ip) {
			t.Errorf("isDisallowedRAGFetchIP(%s) = true, want false", s)
		}
	}
}

func TestWikipediaArticleEndpoint(t *testing.T) {
	target, err := url.Parse("https://de.wikipedia.org/wiki/Retrieval-Augmented_Generation#Geschichte")
	if err != nil {
		t.Fatal(err)
	}
	endpoint, ok := wikipediaArticleEndpoint(target)
	if !ok {
		t.Fatal("Wikipedia article URL was not recognized")
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Host != "de.wikipedia.org" || parsed.Path != "/w/api.php" || parsed.Query().Get("titles") != "Retrieval-Augmented Generation" || parsed.Query().Get("explaintext") != "1" {
		t.Fatalf("endpoint = %q", endpoint)
	}

	notArticle, _ := url.Parse("https://example.com/wiki/RAG")
	if _, ok := wikipediaArticleEndpoint(notArticle); ok {
		t.Fatal("non-Wikipedia URL was recognized as a Wikipedia article")
	}
}

func TestDecodeWikipediaArticle(t *testing.T) {
	title, text, err := decodeWikipediaArticle([]byte(`{"query":{"pages":[{"title":"Berlin","extract":"Berlin ist die Hauptstadt Deutschlands."}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if title != "Berlin" || !strings.Contains(text, "Hauptstadt") {
		t.Fatalf("title=%q text=%q", title, text)
	}
}

func TestExtractRAGHTML(t *testing.T) {
	title, text := extractRAGHTML([]byte(`<html><head><title>A &amp; B</title><style>.hidden{}</style></head><body><nav>Skip this</nav><main><h1>Heading</h1><p>Useful&nbsp;text &amp; facts.</p></main><script>skip()</script></body></html>`))
	if title != "A & B" {
		t.Fatalf("title = %q", title)
	}
	if !strings.Contains(text, "Useful text & facts.") || strings.Contains(text, "Skip this") || strings.Contains(text, "skip()") {
		t.Fatalf("text = %q", text)
	}
}

func TestFetchContentAllowed(t *testing.T) {
	cases := []struct {
		contentType string
		wantHTML    bool
		wantOK      bool
	}{
		{"", false, true},
		{"text/plain; charset=utf-8", false, true},
		{"text/markdown", false, true},
		{"application/json", false, true},
		{"text/html; charset=utf-8", true, true},
		{"application/xhtml+xml", true, true},
		{"image/png", false, false},
		{"application/pdf", false, false},
		{"application/octet-stream", false, false},
	}
	for _, c := range cases {
		gotHTML, gotOK := fetchContentAllowed(c.contentType)
		if gotHTML != c.wantHTML || gotOK != c.wantOK {
			t.Errorf("fetchContentAllowed(%q) = (%v, %v), want (%v, %v)", c.contentType, gotHTML, gotOK, c.wantHTML, c.wantOK)
		}
	}
}
