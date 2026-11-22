package hestia

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
)

type researchTransportFunc func(*http.Request) (*http.Response, error)

func (f researchTransportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func searchTestResponse(code int, statusBody io.Reader) *http.Response {
	return &http.Response{StatusCode: code, Header: make(http.Header), Body: io.NopCloser(statusBody)}
}

func TestWikipediaSearchSendsOnlyQueryAndFormatsCitations(t *testing.T) {
	provider, err := NewWikipediaSearch(" DE ")
	if err != nil {
		t.Fatal(err)
	}
	wiki := provider.(*wikipediaSearch)
	wiki.client.Transport = researchTransportFunc(func(req *http.Request) (*http.Response, error) {
		params := req.URL.Query()
		if req.Method != http.MethodGet || req.URL.Scheme != "https" || req.URL.Host != "de.wikipedia.org" ||
			req.URL.Path != "/w/rest.php/v1/search/page" || len(params) != 2 ||
			params.Get("q") != "Wärme & Wind" || params.Get("limit") != "5" ||
			req.Header.Get("Accept") != "application/json" || req.Header.Get("User-Agent") == "" {
			t.Errorf("unexpected Wikipedia request: %+v", req)
		}
		return searchTestResponse(http.StatusOK, strings.NewReader(`{"pages":[
			{"key":"Rhein-Main-Gebiet","title":"Rhein-Main-Gebiet","excerpt":"Das <span>Rhein-Main</span>-Gebiet &amp; Umgebung."},
			{"key":"Stadt_/_Land","title":"Stadt und Land","description":"Ein Beispiel"}
		]}`)), nil
	})
	results, err := wiki.Search(t.Context(), "  Wärme & Wind  ")
	if err != nil || len(results) != 2 {
		t.Fatalf("Search() = %+v, %v", results, err)
	}
	if results[0].Snippet != "Das Rhein-Main-Gebiet & Umgebung." || results[0].URL != "https://de.wikipedia.org/wiki/Rhein-Main-Gebiet" ||
		results[1].Snippet != "Ein Beispiel" || results[1].URL != "https://de.wikipedia.org/wiki/Stadt_%2F_Land" {
		t.Fatalf("unexpected citations: %+v", results)
	}
}

func TestWebSearchUsesFormAndBoundsUntrustedResults(t *testing.T) {
	provider, err := NewWebSearch("http://127.0.0.1:8888")
	if err != nil {
		t.Fatal(err)
	}
	web := provider.(*searxSearch)
	web.client.Transport = researchTransportFunc(func(req *http.Request) (*http.Response, error) {
		if err := req.ParseForm(); err != nil {
			t.Fatal(err)
		}
		if req.Method != http.MethodPost || req.URL.String() != "http://127.0.0.1:8888/search" ||
			req.PostForm.Get("q") != "Wärme & Wind" || req.PostForm.Get("format") != "json" ||
			req.Header.Get("Authorization") != "" || req.Header.Get("Cookie") != "" || req.Header.Get("Referer") != "" {
			t.Errorf("unexpected SearXNG request: %+v", req)
		}
		var rows []string
		rows = append(rows,
			`{"title":"bad","url":"javascript:alert(1)"}`,
			`{"title":"secret","url":"https://u:p@example.com/"}`,
			`{"title":"empty","url":"//example.com/"}`,
		)
		for i := range 8 {
			rows = append(rows, fmt.Sprintf(`{"title":"<b>Example</b>","url":"https://example.com/%d","content":%q}`, i, strings.Repeat("ä", 500)))
		}
		return searchTestResponse(http.StatusOK, strings.NewReader(`{"results":[`+strings.Join(rows, ",")+`]}`)), nil
	})
	results, err := web.Search(t.Context(), "Wärme & Wind")
	if err != nil || len(results) != searchResultLimit {
		t.Fatalf("Search() returned %d results, err=%v", len(results), err)
	}
	if results[0].Title != "Example" || results[0].URL != "https://example.com/0" || results[0].Snippet != strings.Repeat("ä", 400) {
		t.Fatalf("unexpected first result: %+v", results[0])
	}
}

func TestSearchRejectsExfiltrationPathsAndBoundsInput(t *testing.T) {
	for _, raw := range []string{"", "https://name:password@example.com", "http://192.168.1.100", "http://example.com", "https://example.com/#x", "https://example.com/?q=x", "file:///etc/hosts"} {
		if _, err := NewWebSearch(raw); err == nil {
			t.Errorf("NewWebSearch(%q) unexpectedly succeeded", raw)
		}
	}
	if _, err := NewWikipediaSearch("en.example.net"); err == nil {
		t.Fatal("NewWikipediaSearch accepted an invalid edition")
	}
	provider, err := NewWebSearch("https://example.com/search")
	if err != nil {
		t.Fatal(err)
	}
	web := provider.(*searxSearch)
	if web.client.Jar != nil || web.client.Transport.(*http.Transport).Proxy != nil {
		t.Fatal("search transport uses cookies or environmental proxy settings")
	}
	web.client.Transport = researchTransportFunc(func(req *http.Request) (*http.Response, error) {
		t.Fatal("invalid query reached provider")
		return nil, nil
	})
	for _, query := range []string{"  ", strings.Repeat("ä", searchQueryLimit+1), "\xff"} {
		if _, err := web.Search(t.Context(), query); err == nil {
			t.Error("invalid query accepted")
		}
	}
}

func TestSearchRedirectsAndSecretsStayInProvider(t *testing.T) {
	provider, err := NewWikipediaSearch("en")
	if err != nil {
		t.Fatal(err)
	}
	wiki := provider.(*wikipediaSearch)
	const query = "private search terms"
	wiki.client.Transport = researchTransportFunc(func(req *http.Request) (*http.Response, error) {
		if req.URL.Host != "en.wikipedia.org" {
			t.Fatalf("followed redirect to a different host: %s", req.URL.Host)
		}
		resp := searchTestResponse(http.StatusTemporaryRedirect, strings.NewReader(query))
		resp.Header.Set("Location", "https://elsewhere.example.com/?q=private+search+terms")
		return resp, nil
	})
	if _, err := wiki.Search(t.Context(), query); err == nil || !strings.Contains(err.Error(), "HTTP 307") || strings.Contains(err.Error(), query) {
		t.Fatalf("redirect error = %v", err)
	}
	wiki.client.Transport = researchTransportFunc(func(req *http.Request) (*http.Response, error) {
		return nil, errors.New(query)
	})
	if _, err := wiki.Search(t.Context(), query); err == nil || strings.Contains(err.Error(), "private") {
		t.Fatalf("transport error = %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := wiki.Search(ctx, query); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled search = %v", err)
	}
	wiki.client.Transport = researchTransportFunc(func(req *http.Request) (*http.Response, error) {
		return searchTestResponse(http.StatusOK, strings.NewReader(strings.Repeat(" ", searchBodyLimit+1))), nil
	})
	if _, err := wiki.Search(t.Context(), query); err == nil || !strings.Contains(err.Error(), "large") {
		t.Fatalf("large response error = %v", err)
	}
}
