package server

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestChatUIStaticAssetRevalidation(t *testing.T) {
	h := NewHandler(nil, HandlerOptions{ChatUI: true})
	defer h.Close()
	for _, assetPath := range []string{"/script.js", "/style.css", "/audio.js", "/vision.js", "/audio-worklet.js"} {
		t.Run(assetPath, func(t *testing.T) {
			var plainETag string
			for _, encoding := range []string{"identity", "gzip"} {
				req := httptest.NewRequest(http.MethodGet, assetPath, nil)
				req.Header.Set("Accept-Encoding", encoding)
				initial := httptest.NewRecorder()
				h.ServeHTTP(initial, req)
				etag := initial.Header().Get("ETag")
				if initial.Code != http.StatusOK || initial.Body.Len() == 0 || etag == "" {
					t.Fatalf("initial %s response: status=%d size=%d etag=%q", encoding, initial.Code, initial.Body.Len(), etag)
				}
				if got := initial.Header().Get("Cache-Control"); got != "private, max-age=0, must-revalidate" {
					t.Fatalf("cache-control=%q", got)
				}
				if initial.Header().Get("Vary") != "Accept-Encoding" || initial.Header().Get("Pragma") != "" {
					t.Fatalf("invalid revalidation headers: %v", initial.Header())
				}
				if got := initial.Header().Get("Content-Length"); got != strconv.Itoa(initial.Body.Len()) {
					t.Fatalf("content-length=%q, size=%d", got, initial.Body.Len())
				}
				if encoding == "identity" {
					plainETag = etag
				} else {
					if etag == plainETag {
						t.Fatal("compressed and plain representations share a strong ETag")
					}
					gz, err := gzip.NewReader(initial.Body)
					if err != nil {
						t.Fatal(err)
					}
					if _, err := io.ReadAll(gz); err != nil {
						t.Fatal(err)
					}
					gz.Close()
				}
				for _, validator := range []string{etag, "W/" + etag, `"another-build", W/` + etag, "*"} {
					req.Header.Set("If-None-Match", validator)
					cached := httptest.NewRecorder()
					h.ServeHTTP(cached, req)
					if cached.Code != http.StatusNotModified || cached.Body.Len() != 0 || cached.Header().Get("ETag") != etag {
						t.Fatalf("validator %q: status=%d size=%d headers=%v", validator, cached.Code, cached.Body.Len(), cached.Header())
					}
				}
				req.Header.Set("If-None-Match", `"previous-build"`)
				changed := httptest.NewRecorder()
				h.ServeHTTP(changed, req)
				if changed.Code != http.StatusOK || changed.Body.Len() == 0 {
					t.Fatal("stale validator suppressed the new asset")
				}
				req.Method = http.MethodHead
				head := httptest.NewRecorder()
				h.ServeHTTP(head, req)
				if head.Code != http.StatusOK || head.Body.Len() != 0 || head.Header().Get("Content-Length") != initial.Header().Get("Content-Length") {
					t.Fatalf("HEAD: status=%d size=%d headers=%v", head.Code, head.Body.Len(), head.Header())
				}
			}
		})
	}
	if newChatStaticAsset("first build").plainETag == newChatStaticAsset("next build").plainETag {
		t.Fatal("asset content changes must invalidate its ETag")
	}
}

func TestChatUIGzipNegotiation(t *testing.T) {
	for _, tc := range []struct {
		header string
		want   bool
	}{
		{"", false}, {"br", false}, {"gzip", true}, {"br, GZip", true},
		{"gzip;q=0", false}, {"br, gzip; q=0.0", false}, {"gzip; q=0.25", true},
		{"gzip;q=invalid", false}, {"gzip;q=-1", false}, {"gzip;q=2", false},
	} {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		req.Header.Set("Accept-Encoding", tc.header)
		if got := acceptsGzip(req); got != tc.want {
			t.Errorf("acceptsGzip(%q)=%v, want %v", tc.header, got, tc.want)
		}
	}
	h := NewHandler(nil, HandlerOptions{ChatUI: true})
	defer h.Close()
	for _, assetPath := range []string{"/chat", "/script.js"} {
		req := httptest.NewRequest(http.MethodGet, assetPath, nil)
		req.Header.Set("Accept-Encoding", "gzip;q=0")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Header().Get("Content-Encoding") != "" {
			t.Fatalf("%s ignored gzip;q=0", assetPath)
		}
		if assetPath == "/chat" && w.Header().Get("Cache-Control") != "no-store" {
			t.Fatal("chat HTML containing model and admin state must remain no-store")
		}
	}
}

func TestChatPageCacheSeparatesAndBoundsVariants(t *testing.T) {
	var cache chatPageCache
	data := chatTemplateData{Title: "Chat", Model: "first-model", HasModel: true}
	first, err := cache.render(data)
	if err != nil {
		t.Fatal(err)
	}
	again, err := cache.render(data)
	if err != nil || again != first {
		t.Fatal("identical template data did not reuse the rendered page")
	}
	for _, variant := range []chatTemplateData{
		{Title: "Chat", Model: "second-model", HasModel: true},
		{Title: "Chat", Model: "first-model", HasModel: true, AdminAuthorized: true},
		{Title: "Chat", Model: "first-model", HasModel: true, MermaidCDN: "unpkg", MermaidScript: mermaidCDNs["unpkg"].Script},
	} {
		page, err := cache.render(variant)
		if err != nil {
			t.Fatal(err)
		}
		if page == first || bytes.Equal(page.plain, first.plain) {
			t.Fatal("changed model/admin/CDN reused another template variant")
		}
		var expected bytes.Buffer
		if err := chatTemplate.Execute(&expected, variant); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(page.plain, expected.Bytes()) {
			t.Fatal("cached page does not match current template data")
		}
		gz, err := gzip.NewReader(bytes.NewReader(page.gzip))
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := io.ReadAll(gz)
		gz.Close()
		if err != nil || !bytes.Equal(decoded, expected.Bytes()) {
			t.Fatal("compressed page does not match current template data")
		}
	}
	for i := range 40 {
		data.Model = fmt.Sprintf("model-%d", i)
		if _, err := cache.render(data); err != nil {
			t.Fatal(err)
		}
		if len(cache.pages) > 16 {
			t.Fatal("model changes retained unbounded rendered pages")
		}
	}
}

func TestChatPageCacheConcurrentRequests(t *testing.T) {
	var cache chatPageCache
	var wg sync.WaitGroup
	for i := range 16 {
		wg.Go(func() {
			data := chatTemplateData{Title: "Chat", Model: "model", AdminAuthorized: i%2 == 0}
			for range 10 {
				page, err := cache.render(data)
				if err != nil {
					t.Error(err)
					return
				}
				if !strings.Contains(string(page.plain), fmt.Sprintf(`data-admin-authorized="%v"`, data.AdminAuthorized)) {
					t.Error("concurrent requests mixed authorization variants")
				}
			}
		})
	}
	wg.Wait()
}

func TestChatAssetsBypassModelWriterLock(t *testing.T) {
	state := &runnerState{}
	state.mu.Lock()
	defer state.mu.Unlock()
	h := remoteOrLoadedModel(state, newRemoteState(), make(chan struct{}, 1), http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	for _, assetPath := range []string{"/script.js", "/style.css", "/health", "/deployment"} {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		req := httptest.NewRequest(http.MethodGet, assetPath, nil).WithContext(ctx)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != http.StatusNoContent {
			t.Fatalf("%s waited for the model lock: status=%d", assetPath, w.Code)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code == http.StatusNoContent {
		t.Fatal("inference route bypassed its loaded-model guard")
	}
}

// Discard response bodies so these benchmarks measure serving work rather
// than httptest's allocation and copying of every full response body.
type chatBenchmarkWriter struct {
	header http.Header
}

func (w *chatBenchmarkWriter) Header() http.Header            { return w.header }
func (w *chatBenchmarkWriter) Write(body []byte) (int, error) { return len(body), nil }
func (w *chatBenchmarkWriter) WriteHeader(int)                {}

func BenchmarkChatUIPage(b *testing.B) {
	for _, encoding := range []string{"identity", "gzip"} {
		b.Run(encoding, func(b *testing.B) {
			h := NewHandler(nil, HandlerOptions{ChatUI: true})
			defer h.Close()
			req := httptest.NewRequest(http.MethodGet, "/chat", nil)
			req.Header.Set("Accept-Encoding", encoding)
			w := &chatBenchmarkWriter{header: make(http.Header)}
			h.ServeHTTP(w, req)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				h.ServeHTTP(w, req)
			}
		})
	}
}

func BenchmarkChatUIStaticAsset(b *testing.B) {
	for _, revalidate := range []bool{false, true} {
		name := "full"
		if revalidate {
			name = "revalidated"
		}
		b.Run(name, func(b *testing.B) {
			h := NewHandler(nil, HandlerOptions{ChatUI: true})
			defer h.Close()
			req := httptest.NewRequest(http.MethodGet, "/script.js", nil)
			req.Header.Set("Accept-Encoding", "gzip")
			w := &chatBenchmarkWriter{header: make(http.Header)}
			h.ServeHTTP(w, req)
			if revalidate {
				etag := w.header.Get("ETag")
				if etag == "" {
					etag = `"missing-validator"`
				}
				req.Header.Set("If-None-Match", etag)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				h.ServeHTTP(w, req)
			}
		})
	}
}
