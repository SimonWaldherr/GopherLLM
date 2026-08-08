package server

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestManualBrowserCheckServer is a throwaway harness, not a real test: it
// spins up a fake Hugging Face Hub plus a real GopherLLM chat server so a
// browser can exercise the new "download a model" UI end to end without
// touching the real network or downloading a multi-gigabyte file. Deleted
// after manual verification.
func TestManualBrowserCheckServer(t *testing.T) {
	if testing.Short() {
		t.Skip("manual-only harness")
	}
	weights := buildTinyLlamaGGUF()
	hf := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/tree/"):
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `[{"path":"tinyllama-Q4_K_M.gguf","type":"file","size":%d},{"path":"tinyllama-Q8_0.gguf","type":"file","size":%d}]`, len(weights), len(weights)*2)
		case strings.Contains(r.URL.Path, "/resolve/"):
			w.Header().Set("X-Linked-ETag", `"tiny-blob-`+filepathBaseForTest(r.URL.Path)+`"`)
			if r.Method == http.MethodGet {
				time.Sleep(400 * time.Millisecond) // slow enough to see progress in the UI
				_, _ = w.Write(weights)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer hf.Close()
	t.Setenv("HF_ENDPOINT", hf.URL)
	t.Setenv("HF_HOME", t.TempDir())

	modelDir := t.TempDir()
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: modelDir, ChatUI: true}))
	defer srv.Close()

	t.Logf("MANUAL CHECK SERVER: %s/chat", srv.URL)
	t.Logf("MANUAL CHECK MODEL DIR: %s", modelDir)
	time.Sleep(6 * time.Minute)
}

func filepathBaseForTest(p string) string {
	idx := strings.LastIndexByte(p, '/')
	if idx < 0 {
		return p
	}
	return p[idx+1:]
}
