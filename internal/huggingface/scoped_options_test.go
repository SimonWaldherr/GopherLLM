package huggingface

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExplicitHubCacheAndTokenStayScoped(t *testing.T) {
	ambient := t.TempDir()
	destination := t.TempDir()
	t.Setenv("HF_HUB_OFFLINE", "0")
	t.Setenv("HF_HUB_CACHE", ambient)
	t.Setenv("HF_TOKEN", "ambient")
	t.Setenv("HF_ENDPOINT", "http://127.0.0.1:1")
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if token := r.Header.Get("Authorization"); token != "" {
			t.Error("Hub credential was forwarded to redirected download host")
		}
		_, _ = io.WriteString(w, "data")
	}))
	defer cdn.Close()
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if token := r.Header.Get("Authorization"); token != "Bearer explicit" {
			t.Errorf("request to the chosen Hugging Face origin had authorization %q", token)
		}
		if strings.Contains(r.URL.Path, "/tree/") {
			w.Header().Set("X-Repo-Commit", "test-commit")
			_, _ = io.WriteString(w, `[{"path":"model.fermion","type":"file","size":4}]`)
			return
		}
		if r.Method == http.MethodHead {
			w.Header().Set("ETag", `"test-blob"`)
			return
		}
		http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
	}))
	defer hub.Close()
	token := "explicit"
	opts := Options{CacheDir: destination, Endpoint: hub.URL, Token: &token}
	paths, err := DownloadFiles(t.Context(), "org/model", []string{"model.fermion"}, io.Discard, opts)
	if err != nil || len(paths) != 1 {
		t.Fatalf("paths = %v, err = %v", paths, err)
	}
	if !strings.HasPrefix(paths[0], filepath.Join(destination, "models--org--model")+string(filepath.Separator)) {
		t.Fatalf("used cache outside the explicit scope: %s", paths[0])
	}
	if data, err := os.ReadFile(paths[0]); err != nil || string(data) != "data" {
		t.Fatalf("downloaded model = %q, err = %v", data, err)
	}
	if entries, err := os.ReadDir(ambient); err != nil || len(entries) != 0 {
		t.Fatalf("ambient cache was touched: %v, %v", entries, err)
	}
	opts.Offline = true
	opts.Endpoint = "http://127.0.0.1:1"
	if _, err := DownloadFiles(t.Context(), "org/model", []string{"model.fermion"}, io.Discard, opts); err != nil {
		t.Fatalf("explicit cache miss in offline mode: %v", err)
	}
}
