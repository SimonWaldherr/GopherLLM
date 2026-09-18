package huggingface

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestSearchModelsAcrossTasks(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("pipeline_tag") != "automatic-speech-recognition" || r.URL.Query().Get("filter") != "safetensors" {
			t.Errorf("query: %s", r.URL.RawQuery)
		}
		io.WriteString(w, `[{"id":"org/speech","pipeline_tag":"automatic-speech-recognition","library_name":"transformers","tags":["safetensors"],"gated":"manual"}]`)
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	got, err := SearchModels(context.Background(), SearchOptions{Query: "speech", PipelineTag: "automatic-speech-recognition", Format: "safetensors"}, Options{})
	if err != nil || len(got) != 1 {
		t.Fatalf("%+v %v", got, err)
	}
	if got[0].GGUF || !got[0].Gated || got[0].PipelineTag != "automatic-speech-recognition" || got[0].LibraryName != "transformers" {
		t.Fatalf("%+v", got[0])
	}
}

func TestArtifactsPinnedDownloadAndOffline(t *testing.T) {
	var downloads atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/tree/") {
			w.Header().Set("X-Repo-Commit", "commit123")
			io.WriteString(w, `[{"path":"encoder/model.safetensors","type":"file","size":4},{"path":"decoder/model.onnx","type":"file","size":4},{"path":"tokenizer.json","type":"file","size":4},{"path":"preprocessor_config.json","type":"file","size":4},{"path":"mmproj-f16.gguf","type":"file","size":4}]`)
			return
		}
		if !strings.Contains(r.URL.Path, "/resolve/commit123/") {
			t.Errorf("unpinned request %s", r.URL.Path)
		}
		w.Header().Set("ETag", fmt.Sprintf(`"blob-%s"`, strings.ReplaceAll(r.URL.Path, "/", "-")))
		if r.Method == http.MethodGet {
			downloads.Add(1)
			io.WriteString(w, "data")
		}
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	ctx := context.Background()
	manifest, err := Inspect(ctx, "org/model", Options{})
	if err != nil || len(manifest.Files) != 5 {
		t.Fatalf("%+v %v", manifest, err)
	}
	roles := map[string]string{}
	for _, f := range manifest.Files {
		roles[f.Path] = f.Role
	}
	if roles["mmproj-f16.gguf"] != "projector" || roles["preprocessor_config.json"] != "processor" {
		t.Fatal(roles)
	}
	names := []string{"encoder/model.safetensors", "decoder/model.onnx", "tokenizer.json"}
	paths, err := DownloadFiles(ctx, "org/model", names, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(paths[0], filepath.FromSlash("encoder/model.safetensors")) {
		t.Fatal(paths)
	}
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil || string(b) != "data" {
			t.Fatalf("%s: %q %v", p, b, err)
		}
	}
	if _, err := DownloadFiles(ctx, "org/model", names, nil, Options{}); err != nil {
		t.Fatal(err)
	}
	if downloads.Load() != 3 {
		t.Fatalf("downloads=%d", downloads.Load())
	}
	t.Setenv("HF_ENDPOINT", "http://127.0.0.1:1")
	if _, err := DownloadFiles(ctx, "org/model", names, nil, Options{Offline: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadFiles(ctx, "org/model", []string{"preprocessor_config.json"}, nil, Options{Offline: true}); err == nil {
		t.Fatal("uncached companion accepted")
	}
	if err := os.Remove(paths[1]); err != nil {
		t.Fatal(err)
	}
	if _, err := DownloadFiles(ctx, "org/model", names, nil, Options{Offline: true}); err == nil {
		t.Fatal("incomplete set accepted")
	}
}

func TestArtifactValidationBeforeNetwork(t *testing.T) {
	for _, path := range []string{"../escape", "/absolute", "nested/../../escape", `dir\evil`, "C:evil", "a\x00b", ""} {
		if _, err := DownloadFiles(context.Background(), "org/model", []string{path}, nil, Options{}); err == nil {
			t.Fatalf("accepted %q", path)
		}
	}
	for _, ref := range []string{"org/..", "../model", "org/model@..", "org/model@a/../b"} {
		if _, err := ParseHuggingFaceReference(ref); err == nil {
			t.Fatalf("accepted %q", ref)
		}
	}
	if safeHFETag("..") != "" {
		t.Fatal("dot etag accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Inspect(ctx, "org/model", Options{}); err != context.Canceled {
		t.Fatal(err)
	}
}

func TestPaginationRejectsForeignOriginAndCycles(t *testing.T) {
	for _, link := range []string{`<https://other.invalid/page>; rel="next"`, `</api/models/org/model/tree/main?recursive=true>; rel="next"`} {
		t.Run(link, func(t *testing.T) {
			var requests atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				w.Header().Set("Link", link)
				io.WriteString(w, `[]`)
			}))
			defer srv.Close()
			t.Setenv("HF_ENDPOINT", srv.URL)
			if _, _, err := hfListFiles(context.Background(), hfReference{Repository: "org/model", Revision: "main"}); err == nil {
				t.Fatal("invalid pagination accepted")
			}
			if requests.Load() != 1 {
				t.Fatalf("requests=%d", requests.Load())
			}
		})
	}
}

func TestExactGGUFSelectorsSeparateProjector(t *testing.T) {
	entries := []hfTreeEntry{{Path: "model-F16.gguf", Type: "file"}, {Path: "mmproj-F16.gguf", Type: "file"}}
	options := ggufOptions(entries, "")
	for _, option := range options {
		selector := hfVariantSelector("org/model", "main", option, options)
		r, err := ParseHuggingFaceReference(selector)
		if err != nil {
			t.Fatal(err)
		}
		files, err := selectHFGGUF(entries, r.Quant)
		if err != nil || len(files) != 1 || files[0] != option.File {
			t.Fatalf("%s: %v %v", selector, files, err)
		}
	}
}

func TestBadResumeRangePreservesPartialFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"blob"`)
		if r.Method == http.MethodHead {
			return
		}
		w.Header().Set("Content-Range", "bytes 0-4/10")
		w.WriteHeader(http.StatusPartialContent)
		io.WriteString(w, "wrong")
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	cache := hfCache{root: t.TempDir()}
	os.MkdirAll(filepath.Join(cache.root, "blobs"), 0755)
	partial := filepath.Join(cache.root, "blobs", "blob.incomplete")
	os.WriteFile(partial, []byte("hello"), 0644)
	if _, err := hfDownload(context.Background(), hfReference{Repository: "org/model", Revision: "main"}, "model.onnx", cache, 10, nil, nil, nil); err == nil {
		t.Fatal("invalid range accepted")
	}
	b, err := os.ReadFile(partial)
	if err != nil || string(b) != "hello" {
		t.Fatalf("%q %v", b, err)
	}
}

func TestArtifactsSharingOneBlobDownloadOnce(t *testing.T) {
	var downloads atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/tree/") {
			w.Header().Set("X-Repo-Commit", "same-content")
			io.WriteString(w, `[{"path":"a/config.json","type":"file","size":2},{"path":"b/config.json","type":"file","size":2},{"path":"c/config.json","type":"file","size":2}]`)
			return
		}
		w.Header().Set("ETag", `"shared-blob"`)
		if r.Method == http.MethodGet {
			downloads.Add(1)
			io.WriteString(w, `{}`)
		}
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	files, err := DownloadFiles(context.Background(), "org/model", []string{"a/config.json", "b/config.json", "c/config.json"}, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if downloads.Load() != 1 {
		t.Fatalf("downloads=%d", downloads.Load())
	}
	for _, file := range files {
		b, err := os.ReadFile(file)
		if err != nil || string(b) != "{}" {
			t.Fatalf("%q %v", b, err)
		}
	}
}

func TestFailedArtifactSetDoesNotPublishRevision(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/tree/") {
			w.Header().Set("X-Repo-Commit", "broken-set")
			io.WriteString(w, `[{"path":"model.safetensors","type":"file","size":20}]`)
			return
		}
		w.Header().Set("ETag", `"incomplete-weights"`)
		if r.Method == http.MethodGet {
			io.WriteString(w, "short")
		}
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("HF_HUB_CACHE", t.TempDir())
	if _, err := DownloadFiles(context.Background(), "org/model", []string{"model.safetensors"}, nil, Options{}); err == nil {
		t.Fatal("truncated file accepted")
	}
	if snapshot := newHFCache(hfReference{Repository: "org/model"}).snapshotForRef("main"); snapshot != "" {
		t.Fatal("published incomplete revision", snapshot)
	}
}
