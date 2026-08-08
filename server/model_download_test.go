package server

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"

	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestModelsDownloadVariantsRequiresModelDir(t *testing.T) {
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{}))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/models/download/variants?ref=org/model")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
}

func TestModelsDownloadVariantsListsQuantizations(t *testing.T) {
	hf := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"path":"model-Q4_K_M.gguf","type":"file","size":123},{"path":"model-Q8_0.gguf","type":"file","size":456}]`)
	}))
	defer hf.Close()
	t.Setenv("HF_ENDPOINT", hf.URL)

	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: t.TempDir()}))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/models/download/variants?ref=org/model")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}
	var data struct {
		Repository string `json:"repository"`
		Revision   string `json:"revision"`
		Variants   []struct {
			Quant     string `json:"quant"`
			SizeBytes int64  `json:"size_bytes"`
			Shards    int    `json:"shards"`
			Selector  string `json:"selector"`
		} `json:"variants"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatal(err)
	}
	if data.Repository != "org/model" || data.Revision != "main" || len(data.Variants) != 2 {
		t.Fatalf("unexpected response: %+v", data)
	}
	for _, v := range data.Variants {
		if v.Selector != "hf:org/model:"+v.Quant {
			t.Fatalf("selector = %q for quant %q", v.Selector, v.Quant)
		}
	}
}

func TestModelsDownloadVariantsRejectsMissingRef(t *testing.T) {
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: t.TempDir()}))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/models/download/variants")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

func TestModelsSearchFindsGGUFRepositories(t *testing.T) {
	hf := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/models" {
			t.Fatalf("request = %s %s", r.Method, r.URL.String())
		}
		query := r.URL.Query()
		if query.Get("search") != "qwen" || query.Get("filter") != "gguf" || query.Get("limit") != "2" || query.Get("sort") != "downloads" || query.Get("direction") != "-1" {
			t.Fatalf("unexpected search query: %q", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"id":"Qwen/Qwen3-4B-GGUF","downloads":123,"likes":45,"lastModified":"2026-08-09T12:00:00.000Z","private":false,"gated":false}]`)
	}))
	defer hf.Close()
	t.Setenv("HF_ENDPOINT", hf.URL)

	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: t.TempDir()}))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/models/search?q=qwen&limit=2")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}
	if got := resp.Header.Get("Cache-Control"); got != "no-store" {
		t.Fatalf("cache-control = %q", got)
	}
	var data struct {
		Models []struct {
			ID        string `json:"id"`
			Name      string `json:"name"`
			Downloads int64  `json:"downloads"`
			Likes     int64  `json:"likes"`
			UpdatedAt string `json:"updated_at"`
			GGUF      bool   `json:"gguf"`
			Private   bool   `json:"private"`
			Gated     bool   `json:"gated"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Fatal(err)
	}
	if len(data.Models) != 1 {
		t.Fatalf("models = %#v", data.Models)
	}
	model := data.Models[0]
	if model.ID != "Qwen/Qwen3-4B-GGUF" || model.Name != model.ID || model.Downloads != 123 || model.Likes != 45 || model.UpdatedAt != "2026-08-09T12:00:00.000Z" || !model.GGUF || model.Private || model.Gated {
		t.Fatalf("model = %#v", model)
	}
}

func TestModelsSearchValidatesRequest(t *testing.T) {
	disabled := httptest.NewServer(NewHandler(nil, HandlerOptions{}))
	defer disabled.Close()
	resp, err := http.Get(disabled.URL + "/models/search?q=qwen")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled search status = %d, want 404", resp.StatusCode)
	}

	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: t.TempDir()}))
	defer srv.Close()
	for _, target := range []string{
		"/models/search",
		"/models/search?q=qwen&limit=0",
		"/models/search?q=qwen&limit=21",
	} {
		response, err := http.Get(srv.URL + target)
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != http.StatusBadRequest {
			t.Fatalf("GET %s status = %d, want 400", target, response.StatusCode)
		}
	}
	response, err := http.Post(srv.URL+"/models/search?q=qwen", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST search status = %d, want 405", response.StatusCode)
	}
}

func TestModelsDownloadRejectsInvalidRef(t *testing.T) {
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: t.TempDir()}))
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/models/download", "application/json", strings.NewReader(`{"ref":"not-a-repository"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}
}

// fakeHFDownloadServer serves a single-file "repository" containing weights,
// mimicking the tree/resolve endpoints internal/huggingface talks to.
// onResolveGET, if set, runs synchronously inside the GET /resolve/ handler
// before the body is written (used to coordinate concurrency tests).
func fakeHFDownloadServer(t *testing.T, filename string, weights []byte, onResolveGET func()) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/tree/"):
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `[{"path":%q,"type":"file","size":%d}]`, filename, len(weights))
		case strings.Contains(r.URL.Path, "/resolve/"):
			w.Header().Set("X-Linked-ETag", `"tiny-blob"`)
			if r.Method == http.MethodGet {
				if onResolveGET != nil {
					onResolveGET()
				}
				_, _ = w.Write(weights)
			}
		default:
			http.NotFound(w, r)
		}
	}))
}

func TestModelsDownloadPlacesFileAndAppearsInCatalog(t *testing.T) {
	weights := buildTinyLlamaGGUF()
	hf := fakeHFDownloadServer(t, "tiny-Q4_K_M.gguf", weights, nil)
	defer hf.Close()
	t.Setenv("HF_ENDPOINT", hf.URL)
	t.Setenv("HF_HOME", t.TempDir())

	modelDir := t.TempDir()
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: modelDir}))
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/models/download", "application/json", strings.NewReader(`{"ref":"org/tiny:Q4_K_M"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/x-ndjson" {
		t.Fatalf("content-type = %q", ct)
	}

	var events []map[string]any
	dec := json.NewDecoder(resp.Body)
	for {
		var event map[string]any
		if err := dec.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if len(events) == 0 {
		t.Fatal("expected at least one NDJSON event")
	}
	last := events[len(events)-1]
	if last["status"] != "success" {
		t.Fatalf("final event = %v", last)
	}
	placedPath, _ := last["path"].(string)
	if placedPath == "" {
		t.Fatalf("final event missing path: %v", last)
	}
	if _, err := os.Stat(placedPath); err != nil {
		t.Fatalf("placed file missing: %v", err)
	}
	if rel, err := filepath.Rel(modelDir, placedPath); err != nil || strings.HasPrefix(rel, "..") {
		t.Fatalf("placed path %q escaped model dir %q (rel=%q, err=%v)", placedPath, modelDir, rel, err)
	}

	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, e := range entries {
		if e.Path == placedPath {
			found = true
			if !e.IsSupported {
				t.Fatalf("downloaded model not marked supported: %+v", e)
			}
		}
	}
	if !found {
		t.Fatalf("downloaded model %q not found via DiscoverModels: %+v", placedPath, entries)
	}

	modelsResp, err := http.Get(srv.URL + "/models")
	if err != nil {
		t.Fatal(err)
	}
	defer modelsResp.Body.Close()
	var list struct {
		Models []struct {
			Path      string `json:"path"`
			Supported bool   `json:"supported"`
		} `json:"models"`
	}
	if err := json.NewDecoder(modelsResp.Body).Decode(&list); err != nil {
		t.Fatal(err)
	}
	seen := false
	for _, m := range list.Models {
		if m.Path == placedPath {
			seen = true
		}
	}
	if !seen {
		t.Fatalf("/models did not list the downloaded model: %+v", list.Models)
	}
}

func TestModelsDownloadRejectsDuplicateConcurrentRef(t *testing.T) {
	weights := buildTinyLlamaGGUF()
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	hf := fakeHFDownloadServer(t, "tiny-Q4_K_M.gguf", weights, func() {
		select {
		case started <- struct{}{}:
		default:
		}
		<-release
	})
	defer hf.Close()
	t.Setenv("HF_ENDPOINT", hf.URL)
	t.Setenv("HF_HOME", t.TempDir())

	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ModelDir: t.TempDir()}))
	defer srv.Close()

	type result struct {
		resp *http.Response
		err  error
	}
	done := make(chan result, 1)
	go func() {
		resp, err := http.Post(srv.URL+"/models/download", "application/json", strings.NewReader(`{"ref":"org/tiny:Q4_K_M"}`))
		done <- result{resp, err}
	}()

	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("first download never reached the Hub")
	}

	resp2, err := http.Post(srv.URL+"/models/download", "application/json", strings.NewReader(`{"ref":"org/tiny:Q4_K_M"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusConflict {
		body, _ := io.ReadAll(resp2.Body)
		t.Fatalf("second concurrent download status = %d: %s", resp2.StatusCode, body)
	}

	close(release)
	first := <-done
	if first.err != nil {
		t.Fatal(first.err)
	}
	first.resp.Body.Close()
}

func TestHFRepoDirNameNeverEscapesModelDir(t *testing.T) {
	base := t.TempDir()
	for _, repo := range []string{"owner/repo", "../secret", "../../etc/passwd", "a/../../b", "..", ".", "", `..\secret`} {
		dir := filepath.Join(base, hfRepoDirName(repo))
		rel, err := filepath.Rel(base, dir)
		if err != nil {
			t.Fatalf("repo %q: %v", repo, err)
		}
		if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
			t.Fatalf("repo %q escaped base dir: joined=%q rel=%q", repo, dir, rel)
		}
	}
}
