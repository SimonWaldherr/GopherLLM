package huggingface

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestParseHuggingFaceReference(t *testing.T) {
	r, err := ParseHuggingFaceReference("hf:bartowski/Qwen3-4B-GGUF:Q4_K_M@abc123")
	if err != nil {
		t.Fatal(err)
	}
	if r.Repository != "bartowski/Qwen3-4B-GGUF" || r.Quant != "Q4_K_M" || r.Revision != "abc123" {
		t.Fatalf("unexpected reference: %#v", r)
	}
	if _, err := ParseHuggingFaceReference("hf:not-a-repository"); err == nil {
		t.Fatal("expected invalid repository error")
	}
}

func TestResolveHuggingFaceModelDownloadsSplitAndReusesCache(t *testing.T) {
	var downloads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("Authorization = %q", got)
		}
		switch {
		case strings.Contains(r.URL.Path, "/tree/"):
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Repo-Commit", "0123456789012345678901234567890123456789")
			_, _ = io.WriteString(w, `[{"path":"Qwen-Q4_K_M-00001-of-00002.gguf","type":"file"},{"path":"Qwen-Q4_K_M-00002-of-00002.gguf","type":"file"},{"path":"Qwen-Q8_0.gguf","type":"file"}]`)
		case strings.Contains(r.URL.Path, "/resolve/"):
			blob := "blob-1"
			if strings.Contains(r.URL.Path, "-00002-of-") {
				blob = "blob-2"
			}
			w.Header().Set("X-Linked-ETag", `"`+blob+`"`)
			if r.Method == http.MethodGet {
				downloads.Add(1)
				_, _ = io.WriteString(w, "gguf-data")
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	cacheHome := t.TempDir()
	t.Setenv("HF_ENDPOINT", server.URL)
	t.Setenv("HF_HOME", cacheHome)
	t.Setenv("HF_TOKEN", "test-token")

	path, err := ResolveHuggingFaceModel("hf:org/model:Q4_K_M@revision", io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(path, "Qwen-Q4_K_M-00001-of-00002.gguf") {
		t.Fatalf("unexpected selected path %q", path)
	}
	if got, err := os.ReadFile(path); err != nil || string(got) != "gguf-data" {
		t.Fatalf("downloaded file = %q, %v", got, err)
	}
	if downloads.Load() != 2 {
		t.Fatalf("downloads = %d, want 2", downloads.Load())
	}
	_, err = ResolveHuggingFaceModel("hf:org/model:Q4_K_M@revision", io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if downloads.Load() != 2 {
		t.Fatalf("cached download count = %d, want 2", downloads.Load())
	}
	t.Setenv("HF_ENDPOINT", "http://127.0.0.1:1")
	t.Setenv("HF_HUB_OFFLINE", "1")
	offlinePath, err := ResolveHuggingFaceModel("hf:org/model:Q4_K_M@revision", io.Discard)
	if err != nil || offlinePath != path {
		t.Fatalf("offline cache path = %q, %v", offlinePath, err)
	}
	var listing bytes.Buffer
	if err := ListGGUF("org/model@revision", &listing); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Q4_K_M", "2", "hf:org/model:Q4_K_M@revision"} {
		if !strings.Contains(listing.String(), want) {
			t.Fatalf("offline listing missing %q:\n%s", want, listing.String())
		}
	}
	if _, err := os.Stat(filepath.Join(cacheHome, "hub", "models--org--model", "blobs", "blob-1")); err != nil {
		t.Fatalf("first content blob missing: %v", err)
	}
	ref, err := os.ReadFile(filepath.Join(cacheHome, "hub", "models--org--model", "refs", "revision"))
	if err != nil || strings.TrimSpace(string(ref)) != "0123456789012345678901234567890123456789" {
		t.Fatalf("cached ref = %q, %v", ref, err)
	}
	if _, err := os.Stat(filepath.Dir(path)); err != nil {
		t.Fatal(err)
	}
}

func TestHFListFilesFollowsPagination(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Repo-Commit", "commit")
		if r.URL.Query().Get("cursor") == "next" {
			_, _ = io.WriteString(w, `[{"path":"second.gguf","type":"file"}]`)
			return
		}
		w.Header().Set("Link", `</api/models/org/model/tree/main?recursive=true&cursor=next>; rel="next"`)
		_, _ = io.WriteString(w, `[{"path":"first.gguf","type":"file"}]`)
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)
	entries, commit, err := hfListFiles(context.Background(), hfReference{Repository: "org/model", Revision: "main"})
	if err != nil {
		t.Fatal(err)
	}
	if commit != "commit" || len(entries) != 2 || entries[1].Path != "second.gguf" {
		t.Fatalf("list = %#v, commit %q", entries, commit)
	}
}

func TestListGGUFShowsSizesAndRunnableSelectors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"path":"model-Q4_K_M-00001-of-00002.gguf","type":"file","size":1048576},{"path":"model-Q4_K_M-00002-of-00002.gguf","type":"file","size":1048576},{"path":"model-Q8_0.gguf","type":"file","size":3221225472}]`)
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)
	var output bytes.Buffer
	if err := ListGGUF("org/model@revision", &output); err != nil {
		t.Fatal(err)
	}
	text := output.String()
	for _, want := range []string{"Q4_K_M", "2.0 MiB", "2", "hf:org/model:Q4_K_M@revision", "Q8_0", "3.00 GiB"} {
		if !strings.Contains(text, want) {
			t.Fatalf("listing missing %q:\n%s", want, text)
		}
	}
}

func TestResolveHuggingFaceModelExplainsHowToChooseAnAmbiguousModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"path":"model-Q4_K_M.gguf","type":"file"},{"path":"model-Q8_0.gguf","type":"file"}]`)
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)
	t.Setenv("HF_HOME", t.TempDir())
	_, err := ResolveHuggingFaceModel("hf:org/model@revision", io.Discard)
	if err == nil || !strings.Contains(err.Error(), "gopherllm --hf-list org/model@revision") {
		t.Fatalf("ambiguous error = %v", err)
	}
}

func TestHFDownloadResumesAnIncompleteBlob(t *testing.T) {
	var gotRange string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Linked-ETag", `"model-blob"`)
		if r.Method == http.MethodHead {
			return
		}
		gotRange = r.Header.Get("Range")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = io.WriteString(w, "world")
	}))
	defer server.Close()
	cache := hfCache{root: t.TempDir()}
	if err := os.MkdirAll(filepath.Join(cache.root, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}
	incomplete := filepath.Join(cache.root, "blobs", "model-blob.incomplete")
	if err := os.WriteFile(incomplete, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_ENDPOINT", server.URL)
	blob, err := hfDownload(context.Background(), hfReference{Repository: "org/model", Revision: "main"}, "model.gguf", cache, 10, io.Discard, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gotRange != "bytes=5-" {
		t.Fatalf("Range = %q", gotRange)
	}
	if got, err := os.ReadFile(blob); err != nil || string(got) != "helloworld" {
		t.Fatalf("resumed blob = %q, %v", got, err)
	}
	if _, err := os.Stat(incomplete); !os.IsNotExist(err) {
		t.Fatalf("incomplete file still exists: %v", err)
	}
}

func TestResolveHuggingFaceModelContextStopsBeforeNetworkWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := ResolveHuggingFaceModelContext(ctx, "hf:org/model:Q4_K_M", io.Discard)
	if err != context.Canceled {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

func TestResolveHuggingFaceModelOfflineNeverContactsHub(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		http.Error(w, "network access is forbidden", http.StatusInternalServerError)
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)
	t.Setenv("HF_HOME", t.TempDir())
	_, err := ResolveHuggingFaceModelContextWithOptions(context.Background(), "hf:org/model:Q4_K_M@revision", io.Discard, Options{Offline: true})
	if err == nil || !strings.Contains(err.Error(), "offline mode is enabled") {
		t.Fatalf("offline error = %v", err)
	}
	if requests.Load() != 0 {
		t.Fatalf("offline mode made %d Hub requests", requests.Load())
	}
}

func TestDownloadHFFilesUsesABoundedWorkerPool(t *testing.T) {
	var active, peak atomic.Int32
	started := make(chan struct{}, 3)
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := filepath.Base(r.URL.Path)
		w.Header().Set("X-Linked-ETag", `"`+name+`"`)
		if r.Method == http.MethodHead {
			return
		}
		current := active.Add(1)
		for old := peak.Load(); current > old && !peak.CompareAndSwap(old, current); old = peak.Load() {
		}
		started <- struct{}{}
		<-release
		active.Add(-1)
		_, _ = io.WriteString(w, "x")
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)
	cache := hfCache{root: t.TempDir()}
	tasks := []hfDownloadTask{
		{name: "a.gguf", path: filepath.Join(t.TempDir(), "a.gguf"), size: 1},
		{name: "b.gguf", path: filepath.Join(t.TempDir(), "b.gguf"), size: 1},
		{name: "c.gguf", path: filepath.Join(t.TempDir(), "c.gguf"), size: 1},
	}
	done := make(chan error, 1)
	go func() {
		done <- downloadHFFiles(context.Background(), hfReference{Repository: "org/model", Revision: "main"}, cache, tasks, io.Discard, nil)
	}()
	for range tasks {
		select {
		case <-started:
		case <-time.After(2 * time.Second):
			t.Fatal("workers did not start concurrently")
		}
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if peak.Load() != 3 {
		t.Fatalf("peak concurrent downloads = %d, want 3", peak.Load())
	}
}

func TestSelectHFGGUFRequiresQuantForMultipleModels(t *testing.T) {
	entries := []hfTreeEntry{{Path: "a-Q4_K_M.gguf", Type: "file"}, {Path: "a-Q8_0.gguf", Type: "file"}}
	if _, err := selectHFGGUF(entries, ""); err == nil {
		t.Fatal("expected ambiguity error")
	}
	files, err := selectHFGGUF(entries, "Q8_0")
	if err != nil || len(files) != 1 || files[0] != "a-Q8_0.gguf" {
		t.Fatalf("select = %v, %v", files, err)
	}
}

func TestSelectHFGGUFRejectsIncompleteSplit(t *testing.T) {
	entries := []hfTreeEntry{
		{Path: "a-Q4_K_M-00001-of-00003.gguf", Type: "file"},
		{Path: "a-Q4_K_M-00003-of-00003.gguf", Type: "file"},
	}
	_, err := selectHFGGUF(entries, "Q4_K_M")
	if err == nil || !strings.Contains(err.Error(), "split is incomplete") {
		t.Fatalf("incomplete split error = %v", err)
	}
}

func TestRepositoryVariantsReturnsStructuredSelectors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"path":"model-Q4_K_M-00001-of-00002.gguf","type":"file","size":1048576},{"path":"model-Q4_K_M-00002-of-00002.gguf","type":"file","size":1048576},{"path":"model-Q8_0.gguf","type":"file","size":3221225472}]`)
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)

	info, err := RepositoryVariants(context.Background(), "org/model@revision", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if info.Repository != "org/model" || info.Revision != "revision" {
		t.Fatalf("unexpected repository info: %#v", info)
	}
	if len(info.Variants) != 2 {
		t.Fatalf("variants = %#v, want 2", info.Variants)
	}
	byQuant := map[string]RepositoryVariant{}
	for _, v := range info.Variants {
		byQuant[v.Quant] = v
	}
	if q4 := byQuant["Q4_K_M"]; q4.Shards != 2 || q4.SizeBytes != 2*1048576 || q4.Selector != "hf:org/model:Q4_K_M@revision" {
		t.Fatalf("Q4_K_M variant = %#v", q4)
	}
	if q8 := byQuant["Q8_0"]; q8.Shards != 1 || q8.SizeBytes != 3221225472 || q8.Selector != "hf:org/model:Q8_0@revision" {
		t.Fatalf("Q8_0 variant = %#v", q8)
	}
}

func TestRepositoryVariantsPropagatesNotFoundError(t *testing.T) {
	if _, err := RepositoryVariants(context.Background(), "hf:not-a-repository", Options{}); err == nil {
		t.Fatal("expected invalid repository error")
	}
}

func TestSearchGGUFRepositoriesUsesBoundedGGUFFilter(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/models" {
			t.Fatalf("request = %s %s", r.Method, r.URL.String())
		}
		query := r.URL.Query()
		for key, want := range map[string]string{
			"search":    "ministral",
			"filter":    "gguf",
			"sort":      "downloads",
			"direction": "-1",
			"limit":     "2",
			"full":      "true",
		} {
			if got := query.Get(key); got != want {
				t.Errorf("query %s = %q, want %q", key, got, want)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[
			{"id":"mistralai/Ministral-3-3B-Instruct-GGUF","downloads":1234,"likes":56,"lastModified":"2026-08-09T12:00:00.000Z","private":false,"gated":false},
			{"modelId":"community/private-gguf","downloads":7,"likes":2,"lastModified":"2026-08-08T12:00:00.000Z","private":true,"gated":"auto"},
			{"id":"not-a-repository","downloads":999}
		]`)
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)

	results, err := SearchGGUFRepositories(context.Background(), " ministral ", 2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("results = %#v, want 2 valid repositories", results)
	}
	first := results[0]
	if first.ID != "mistralai/Ministral-3-3B-Instruct-GGUF" || first.Name != first.ID || first.Downloads != 1234 || first.Likes != 56 || first.UpdatedAt != "2026-08-09T12:00:00.000Z" || !first.GGUF || first.Private || first.Gated {
		t.Fatalf("first result = %#v", first)
	}
	second := results[1]
	if second.ID != "community/private-gguf" || !second.Private || !second.Gated || !second.GGUF {
		t.Fatalf("second result = %#v", second)
	}
}

func TestSearchGGUFRepositoriesValidatesAndHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := SearchGGUFRepositories(ctx, "qwen", 1, Options{}); err != context.Canceled {
		t.Fatalf("cancelled search error = %v, want context.Canceled", err)
	}
	if _, err := SearchGGUFRepositories(context.Background(), "", 1, Options{}); err == nil {
		t.Fatal("expected empty query to be rejected")
	}
	if _, err := SearchGGUFRepositories(context.Background(), strings.Repeat("x", maxSearchQueryRunes+1), 1, Options{}); err == nil {
		t.Fatal("expected overlong query to be rejected")
	}
	if _, err := SearchGGUFRepositories(context.Background(), "qwen", MaxSearchLimit+1, Options{}); err == nil {
		t.Fatal("expected overlarge limit to be rejected")
	}
}

func TestResolveHuggingFaceModelFilesContextReturnsAllShardsInOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/tree/"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `[{"path":"Qwen-Q4_K_M-00001-of-00002.gguf","type":"file"},{"path":"Qwen-Q4_K_M-00002-of-00002.gguf","type":"file"}]`)
		case strings.Contains(r.URL.Path, "/resolve/"):
			blob := "blob-1"
			if strings.Contains(r.URL.Path, "-00002-of-") {
				blob = "blob-2"
			}
			w.Header().Set("X-Linked-ETag", `"`+blob+`"`)
			if r.Method == http.MethodGet {
				_, _ = io.WriteString(w, "gguf-data-"+blob)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)
	t.Setenv("HF_HOME", t.TempDir())

	files, err := ResolveHuggingFaceModelFilesContextWithOptions(context.Background(), "hf:org/model:Q4_K_M@revision", io.Discard, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 || !strings.HasSuffix(files[0], "00001-of-00002.gguf") || !strings.HasSuffix(files[1], "00002-of-00002.gguf") {
		t.Fatalf("unexpected shard set: %v", files)
	}
	for i, want := range []string{"gguf-data-blob-1", "gguf-data-blob-2"} {
		got, err := os.ReadFile(files[i])
		if err != nil || string(got) != want {
			t.Fatalf("shard %d = %q, %v; want %q", i, got, err, want)
		}
	}
}

func TestResolveHuggingFaceModelReportsProgress(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/tree/"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `[{"path":"model-Q4_K_M.gguf","type":"file","size":5}]`)
		case strings.Contains(r.URL.Path, "/resolve/"):
			w.Header().Set("X-Linked-ETag", `"blob"`)
			if r.Method == http.MethodGet {
				_, _ = io.WriteString(w, "hello")
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	t.Setenv("HF_ENDPOINT", server.URL)
	t.Setenv("HF_HOME", t.TempDir())

	var mu sync.Mutex
	var events []ProgressEvent
	_, err := ResolveHuggingFaceModelContextWithOptions(context.Background(), "hf:org/model:Q4_K_M", io.Discard, Options{
		OnProgress: func(ev ProgressEvent) {
			mu.Lock()
			defer mu.Unlock()
			events = append(events, ev)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(events) == 0 {
		t.Fatal("expected at least one progress event")
	}
	last := events[len(events)-1]
	if !last.Done || last.Err != nil || last.Completed != 5 || last.Total != 5 {
		t.Fatalf("final progress event = %#v", last)
	}
}

func TestPlaceGGUFFilesLinksShardsIntoDestDir(t *testing.T) {
	src := t.TempDir()
	shard1 := filepath.Join(src, "model-00001-of-00002.gguf")
	shard2 := filepath.Join(src, "model-00002-of-00002.gguf")
	if err := os.WriteFile(shard1, []byte("part1"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(shard2, []byte("part2"), 0o644); err != nil {
		t.Fatal(err)
	}
	destDir := filepath.Join(t.TempDir(), "owner--repo")

	placed, err := PlaceGGUFFiles([]string{shard1, shard2}, destDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(placed) != 2 {
		t.Fatalf("placed = %v", placed)
	}
	for i, want := range []string{"part1", "part2"} {
		if filepath.Dir(placed[i]) != destDir {
			t.Fatalf("placed[%d] dir = %q, want %q", i, filepath.Dir(placed[i]), destDir)
		}
		got, err := os.ReadFile(placed[i])
		if err != nil || string(got) != want {
			t.Fatalf("placed[%d] = %q, %v; want %q", i, got, err, want)
		}
	}
}

func TestPlaceGGUFFilesRejectsEmptyInput(t *testing.T) {
	if _, err := PlaceGGUFFiles(nil, t.TempDir()); err == nil {
		t.Fatal("expected error for an empty file list")
	}
}

func TestRepositoryReturnsOwnerAndName(t *testing.T) {
	repo, err := Repository("hf:bartowski/Qwen3-4B-GGUF:Q4_K_M@main")
	if err != nil || repo != "bartowski/Qwen3-4B-GGUF" {
		t.Fatalf("Repository = %q, %v", repo, err)
	}
	if _, err := Repository("hf:not-a-repository"); err == nil {
		t.Fatal("expected error for an invalid repository")
	}
}
