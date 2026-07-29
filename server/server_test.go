package server

import (
	gopherllm "github.com/SimonWaldherr/GopherLLM"

	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDisplayServerURLNormalizesWildcardAddress(t *testing.T) {
	got := displayServerURL(":8081", true)
	if got != "http://localhost:8081/chat" {
		t.Fatalf("url = %q", got)
	}
}

func TestDisplayServerURLKeepsExplicitHost(t *testing.T) {
	got := displayServerURL("127.0.0.1:9090", false)
	if got != "http://127.0.0.1:9090" {
		t.Fatalf("url = %q", got)
	}
}

func TestChatTemplateUsesServerDefaults(t *testing.T) {
	var out strings.Builder
	err := chatTemplate.Execute(&out, chatTemplateData{
		Title:         "GopherLLM Chat",
		Model:         "test-model",
		MaxTokens:     123,
		Temperature:   0.25,
		TopP:          0.85,
		TopK:          10,
		MinP:          0.05,
		RepeatPenalty: 1.2,
	})
	if err != nil {
		t.Fatal(err)
	}
	html := out.String()
	if !strings.Contains(html, `value="123"`) {
		t.Fatalf("max token default missing from template")
	}
	if !strings.Contains(html, `value="0.25"`) {
		t.Fatalf("temperature default missing from template")
	}
	if !strings.Contains(html, `value="0.85"`) {
		t.Fatalf("top-p default missing from template")
	}
	if !strings.Contains(html, `value="10"`) {
		t.Fatalf("top-k default missing from template")
	}
	if !strings.Contains(html, `value="0.05"`) {
		t.Fatalf("min-p default missing from template")
	}
	if !strings.Contains(html, `value="1.2"`) {
		t.Fatalf("repeat penalty default missing from template")
	}
}

func TestChatUIAssetsArePrivateAndSelfContained(t *testing.T) {
	srv := newTestServer(t, HandlerOptions{ChatUI: true})
	for _, want := range []struct {
		path        string
		contentType string
		marker      string
	}{
		{path: "/chat", contentType: "text/html", marker: `id="chatList"`},
		{path: "/chat", contentType: "text/html", marker: `id="briefChat"`},
		{path: "/chat", contentType: "text/html", marker: `id="shareChat"`},
		{path: "/style.css", contentType: "text/css", marker: ".sidebar"},
		{path: "/script.js", contentType: "text/javascript", marker: "IndexedDB"},
	} {
		resp, err := http.Get(srv.URL + want.path)
		if err != nil {
			t.Fatalf("GET %s: %v", want.path, err)
		}
		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			t.Fatalf("read %s: %v", want.path, readErr)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("GET %s status = %d", want.path, resp.StatusCode)
		}
		if !strings.HasPrefix(resp.Header.Get("Content-Type"), want.contentType) {
			t.Fatalf("GET %s content-type = %q", want.path, resp.Header.Get("Content-Type"))
		}
		if resp.Header.Get("Cache-Control") != "no-store" {
			t.Fatalf("GET %s cache-control = %q", want.path, resp.Header.Get("Cache-Control"))
		}
		if !strings.Contains(resp.Header.Get("Content-Security-Policy"), "default-src 'self'") {
			t.Fatalf("GET %s missing CSP: %q", want.path, resp.Header.Get("Content-Security-Policy"))
		}
		if !strings.Contains(string(body), want.marker) {
			t.Fatalf("GET %s missing %q", want.path, want.marker)
		}
	}
}

func TestHandlerStartsWithoutModel(t *testing.T) {
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ChatUI: true}))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/chat")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status = %d", resp.StatusCode)
	}
	resp.Body.Close()

	resp, err = http.Post(srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("chat without model status = %d body=%s", resp.StatusCode, body)
	}
}

func TestApplyRequestOptionsPreservesDefaultStopsWhenStopOmitted(t *testing.T) {
	def := gopherllm.DefaultGenerationOptions()
	def.StopSequences = []string{"</s>"}

	got := applyRequestOptions(def, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, "")
	if len(got.StopSequences) != 1 || got.StopSequences[0] != "</s>" {
		t.Fatalf("stop sequences = %#v", got.StopSequences)
	}
}

func TestApplyRequestOptionsOverridesStopsWhenProvided(t *testing.T) {
	def := gopherllm.DefaultGenerationOptions()
	def.StopSequences = []string{"</s>"}

	got := applyRequestOptions(def, nil, nil, nil, nil, nil, nil, nil, nil, []any{"END", "STOP"}, nil, "")
	if len(got.StopSequences) != 2 || got.StopSequences[0] != "END" || got.StopSequences[1] != "STOP" {
		t.Fatalf("stop sequences = %#v", got.StopSequences)
	}
}

// TestAutoTuneEndpointsReportStatusAndInvalidateOnSwap exercises the web UI's
// auto-tune integration end to end: GET /autotune before anything has run,
// POST /autotune/run measuring and applying a tuning, GET /autotune
// reflecting it as active, and a model hot-swap clearing "active" (a fresh
// gopherllm.Runner has had no tuning applied to it this session) while "cached" still
// reports the on-disk result from before the swap.
func TestAutoTuneEndpointsReportStatusAndInvalidateOnSwap(t *testing.T) {
	baseline := gopherllm.CaptureRuntimeTuning()
	defer baseline.Apply()

	cacheDir := t.TempDir()
	t.Setenv("LOCALAPPDATA", cacheDir) // os.UserCacheDir on Windows
	t.Setenv("XDG_CACHE_HOME", cacheDir)
	t.Setenv("HOME", cacheDir)

	modelDir := t.TempDir()
	modelPath := filepath.Join(modelDir, "catalog", "tiny.gguf")
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath, buildTinyLlamaGGUF(), 0o600); err != nil {
		t.Fatal(err)
	}

	runner, _, err := gopherllm.RunnerFromPath(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()

	srv := newManagedTestServer(t, NewHandler(runner, HandlerOptions{
		ModelDir:              modelDir,
		ModelPath:             modelPath,
		BaselineRuntimeTuning: &baseline,
	}))

	type status struct {
		Active bool `json:"active"`
		Cached bool `json:"cached"`
		Result *struct {
			Threads int `json:"threads"`
		} `json:"result"`
	}
	getStatus := func() status {
		t.Helper()
		resp, err := http.Get(srv.URL + "/autotune")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("GET /autotune status = %d", resp.StatusCode)
		}
		var s status
		if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
			t.Fatal(err)
		}
		return s
	}

	if s := getStatus(); s.Active || s.Cached {
		t.Fatalf("fresh server should report no tuning yet, got %+v", s)
	}

	if resp, err := http.Get(srv.URL + "/autotune/run"); err != nil {
		t.Fatal(err)
	} else {
		resp.Body.Close()
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Fatalf("GET /autotune/run status = %d, want 405", resp.StatusCode)
		}
	}
	invalidBody, _ := json.Marshal(map[string]any{"effort": "turbo"})
	resp, err := http.Post(srv.URL+"/autotune/run", "application/json", strings.NewReader(string(invalidBody)))
	if err != nil {
		t.Fatal(err)
	}
	invalidRespBody := readAll(t, resp)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("POST /autotune/run invalid effort status = %d body=%s, want 400", resp.StatusCode, invalidRespBody)
	}

	runBody, _ := json.Marshal(map[string]any{"effort": "quick"})
	resp, err = http.Post(srv.URL+"/autotune/run", "application/json", strings.NewReader(string(runBody)))
	if err != nil {
		t.Fatal(err)
	}
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /autotune/run status = %d body=%s", resp.StatusCode, body)
	}
	var ran struct {
		Cached bool `json:"cached"`
		Result struct {
			Threads int `json:"threads"`
		} `json:"result"`
	}
	if err := json.Unmarshal([]byte(body), &ran); err != nil {
		t.Fatal(err)
	}
	if ran.Cached {
		t.Fatal("first run should measure, not reuse a cache")
	}
	if ran.Result.Threads < 1 {
		t.Fatalf("tuned threads = %d, want >= 1", ran.Result.Threads)
	}

	if s := getStatus(); !s.Active || !s.Cached || s.Result == nil || s.Result.Threads < 1 {
		t.Fatalf("status after tuning = %+v, want active+cached with a result", s)
	}
	// Make a runtime setting visibly differ from the captured baseline. The
	// hot-swap below must restore the baseline rather than leave any setting
	// that was active while the previous model was tuned.
	changedChunk := baseline.PrefillChunk() + 1
	if changedChunk < 1 {
		changedChunk = 1
	}
	gopherllm.SetPrefillChunk(changedChunk)

	// Hot-swap to a runner built from the same catalog file. Even though the
	// tuning cache key ends up identical, the newly loaded gopherllm.Runner itself has
	// never had a tuning applied, so "active" must go back to false while
	// "cached" keeps reporting the on-disk result from before the swap.
	loadBody, _ := json.Marshal(map[string]string{"path": modelPath})
	resp, err = http.Post(srv.URL+"/models/load", "application/json", strings.NewReader(string(loadBody)))
	if err != nil {
		t.Fatal(err)
	}
	loadRespBody := readAll(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /models/load status = %d body=%s", resp.StatusCode, loadRespBody)
	}

	if s := getStatus(); s.Active {
		t.Fatalf("status after hot-swap should not report active, got %+v", s)
	} else if !s.Cached {
		t.Fatalf("status after hot-swap should still report the on-disk cache, got %+v", s)
	}
	if got := gopherllm.CaptureRuntimeTuning(); got != baseline {
		t.Fatalf("hot-swap runtime tuning = %+v, want captured baseline %+v", got.PrefillChunk(), baseline.PrefillChunk())
	}
}
