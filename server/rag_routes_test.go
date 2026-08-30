package server

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type ragRoundTripperFunc func(*http.Request) (*http.Response, error)

func (f ragRoundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

// TestRAGRoutesNotRegisteredWithoutFeature covers the same promise
// TestDisabledFeaturesAreNotRegistered checks for the other optional
// capabilities: a server that never enabled "rag" answers 404 for its whole
// route family, not merely an empty result.
func TestRAGRoutesNotRegisteredWithoutFeature(t *testing.T) {
	h := NewHandler(nil, HandlerOptions{ChatUI: true})
	t.Cleanup(func() { _ = h.Close() })
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)

	for _, path := range []string{"/rag/status", "/rag/documents", "/rag/upload", "/rag/fetch", "/rag/reload", "/rag/search"} {
		resp, err := srv.Client().Get(srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("GET %s = %d, want 404 with the rag feature disabled", path, resp.StatusCode)
		}
	}
}

func TestRAGUploadIndexesTextFilesAndReportsSkippedFiles(t *testing.T) {
	srv := ragTestServer(t, HandlerOptions{})

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	good, err := writer.CreateFormFile("files", "field-notes.md")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = good.Write([]byte("The observatory uses a heliostat calibration sequence every Tuesday."))
	bad, err := writer.CreateFormFile("files", "diagram.png")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = bad.Write([]byte{0x89, 'P', 'N', 'G'})
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest(http.MethodPost, srv.URL+"/rag/upload", &body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST upload = %d: %+v", resp.StatusCode, result)
	}
	if docs, _ := result["documents"].([]any); len(docs) != 1 {
		t.Fatalf("uploaded documents = %+v, want one indexed text file", result)
	}
	if skipped, _ := result["skipped"].([]any); len(skipped) != 1 {
		t.Fatalf("skipped = %+v, want the PNG reported", result)
	}

	search, err := srv.Client().Post(srv.URL+"/rag/search", "application/json", strings.NewReader(`{"query":"heliostat calibration"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer search.Body.Close()
	var searched map[string]any
	if err := json.NewDecoder(search.Body).Decode(&searched); err != nil {
		t.Fatal(err)
	}
	if hits, _ := searched["hits"].([]any); len(hits) == 0 {
		t.Fatalf("search over upload = %+v, want a hit", searched)
	}
}

func TestRAGFetchExtractsWebPageTitleAndContent(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(`<!doctype html><html><head><title>Orchard Manual</title><script>secretNavigationNoise()</script></head><body><nav>Site links</nav><main><h1>Pruning</h1><p>Prune the espalier after the winter solstice.</p></main></body></html>`))
	}))
	defer upstream.Close()
	previousClient := ragFetchClientFunc
	ragFetchClientFunc = func() *http.Client { return upstream.Client() }
	t.Cleanup(func() { ragFetchClientFunc = previousClient })

	srv := ragTestServer(t, HandlerOptions{})
	resp, err := srv.Client().Post(srv.URL+"/rag/fetch", "application/json", strings.NewReader(`{"url":`+strconv.Quote(upstream.URL+"/manual")+`}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var created map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || created["title"] != "Orchard Manual" || created["kind"] != "web" {
		t.Fatalf("POST fetch = %d: %+v", resp.StatusCode, created)
	}

	search, err := srv.Client().Post(srv.URL+"/rag/search", "application/json", strings.NewReader(`{"query":"espalier winter solstice"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer search.Body.Close()
	var searched map[string]any
	if err := json.NewDecoder(search.Body).Decode(&searched); err != nil {
		t.Fatal(err)
	}
	hits, _ := searched["hits"].([]any)
	if len(hits) == 0 {
		t.Fatalf("search over fetched page = %+v, want a hit", searched)
	}
	if text, _ := hits[0].(map[string]any)["text"].(string); strings.Contains(text, "secretNavigationNoise") || strings.Contains(text, "Site links") {
		t.Fatalf("fetched text retained ignored page chrome: %q", text)
	}
}

func TestRAGFetchWikipediaUsesPlaintextArticleAPI(t *testing.T) {
	var requestedURL string
	previousClient := ragFetchClientFunc
	ragFetchClientFunc = func() *http.Client {
		return &http.Client{Transport: ragRoundTripperFunc(func(req *http.Request) (*http.Response, error) {
			requestedURL = req.URL.String()
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"query":{"pages":[{"title":"Retrieval-Augmented Generation","extract":"Retrieval augmentation combines a language model with an external knowledge index."}]}}`,
				)),
				Request: req,
			}, nil
		})}
	}
	t.Cleanup(func() { ragFetchClientFunc = previousClient })

	srv := ragTestServer(t, HandlerOptions{})
	resp, err := srv.Client().Post(srv.URL+"/rag/fetch", "application/json", strings.NewReader(
		`{"url":"https://en.wikipedia.org/wiki/Retrieval-Augmented_Generation"}`,
	))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var created map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || created["title"] != "Retrieval-Augmented Generation" || created["kind"] != "wikipedia" {
		t.Fatalf("POST Wikipedia fetch = %d: %+v", resp.StatusCode, created)
	}
	if !strings.Contains(requestedURL, "/w/api.php?") || !strings.Contains(requestedURL, "explaintext=1") || !strings.Contains(requestedURL, "redirects=1") {
		t.Fatalf("Wikipedia import requested %q, want the plaintext action API", requestedURL)
	}
}

func ragTestServer(t *testing.T, opts HandlerOptions) *httptest.Server {
	t.Helper()
	opts.Features.RAG = true
	h := NewHandler(nil, opts)
	t.Cleanup(func() { _ = h.Close() })
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv
}

func TestRAGDocumentLifecycle(t *testing.T) {
	srv := ragTestServer(t, HandlerOptions{})
	client := srv.Client()

	// A fresh knowledge base starts empty.
	var status map[string]any
	getJSON(t, client, srv.URL+"/rag/status", &status)
	if int(status["documents"].(float64)) != 0 || int(status["chunks"].(float64)) != 0 {
		t.Fatalf("status = %+v, want an empty knowledge base", status)
	}

	// Adding a document without text is rejected.
	resp, err := client.Post(srv.URL+"/rag/documents", "application/json", strings.NewReader(`{"title":"Empty"}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("POST empty text = %d, want 400", resp.StatusCode)
	}

	// Adding a real document indexes it and reports it back.
	resp, err = client.Post(srv.URL+"/rag/documents", "application/json", strings.NewReader(
		`{"id":"handbook","title":"Return Policy","text":"Products may be returned within 30 days of purchase for a full refund."}`))
	if err != nil {
		t.Fatal(err)
	}
	var created map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&created); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST document = %d: %+v", resp.StatusCode, created)
	}
	if created["id"] != "handbook" || created["title"] != "Return Policy" {
		t.Fatalf("created = %+v", created)
	}

	// A document without an explicit ID gets one generated.
	resp, err = client.Post(srv.URL+"/rag/documents", "application/json", strings.NewReader(
		`{"text":"Orders ship within two business days via standard courier."}`))
	if err != nil {
		t.Fatal(err)
	}
	var autoID map[string]any
	json.NewDecoder(resp.Body).Decode(&autoID)
	resp.Body.Close()
	if id, _ := autoID["id"].(string); id == "" {
		t.Fatalf("autoID = %+v, want a generated id", autoID)
	}

	getJSON(t, client, srv.URL+"/rag/status", &status)
	if int(status["documents"].(float64)) != 2 {
		t.Fatalf("status after adding two docs = %+v", status)
	}

	var listed map[string]any
	getJSON(t, client, srv.URL+"/rag/documents", &listed)
	docs, _ := listed["documents"].([]any)
	if len(docs) != 2 {
		t.Fatalf("documents = %+v, want 2", listed)
	}

	// Search finds the seeded document by its content.
	resp, err = client.Post(srv.URL+"/rag/search", "application/json", strings.NewReader(`{"query":"return refund"}`))
	if err != nil {
		t.Fatal(err)
	}
	var searched map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&searched); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	hits, _ := searched["hits"].([]any)
	if len(hits) == 0 {
		t.Fatalf("search hits = %+v, want at least one match", searched)
	}
	first := hits[0].(map[string]any)
	if first["doc_id"] != "handbook" {
		t.Fatalf("first hit = %+v, want the handbook doc", first)
	}

	// An empty query is rejected rather than matching everything.
	resp, err = client.Post(srv.URL+"/rag/search", "application/json", strings.NewReader(`{"query":""}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty query search = %d, want 400", resp.StatusCode)
	}

	// Deleting an unknown document is a 404, not a silent success.
	req, _ := http.NewRequest(http.MethodDelete, srv.URL+"/rag/documents", strings.NewReader(`{"id":"missing"}`))
	resp, err = client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("DELETE missing doc = %d, want 404", resp.StatusCode)
	}

	// Deleting the seeded document removes it from the index.
	req, _ = http.NewRequest(http.MethodDelete, srv.URL+"/rag/documents", strings.NewReader(`{"id":"handbook"}`))
	resp, err = client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("DELETE handbook = %d, want 204", resp.StatusCode)
	}
	getJSON(t, client, srv.URL+"/rag/status", &status)
	if int(status["documents"].(float64)) != 1 {
		t.Fatalf("status after delete = %+v, want 1 remaining document", status)
	}
}

// TestRAGDocumentMutationsAreAdminOnlyInManagedMode checks that RAG follows
// the same rule /remote does: any user may read the shared knowledge base
// (list, search, status), but changing what is in it — a corpus every other
// user's chat then searches — requires the administrator token.
func TestRAGDocumentMutationsAreAdminOnlyInManagedMode(t *testing.T) {
	const token = "not-in-any-response"
	srv := ragTestServer(t, HandlerOptions{DeploymentMode: DeploymentManaged, AdminToken: token})
	client := srv.Client()

	do := func(method, path, body string, admin bool) *http.Response {
		t.Helper()
		req, err := http.NewRequest(method, srv.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		if admin {
			req.Header.Set(AdminTokenHeader, token)
		}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	resp := do(http.MethodPost, "/rag/documents", `{"text":"hello"}`, false)
	resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("unauthorized POST /rag/documents = %d, want 403", resp.StatusCode)
	}

	resp = do(http.MethodGet, "/rag/documents", "", false)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("unauthorized GET /rag/documents = %d, want 200 (reads stay public)", resp.StatusCode)
	}

	resp = do(http.MethodPost, "/rag/documents", `{"text":"hello"}`, true)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("authorized POST /rag/documents = %d, want 200", resp.StatusCode)
	}
}

// TestRAGDocsDirSeedsTheKnowledgeBaseAtStartup covers HandlerOptions.RAGDocsDir:
// an operator pointing --rag-docs at a folder should see it searchable from
// the very first request, with no separate POST /rag/documents step.
func TestRAGDocsDirSeedsTheKnowledgeBaseAtStartup(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "handbook.md"), []byte("Products may be returned within 30 days of purchase for a full refund."), 0o644); err != nil {
		t.Fatal(err)
	}

	srv := ragTestServer(t, HandlerOptions{RAGDocsDir: dir})
	client := srv.Client()

	var status map[string]any
	getJSON(t, client, srv.URL+"/rag/status", &status)
	if int(status["documents"].(float64)) != 1 {
		t.Fatalf("status = %+v, want the seeded document indexed at startup", status)
	}

	resp, err := client.Post(srv.URL+"/rag/search", "application/json", strings.NewReader(`{"query":"return refund"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var searched map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&searched); err != nil {
		t.Fatal(err)
	}
	hits, _ := searched["hits"].([]any)
	if len(hits) == 0 {
		t.Fatalf("search over seeded docs = %+v, want at least one hit", searched)
	}
}

func getJSON(t *testing.T, client *http.Client, url string, v any) {
	t.Helper()
	resp, err := client.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		t.Fatal(err)
	}
}
