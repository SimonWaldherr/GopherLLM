package server

import (
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testWorkspace = `{"format":"gopherllm-chat-workspace","version":2,"activeID":"","preferences":{},"conversations":[]}`

func TestChatHistoryStoreCompressesAndDetectsConflicts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "workspace.gz")
	store := newChatHistoryStore(path)
	etag, err := store.write(t.Context(), []byte(testWorkspace), "")
	if err != nil {
		t.Fatal(err)
	}
	if etag == "" {
		t.Fatal("write returned an empty ETag")
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := gzip.NewReader(file); err != nil {
		file.Close()
		t.Fatalf("history is not gzip-compressed: %v", err)
	}
	file.Close()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("history permissions = %o, want 600", info.Mode().Perm())
	}
	got, gotETag, err := store.read(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != testWorkspace || gotETag != etag {
		t.Fatalf("read = %q etag=%q, want %q/%q", got, gotETag, testWorkspace, etag)
	}
	if _, err := store.write(t.Context(), []byte(testWorkspace), "stale"); err != errChatHistoryConflict {
		t.Fatalf("stale write error = %v, want conflict", err)
	}
}

func TestChatHistoryEndpoints(t *testing.T) {
	path := filepath.Join(t.TempDir(), "workspace.gz")
	srv := httptest.NewServer(NewHandler(nil, HandlerOptions{ChatHistoryPath: path}))
	t.Cleanup(srv.Close)

	status, err := srv.Client().Get(srv.URL + "/chat/storage")
	if err != nil {
		t.Fatal(err)
	}
	if status.StatusCode != http.StatusOK {
		t.Fatalf("storage status = %d", status.StatusCode)
	}
	status.Body.Close()

	put, err := http.NewRequest(http.MethodPut, srv.URL+"/chat/workspace", strings.NewReader(testWorkspace))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := srv.Client().Do(put)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || resp.Header.Get("ETag") == "" {
		t.Fatalf("PUT status=%d etag=%q", resp.StatusCode, resp.Header.Get("ETag"))
	}
	etag := resp.Header.Get("ETag")

	get, err := srv.Client().Get(srv.URL + "/chat/workspace")
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(get.Body)
	get.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if get.StatusCode != http.StatusOK || string(body) != testWorkspace || get.Header.Get("ETag") != etag {
		t.Fatalf("GET status=%d body=%q etag=%q", get.StatusCode, body, get.Header.Get("ETag"))
	}

	stale, err := http.NewRequest(http.MethodPut, srv.URL+"/chat/workspace", strings.NewReader(testWorkspace))
	if err != nil {
		t.Fatal(err)
	}
	stale.Header.Set("If-Match", `"stale"`)
	conflict, err := srv.Client().Do(stale)
	if err != nil {
		t.Fatal(err)
	}
	conflict.Body.Close()
	if conflict.StatusCode != http.StatusPreconditionFailed {
		t.Fatalf("stale PUT status=%d, want 412", conflict.StatusCode)
	}
}
