package server

import (
	"net/http/httptest"
	"testing"
)

// newManagedTestServer closes the listener before releasing the handler's
// active runners. That ordering matters on Windows, where a live mmap keeps
// its GGUF file undeletable during TempDir cleanup.
func newManagedTestServer(t *testing.T, handler *Handler) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(func() {
		srv.Close()
		if err := handler.Close(); err != nil {
			t.Errorf("close handler: %v", err)
		}
	})
	return srv
}
