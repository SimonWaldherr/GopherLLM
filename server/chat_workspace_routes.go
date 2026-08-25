package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
)

// registerChatWorkspaceRoutes registers the opt-in server-side browser
// workspace HTTP surface: /chat/storage (capability discovery) and
// /chat/workspace (read/replace/clear the stored history). Extracted from
// NewHandler's inline handlers for these two routes.
func registerChatWorkspaceRoutes(mux *http.ServeMux, history *chatHistoryStore) {
	mux.HandleFunc("/chat/storage", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		configured := history.enabled()
		mode := "browser"
		if configured {
			mode = "server"
		}
		writeJSON(w, map[string]any{
			"configured": configured,
			"mode":       mode,
			"compressed": configured,
			"max_bytes":  maxChatHistoryBytes,
		})
	})
	mux.HandleFunc("/chat/workspace", func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if !history.enabled() {
			http.Error(w, "server chat storage is not configured; use browser storage or set ChatHistoryPath", http.StatusNotFound)
			return
		}
		switch req.Method {
		case http.MethodGet:
			data, etag, err := history.read(req.Context())
			if errors.Is(err, os.ErrNotExist) {
				http.Error(w, "server chat history is empty", http.StatusNotFound)
				return
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json; charset=utf-8")
			w.Header().Set("ETag", `"`+etag+`"`)
			_, _ = w.Write(data)
		case http.MethodPut:
			data, err := io.ReadAll(io.LimitReader(req.Body, maxChatHistoryBytes+1))
			if err != nil {
				http.Error(w, "read chat history: "+err.Error(), http.StatusBadRequest)
				return
			}
			if len(data) > maxChatHistoryBytes {
				http.Error(w, fmt.Sprintf("chat history exceeds %d bytes", maxChatHistoryBytes), http.StatusRequestEntityTooLarge)
				return
			}
			etag, err := history.write(req.Context(), data, req.Header.Get("If-Match"))
			if errors.Is(err, errChatHistoryConflict) {
				if etag != "" {
					w.Header().Set("ETag", `"`+etag+`"`)
				}
				http.Error(w, err.Error(), http.StatusPreconditionFailed)
				return
			}
			if err != nil {
				status := http.StatusBadRequest
				if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
					status = http.StatusRequestTimeout
				}
				http.Error(w, err.Error(), status)
				return
			}
			w.Header().Set("ETag", `"`+etag+`"`)
			writeJSON(w, map[string]any{"ok": true, "etag": etag, "bytes": len(data)})
		case http.MethodDelete:
			if err := history.clear(req.Context()); err != nil && !errors.Is(err, os.ErrNotExist) {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})
}
