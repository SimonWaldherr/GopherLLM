package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// registerSystemRoutes registers the small standalone status/config endpoints
// that don't belong to a protocol family: /health, /deployment, /privacy,
// /remote, /remote/models, and /batch/parse. Extracted from NewHandler's
// inline handlers for these routes.
func registerSystemRoutes(mux *http.ServeMux, state *runnerState, deployment deploymentAccess, remote *remoteState, history *chatHistoryStore) {
	mux.HandleFunc("/health", func(w http.ResponseWriter, req *http.Request) {
		status := deployment.status(req)
		status["ok"] = true
		status["model"] = modelID(state.get())
		status["remote"] = remote.enabled()
		writeJSON(w, status)
	})
	mux.HandleFunc("/deployment", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, deployment.status(req))
	})
	mux.HandleFunc("/batch/parse", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		data, err := io.ReadAll(io.LimitReader(req.Body, maxSpreadsheetBytes+1))
		if err != nil {
			http.Error(w, "read spreadsheet: "+err.Error(), http.StatusBadRequest)
			return
		}
		if len(data) > maxSpreadsheetBytes {
			http.Error(w, fmt.Sprintf("spreadsheet exceeds %d bytes", maxSpreadsheetBytes), http.StatusRequestEntityTooLarge)
			return
		}
		result, err := parseSpreadsheet(data, req.URL.Query().Get("filename"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, result)
	})
	mux.HandleFunc("/privacy", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, map[string]any{
			"report": gopherllm.DefaultPrivacyReport(),
			"chat_history": map[string]any{
				"configured": history.enabled(),
				"mode":       map[bool]string{true: "server-file-opt-in", false: "browser-default"}[history.enabled()],
			},
			"research_tools": map[string]any{
				"default":              "disabled",
				"request_flags":        []string{"gopherllm_wikimedia", "gopherllm_openstreetmap"},
				"openstreetmap_notice": OSMUsageNotice,
			},
		})
	})
	mux.HandleFunc("/remote", func(w http.ResponseWriter, req *http.Request) {
		switch req.Method {
		case http.MethodGet:
			writeJSON(w, remote.publicConfig())
		case http.MethodDelete:
			remote.clear()
			writeJSON(w, remote.publicConfig())
		case http.MethodPost:
			var body remoteConfigRequest
			if err := json.NewDecoder(io.LimitReader(req.Body, 64<<10)).Decode(&body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if err := remote.configure(body); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			writeJSON(w, remote.publicConfig())
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/remote/models", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		models, err := remote.listModels(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, models)
	})
}
