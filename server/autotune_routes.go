package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// registerAutoTuneRoutes registers /autotune and /autotune/run. Extracted
// from NewHandler's inline handlers for these routes; it creates its own
// autoTuneMu locally since it is the only route group that needs it.
func registerAutoTuneRoutes(mux *http.ServeMux, state *runnerState, sem chan struct{}, logw io.Writer) {
	var autoTuneMu sync.Mutex
	mux.HandleFunc("/autotune", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, state.autoTuneStatus())
	})
	mux.HandleFunc("/autotune/run", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Effort  string `json:"effort"`
			Refresh bool   `json:"refresh"`
		}
		if req.Body != nil {
			if err := json.NewDecoder(req.Body).Decode(&body); err != nil && err != io.EOF {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		if !autoTuneMu.TryLock() {
			http.Error(w, "auto-tuning is already running", http.StatusConflict)
			return
		}
		defer autoTuneMu.Unlock()
		body.Effort = strings.TrimSpace(body.Effort)
		if !gopherllm.ValidAutoTuneEffort(body.Effort) {
			http.Error(w, "effort must be quick, balanced, or thorough", http.StatusBadRequest)
			return
		}
		runOpts := gopherllm.AutoTuneOptionsForEffort(body.Effort)
		runOpts.LogWriter = logw
		res, cached, err := state.runAutoTune(runOpts, body.Refresh)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(logw, "Auto-tune via web UI: %s\n", res.SettingsLine())
		writeJSON(w, map[string]any{"cached": cached, "result": res})
	}))
}
