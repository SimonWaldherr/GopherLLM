package server

import (
	"encoding/json"
	"net/http"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/agentos"
)

// registerAgentOSRoutes registers /agentos/status, /agentos/propose, and
// /agentos/execute. Extracted from NewHandler's inline handlers for these
// routes (all already guarded by opts.AgentOS != nil).
func registerAgentOSRoutes(mux *http.ServeMux, state *runnerState, sem chan struct{}, opts HandlerOptions) {
	mux.HandleFunc("/agentos/status", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.AgentOS == nil {
			writeJSON(w, map[string]any{"enabled": false})
			return
		}
		writeJSON(w, map[string]any{
			"enabled": true,
			"policy":  string(opts.AgentOS.Policy),
			"allowed": opts.AgentOS.Allowed,
		})
	})
	mux.HandleFunc("/agentos/propose", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.AgentOS == nil {
			http.Error(w, "the agentic OS-command feature is not enabled on this server", http.StatusNotFound)
			return
		}
		var body struct {
			Instruction string `json:"instruction"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		body.Instruction = strings.TrimSpace(body.Instruction)
		if body.Instruction == "" {
			http.Error(w, "instruction must not be empty", http.StatusBadRequest)
			return
		}
		state.withRunner(func(r *gopherllm.Runner) {
			options := opts.Defaults
			options.SystemPrompt = agentos.SystemPrompt
			options.Tools = nil
			options.ToolChoice = "none"
			options = withRequestContext(options, req)
			result, err := r.GenerateChat([]gopherllm.ChatMessage{gopherllm.UserMessage(body.Instruction)}, options)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			proposal, err := agentos.ParseProposal(result.Text)
			if err != nil {
				http.Error(w, "model did not return a usable proposal: "+err.Error(), http.StatusBadGateway)
				return
			}
			decision := opts.AgentOS.Evaluate(proposal)
			response := map[string]any{"proposal": proposal, "decision": decision}
			// AutoRun means the operator's own policy (whitelist/allow), not the
			// model's self-reported "safe" field, already authorizes this — see
			// the agentos package comment on why Evaluate never reads Safe.
			if decision.AutoRun {
				res, dec, execErr := opts.AgentOS.Execute(req.Context(), proposal, false)
				response["decision"] = dec
				if execErr != nil {
					response["error"] = execErr.Error()
				} else {
					response["result"] = res
				}
			}
			writeJSON(w, response)
		})
	}))
	mux.HandleFunc("/agentos/execute", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.AgentOS == nil {
			http.Error(w, "the agentic OS-command feature is not enabled on this server", http.StatusNotFound)
			return
		}
		var body struct {
			Proposal agentos.Proposal `json:"proposal"`
			// Approved is the human's out-of-band decision — a button click in
			// the browser, never anything read from the model's own output. See
			// Runner.Execute's doc comment.
			Approved bool `json:"approved"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		res, dec, err := opts.AgentOS.Execute(req.Context(), body.Proposal, body.Approved)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"decision": dec, "result": res})
	}))
}
