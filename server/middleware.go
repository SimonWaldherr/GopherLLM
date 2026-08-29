package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

var inferenceRequestSeq atomic.Uint64

func withLimit(sem chan struct{}, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		select {
		case sem <- struct{}{}:
		case <-r.Context().Done():
			return
		}
		defer func() { <-sem }()
		h(w, r)
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("content-type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// writeContextWindowHeaders exposes the exact rendered prompt accounting to
// the bundled web UI without changing OpenAI's streaming event schema. They
// are only emitted for explicit bounded-context extensions.
func writeContextWindowHeaders(w http.ResponseWriter, info gopherllm.ContextWindowInfo) {
	w.Header().Set("X-GopherLLM-Context-Mode", string(info.Mode))
	w.Header().Set("X-GopherLLM-Context-Length", strconv.Itoa(info.ContextLength))
	w.Header().Set("X-GopherLLM-Context-Budget", strconv.Itoa(info.PromptBudget))
	w.Header().Set("X-GopherLLM-Context-Prompt-Tokens", strconv.Itoa(info.PromptTokens))
	w.Header().Set("X-GopherLLM-Context-Input-Messages", strconv.Itoa(info.InputMessages))
	w.Header().Set("X-GopherLLM-Context-Retained-Messages", strconv.Itoa(info.RetainedMessages))
	w.Header().Set("X-GopherLLM-Context-Dropped-Messages", strconv.Itoa(info.DroppedMessages))
	w.Header().Set("X-GopherLLM-Context-Compressed-Messages", strconv.Itoa(info.CompressedMessages))
}

func ensureRequestID(w http.ResponseWriter, req *http.Request) string {
	id := strings.TrimSpace(req.Header.Get("X-Request-ID"))
	if id == "" {
		id = fmt.Sprintf("gopherllm-%d-%d", time.Now().UnixNano(), inferenceRequestSeq.Add(1))
	}
	w.Header().Set("X-Request-ID", id)
	return id
}

// withRequestContext binds the HTTP request's context to the generation
// options, so a client that disconnects or times out stops the work it asked
// for: GenerateChatStreamUntil checks the context between prefill chunks and
// decoded tokens, and the agentic loop hands the same context to every
// AgenticTool.Execute.
//
// This used to return options unchanged, which meant a browser tab closed
// mid-answer left the model generating to completion and left any in-flight
// tool fetch running against a caller nobody would read. The cancellation
// machinery was already there on both sides; only this wiring was missing.
//
// The visible consequence is that a cancelled request now ends with a context
// error rather than a full answer. Routes turn that into a 400 nobody is left
// to read, and logInferenceResult records it as the cancellation it is.
func withRequestContext(options gopherllm.GenerationOptions, req *http.Request) gopherllm.GenerationOptions {
	if req == nil {
		return options
	}
	return options.WithContext(req.Context())
}

// requireLoadedModel keeps catalog, UI and model-loading routes available when
// the server starts without weights. Generation-shaped routes return a clear
// 503 until the user chooses a model instead of dereferencing a nil Runner.
func remoteOrLoadedModel(state *runnerState, remote *remoteState, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/v1/chat/completions" && remote.enabled() {
			remote.proxyChat(w, req)
			return
		}
		if state.get() == nil && needsLoadedModel(req.URL.Path) {
			http.Error(w, "no model is loaded; choose one in the Web UI or POST /models/load", http.StatusServiceUnavailable)
			return
		}
		next.ServeHTTP(w, req)
	})
}

func needsLoadedModel(path string) bool {
	switch path {
	case "/generate", "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/generate", "/api/chat", "/api/embeddings", "/api/embed", "/autotune/run":
		return true
	default:
		return false
	}
}
