package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// DefaultAddr is the listen address Serve uses when ServeOptions.Addr is
// empty. Loopback, so a server nobody configured is reachable from this
// machine only; see NetworkExposureWarning for what changes otherwise.
const DefaultAddr = "127.0.0.1:8080"

// Handler is the mountable HTTP API returned by NewHandler. Close releases
// the currently active chat and embedding runners, including memory-mapped
// GGUF files installed through a hot-swap. Hosts should stop their HTTP server
// before calling Close so no new requests can enter.
type Handler struct {
	next      http.Handler
	state     *runnerState
	embedder  *embeddingState
	closeOnce sync.Once
	closeErr  error
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	h.next.ServeHTTP(w, req)
}

func (h *Handler) Close() error {
	if h == nil {
		return nil
	}
	h.closeOnce.Do(func() {
		h.closeErr = errors.Join(h.embedder.close(), h.state.close())
	})
	return h.closeErr
}

// Serve builds the API handler and runs a blocking http.Server on opts.Addr.
// Library consumers who want to control the server lifecycle, add middleware,
// TLS, or mount the API under a path prefix should use NewHandler instead:
//
//	handler := gopherllm.NewHandler(model.Runner(), gopherllm.HandlerOptions{...})
//	mux.Handle("/llm/", http.StripPrefix("/llm", handler))
func Serve(initialRunner *gopherllm.Runner, opts ServeOptions) error {
	mode, err := ParseDeploymentMode(string(opts.DeploymentMode))
	if err != nil {
		return err
	}
	if err := validateDeploymentOptions(mode, opts.AdminToken, opts.WasmDir, opts.ChatUI); err != nil {
		return err
	}
	if strings.TrimSpace(opts.Addr) == "" {
		opts.Addr = DefaultAddr
	}
	networkExposed := !isLoopbackListenAddress(opts.Addr)
	if mode == DeploymentBrowser && initialRunner != nil {
		return errors.New("browser deployment must not be started with a server-side model runner")
	}
	logw := opts.LogWriter
	if logw == nil {
		logw = os.Stderr
	}
	handler := NewHandler(initialRunner, HandlerOptions{
		Features:              opts.Features,
		NetworkExposed:        networkExposed,
		DeploymentMode:        mode,
		AdminToken:            opts.AdminToken,
		Defaults:              opts.Defaults,
		MaxConcurrentRequests: opts.MaxConcurrentConnections,
		ChatUI:                opts.ChatUI,
		ChatHistoryPath:       opts.ChatHistoryPath,
		ModelDir:              opts.ModelDir,
		ModelPath:             opts.ModelPath,
		WasmDir:               opts.WasmDir,
		ModelLoadOptions:      opts.ModelLoadOptions,
		ModelLoaded:           opts.ModelLoaded,
		SkillsDir:             opts.SkillsDir,
		AppliedAutoTune:       opts.AppliedAutoTune,
		BaselineRuntimeTuning: opts.BaselineRuntimeTuning,
		LogWriter:             logw,
		AgentOS:               opts.AgentOS,
		OSMSearchURL:          opts.OSMSearchURL,
	})
	defer handler.Close()
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	server := &http.Server{
		Addr:              opts.Addr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
		BaseContext:       func(net.Listener) context.Context { return ctx },
	}
	fmt.Fprintf(logw, "Serving on %s\n", displayServerURL(opts.Addr, opts.ChatUI))
	if enabled := opts.Features.EnabledNames(); len(enabled) > 0 {
		fmt.Fprintf(logw, "Optional features: %s\n", strings.Join(enabled, ", "))
	} else {
		fmt.Fprintln(logw, "Optional features: none (chat and completions only; see --enable)")
	}
	if warning := NetworkExposureWarning(mode, opts.Addr, opts.AdminToken); warning != "" {
		fmt.Fprint(logw, warning)
	}
	shutdownDone := make(chan struct{})
	shutdownFinished := make(chan struct{})
	go func() {
		defer close(shutdownFinished)
		select {
		case <-ctx.Done():
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := server.Shutdown(shutdownCtx); err != nil {
				_ = server.Close()
			}
		case <-shutdownDone:
		}
	}()
	err = server.ListenAndServe()
	close(shutdownDone)
	if errors.Is(err, http.ErrServerClosed) && ctx.Err() != nil {
		<-shutdownFinished
		return nil
	}
	return err
}

// HandlerForModel serves a Model opened through the high-level API. It is the
// replacement for the former gopherllm.Model.HTTPHandler method, which had to
// go when HTTP serving moved out of the inference package: the handler shares
// the Model's underlying Runner, so requests serialize with direct Model calls.
func HandlerForModel(m *gopherllm.Model, opts HandlerOptions) *Handler {
	return NewHandler(m.Runner(), opts)
}

// NewHandler returns the complete GopherLLM HTTP API (OpenAI-compatible,
// Ollama-compatible, and native endpoints — see the README's endpoint table)
// as a mountable http.Handler. It owns no listener, manages runners installed
// by model hot-swaps, and writes nothing except to opts.LogWriter, so it
// composes with any router, middleware stack, or server the host application
// already has. After stopping that server, call Handler.Close to release the
// active runners and their memory-mapped GGUF files.
func NewHandler(initialRunner *gopherllm.Runner, opts HandlerOptions) *Handler {
	logw := opts.LogWriter
	if logw == nil {
		logw = io.Discard
	}
	if _, err := ParseDeploymentMode(string(opts.DeploymentMode)); err != nil {
		// NewHandler predates error returns. Keep this direct-library path
		// fail-closed: newDeploymentAccess turns an invalid mode into managed
		// mode without a usable token, so privileged routes stay unavailable.
		fmt.Fprintf(logw, "Warning: invalid deployment mode: %v; privileged routes are disabled\n", err)
	}
	deployment := newDeploymentAccess(opts.DeploymentMode, opts.AdminToken, opts.NetworkExposed)
	if deployment.mode != DeploymentLocal && strings.TrimSpace(opts.ChatHistoryPath) != "" {
		// A single shared history file has no user identity boundary. Managed
		// and browser deployments therefore keep workspaces in each browser,
		// rather than letting one user read or replace another user's chats.
		fmt.Fprintln(logw, "Chat history: server-side shared storage is disabled outside local deployment")
		opts.ChatHistoryPath = ""
	}
	if deployment.mode.browserOnly() {
		// Browser deployment is an on-device inference profile, not a thin UI
		// in front of an accidentally resident server model. The route policy
		// below blocks server inference too; clearing the catalog prevents the
		// UI from discovering or changing server-side model state.
		if initialRunner != nil {
			// NewHandler cannot return a configuration error. Release the runner
			// rather than retaining model weights that this deployment promises
			// never to execute on the server. Serve rejects this configuration
			// before it gets here; this is the fail-closed library fallback.
			_ = initialRunner.Close()
			initialRunner = nil
			fmt.Fprintln(logw, "Browser deployment: released the supplied server-side model runner")
		}
		opts.ModelDir = ""
		opts.ModelPath = ""
		opts.AgentOS = nil
	}
	opts.ModelDir = strings.TrimSpace(opts.ModelDir)
	if opts.MaxConcurrentRequests <= 0 {
		opts.MaxConcurrentRequests = 8
	}
	skills, err := gopherllm.LoadSkills(opts.SkillsDir)
	if err != nil {
		fmt.Fprintf(logw, "Warning: skills: %v (continuing without skills)\n", err)
	} else if len(skills) > 0 {
		names := make([]string, len(skills))
		for i, s := range skills {
			names[i] = s.Name
		}
		fmt.Fprintf(logw, "Skills: loaded %d (%s)\n", len(skills), strings.Join(names, ", "))
	}
	wikimediaTools := NewResearchTools(ResearchOptions{Wikimedia: true})
	osmTools := NewResearchTools(ResearchOptions{OpenStreetMap: true, OSMSearchURL: opts.OSMSearchURL})
	skillsFor := func(enabled bool) []gopherllm.Skill {
		if !enabled {
			return nil
		}
		return skills
	}
	agenticToolsFor := func(wikimedia, openStreetMap bool) []gopherllm.AgenticTool {
		// A request asking for a lookup tool the operator did not enable gets
		// a normal answer without it, not an error: the flags are hints from
		// the client, and the server decides what exists.
		if !opts.Features.WebLookup {
			return nil
		}
		var tools []gopherllm.AgenticTool
		if wikimedia {
			tools = append(tools, wikimediaTools...)
		}
		if openStreetMap {
			tools = append(tools, osmTools...)
		}
		return tools
	}
	baseline := gopherllm.CaptureRuntimeTuning()
	if opts.BaselineRuntimeTuning != nil {
		baseline = *opts.BaselineRuntimeTuning
	}
	state := &runnerState{r: initialRunner, path: opts.ModelPath, baseline: baseline, autoTune: opts.AppliedAutoTune}
	embedder := &embeddingState{}
	history := newChatHistoryStore(opts.ChatHistoryPath)
	remote := newRemoteState()
	sem := make(chan struct{}, opts.MaxConcurrentRequests)
	// Serializes replacement plus its host callback. This prevents two nearly
	// simultaneous hot-swaps from recording the models out of their actual
	// swap order (for example, in a host that persists the active model).
	var modelLoadMu sync.Mutex
	mux := http.NewServeMux()

	// Optional capabilities are not registered unless the host asked for them,
	// so a disabled one answers 404 rather than presenting a permission check.
	registerSystemRoutes(mux, state, deployment, remote, history, opts.Features)
	registerChatWorkspaceRoutes(mux, history)
	registerOpenAIRoutes(mux, state, embedder, sem, opts, skills, skillsFor, agenticToolsFor, logw)
	registerOllamaRoutes(mux, state, embedder, sem, opts, skills, agenticToolsFor, logw)
	registerModelRoutes(mux, state, embedder, sem, opts, deployment, &modelLoadMu, logw)
	if opts.Features.AutoTune {
		registerAutoTuneRoutes(mux, state, sem, logw)
	}
	registerAgentOSRoutes(mux, state, sem, opts)
	if opts.ChatUI {
		registerChatUIRoutes(mux, state, opts, deployment, logw)
	}

	return &Handler{
		next:     deployment.wrap(remoteOrLoadedModel(state, remote, mux)),
		state:    state,
		embedder: embedder,
	}
}

func displayServerURL(addr string, chatUI bool) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		host = strings.Trim(addr, "[]")
		port = ""
	}
	if host == "" || host == "::" || host == "0.0.0.0" || host == "[::]" {
		host = "localhost"
	}
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") {
		host = "[" + host + "]"
	}
	url := "http://" + host
	if port != "" {
		url += ":" + port
	}
	if chatUI {
		url += "/chat"
	}
	return url
}
