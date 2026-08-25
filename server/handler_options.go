package server

import (
	"context"
	"io"
	"sync"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/agentos"
)

// HandlerOptions configures the mountable HTTP API handler.
type HandlerOptions struct {
	// DeploymentMode selects the ownership boundary for inference and shared
	// server settings. The zero value is DeploymentLocal for backward
	// compatibility. See DeploymentMode for the semantics of local, managed,
	// and browser deployments.
	DeploymentMode DeploymentMode
	// AdminToken protects server-wide management routes in managed mode. It is
	// never returned by an endpoint or inserted into the Web UI. API clients may
	// send it through X-GopherLLM-Admin-Token or Authorization: Bearer.
	AdminToken string
	// Defaults are the generation settings requests inherit unless they
	// override individual fields.
	Defaults gopherllm.GenerationOptions
	// MaxConcurrentRequests bounds in-flight generation requests (default 8).
	// Requests beyond the bound queue rather than failing.
	MaxConcurrentRequests int
	// ChatUI serves the embedded browser chat at /chat (plus its assets).
	ChatUI bool
	// ChatHistoryPath enables the opt-in server-side browser workspace. The
	// history is compressed and atomically replaced at this path; an empty path
	// keeps browser-local IndexedDB/localStorage as the default.
	ChatHistoryPath string
	// ModelDir enables GET /models discovery and POST /models/load hot-swap.
	// A load request is resolved against this directory's discovered,
	// supported GGUF files; arbitrary filesystem paths are never loaded.
	ModelDir string
	// ModelPath is the initially loaded model's path (reported by /models).
	ModelPath string
	// WasmDir, if it contains both gopherllm.wasm and wasm_exec.js (see
	// `make wasm-build`), enables the chat UI's "run locally in this
	// browser tab" mode: those two files are served at /wasm/gopherllm.wasm
	// and /wasm/wasm_exec.js, and chatTemplateData.HasLocalRuntime is set so
	// the page can offer the toggle. Either file missing (or WasmDir unset)
	// means that mode simply isn't offered — the rest of the server is
	// unaffected either way.
	WasmDir string
	// ModelLoadOptions are retained for catalog hot-swaps, so a server started
	// in out-of-core mode does not accidentally load the next model eagerly.
	ModelLoadOptions gopherllm.LoadOptions
	// ModelLoaded is called after a local model has been successfully loaded or
	// hot-swapped. It is useful for hosts that persist the active selection.
	// Callback failures are the host's responsibility and never undo a load.
	ModelLoaded func(path string)
	// SkillsDir, if set, is scanned once at handler construction for SKILL.md
	// files (see skills.go). Every chat/generate endpoint offers a load_skill
	// tool and resolves it server-side via gopherllm.RunAgenticChat.
	SkillsDir string
	// AppliedAutoTune, if set, is a tuning already applied to initialRunner
	// before the handler was built (e.g. by the CLI's --auto flag). GET
	// /autotune reports it as active from the very first request, rather than
	// only after someone hits POST /autotune/run through the web UI.
	AppliedAutoTune *gopherllm.AutoTuneResult
	// BaselineRuntimeTuning is the process-wide configuration to restore after
	// a hot-swap to an uncalibrated model. When absent, NewHandler captures the
	// settings visible at construction time.
	BaselineRuntimeTuning *gopherllm.RuntimeTuning
	// LogWriter receives handler diagnostics (skill load notes). Defaults to
	// io.Discard.
	LogWriter io.Writer
	// AgentOS enables the agentic OS-command feature (a model proposes a local
	// shell command, this Runner's Policy decides whether it needs a human
	// click before it runs) when non-nil. Nil, the default, registers no
	// /agentos endpoints at all — the feature does not exist on this server
	// unless an operator deliberately configured a policy for it. See the
	// agentos package for the safety model.
	AgentOS *agentos.Runner
	// OSMSearchURL optionally replaces the public Nominatim endpoint with an
	// operator-managed compatible endpoint. The source remains disabled until
	// a request explicitly sets gopherllm_openstreetmap to true.
	OSMSearchURL string
}

// ServeOptions is HandlerOptions plus the listen address, for the Serve
// convenience wrapper (used by the CLI). ChatHistoryLock remains for source
// compatibility with older hosts; the handler serializes its own file access.
type ServeOptions struct {
	// Context controls the lifetime of the listener. Cancelling it gracefully
	// stops accepting requests and releases the active runner.
	Context                  context.Context
	Addr                     string
	DeploymentMode           DeploymentMode
	AdminToken               string
	Defaults                 gopherllm.GenerationOptions
	MaxConcurrentConnections int
	ChatUI                   bool
	ChatHistoryPath          string
	ChatHistoryLock          *sync.Mutex
	ModelDir                 string
	ModelPath                string
	// WasmDir is forwarded to HandlerOptions.WasmDir.
	WasmDir          string
	ModelLoadOptions gopherllm.LoadOptions
	ModelLoaded      func(path string)
	SkillsDir        string
	// AppliedAutoTune carries forward a tuning already applied before Serve
	// was called (e.g. by --auto), so GET /autotune reports it from the start.
	AppliedAutoTune *gopherllm.AutoTuneResult
	// BaselineRuntimeTuning forwards the pre-auto runtime settings captured by
	// a host such as the CLI, so a later model hot-swap can restore them.
	BaselineRuntimeTuning *gopherllm.RuntimeTuning
	// LogWriter receives startup and handler diagnostics; Serve defaults it
	// to os.Stderr (CLI behavior), unlike NewHandler's io.Discard.
	LogWriter io.Writer
	// AgentOS enables the agentic OS-command feature; see HandlerOptions.AgentOS.
	AgentOS *agentos.Runner
	// OSMSearchURL is forwarded to HandlerOptions.OSMSearchURL.
	OSMSearchURL string
}
