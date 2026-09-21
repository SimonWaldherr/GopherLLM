package server

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/SimonWaldherr/GopherLLM/voiceweb"
)

//go:embed web_ui/chat.html
var chatHTMLTmpl string

//go:embed web_ui/style.css
var chatCSS string

//go:embed web_ui/script.js
var chatJS string

//go:embed web_ui/audio.js
var audioJS string

// audioWorkletJS is the shared voiceweb.AudioWorkletJS, not a second copy
// of the source -- see that package's doc comment for why only the
// capture/resample worklet (not this page's own audio.js orchestration)
// is factored out.
var audioWorkletJS = voiceweb.AudioWorkletJS

//go:embed web_ui/wasm-bridge.js
var wasmBridgeJS string

var chatTemplate = template.Must(template.New("chat").Parse(chatHTMLTmpl))

// gzipBytes compresses a static asset once at process startup rather than on
// every request. setChatUIHeaders below deliberately keeps Cache-Control at
// no-store (so a local rebuild is always visible on the next reload without
// a hard refresh), which means every request re-transfers the full asset --
// script.js alone is ~340KB. Since it can't be cached, compressing it is the
// next best thing, and these strings are immutable for the process lifetime
// so compressing once up front costs nothing per request.
func gzipBytes(s string) []byte {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	_, _ = io.WriteString(gz, s)
	_ = gz.Close()
	return buf.Bytes()
}

var (
	chatCSSGZ        = gzipBytes(chatCSS)
	chatJSGZ         = gzipBytes(chatJS)
	audioJSGZ        = gzipBytes(audioJS)
	audioWorkletJSGZ = gzipBytes(audioWorkletJS)
	wasmBridgeJSGZ   = gzipBytes(wasmBridgeJS)
)

// acceptsGzip reports whether the client's Accept-Encoding header lists
// gzip -- every browser that would load this chat UI does, but a curl/API
// client asking for the raw asset should still get an uncompressed body.
func acceptsGzip(req *http.Request) bool {
	for _, part := range strings.Split(req.Header.Get("Accept-Encoding"), ",") {
		if strings.EqualFold(strings.TrimSpace(strings.SplitN(part, ";", 2)[0]), "gzip") {
			return true
		}
	}
	return false
}

// writeStaticAsset serves a precompressed static asset, falling back to the
// plain string for a client that didn't ask for gzip.
func writeStaticAsset(w http.ResponseWriter, req *http.Request, plain string, gz []byte) {
	if acceptsGzip(req) {
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Set("Vary", "Accept-Encoding")
		_, _ = w.Write(gz)
		return
	}
	_, _ = io.WriteString(w, plain)
}

type chatTemplateData struct {
	Title string
	Model string
	// HasModel is false exactly when Model has already been replaced with
	// the "No model selected" placeholder below — the template uses it to
	// render the first-contact empty state honestly instead of showing the
	// same "ready to chat" welcome screen whether or not a model is loaded.
	HasModel      bool
	MaxTokens     int
	Temperature   float32
	TopP          float32
	TopK          int
	MinP          float32
	RepeatPenalty float32
	// MermaidCDN is the validated CDN key ("" when diagrams are off) and
	// MermaidScript the full script URL the page should load.
	MermaidCDN    string
	MermaidScript string
	// HasLocalRuntime reports whether this server was started with a valid
	// WasmDir (see HandlerOptions.WasmDir) — the page only offers the
	// "run locally in this browser tab" inference-mode toggle when true.
	HasLocalRuntime bool
	// Features is the space-separated list of optional capabilities this
	// server actually backs (see Features.EnabledNames). The page removes the
	// panels for everything absent from it, so the UI a beginner opens is the
	// UI this server can honour rather than the full catalogue of options.
	Features string
	// NetworkExposed tells the page that this listener is reachable beyond
	// loopback, so it can say so instead of leaving it to the terminal.
	NetworkExposed bool
	// DeploymentMode describes the server policy without exposing any secret.
	// BrowserOnly forces the UI into its on-device WASM/WebGPU path, while
	// AdminRequired lets a managed deployment keep ordinary chat preferences
	// visible but hide server-wide controls until an administrator authorizes.
	DeploymentMode  string
	BrowserOnly     bool
	AdminRequired   bool
	AdminAuthorized bool
}

// registerChatUIRoutes registers the embedded browser chat UI: the redirect
// at "/", the chat page itself, its static assets, and (when the server was
// given local wasm runtime assets) the /wasm/ routes that let the browser run
// inference on-device. Extracted from NewHandler's opts.ChatUI block.
func registerChatUIRoutes(mux *http.ServeMux, state *runnerState, opts HandlerOptions, deployment deploymentAccess, logw io.Writer) {
	// wasmPath/wasmExecPath point at fixed filenames under opts.WasmDir
	// (never request-derived), so serving them directly is not a path-
	// traversal risk. hasLocalRuntime gates both the /wasm/ routes below
	// and the chat page's advertised HasLocalRuntime/CSP -- a WasmDir
	// missing either file behaves exactly like an unset WasmDir.
	wasmPath := filepath.Join(opts.WasmDir, "gopherllm.wasm")
	wasmExecPath := filepath.Join(opts.WasmDir, "wasm_exec.js")
	hasLocalRuntime := opts.WasmDir != "" && fileReadable(wasmPath) && fileReadable(wasmExecPath)
	if hasLocalRuntime {
		mux.HandleFunc("/wasm/gopherllm.wasm", func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set("Content-Type", "application/wasm")
			// The filename remains stable across local builds, so require a
			// revalidation rather than serving stale bytes. Unlike no-store,
			// this lets the browser retain the downloaded and compiled module
			// when it has not changed.
			w.Header().Set("Cache-Control", "private, max-age=0, must-revalidate")
			http.ServeFile(w, req, wasmPath)
		})
		mux.HandleFunc("/wasm/wasm_exec.js", func(w http.ResponseWriter, req *http.Request) {
			w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
			w.Header().Set("Cache-Control", "private, max-age=0, must-revalidate")
			http.ServeFile(w, req, wasmExecPath)
		})
		fmt.Fprintf(logw, "Local browser inference: serving %s at /wasm/\n", opts.WasmDir)
	}
	mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/" {
			http.NotFound(w, req)
			return
		}
		http.Redirect(w, req, "/chat", http.StatusFound)
	})
	mux.HandleFunc("/chat", func(w http.ResponseWriter, req *http.Request) {
		// The diagram renderer is chosen in the browser's Settings, but the
		// CSP that permits it is a response header, so the choice travels
		// back as a query parameter and is validated against the map above.
		mermaid := mermaidChoice(req.URL.Query().Get("mermaid"))
		setChatUIHeaders(w, mermaid, hasLocalRuntime)
		w.Header().Set("content-type", "text/html; charset=utf-8")
		model := modelID(state.get())
		hasModel := model != ""
		if !hasModel {
			model = "No model selected"
		}
		data := chatTemplateData{
			Title:           "GopherLLM Chat",
			Model:           model,
			HasModel:        hasModel,
			MaxTokens:       opts.Defaults.MaxTokens,
			Temperature:     opts.Defaults.Sampler.Temperature,
			TopP:            opts.Defaults.Sampler.TopP,
			TopK:            opts.Defaults.Sampler.TopK,
			MinP:            opts.Defaults.Sampler.MinP,
			RepeatPenalty:   opts.Defaults.Sampler.RepeatPenalty,
			MermaidCDN:      mermaid,
			HasLocalRuntime: hasLocalRuntime,
			Features:        strings.Join(opts.Features.EnabledNames(), " "),
			NetworkExposed:  opts.NetworkExposed,
			DeploymentMode:  string(deployment.mode),
			BrowserOnly:     deployment.mode.browserOnly(),
			AdminRequired:   deployment.mode.adminRequired(),
			AdminAuthorized: deployment.adminAuthorized(req),
		}
		if cdn, ok := mermaidCDNs[mermaid]; ok {
			data.MermaidScript = cdn.Script
		}
		var page bytes.Buffer
		if err := chatTemplate.Execute(&page, data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Rendered per request (model name, settings, feature flags), so
		// unlike the static assets above this can't be gzipped once at
		// startup -- still worth compressing on the fly since Cache-Control
		// stays no-store and this is the single largest response most page
		// loads make.
		if acceptsGzip(req) {
			w.Header().Set("Content-Encoding", "gzip")
			w.Header().Set("Vary", "Accept-Encoding")
			gz := gzip.NewWriter(w)
			if _, err := gz.Write(page.Bytes()); err != nil {
				fmt.Fprintf(logw, "Warning: write chat page: %v\n", err)
			}
			if err := gz.Close(); err != nil {
				fmt.Fprintf(logw, "Warning: close chat page gzip writer: %v\n", err)
			}
			return
		}
		if _, err := w.Write(page.Bytes()); err != nil {
			fmt.Fprintf(logw, "Warning: write chat page: %v\n", err)
		}
	})
	mux.HandleFunc("/style.css", func(w http.ResponseWriter, req *http.Request) {
		setChatUIHeaders(w, "", hasLocalRuntime)
		w.Header().Set("content-type", "text/css; charset=utf-8")
		writeStaticAsset(w, req, chatCSS, chatCSSGZ)
	})
	mux.HandleFunc("/audio.js", func(w http.ResponseWriter, req *http.Request) {
		setChatUIHeaders(w, "", hasLocalRuntime)
		w.Header().Set("content-type", "text/javascript; charset=utf-8")
		writeStaticAsset(w, req, audioJS, audioJSGZ)
	})
	mux.HandleFunc("/audio-worklet.js", func(w http.ResponseWriter, req *http.Request) {
		setChatUIHeaders(w, "", hasLocalRuntime)
		w.Header().Set("content-type", "text/javascript; charset=utf-8")
		writeStaticAsset(w, req, audioWorkletJS, audioWorkletJSGZ)
	})
	mux.HandleFunc("/script.js", func(w http.ResponseWriter, req *http.Request) {
		setChatUIHeaders(w, "", hasLocalRuntime)
		w.Header().Set("content-type", "text/javascript; charset=utf-8")
		writeStaticAsset(w, req, chatJS, chatJSGZ)
	})
	if hasLocalRuntime {
		mux.HandleFunc("/wasm-bridge.js", func(w http.ResponseWriter, req *http.Request) {
			setChatUIHeaders(w, "", hasLocalRuntime)
			w.Header().Set("content-type", "text/javascript; charset=utf-8")
			writeStaticAsset(w, req, wasmBridgeJS, wasmBridgeJSGZ)
		})
	}
}

// setChatUIHeaders keeps the local browser workspace private to this origin:
// chat HTML and its assets are never cached and cannot load third-party code.
// mermaidCDNs are the only origins the chat page may load a diagram renderer
// from. Mermaid is ~2.8 MB, so embedding it would inflate every binary for a
// feature most sessions never use; loading it from a CDN is the alternative,
// and that means punching a hole in the CSP. The hole is kept as small as
// possible: one operator-chosen origin, named here rather than assembled from
// user input, and absent entirely unless a choice was made.
var mermaidCDNs = map[string]struct {
	Origin string
	Script string
}{
	"jsdelivr": {"https://cdn.jsdelivr.net", "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"},
	"unpkg":    {"https://unpkg.com", "https://unpkg.com/mermaid@11/dist/mermaid.min.js"},
	"cdnjs":    {"https://cdnjs.cloudflare.com", "https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.4.1/mermaid.min.js"},
}

// mermaidChoice validates a requested CDN key, returning "" for "no diagrams".
func mermaidChoice(raw string) string {
	key := strings.ToLower(strings.TrimSpace(raw))
	if _, ok := mermaidCDNs[key]; ok {
		return key
	}
	return ""
}

func setChatUIHeaders(w http.ResponseWriter, mermaidCDN string, allowWasm bool) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Referrer-Policy", "same-origin")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	script, style := "'self'", "'self'"
	if cdn, ok := mermaidCDNs[mermaidCDN]; ok {
		script += " " + cdn.Origin
		// Mermaid styles the SVG it builds with inline <style>, so drawing a
		// diagram needs this. It is scoped to the page that opted in: with no
		// CDN chosen the policy stays strict.
		style += " 'unsafe-inline'"
	}
	if allowWasm {
		// Compiling same-origin-fetched WebAssembly bytes needs this even
		// though the page never uses JS eval() — browsers gate
		// WebAssembly.instantiate/compile behind 'wasm-unsafe-eval'
		// specifically (not 'unsafe-eval', which stays absent) once a CSP
		// declares script-src at all. Only added when this server was
		// actually started with local wasm assets to offer.
		script += " 'wasm-unsafe-eval'"
	}
	w.Header().Set("Content-Security-Policy", "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; img-src 'self' data:; script-src "+script+"; style-src "+style)
}

// fileReadable reports whether path exists and is a regular, openable file
// (not a directory) -- used only to decide whether to advertise/serve the
// optional local-wasm-runtime assets, never on request-derived paths.
func fileReadable(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}
