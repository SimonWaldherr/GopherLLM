package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/huggingface"
)

const (
	hfSearchTimeout     = 10 * time.Second
	hfSearchConcurrency = 2
)

// modelLoadRequest accepts the model catalog ID used by the browser UI and
// API clients. Path remains for compatibility with older clients, but is
// treated only as a selector for a model already discovered in ModelDir.
type modelLoadRequest struct {
	Model string `json:"model"`
	ID    string `json:"id"`
	Path  string `json:"path"`
}

func (r modelLoadRequest) selector() string {
	for _, value := range []string{r.Model, r.ID, r.Path} {
		if selector := strings.TrimSpace(value); selector != "" {
			return selector
		}
	}
	return ""
}

// modelSearchRequest is deliberately query-string-only: the browser can
// cancel a GET while a user keeps typing, and the request does not carry any
// sensitive model-download credential or arbitrary remote URL.
type modelSearchRequest struct {
	Query string
	Limit int
}

func parseModelSearchRequest(req *http.Request) (modelSearchRequest, error) {
	query := strings.TrimSpace(req.URL.Query().Get("q"))
	if query == "" {
		return modelSearchRequest{}, errors.New("missing q query parameter")
	}
	limit := huggingface.DefaultSearchLimit
	if raw := strings.TrimSpace(req.URL.Query().Get("limit")); raw != "" {
		value, err := strconv.Atoi(raw)
		if err != nil || value < 1 || value > huggingface.MaxSearchLimit {
			return modelSearchRequest{}, fmt.Errorf("limit must be between 1 and %d", huggingface.MaxSearchLimit)
		}
		limit = value
	}
	return modelSearchRequest{Query: query, Limit: limit}, nil
}

// hfRepoDirName turns a Hugging Face "owner/repository" reference into a
// single path-safe directory component, mirroring the flattening
// internal/huggingface already applies to its own on-disk cache layout
// (newHFCache). filepath.Join-ing an unflattened repository string would let
// a "/"-containing (or, worse, ".."-containing) value escape opts.ModelDir;
// collapsing every separator into "--" first guarantees the result is always
// exactly one path component appended under ModelDir.
func hfRepoDirName(repository string) string {
	name := strings.ReplaceAll(repository, "\\", "--")
	name = strings.ReplaceAll(name, "/", "--")
	if name == "" || name == "." || name == ".." {
		name = "hf-model"
	}
	return name
}

// registerModelRoutes registers the model-catalog HTTP surface: /models,
// /models/architecture, /models/load, /models/embed/load, /models/embed,
// /models/download/variants, /models/search, and /models/download.
// Extracted from NewHandler's inline handlers for these routes. It creates
// its own hfSearchSem/downloadMu/activeDownloads locally since those guard
// only routes registered here; modelLoadMu is shared with the caller because
// it also serializes NewHandler's own construction-time model swap bookkeeping.
// registerModelRoutes registers the model catalog and, separately, the
// Hugging Face download routes. They are split because they differ in kind:
// the catalog only reads and hot-swaps GGUFs the operator already placed in
// ModelDir, while the download routes make outbound requests and write new
// files. Features decides which of the two exists.
func registerModelRoutes(mux *http.ServeMux, state *runnerState, embedder *embeddingState, sem chan struct{}, opts HandlerOptions, deployment deploymentAccess, modelLoadMu *sync.Mutex, logw io.Writer) {
	if opts.Features.ModelCatalog {
		registerModelCatalogRoutes(mux, state, embedder, sem, opts, deployment, modelLoadMu, logw)
	}
	if opts.Features.ModelDownload {
		registerModelDownloadRoutes(mux, state, opts, logw)
	}
}

func registerModelCatalogRoutes(mux *http.ServeMux, state *runnerState, embedder *embeddingState, sem chan struct{}, opts HandlerOptions, deployment deploymentAccess, modelLoadMu *sync.Mutex, logw io.Writer) {
	mux.HandleFunc("/models", func(w http.ResponseWriter, req *http.Request) {
		type modelInfo struct {
			ID            string  `json:"id"`
			Name          string  `json:"name"`
			Path          string  `json:"path,omitempty"`
			Architecture  string  `json:"architecture"`
			ContextLength int     `json:"context_length"`
			SizeGB        float64 `json:"size_gb"`
			Supported     bool    `json:"supported"`
			Loaded        bool    `json:"loaded"`
			Embedding     bool    `json:"embedding"`
			Vision        bool    `json:"vision"`
			Reasoning     bool    `json:"reasoning"`
		}
		if opts.ModelDir == "" {
			writeJSON(w, map[string]any{"models": []modelInfo{}})
			return
		}
		entries, err := discoverCatalog(opts.ModelDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		loadedPath := state.getPath()
		includePaths := deployment.mode == DeploymentLocal || deployment.adminAuthorized(req)
		models := make([]modelInfo, 0, len(entries))
		for _, e := range entries {
			if e.IsProjector {
				continue
			}
			name := e.ModelName
			if name == "" {
				name = e.FileName
			}
			info := modelInfo{
				ID:            e.ID,
				Name:          name,
				Architecture:  e.Architecture,
				ContextLength: e.ContextLength,
				SizeGB:        float64(e.SizeBytes) / (1024 * 1024 * 1024),
				Supported:     e.IsSupported,
				Loaded:        e.Path == loadedPath,
				Embedding:     e.IsEmbedding && e.IsSupported && !e.IsProjector,
				Vision:        e.ProjectorPath != "",
				Reasoning:     e.Reasoning,
			}
			if includePaths {
				info.Path = e.Path
			}
			models = append(models, info)
		}
		writeJSON(w, map[string]any{"models": models})
	})
	mux.HandleFunc("/models/architecture", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		requested := strings.TrimSpace(req.URL.Query().Get("model"))
		if requested == "" {
			requested = strings.TrimSpace(req.URL.Query().Get("id"))
		}
		g, ok := resolveArchGraph(state, opts.ModelDir, requested)
		if !ok {
			http.Error(w, "model not found", http.StatusNotFound)
			return
		}
		writeJSON(w, g)
	})
	mux.HandleFunc("/models/load", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.ModelDir == "" {
			http.Error(w, "model hot-swap is disabled: configure HandlerOptions.ModelDir", http.StatusNotFound)
			return
		}
		var body modelLoadRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		selector := body.selector()
		if selector == "" {
			http.Error(w, "missing model selector", http.StatusBadRequest)
			return
		}
		entries, err := discoverCatalog(opts.ModelDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		entry, err := gopherllm.SelectModel(entries, selector)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		loadOptions := opts.ModelLoadOptions
		if loadOptions.VisionProjectorPath == "" && len(loadOptions.VisionProjectorBytes) == 0 && entry.ProjectorPath != "" {
			loadOptions.VisionProjectorPath = entry.ProjectorPath
		}
		newRunner, _, err := gopherllm.RunnerFromPathWithOptions(entry.Path, loadOptions)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		func() {
			modelLoadMu.Lock()
			defer modelLoadMu.Unlock()
			state.swap(newRunner, entry.Path)
			if opts.ModelLoaded != nil {
				opts.ModelLoaded(entry.Path)
			}
		}()
		writeJSON(w, map[string]any{"ok": true, "id": entry.ID, "model": modelID(newRunner), "context_length": newRunner.Config().MaxSeqLen, "out_of_core": newRunner.OutOfCore(), "vision": newRunner.HasVision()})
	}))
	mux.HandleFunc("/models/embed/load", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.ModelDir == "" {
			http.Error(w, "embedding-model loading is disabled: configure HandlerOptions.ModelDir", http.StatusNotFound)
			return
		}
		var body modelLoadRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		entries, err := discoverCatalog(opts.ModelDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		entry, err := gopherllm.SelectModel(entries, body.selector())
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if !entry.IsEmbedding {
			http.Error(w, "selected model is not an embedding model", http.StatusBadRequest)
			return
		}
		runner, _, err := gopherllm.RunnerFromPathWithOptions(entry.Path, opts.ModelLoadOptions)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		embedder.swap(runner)
		writeJSON(w, map[string]any{"ok": true, "id": entry.ID, "model": modelID(runner), "context_length": runner.Config().MaxSeqLen, "out_of_core": runner.OutOfCore()})
	}))
	mux.HandleFunc("/models/embed", withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(body.Input) == 0 || len(body.Input) > 256 {
			http.Error(w, "input must contain between 1 and 256 texts", http.StatusBadRequest)
			return
		}
		var vectors [][]float32
		var model string
		var embedErr error
		embedder.withRunner(func(runner *gopherllm.Runner) {
			model = modelID(runner)
			vectors, _, embedErr = embedTexts(runner, body.Input)
		})
		if embedErr != nil {
			status := http.StatusBadRequest
			if embedErr.Error() == "no embedding model is loaded" {
				status = http.StatusConflict
			}
			http.Error(w, embedErr.Error(), status)
			return
		}
		writeJSON(w, map[string]any{"model": model, "embeddings": vectors})
	}))
}

func registerModelDownloadRoutes(mux *http.ServeMux, state *runnerState, opts HandlerOptions, logw io.Writer) {
	// Model search has its own tiny outbound-request budget so repeated
	// type-ahead queries cannot consume inference capacity or fan out into an
	// unbounded number of Hub requests.
	hfSearchSem := make(chan struct{}, hfSearchConcurrency)
	// Guards against two requests downloading the same Hugging Face reference
	// concurrently, which would otherwise let two goroutines append to the
	// same .incomplete blob at once and corrupt it. Keyed by the exact
	// normalized ref string, not a distributed lock: it only protects against
	// this one server process racing itself (e.g. a doubled click).
	var downloadMu sync.Mutex
	activeDownloads := map[string]struct{}{}

	mux.HandleFunc("/models/download/variants", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.ModelDir == "" {
			http.Error(w, "model download is disabled: configure HandlerOptions.ModelDir", http.StatusNotFound)
			return
		}
		ref := strings.TrimSpace(req.URL.Query().Get("ref"))
		if ref == "" {
			http.Error(w, "missing ref query parameter", http.StatusBadRequest)
			return
		}
		info, err := huggingface.RepositoryVariants(req.Context(), ref, huggingface.Options{})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		variants := make([]map[string]any, len(info.Variants))
		for i, v := range info.Variants {
			variants[i] = map[string]any{"quant": v.Quant, "size_bytes": v.SizeBytes, "shards": v.Shards, "selector": v.Selector}
		}
		writeJSON(w, map[string]any{"repository": info.Repository, "revision": info.Revision, "variants": variants})
	})
	mux.HandleFunc("/models/search", withLimit(hfSearchSem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.ModelDir == "" {
			http.Error(w, "model download is disabled: configure HandlerOptions.ModelDir", http.StatusNotFound)
			return
		}
		search, err := parseModelSearchRequest(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		ctx, cancel := context.WithTimeout(req.Context(), hfSearchTimeout)
		defer cancel()
		models, err := huggingface.SearchGGUFRepositories(ctx, search.Query, search.Limit, huggingface.DefaultOptions())
		if err != nil {
			if errors.Is(err, context.Canceled) && req.Context().Err() != nil {
				// The browser aborted a type-ahead request. There is no client left
				// to receive an error, and the request context already stopped the
				// outbound Hub operation.
				return
			}
			status := http.StatusBadGateway
			if errors.Is(err, context.DeadlineExceeded) {
				status = http.StatusGatewayTimeout
			}
			http.Error(w, err.Error(), status)
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		writeJSON(w, map[string]any{"models": models})
	}))
	mux.HandleFunc("/models/download", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if opts.ModelDir == "" {
			http.Error(w, "model download is disabled: configure HandlerOptions.ModelDir", http.StatusNotFound)
			return
		}
		var body struct {
			Ref string `json:"ref"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		ref := strings.TrimSpace(body.Ref)
		if ref == "" {
			http.Error(w, "missing model reference", http.StatusBadRequest)
			return
		}
		if !strings.HasPrefix(strings.ToLower(ref), "hf:") {
			ref = "hf:" + ref
		}
		repository, err := huggingface.Repository(ref)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		downloadMu.Lock()
		if _, busy := activeDownloads[ref]; busy {
			downloadMu.Unlock()
			http.Error(w, "this model is already downloading", http.StatusConflict)
			return
		}
		activeDownloads[ref] = struct{}{}
		downloadMu.Unlock()
		defer func() {
			downloadMu.Lock()
			delete(activeDownloads, ref)
			downloadMu.Unlock()
		}()
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("content-type", "application/x-ndjson")
		w.Header().Set("cache-control", "no-store")
		send := func(event map[string]any) { _ = writeNDJSON(w, flusher, event) }
		send(map[string]any{"status": "resolving", "ref": ref})
		files, err := huggingface.ResolveHuggingFaceModelFilesContextWithOptions(req.Context(), ref, io.Discard, huggingface.Options{
			OnProgress: func(ev huggingface.ProgressEvent) {
				if ev.Err != nil {
					return
				}
				send(map[string]any{"status": "downloading", "file": ev.File, "completed": ev.Completed, "total": ev.Total})
			},
		})
		if err != nil {
			send(map[string]any{"status": "error", "error": err.Error()})
			return
		}
		send(map[string]any{"status": "placing"})
		destDir := filepath.Join(opts.ModelDir, hfRepoDirName(repository))
		placed, err := huggingface.PlaceGGUFFiles(files, destDir)
		if err != nil {
			send(map[string]any{"status": "error", "error": err.Error()})
			return
		}
		id := strings.TrimSuffix(filepath.Base(placed[0]), ".gguf")
		if rel, relErr := filepath.Rel(opts.ModelDir, placed[0]); relErr == nil {
			id = strings.TrimSuffix(rel, ".gguf")
		}
		send(map[string]any{"status": "success", "id": id, "path": placed[0], "file": filepath.Base(placed[0])})
	})
}

// discoverCatalog lists the GGUFs under dir. A directory that does not exist
// is an empty catalog rather than an error: the CLI ships a default model
// directory, and someone who has not created it yet should see "no models
// found" in the Web UI instead of a 500 on their first page load.
func discoverCatalog(dir string) ([]gopherllm.ModelEntry, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, nil
	}
	entries, err := gopherllm.DiscoverModels(dir, io.Discard)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	return entries, err
}
