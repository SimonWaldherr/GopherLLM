package server

import (
	"bytes"
	"context"
	"errors"
	"image"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/huggingface"
)

const (
	detectionMaxBytes  = 16 << 20
	detectionMaxPixels = 40_000_000
	// detectionCacheSize bounds how many YOLO networks stay resident. They
	// are small (6-140 MB in memory), and a demo typically alternates between
	// at most a stock model and one custom model.
	detectionCacheSize = 2
)

// detectionModel is one entry of GET /models/detection.
type detectionModel struct {
	ID string `json:"id"`
	// Source is "local" for a checkpoint under the model directory, or
	// "stock" for a pinned Hugging Face checkpoint (see
	// gopherllm.YOLOCheckpointReference).
	Source string `json:"source"`
	// Available is false for a stock model not yet in the HF cache. Such a
	// model is fetched through the admin-controlled /models/download route
	// (DownloadRef/DownloadFile), never implicitly by a detection request.
	Available    bool   `json:"available"`
	DownloadRef  string `json:"download_ref,omitempty"`
	DownloadFile string `json:"download_file,omitempty"`
	path         string
}

// yoloModelCache keeps recently used YOLO networks loaded across requests.
// YOLOModel is safe for concurrent Detect calls, so entries are shared.
type yoloModelCache struct {
	mu     sync.Mutex
	models map[string]*gopherllm.YOLOModel
	order  []string // least recently used first
}

func (c *yoloModelCache) open(path string) (*gopherllm.YOLOModel, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if m, ok := c.models[path]; ok {
		c.touch(path)
		return m, nil
	}
	// Probe before loading: LoadYOLO reads the whole file, which for a
	// multi-gigabyte LLM safetensors elsewhere in the model directory would
	// be an expensive way to find out it is not a detector.
	if !gopherllm.IsYOLOCheckpoint(path) {
		return nil, errors.New("not a YOLO detection checkpoint")
	}
	m, err := gopherllm.LoadYOLO(path)
	if err != nil {
		return nil, err
	}
	if m.ClassNames == nil && m.NumClasses == len(gopherllm.COCOClassNames) {
		m.ClassNames = gopherllm.COCOClassNames
	}
	if c.models == nil {
		c.models = map[string]*gopherllm.YOLOModel{}
	}
	if len(c.order) >= detectionCacheSize {
		delete(c.models, c.order[0])
		c.order = c.order[1:]
	}
	c.models[path] = m
	c.order = append(c.order, path)
	return m, nil
}

func (c *yoloModelCache) touch(path string) {
	for i, p := range c.order {
		if p == path {
			c.order = append(append(c.order[:i:i], c.order[i+1:]...), path)
			return
		}
	}
}

// discoverDetectionModels lists YOLO checkpoints (.safetensors/.pt) under
// dir plus the stock models, resolving stock ones only against the local HF
// cache.
func discoverDetectionModels(ctx context.Context, dir string) ([]detectionModel, error) {
	var models []detectionModel
	if strings.TrimSpace(dir) != "" {
		err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			ext := strings.ToLower(filepath.Ext(path))
			if d.IsDir() || (ext != ".safetensors" && ext != ".pt") {
				return nil
			}
			if _, err := localDetectionPath(dir, path); err != nil {
				return nil
			}
			if !gopherllm.IsYOLOCheckpoint(path) {
				return nil
			}
			id := path
			if rel, err := filepath.Rel(dir, path); err == nil {
				id = filepath.ToSlash(rel)
			}
			models = append(models, detectionModel{ID: id, Source: "local", Available: true, path: path})
			return nil
		})
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, err
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID < models[j].ID })
	for _, name := range gopherllm.YOLOStockModels {
		ref, _ := gopherllm.YOLOCheckpointReference(name)
		repo, file := splitFileReference(ref)
		entry := detectionModel{ID: name, Source: "stock", DownloadRef: repo, DownloadFile: file}
		if paths, err := huggingface.DownloadFiles(ctx, repo, []string{file}, io.Discard, huggingface.Options{Offline: true}); err == nil {
			entry.Available, entry.path = true, paths[0]
		}
		models = append(models, entry)
	}
	return models, nil
}

var errStockNotDownloaded = errors.New("stock detection model is not downloaded yet; fetch it via /models/download with the download_ref and download_file from /models/detection")

// resolveDetectionModel maps an id from /models/detection to a file without
// rescanning the model directory: a stock name resolves against the HF
// cache only, anything else must be a local, non-escaping .safetensors or
// .pt path below dir. Whether that file really is a YOLO checkpoint is
// checked by yoloModelCache.open, once per file rather than per request.
func resolveDetectionModel(ctx context.Context, dir, id string) (string, error) {
	if slices.Contains(gopherllm.YOLOStockModels, id) {
		ref, _ := gopherllm.YOLOCheckpointReference(id)
		repo, file := splitFileReference(ref)
		paths, err := huggingface.DownloadFiles(ctx, repo, []string{file}, io.Discard, huggingface.Options{Offline: true})
		if err != nil {
			return "", errStockNotDownloaded
		}
		return paths[0], nil
	}
	notFound := errors.New("detection model not found; use an id from /models/detection")
	rel := filepath.FromSlash(id)
	ext := strings.ToLower(filepath.Ext(rel))
	if strings.TrimSpace(dir) == "" || !filepath.IsLocal(rel) || (ext != ".safetensors" && ext != ".pt") {
		return "", notFound
	}
	path := filepath.Join(dir, rel)
	path, err := localDetectionPath(dir, path)
	if err != nil {
		return "", notFound
	}
	return path, nil
}

// localDetectionPath follows symlinks and accepts only regular files that
// still reside beneath the configured model directory.
func localDetectionPath(dir, path string) (string, error) {
	root, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || !filepath.IsLocal(rel) {
		return "", fs.ErrPermission
	}
	info, err := os.Stat(resolved)
	if err != nil || !info.Mode().IsRegular() {
		return "", fs.ErrInvalid
	}
	return resolved, nil
}

// splitFileReference turns "hf:owner/repo:file@rev" into the "owner/repo@rev"
// repository reference and file path DownloadFiles expects.
func splitFileReference(ref string) (repo, file string) {
	spec, rev, hasRev := strings.Cut(strings.TrimPrefix(ref, "hf:"), "@")
	repo, file, _ = strings.Cut(spec, ":")
	if hasRev {
		repo += "@" + rev
	}
	return repo, file
}

// registerVisionRoutes wires YOLO object detection:
//
//	GET  /models/detection      list local and stock YOLO checkpoints
//	POST /v1/vision/detections  multipart model, file (JPEG/PNG), optional
//	                            conf, iou, size, classes (comma-separated labels)
//
// Like the audio routes, a request selects a model only by an ID returned
// from the listing, never by a filesystem path, and never downloads.
func registerVisionRoutes(mux *http.ServeMux, sem chan struct{}, opts HandlerOptions, cache *yoloModelCache) {
	if !opts.Features.ModelCatalog {
		return
	}
	mux.HandleFunc("/models/detection", func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		models, err := discoverDetectionModels(req.Context(), opts.ModelDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{"models": models, "max_bytes": detectionMaxBytes})
	})
	detectSem := make(chan struct{}, 2)
	mux.HandleFunc("/v1/vision/detections", withLimit(detectSem, withLimit(sem, func(w http.ResponseWriter, req *http.Request) {
		if req.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		req.Body = http.MaxBytesReader(w, req.Body, detectionMaxBytes+1<<20)
		defer req.Body.Close()
		err := req.ParseMultipartForm(detectionMaxBytes)
		if req.MultipartForm != nil {
			defer req.MultipartForm.RemoveAll()
		}
		if err != nil {
			var tooLarge *http.MaxBytesError
			if errors.As(err, &tooLarge) {
				http.Error(w, "image upload exceeds 16 MiB", http.StatusRequestEntityTooLarge)
			} else {
				http.Error(w, "expected multipart form with model and file fields", http.StatusBadRequest)
			}
			return
		}
		cfg, classes, err := detectionConfig(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		selector := strings.TrimSpace(req.FormValue("model"))
		if selector == "" {
			http.Error(w, "choose a detection model", http.StatusBadRequest)
			return
		}
		path, err := resolveDetectionModel(req.Context(), opts.ModelDir, selector)
		if err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, errStockNotDownloaded) {
				status = http.StatusConflict
			}
			http.Error(w, err.Error(), status)
			return
		}
		file, header, err := req.FormFile("file")
		if err != nil {
			http.Error(w, "missing image file", http.StatusBadRequest)
			return
		}
		defer file.Close()
		if header.Size > detectionMaxBytes {
			http.Error(w, "image upload exceeds 16 MiB", http.StatusRequestEntityTooLarge)
			return
		}
		raw, err := io.ReadAll(io.LimitReader(file, detectionMaxBytes+1))
		if err != nil {
			http.Error(w, "cannot read image file", http.StatusBadRequest)
			return
		}
		if len(raw) > detectionMaxBytes {
			http.Error(w, "image upload exceeds 16 MiB", http.StatusRequestEntityTooLarge)
			return
		}
		// Check dimensions before decoding, so a tiny file declaring an
		// enormous canvas cannot make the server allocate gigabytes.
		dims, _, err := image.DecodeConfig(bytes.NewReader(raw))
		if err != nil {
			http.Error(w, "unsupported image (send JPEG or PNG)", http.StatusBadRequest)
			return
		}
		if dims.Width <= 0 || dims.Height <= 0 || dims.Width > detectionMaxPixels/dims.Height {
			http.Error(w, "image dimensions are out of range", http.StatusBadRequest)
			return
		}
		img, err := gopherllm.DecodeImageBytes(raw)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		model, err := cache.open(path)
		if err != nil {
			http.Error(w, "loading detection model failed: "+err.Error(), http.StatusBadRequest)
			return
		}
		dets, err := model.Detect(img, cfg)
		if req.Context().Err() != nil {
			return
		}
		if err != nil {
			http.Error(w, "detection failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if len(classes) > 0 {
			kept := dets[:0]
			for _, d := range dets {
				if classes[strings.ToLower(d.Label)] {
					kept = append(kept, d)
				}
			}
			dets = kept
		}
		if dets == nil {
			dets = []gopherllm.Detection{}
		}
		b := img.Bounds()
		writeJSON(w, map[string]any{
			"model": selector, "version": model.Version,
			"width": b.Dx(), "height": b.Dy(), "detections": dets,
		})
	})))
}

// detectionConfig reads the optional tuning fields of a detection request.
func detectionConfig(req *http.Request) (gopherllm.YOLOConfig, map[string]bool, error) {
	cfg := gopherllm.DefaultYOLOConfig()
	float := func(name string, dst *float32) error {
		v := strings.TrimSpace(req.FormValue(name))
		if v == "" {
			return nil
		}
		f, err := strconv.ParseFloat(v, 32)
		if err != nil || f <= 0 || f > 1 {
			return errors.New(name + " must be a number in (0, 1]")
		}
		*dst = float32(f)
		return nil
	}
	if err := float("conf", &cfg.ScoreThreshold); err != nil {
		return cfg, nil, err
	}
	if err := float("iou", &cfg.IoUThreshold); err != nil {
		return cfg, nil, err
	}
	if v := strings.TrimSpace(req.FormValue("size")); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 32 || n > 1920 || n%32 != 0 {
			return cfg, nil, errors.New("size must be a multiple of 32 between 32 and 1920")
		}
		cfg.InputWidth, cfg.InputHeight = n, n
	}
	var classes map[string]bool
	if v := strings.TrimSpace(req.FormValue("classes")); v != "" {
		classes = map[string]bool{}
		for _, c := range strings.Split(v, ",") {
			if c = strings.ToLower(strings.TrimSpace(c)); c != "" {
				classes[c] = true
			}
		}
	}
	return cfg, classes, nil
}
