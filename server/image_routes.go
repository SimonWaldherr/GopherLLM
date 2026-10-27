package server

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

//go:embed web_ui/images.html
var imagesHTML string

//go:embed web_ui/images.js
var imagesJS string

//go:embed web_ui/images.css
var imagesCSS string

var imagesJSAsset = newChatStaticAsset(imagesJS)
var imagesCSSAsset = newChatStaticAsset(imagesCSS)

type imageGenerationFunc func(context.Context, gopherllm.ImageGenerationOptions, bool) (*image.NRGBA, error)

type imageGenerationModel = gopherllm.ImageGenerationModel

// imageModelCache is serialized by the routes' shared admission slot. Shutdown
// waits for active requests before closing it, just like explicit unloading.
type imageModelCache struct {
	open  func(context.Context) (imageGenerationModel, error)
	model imageGenerationModel
}

func (c *imageModelCache) generate(ctx context.Context, opts gopherllm.ImageGenerationOptions, keepAlive bool) (img *image.NRGBA, err error) {
	if c.model == nil {
		model, openErr := c.open(ctx)
		if openErr != nil {
			return nil, openErr
		}
		c.model = model
	}
	defer func() {
		// Failed or cancelled work never pins a partially used model in memory.
		if !keepAlive || err != nil || ctx.Err() != nil {
			c.close()
		}
	}()
	return c.model.Generate(ctx, opts)
}

func (c *imageModelCache) close() {
	if c.model != nil {
		_ = c.model.Close()
		c.model = nil
	}
}

// registerImageRoutes admits one image at a time, without a waiting queue.
// Its closer cancels active inference and waits for the model to be released.
func registerImageRoutes(mux *http.ServeMux, opts HandlerOptions, generate imageGenerationFunc, release func()) func() {
	if opts.ImageModelDir == "" {
		return func() {}
	}
	lifetime, closeLifetime := context.WithCancel(context.Background())
	var active sync.RWMutex
	if release == nil {
		release = func() {}
	}
	admission := make(chan struct{}, 1)
	mux.HandleFunc("POST /v1/images/generations", withLimit(admission, func(w http.ResponseWriter, req *http.Request) {
		active.RLock()
		defer active.RUnlock()
		if lifetime.Err() != nil {
			writeAPIError(w, 503, "unavailable", "", "image generation has stopped")
			return
		}
		ctx, cancel := context.WithCancel(req.Context())
		stop := context.AfterFunc(lifetime, cancel)
		defer stop()
		defer cancel()
		defer func() {
			if ctx.Err() != nil {
				release()
			}
		}()
		var input struct {
			KeepAlive      bool   `json:"keep_alive"`
			Prompt         string `json:"prompt"`
			Size           string `json:"size"`
			Seed           int64  `json:"seed"`
			N              int    `json:"n"`
			ResponseFormat string `json:"response_format"`
		}
		dec := json.NewDecoder(http.MaxBytesReader(w, req.Body, 32<<10))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&input); err != nil {
			writeAPIError(w, 400, "invalid_request", "", "invalid image request JSON")
			return
		}
		if err := dec.Decode(new(any)); err != io.EOF {
			writeAPIError(w, 400, "invalid_request", "", "expected one JSON object")
			return
		}
		input.Prompt = strings.TrimSpace(input.Prompt)
		if input.Prompt == "" || len(input.Prompt) > 4096 {
			writeAPIError(w, 400, "invalid_request", "prompt", "prompt must contain 1–4096 bytes")
			return
		}
		size := 256
		switch input.Size {
		case "", "256x256":
		case "512x512":
			size = 512
		default:
			writeAPIError(w, 400, "invalid_request", "size", "supported sizes: 256x256, 512x512")
			return
		}
		if input.N != 0 && input.N != 1 || input.ResponseFormat != "" && input.ResponseFormat != "b64_json" {
			writeAPIError(w, 400, "invalid_request", "", "only n=1 and response_format=b64_json are supported")
			return
		}
		img, err := generate(ctx, gopherllm.ImageGenerationOptions{Prompt: input.Prompt, Width: size, Height: size, Seed: input.Seed}, input.KeepAlive)
		if ctx.Err() != nil {
			inferenceAPIError(w, ctx.Err())
			return
		}
		if err != nil {
			// Loader errors may contain private filesystem paths. Keep those in
			// operator logs, not in responses on managed/shared deployments.
			if opts.LogWriter != nil {
				_, _ = io.WriteString(opts.LogWriter, "Image generation: "+err.Error()+"\n")
			}
			writeAPIError(w, 500, "generation_failed", "", "image generation failed; check server logs and the configured model directory")
			return
		}
		var output bytes.Buffer
		if err := png.Encode(&output, img); err != nil {
			writeAPIError(w, 500, "encoding_failed", "", "could not encode PNG")
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		writeJSON(w, struct {
			Created int64               `json:"created"`
			Data    []map[string]string `json:"data"`
		}{time.Now().Unix(), []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(output.Bytes())}}})
	}))
	mux.HandleFunc("POST /v1/images/unload", withLimit(admission, func(w http.ResponseWriter, req *http.Request) {
		active.RLock()
		defer active.RUnlock()
		if lifetime.Err() != nil {
			writeAPIError(w, 503, "unavailable", "", "image generation has stopped")
			return
		}
		release()
		w.Header().Set("Cache-Control", "no-store")
		writeJSON(w, map[string]bool{"unloaded": true})
	}))
	if opts.ChatUI {
		mux.HandleFunc("GET /images", func(w http.ResponseWriter, req *http.Request) {
			setChatUIHeaders(w, "", false)
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			if req.Method != http.MethodHead {
				_, _ = io.WriteString(w, imagesHTML)
			}
		})
		mux.HandleFunc("GET /images.js", func(w http.ResponseWriter, req *http.Request) {
			setChatUIHeaders(w, "", false)
			w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
			writeStaticAsset(w, req, imagesJSAsset)
		})
		mux.HandleFunc("GET /images.css", func(w http.ResponseWriter, req *http.Request) {
			setChatUIHeaders(w, "", false)
			w.Header().Set("Content-Type", "text/css; charset=utf-8")
			writeStaticAsset(w, req, imagesCSSAsset)
		})
	}
	return func() { closeLifetime(); active.Lock(); defer active.Unlock(); release() }
}
