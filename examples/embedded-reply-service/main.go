// Embedded Reply Service shows how an application can own its HTTP surface
// while using one GopherLLM Model directly in the same process.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type replyRequest struct {
	Message string `json:"message"`
}

func main() {
	if err := run(); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to local GGUF model (required)")
	addr := flag.String("addr", "127.0.0.1:8091", "listen address")
	flag.Parse()
	if strings.TrimSpace(*modelPath) == "" {
		return errors.New("-model /path/to/model.gguf is required")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	model, err := gopherllm.Open(ctx, *modelPath, gopherllm.WithLogWriter(os.Stderr))
	if err != nil {
		return fmt.Errorf("load local model: %w", err)
	}
	defer model.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ready"}`))
	})
	mux.HandleFunc("POST /reply", func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var input replyRequest
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil || strings.TrimSpace(input.Message) == "" {
			http.Error(w, "JSON {message: non-empty string} required", http.StatusBadRequest)
			return
		}
		requestCtx, cancel := context.WithCancel(r.Context())
		stopCancel := context.AfterFunc(ctx, cancel)
		defer func() { stopCancel(); cancel() }()
		result, err := model.Generate(requestCtx, input.Message,
			gopherllm.WithSystemPrompt("You are a concise local assistant. State uncertainty rather than inventing facts."),
			gopherllm.WithMaxTokens(180), gopherllm.WithTemperature(0.3))
		if err != nil {
			http.Error(w, "local generation failed: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(map[string]string{"reply": result.Text})
	})
	log.Printf("Embedded Reply Service: http://%s", *addr)
	srv := &http.Server{Addr: *addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.ListenAndServe() }()
	select {
	case err := <-serveErr:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
		// Stop accepting requests and cancel generation before unmapping weights.
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			_ = srv.Close()
			return err
		}
		return nil
	}
}
