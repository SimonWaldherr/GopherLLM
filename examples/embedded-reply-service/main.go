// Embedded Reply Service shows how an application can own its HTTP surface
// while using one GopherLLM Model directly in the same process.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

type replyRequest struct {
	Message string `json:"message"`
}

func main() {
	modelPath := flag.String("model", "", "path to local GGUF model (required)")
	addr := flag.String("addr", "127.0.0.1:8091", "listen address")
	flag.Parse()
	if strings.TrimSpace(*modelPath) == "" {
		log.Fatal("-model /path/to/model.gguf is required")
	}
	model, err := gopherllm.Open(context.Background(), *modelPath, gopherllm.WithLogWriter(os.Stderr))
	if err != nil {
		log.Fatalf("load local model: %v", err)
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
		decoder := json.NewDecoder(io.LimitReader(r.Body, 16<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil || strings.TrimSpace(input.Message) == "" {
			http.Error(w, "JSON {message: non-empty string} required", http.StatusBadRequest)
			return
		}
		result, err := model.Generate(r.Context(), input.Message,
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
	log.Fatal(http.ListenAndServe(*addr, mux))
}
