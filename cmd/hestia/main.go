// Command hestia is the entry point for Hestia, a local voice- and
// text-driven home assistant (see CONCEPT.md). This first milestone wires
// the text-command path only: planner (rule-based placeholder), resolver,
// policy/executor and a durable journal, against a simulated lamp. Voice
// input and document retrieval are later steps (CONCEPT.md section 15).
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia"
	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia/adapters"
)

func main() {
	configPath := flag.String("config", "", "path to a JSON config file (defaults if omitted)")
	voxtralModelPath := flag.String("voxtral-model", "", "path to a Voxtral Realtime GGUF; enables voice turns (overrides config)")
	embedModelPath := flag.String("embed-model", "", "path to a GopherLLM-loadable embedding GGUF; enables HestiaRAG document search (overrides config)")
	flag.Parse()

	cfg := hestia.DefaultConfig()
	if *configPath != "" {
		loaded, err := hestia.LoadConfig(*configPath)
		if err != nil {
			log.Fatalf("hestia: %v", err)
		}
		cfg = loaded
	}
	if *voxtralModelPath != "" {
		cfg.VoxtralModelPath = *voxtralModelPath
	}
	if *embedModelPath != "" {
		cfg.EmbedModelPath = *embedModelPath
	}

	if err := os.MkdirAll(cfg.DataDir, 0o700); err != nil {
		log.Fatalf("hestia: creating data dir: %v", err)
	}

	journal, err := hestia.OpenJournal(filepath.Join(cfg.DataDir, "journal.jsonl"))
	if err != nil {
		log.Fatalf("hestia: %v", err)
	}
	defer journal.Close()

	sim := adapters.NewSimulator(
		adapters.Entity{ID: "light_living_floor", Name: "Stehlampe", Room: "Wohnzimmer", Kind: "light"},
		adapters.Entity{ID: "light_living_ceiling", Name: "Deckenlampe", Room: "Wohnzimmer", Kind: "light"},
		adapters.Entity{ID: "light_kitchen", Name: "Lampe", Room: "Küche", Kind: "light"},
	)
	inventory := hestia.Inventory{Devices: []hestia.Device{
		{ID: "light_living_floor", Name: "Stehlampe", Room: "Wohnzimmer", Kind: "light"},
		{ID: "light_living_ceiling", Name: "Deckenlampe", Room: "Wohnzimmer", Kind: "light"},
		{ID: "light_kitchen", Name: "Lampe", Room: "Küche", Kind: "light"},
	}}

	var speech *hestia.SpeechHost
	if cfg.VoxtralModelPath != "" {
		log.Printf("hestia: loading speech model %s (this can take a while)…", cfg.VoxtralModelPath)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		speech, err = hestia.OpenSpeechHost(ctx, cfg.VoxtralModelPath, os.Stderr)
		cancel()
		if err != nil {
			log.Fatalf("hestia: %v", err)
		}
		defer speech.Close()
		log.Println("hestia: speech model ready")
	} else {
		log.Println("hestia: no --voxtral-model given; voice turns are disabled, text still works")
	}

	var knowledge *hestia.KnowledgeBase
	if cfg.EmbedModelPath != "" {
		log.Printf("hestia: loading embedding model %s…", cfg.EmbedModelPath)
		knowledge, err = hestia.OpenKnowledgeBase(cfg.EmbedModelPath, filepath.Join(cfg.DataDir, "knowledge.db"))
		if err != nil {
			log.Fatalf("hestia: %v", err)
		}
		defer knowledge.Close()
		log.Println("hestia: knowledge base ready")
	} else {
		log.Println("hestia: no --embed-model given; document search is disabled, text/voice device control still works")
	}

	app := hestia.NewApp(journal, sim, inventory, speech, knowledge)
	srv := &http.Server{Addr: cfg.Listen, Handler: hestia.NewHandler(app)}

	go func() {
		log.Printf("hestia: listening on http://%s", cfg.Listen)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("hestia: server: %v", err)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()

	log.Println("hestia: shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		fmt.Fprintln(os.Stderr, "hestia: shutdown:", err)
	}
}
