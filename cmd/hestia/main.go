// Command hestia serves a local voice- and text-driven assistant.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/SimonWaldherr/GopherLLM/cmd/hestia/internal/hestia"
)

func main() {
	configPath := flag.String("config", "", "path to a JSON config file (defaults if omitted)")
	listenWord := flag.String("listen-word", "Computer", "activation phrase for hands-free mode (empty disables it)")
	voxtralModelPath := flag.String("voxtral-model", "", "Voxtral Realtime GGUF (overrides the configured speech backend)")
	phononModelPath := flag.String("phonon-model", "", "Phonon .fermion model (overrides the configured speech backend)")
	whisperModelPath := flag.String("whisper-model", "", "Whisper GGML F16/F32 checkpoint (overrides the configured speech backend)")
	whisperLanguage := flag.String("whisper-language", "auto", "Whisper language code or auto, used with -whisper-model")
	embedModelPath := flag.String("embed-model", "", "path to a GopherLLM-loadable embedding GGUF; enables HestiaRAG document search (overrides config)")
	chatModelPath := flag.String("chat-model", "", "local chat GGUF for offline questions (overrides config)")
	flag.Parse()

	cfg := hestia.DefaultConfig()
	if *configPath != "" {
		loaded, err := hestia.LoadConfig(*configPath)
		if err != nil {
			log.Fatalf("hestia: %v", err)
		}
		cfg = loaded
	}
	speechOptions := []hestia.SpeechConfig{
		{Backend: hestia.SpeechVoxtral, ModelPath: *voxtralModelPath},
		{Backend: hestia.SpeechPhonon, ModelPath: *phononModelPath},
		{Backend: hestia.SpeechWhisper, ModelPath: *whisperModelPath, Language: *whisperLanguage},
	}
	var selectedSpeech hestia.SpeechConfig
	for _, option := range speechOptions {
		if option.ModelPath == "" {
			continue
		}
		if selectedSpeech.Backend != "" {
			log.Fatal("hestia: select at most one of -voxtral-model, -phonon-model or -whisper-model")
		}
		selectedSpeech = option
	}
	if selectedSpeech.Backend != "" {
		cfg.VoxtralModelPath = ""
		cfg.Speech = selectedSpeech
	}
	if *embedModelPath != "" {
		cfg.EmbedModelPath = *embedModelPath
	}
	if *chatModelPath != "" {
		cfg.ChatModelPath = *chatModelPath
	}
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "listen-word" {
			cfg.ListenWord = *listenWord
		}
	})
	loadCtx, cancelLoad := context.WithTimeout(context.Background(), 5*time.Minute)
	runtime, err := hestia.OpenRuntime(loadCtx, cfg, os.Getenv("HESTIA_HA_TOKEN"), os.Stderr)
	cancelLoad()
	if err != nil {
		log.Fatalf("hestia: %v", err)
	}
	defer runtime.Close()
	srv := &http.Server{
		Addr: cfg.Listen, Handler: hestia.NewHandler(runtime.App), ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout: 15 * time.Second, IdleTimeout: time.Minute,
	}

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
