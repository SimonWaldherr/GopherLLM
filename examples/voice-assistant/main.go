// Command voice-assistant serves a local browser speech assistant.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	"github.com/SimonWaldherr/GopherLLM/internal/audioinput"
	"github.com/SimonWaldherr/GopherLLM/speech"
)

//go:embed index.html voice.js voice.css
var assets embed.FS

const maxClipBytes = 2 << 20

type turn struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

type answer struct {
	Input string `json:"input"`
	Text  string `json:"text"`
	STT   int64  `json:"stt_ms"`
	LLM   int64  `json:"reply_ms"`
}

type assistant struct {
	transcribe func(context.Context, []byte) (string, error)
	reply      func(context.Context, []gopherllm.ChatMessage) (string, error)
	speak      func(context.Context, string) (gopherllm.SpeechAudio, error)
	busy       chan struct{}
}

func main() {
	if err := run(); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to a chat GGUF, e.g. Ministral 3 Instruct")
	phononPath := flag.String("phonon", "", "extracted Phonon model.fermion beside config.json")
	voxtralPath := flag.String("voxtral", "", "Voxtral speech model GGUF")
	whisperPath := flag.String("whisper", "", "Whisper GGML F16/F32 checkpoint")
	ttsPath := flag.String("tts", "", "optional local MMS/VITS model directory for spoken replies")
	addr := flag.String("listen", "127.0.0.1:8092", "loopback address")
	metal := flag.Bool("metal", false, "use Metal for chat and Voxtral (requires -tags metal)")
	flag.Parse()
	var path string
	var backend speech.Backend
	for _, candidate := range []struct {
		path    string
		backend speech.Backend
	}{{*phononPath, speech.Phonon}, {*voxtralPath, speech.Voxtral}, {*whisperPath, speech.Whisper}} {
		if candidate.path == "" {
			continue
		}
		if path != "" {
			return errors.New("choose only one speech model")
		}
		path, backend = candidate.path, candidate.backend
	}
	if *modelPath == "" || path == "" {
		return errors.New("specify -model and one of -phonon, -voxtral or -whisper")
	}
	if !loopback(*addr) {
		return errors.New("-listen must be a loopback address with a port")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	a := &assistant{busy: make(chan struct{}, 1)}
	recognizer, err := speech.Open(ctx, path, speech.Options{Backend: backend, CPUOnly: backend == speech.Voxtral && !*metal}, os.Stderr)
	if err != nil {
		return err
	}
	defer recognizer.Close()
	a.transcribe = func(ctx context.Context, wav []byte) (string, error) {
		pcm, err := audioinput.Decode16kMonoWAV(wav, 30)
		if err != nil {
			return "", err
		}
		return recognizer.Transcribe(ctx, pcm)
	}
	runner, _, err := gopherllm.RunnerFromPathWithOptions(*modelPath, gopherllm.LoadOptions{UseMetal: *metal})
	if err != nil {
		return err
	}
	defer runner.Close()
	a.reply = func(ctx context.Context, messages []gopherllm.ChatMessage) (string, error) {
		opts := gopherllm.DefaultGenerationOptions().WithContext(ctx)
		opts.MaxTokens = 72
		opts.Sampler.Temperature = 0
		opts.SystemPrompt = "You are a local voice assistant. Reply in the user's language using at most 25 words of plain spoken text."
		result, err := runner.GenerateChat(messages, opts)
		return result.Text, err
	}
	if *ttsPath != "" {
		tts, err := gopherllm.OpenTTS(ctx, *ttsPath)
		if err != nil {
			return err
		}
		defer tts.Close()
		a.speak = func(ctx context.Context, text string) (gopherllm.SpeechAudio, error) {
			return tts.Synthesize(ctx, text, gopherllm.TTSOptions{})
		}
	}
	listener, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	server := &http.Server{Handler: a.routes(), ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, IdleTimeout: 30 * time.Second, WriteTimeout: 2 * time.Minute,
		BaseContext: func(net.Listener) context.Context { return ctx }}
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if server.Shutdown(shutdown) != nil {
			_ = server.Close()
		}
	}()
	log.Printf("Voice assistant at http://%s", listener.Addr())
	err = server.Serve(listener)
	stop()
	<-drained
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func loopback(address string) bool {
	host, port, err := net.SplitHostPort(address)
	if err != nil || port == "" {
		return false
	}
	return host == "localhost" || net.ParseIP(host).IsLoopback()
}

func (a *assistant) routes() http.Handler {
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServer(http.FS(assets)))
	mux.HandleFunc("GET /options", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]bool{"tts": a.speak != nil})
	})
	mux.HandleFunc("POST /turn", a.handleTurn)
	if a.speak != nil {
		mux.HandleFunc("POST /speak", a.handleSpeak)
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !loopback(r.Host) || (r.Header.Get("Origin") != "" && r.Header.Get("Origin") != "http://"+r.Host) {
			http.Error(w, "local origin required", http.StatusForbidden)
			return
		}
		mux.ServeHTTP(w, r)
	})
}

func (a *assistant) handleTurn(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxClipBytes)
	if err := r.ParseMultipartForm(maxClipBytes); err != nil {
		http.Error(w, "expected a text turn or a WAV clip (≤ 2 MiB)", http.StatusBadRequest)
		return
	}
	defer r.MultipartForm.RemoveAll()
	messages, err := conversation(r.PostForm.Get("history"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	input := strings.TrimSpace(r.PostForm.Get("text"))
	files := r.MultipartForm.File["audio"]
	if len(input) > 4000 || len(files) > 1 || (input == "") == (len(files) == 0) {
		http.Error(w, "send exactly one text turn or audio clip", http.StatusBadRequest)
		return
	}
	select {
	case a.busy <- struct{}{}:
		defer func() { <-a.busy }()
	default:
		http.Error(w, "the assistant is processing another turn", http.StatusServiceUnavailable)
		return
	}
	result := answer{Input: input}
	if len(files) != 0 {
		file, err := files[0].Open()
		if err != nil {
			http.Error(w, "cannot open audio", http.StatusBadRequest)
			return
		}
		defer file.Close()
		raw, err := io.ReadAll(file)
		if err != nil {
			http.Error(w, "cannot read audio", http.StatusBadRequest)
			return
		}
		start := time.Now()
		result.Input, err = a.transcribe(r.Context(), raw)
		result.Input = strings.TrimSpace(result.Input)
		result.STT = time.Since(start).Milliseconds()
		if err != nil {
			log.Printf("speech recognition: %v", err)
			http.Error(w, "speech recognition failed. Check the audio and backend.", http.StatusUnprocessableEntity)
			return
		}
		if result.Input == "" || len(result.Input) > 4000 {
			http.Error(w, "no usable speech detected", http.StatusUnprocessableEntity)
			return
		}
	}

	start := time.Now()
	result.Text, err = a.reply(r.Context(), append(messages, gopherllm.UserMessage(result.Input)))
	result.LLM = time.Since(start).Milliseconds()
	if err != nil || strings.TrimSpace(result.Text) == "" {
		log.Printf("generation: %v", err)
		http.Error(w, "the model could not answer this turn", http.StatusUnprocessableEntity)
		return
	}
	writeJSON(w, result)
}

func conversation(raw string) ([]gopherllm.ChatMessage, error) {
	if raw == "" {
		return nil, nil
	}
	var turns []turn
	if len(raw) > 24_000 || json.Unmarshal([]byte(raw), &turns) != nil || len(turns) > 8 || len(turns)%2 != 0 {
		return nil, errors.New("invalid conversation history")
	}
	messages := make([]gopherllm.ChatMessage, 0, len(turns)+1)
	for i, t := range turns {
		text := strings.TrimSpace(t.Text)
		if text == "" || len(text) > 4000 {
			return nil, errors.New("invalid conversation history")
		}
		if i%2 == 0 && t.Role == "user" {
			messages = append(messages, gopherllm.UserMessage(text))
		} else if i%2 == 1 && t.Role == "assistant" {
			messages = append(messages, gopherllm.AssistantMessage(text))
		} else {
			return nil, errors.New("invalid conversation history")
		}
	}
	return messages, nil
}

func (a *assistant) handleSpeak(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 4000))
	if err != nil || strings.TrimSpace(string(raw)) == "" {
		http.Error(w, "invalid speech text", http.StatusBadRequest)
		return
	}
	select {
	case a.busy <- struct{}{}:
		defer func() { <-a.busy }()
	default:
		http.Error(w, "the assistant is busy", http.StatusServiceUnavailable)
		return
	}
	audio, err := a.speak(r.Context(), string(raw))
	if err != nil {
		http.Error(w, "speech synthesis failed", http.StatusUnprocessableEntity)
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	if err := audio.WriteWAV(w); err != nil {
		log.Printf("writing synthesized audio: %v", err)
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}
