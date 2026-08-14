// Model Sources shows every place GopherLLM can get a model from, and that
// they all end at the same thing: a local .gguf path you hand to Open.
//
// There are three, and none of them require a copy or a conversion step:
//
//   - a model directory you control (also finds LM Studio's library)
//   - an Ollama store, reusing models already pulled with `ollama pull`
//   - the Hugging Face Hub, downloaded into a local cache on first use
//
// Only the Hugging Face path touches the network, and only when asked.
//
// Usage:
//
//	go run ./examples/model-sources                        # list what is available locally
//	go run ./examples/model-sources -search qwen3          # search the Hub
//	go run ./examples/model-sources -run "tinyllama:latest" -prompt "Hi"
//	go run ./examples/model-sources -run "hf:owner/repo:model-Q4_K_M.gguf" -prompt "Hi"
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
	hub "github.com/SimonWaldherr/GopherLLM/huggingface"
)

func main() {
	var (
		modelDir = flag.String("model-dir", gopherllm.DefaultModelDir(), "local directory to scan for .gguf files")
		search   = flag.String("search", "", "search the Hugging Face Hub for GGUF repositories and exit")
		limit    = flag.Int("limit", 10, "maximum search results")
		run      = flag.String("run", "", "model to load: a local id, an Ollama name:tag, or hf:owner/repo[:file.gguf]")
		prompt   = flag.String("prompt", "Say hello in one short sentence.", "prompt to send when -run is set")
		offline  = flag.Bool("offline", false, "never contact the Hub; use only what is already cached")
	)
	flag.Parse()

	// Ctrl-C cancels an in-flight download rather than leaving a partial file.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	opts := hub.DefaultOptions()
	opts.Offline = *offline

	if *search != "" {
		if err := runSearch(ctx, *search, *limit, opts); err != nil {
			fmt.Fprintln(os.Stderr, "search:", err)
			os.Exit(1)
		}
		return
	}

	if *run != "" {
		if err := runModel(ctx, *run, *modelDir, *prompt, opts); err != nil {
			fmt.Fprintln(os.Stderr, "run:", err)
			os.Exit(1)
		}
		return
	}

	listLocal(*modelDir)
}

// listLocal prints the two offline sources side by side. Neither is fatal when
// absent: a fresh machine may have no model directory, and most machines have
// no Ollama store.
func listLocal(modelDir string) {
	fmt.Printf("Model directory: %s\n", modelDir)
	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
	switch {
	case err != nil:
		fmt.Printf("  (unavailable: %v)\n", err)
	case len(entries) == 0:
		fmt.Println("  (no .gguf files found)")
	default:
		printEntries(entries)
	}

	root := gopherllm.OllamaModelsRoot()
	if root == "" {
		fmt.Println("\nOllama store: not found (set OLLAMA_MODELS to point at one)")
		return
	}
	fmt.Printf("\nOllama store: %s\n", root)
	ollama, err := gopherllm.DiscoverOllamaModels(root, io.Discard)
	switch {
	case err != nil:
		fmt.Printf("  (unavailable: %v)\n", err)
	case len(ollama) == 0:
		fmt.Println("  (no models pulled yet)")
	default:
		printEntries(ollama)
	}
}

func printEntries(entries []gopherllm.ModelEntry) {
	for _, e := range entries {
		if e.IsProjector {
			continue // companion vision encoder, not independently runnable
		}
		status := "unsupported"
		if e.IsSupported {
			status = e.Architecture
		}
		extra := ""
		if e.ProjectorPath != "" {
			extra = "  [vision]"
		}
		if e.IsEmbedding {
			extra += "  [embedding]"
		}
		fmt.Printf("  %-44s %-14s %6.2f GB%s\n", e.ID, status, float64(e.SizeBytes)/(1<<30), extra)
	}
}

func runSearch(ctx context.Context, query string, limit int, opts hub.Options) error {
	results, err := hub.Search(ctx, query, limit, opts)
	if err != nil {
		return err
	}
	if len(results) == 0 {
		fmt.Println("No GGUF repositories matched.")
		return nil
	}
	fmt.Printf("%d repositories publishing GGUF files:\n\n", len(results))
	for _, r := range results {
		fmt.Printf("  %s\n", r.ID)
	}
	fmt.Println("\nPick one, then:")
	fmt.Println("  go run ./examples/model-sources -run \"hf:<repo>\" -prompt \"Hi\"")
	fmt.Println("A repository with several quantizations reports the choice; append :<file>.gguf to pick one.")
	return nil
}

// runModel is the point of the example: whichever source the reference names,
// it resolves to a local path and the loading code below is identical.
func runModel(ctx context.Context, ref, modelDir, prompt string, opts hub.Options) error {
	path, source, err := resolve(ctx, ref, modelDir, opts)
	if err != nil {
		return err
	}
	fmt.Printf("Loading %s (via %s)\n\n", path, source)

	m, err := gopherllm.Open(ctx, path)
	if err != nil {
		return err
	}
	defer m.Close()

	out, err := m.Stream(ctx, []gopherllm.ChatMessage{gopherllm.UserMessage(prompt)},
		func(delta string) error {
			fmt.Print(delta)
			return nil
		})
	if err != nil {
		return err
	}
	fmt.Printf("\n\n(%d tokens)\n", out.Stats.GeneratedTokens)
	return nil
}

func resolve(ctx context.Context, ref, modelDir string, opts hub.Options) (path, source string, err error) {
	// An explicit hf: prefix always means the Hub — it is the only source that
	// may hit the network, so it is never inferred silently here. Progress
	// goes to stderr so stdout stays just the model output.
	if strings.HasPrefix(ref, "hf:") {
		path, err = hub.Resolve(ctx, ref, os.Stderr, opts)
		return path, "Hugging Face", err
	}

	// An Ollama reference is name:tag, which is also how the store lists it.
	if root := gopherllm.OllamaModelsRoot(); root != "" {
		if entries, derr := gopherllm.DiscoverOllamaModels(root, io.Discard); derr == nil {
			for _, e := range entries {
				if e.ID == ref {
					return e.Path, "Ollama store", nil
				}
			}
		}
	}

	// Otherwise it names something in the model directory.
	entries, err := gopherllm.DiscoverModels(modelDir, io.Discard)
	if err != nil {
		return "", "", err
	}
	entry, err := gopherllm.SelectModel(entries, ref)
	if err == nil {
		return entry.Path, "model directory", nil
	}

	// Last resort: an "owner/repo" that matched nothing locally is almost
	// certainly a Hub reference. Suggest it rather than downloading on a
	// guess — an implicit network fetch is not something an example should do
	// behind the user's back.
	if strings.Contains(ref, "/") {
		return "", "", fmt.Errorf("%q matched nothing locally; if it is a Hub repository, run it as \"hf:%s\"", ref, ref)
	}
	return "", "", fmt.Errorf("%q matched no local model and no Ollama model: %w", ref, err)
}
