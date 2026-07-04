// Command consumer is a minimal external application that exercises the
// public GopherLLM API surface exactly as a third-party importer would:
// loading, generation with options, streaming, embeddings, tokenization,
// GGUF analysis, and mounting the HTTP handler. It is built (not run) by the
// TestExternalConsumerBuilds regression test to prove `go get`-ability; run
// it manually with a model path to smoke-test for real.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("usage: consumer <model.gguf> [prompt]")
		os.Exit(2)
	}
	ctx := context.Background()
	model, err := gopherllm.Open(ctx, os.Args[1], gopherllm.WithLogWriter(os.Stderr))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	// Header analysis.
	gopherllm.AnalyzeGGUF(model.GGUF(), model.Tokenizer()).WriteText(os.Stdout)

	// Tokenization round trip.
	ids := model.Tokenize("hello world")
	fmt.Printf("tokenized to %d ids; round trip: %q\n", len(ids), model.Detokenize(ids))

	prompt := "Say hello in five words."
	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}

	// Streaming generation with options.
	res, err := model.Stream(ctx, []gopherllm.ChatMessage{gopherllm.UserMessage(prompt)},
		func(delta string) error {
			fmt.Print(delta)
			return nil
		},
		gopherllm.WithMaxTokens(64),
		gopherllm.WithTemperature(0),
		gopherllm.WithSeed(1),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nfinish=%s tokens=%d\n", res.FinishReason, res.Stats.GeneratedTokens)

	// Embedding.
	emb, err := model.Embed(ctx, prompt)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("embedding: %d dims\n", len(emb.Embedding))

	// The HTTP surface mounts as a plain handler (not started here).
	var _ http.Handler = model.HTTPHandler(gopherllm.HandlerOptions{})
}
