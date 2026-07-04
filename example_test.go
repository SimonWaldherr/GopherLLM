package gopherllm_test

// Runnable godoc examples for the public API. They have no "Output:" comments
// because running them needs a real GGUF file; `go test` still compiles them,
// so they are guaranteed to stay in sync with the API.

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	gopherllm "github.com/SimonWaldherr/GopherLLM"
)

// Example demonstrates the simplest possible use: load a model, generate a
// completion, print it.
func Example() {
	ctx := context.Background()
	model, err := gopherllm.Open(ctx, "model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	res, err := model.Generate(ctx, "Explain GGUF in one sentence.",
		gopherllm.WithMaxTokens(128),
		gopherllm.WithTemperature(0.7),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)
}

// ExampleModel_Stream shows incremental token delivery with cancellation:
// the context deadline stops generation cleanly mid-stream.
func ExampleModel_Stream() {
	ctx := context.Background()
	model, err := gopherllm.Open(ctx, "model.gguf", gopherllm.WithLogWriter(os.Stderr))
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	messages := []gopherllm.ChatMessage{
		gopherllm.UserMessage("Write a haiku about Go."),
	}
	_, err = model.Stream(ctx, messages, func(delta string) error {
		fmt.Print(delta)
		return nil // return an error to stop generation
	}, gopherllm.WithMaxTokens(64), gopherllm.WithSeed(42))
	if err != nil {
		log.Fatal(err)
	}
}

// ExampleModel_Chat demonstrates multi-turn chat with tool calling: the model
// requests a tool, the application executes it and replays the result.
func ExampleModel_Chat() {
	ctx := context.Background()
	model, err := gopherllm.Open(ctx, "model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	weatherTool := gopherllm.ToolDefinition{
		Type: "function",
		Function: gopherllm.ToolFunctionDef{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			Parameters:  []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		},
	}
	messages := []gopherllm.ChatMessage{gopherllm.UserMessage("What's the weather in Berlin?")}
	res, err := model.Chat(ctx, messages, gopherllm.WithTools(weatherTool))
	if err != nil {
		log.Fatal(err)
	}
	if res.FinishReason == "tool_calls" {
		call := res.ToolCalls[0]
		// ... execute the tool, then continue the conversation:
		messages = append(messages,
			gopherllm.ChatMessage{Role: gopherllm.ChatRoleAssistant, ToolCalls: res.ToolCalls},
			gopherllm.ToolResultMessage(call.ID, call.Function.Name, `{"temperature_c": 18}`),
		)
		res, err = model.Chat(ctx, messages, gopherllm.WithTools(weatherTool))
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Println(res.Text)
}

// ExampleModel_Embed computes a text embedding for semantic search.
func ExampleModel_Embed() {
	ctx := context.Background()
	model, err := gopherllm.Open(ctx, "model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	emb, err := model.Embed(ctx, "semantic search query")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(emb.Embedding), "dimensions from", emb.TokenCount, "tokens")
}

// ExampleNewHandler mounts the OpenAI/Ollama-compatible HTTP API inside an
// existing application server, under a path prefix, with the host's own
// middleware and lifecycle — no bundled server.
func ExampleNewHandler() {
	model, err := gopherllm.Open(context.Background(), "model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	mux := http.NewServeMux()
	mux.Handle("/llm/", http.StripPrefix("/llm", model.HTTPHandler(gopherllm.HandlerOptions{
		Defaults: gopherllm.DefaultGenerationOptions(),
	})))
	// mux also serves the application's own routes...
	log.Fatal(http.ListenAndServe("127.0.0.1:8080", mux))
}

// ExampleAnalyzeGGUF inspects a model file's structure without loading any
// weights.
func ExampleAnalyzeGGUF() {
	mmap, err := gopherllm.OpenMmap("model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer mmap.Close()
	gguf, err := gopherllm.ParseGGUF(mmap.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	tok, _ := gopherllm.TokenizerFromMetadata(gguf.Metadata)
	gopherllm.AnalyzeGGUF(gguf, tok).WriteText(os.Stdout)
}