// Package gopherllm is a pure-Go GGUF inference runtime: it memory-maps a
// quantized model file, runs the transformer forward pass on CPU (with
// optional ARM64 NEON / x86-64 AVX2 assembly kernels), and exposes
// generation, chat, streaming, embeddings, tokenization, and GGUF analysis
// directly to Go programs — no external process, no HTTP round-trips.
//
//	import gopherllm "github.com/SimonWaldherr/GopherLLM"
//
//	model, err := gopherllm.Open(ctx, "model.gguf")
//	if err != nil { ... }
//	defer model.Close()
//
//	res, err := model.Generate(ctx, "Explain GGUF in one sentence.",
//	    gopherllm.WithMaxTokens(128), gopherllm.WithTemperature(0.7))
//	fmt.Println(res.Text)
//
// # API layers
//
//   - Model: the primary embedding API — context-first methods with
//     functional options (Open, Generate, Chat, Stream, Embed, Tokenize,
//     Detokenize). Start here.
//   - server.NewHandler (server/server.go, a separate package): the
//     OpenAI-/Ollama-compatible HTTP API as a mountable http.Handler for
//     applications that expose the model over HTTP themselves.
//   - Runner: the lower-level engine underneath both, exposed
//     for advanced uses (the agentic skill loop via RunAgenticChat, kernel
//     benchmarking, custom loops).
//   - AnalyzeGGUF / SearchTokens: header-only model structure
//     reports and vocabulary inspection without loading weights.
//   - rag (rag/, a separate package): a hybrid BM25 + optional-vector index
//     over your own documents. No dependency on this package's model type
//     beyond the Embedder interface — retrieval works without a model at all.
//   - agent (agent/, a separate package): composes a Model, a rag.Index and a
//     tool set into Ask/Chat with citations. Neither subpackage is imported
//     by this one.
//   - mobile (mobile/, a separate package): the JSON-in, JSON-out surface
//     behind the C ABI (bindings/c) and the Swift, Rust and Python packages.
//
// The library never writes to stdout/stderr on its own; pass WithLogWriter
// (or HandlerOptions.LogWriter) to opt into diagnostics.
//
// The implementation lives in internal/engine. This package preserves the
// public import path with type aliases and forwarding functions, generated
// from the engine's exported declarations by go generate .
//
// The go doc command does not expand methods behind type aliases. From a
// repository checkout, use go doc ./internal/engine.Model.Generate to read
// the implementation's method documentation.
//
// Platform backends, container formats, tokenizers and numeric helpers have
// their own packages under internal/. See docs/PROJECT_STRUCTURE.md for the
// directory map.
package gopherllm

//go:generate go run ./internal/cmd/genapi

// Version is reported by the CLI's --version and usage header.
const Version = "0.3.0-go"
