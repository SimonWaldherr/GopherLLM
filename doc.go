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
//   - Model (api.go): the primary embedding API — context-first methods with
//     functional options (Open, Generate, Chat, Stream, Embed, Tokenize,
//     Detokenize). Start here.
//   - server.NewHandler (server/server.go, a separate package): the
//     OpenAI-/Ollama-compatible HTTP API as a mountable http.Handler for
//     applications that expose the model over HTTP themselves.
//   - Runner (runtime.go): the lower-level engine underneath both, exposed
//     for advanced uses (the agentic skill loop via RunAgenticChat, kernel
//     benchmarking, custom loops).
//   - AnalyzeGGUF / SearchTokens (analyze.go): header-only model structure
//     reports and vocabulary inspection without loading weights.
//   - rag (rag/, a separate package): a hybrid BM25 + optional-vector index
//     over your own documents. No dependency on this package's model type
//     beyond the Embedder interface — retrieval works without a model at all.
//   - agent (agent/, a separate package): composes a Model, a rag.Index and a
//     tool set into Ask/Chat with citations. Neither subpackage is imported
//     by this one.
//
// The library never writes to stdout/stderr on its own; pass WithLogWriter
// (or HandlerOptions.LogWriter) to opt into diagnostics.
//
// A rough map of the internals:
//
//   - gguf.go       GGUF container parsing (header, metadata, tensor table)
//   - mmap.go       public file-mapping facade (backends in internal/mmapfile)
//   - model_*.go    model config, weight loading, transformer forward pass
//   - forward_batch.go  batched prefill (prompt tokens processed per chunk)
//   - simd_*.go, quant_extra.go  matvec/dot kernels + dequantization + pool
//   - *_amd64.s / *_arm64.s  hand-written SIMD kernels behind runtime dispatch
//   - tokenizer.go, tokenizer_merge.go  SentencePiece and GPT-2/Tekken BPE
//   - sampling.go   temperature/top-k/top-p/min-p sampling
//   - runtime.go    Runner: generation loop, chat templates per model family
//   - agent.go, tool_schema.go, extract.go, skills.go  tool calling
//     (including reflection-derived schemas via NewTool), reasoning
//     extraction, the server-side skill loop (wire types in internal/tooling)
//   - catalog.go    model discovery/selection in a models directory
//   - cmd/gopherllm CLI built on all of the above
package gopherllm

// Version is reported by the CLI's --version and usage header.
const Version = "0.3.0-go"
