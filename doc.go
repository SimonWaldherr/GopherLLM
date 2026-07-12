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
//   - NewHandler (server.go): the OpenAI-/Ollama-compatible HTTP API as a
//     mountable http.Handler for applications that expose the model over
//     HTTP themselves.
//   - Runner (runtime.go): the lower-level engine underneath both, exposed
//     for advanced uses (the agentic skill loop via RunAgenticChat, kernel
//     benchmarking, custom loops).
//   - AnalyzeGGUF / SearchTokens (analyze.go): header-only model structure
//     reports and vocabulary inspection without loading weights.
//
// The library never writes to stdout/stderr on its own; pass WithLogWriter
// (or HandlerOptions.LogWriter) to opt into diagnostics.
//
// A rough map of the internals:
//
//   - gguf.go       GGUF container parsing (header, metadata, tensor table)
//   - mmap*.go      memory-mapped file access (per-OS)
//   - model.go      model config, weight loading, transformer forward pass
//   - forward_batch.go  batched prefill (prompt tokens processed per chunk)
//   - simd.go, quant_extra.go  matvec/dot kernels + dequantization + pool
//   - *_amd64.s / *_arm64.s  hand-written SIMD kernels behind runtime dispatch
//   - tokenizer.go  SentencePiece and GPT-2/Tekken BPE tokenizers
//   - sampling.go   temperature/top-k/top-p/min-p sampling
//   - runtime.go    Runner: generation loop, chat templates per model family
//   - tools.go, extract.go, agent.go, skills.go  tool calling, reasoning
//     extraction, the server-side skill loop
//   - catalog.go    model discovery/selection in a models directory
//   - cmd/gopherllm CLI built on all of the above
package gopherllm
