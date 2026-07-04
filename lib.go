// GopherLLM is a pure-Go GGUF inference runtime and CLI: it memory-maps a
// quantized model file, runs the transformer forward pass on CPU (with
// optional ARM64 NEON / x86-64 AVX2 assembly kernels), and exposes the result
// as a one-shot CLI, a REPL, and an HTTP server speaking OpenAI-, Ollama-,
// and GopherLLM-native APIs.
//
// A rough map of the package:
//
//   - gguf.go       GGUF container parsing (header, metadata, tensor table)
//   - mmap*.go      memory-mapped file access (per-OS)
//   - model.go      model config, weight loading, transformer forward pass
//   - forward_batch.go  batched prefill (prompt tokens processed per chunk)
//   - simd.go       matvec/dot kernels + dequantization + worker pool
//   - *_amd64.s / *_arm64.s  hand-written SIMD kernels behind runtime dispatch
//   - tokenizer.go  SentencePiece and GPT-2/Tekken BPE tokenizers
//   - sampling.go   temperature/top-k/top-p/min-p sampling
//   - runtime.go    Runner: ties the above into generate/embed calls,
//     chat-template rendering per model family
//   - tools.go, extract.go, agent.go, skills.go  tool calling, reasoning
//     extraction, the server-side skill loop
//   - catalog.go    model discovery/selection in a models directory
//   - server.go     HTTP API
//   - main.go       CLI flag parsing and command dispatch
package gopherllm

import (
	"io"
	"os"
)

// Version is reported by --version and the usage header.
const Version = "0.3.0-go"

// errWriter is the destination for human-facing diagnostics; tests swap it to
// capture or silence output. Generation results go to stdout, diagnostics
// here, so `gopherllm ... > out.txt` captures only model output.
var errWriter io.Writer = os.Stderr

func stderr() io.Writer {
	if errWriter == nil {
		return os.Stderr
	}
	return errWriter
}
