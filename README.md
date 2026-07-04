# GopherLLM

GopherLLM is a local GGUF inference tool written in Go. It can run one-shot prompts,
interactive REPL sessions, embeddings, model inspection, benchmark runs, and an HTTP
server with OpenAI-compatible, Ollama-compatible, and built-in endpoints.

## Features

- Pure Go runtime with optional ARM64 (NEON) and x86-64 (AVX2 + FMA) assembly kernels.
- Memory-mapped GGUF loading for fast startup and lower copy pressure.
- Quantized matrix kernels for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, and MXFP4 tensors.
- Temperature, top-k, top-p, and min-p sampling with a repetition penalty.
- CLI generation, REPL mode, embeddings, metadata inspection, and tensor listing.
- HTTP API with `/generate`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/api/generate`, `/api/chat`, and `/api/embeddings`.
- Optional browser chat UI served from the embedded `web_ui` assets.
- Model discovery for LM Studio community model directories.

## Requirements

- Go 1.25 or newer.
- A GGUF text model. By default the tool scans:

```sh
~/.cache/lm-studio/models/lmstudio-community
```

You can override that path with `RUSTY_LLM_MODEL_DIR`, `MODEL_DIR`, or
`--model-dir`, depending on how you run the tool.

## Build

```sh
make build
```

The binary is written to `bin/gopherllm`.

To run formatting, tests, vet, and the release build:

```sh
make all
```

To verify release builds for macOS, Linux, and Windows on `amd64` and `arm64`:

```sh
make cross-build
```

## CLI Usage

List discovered GGUF models:

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models/lmstudio-community" --list-models
```

Run a prompt against a selected model:

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models/lmstudio-community" \
  --model "model-name-or-file-fragment" \
  --prompt "Explain local LLM inference in three sentences." \
  --max-tokens 128
```

Start an interactive REPL:

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models/lmstudio-community" \
  --model "model-name-or-file-fragment" \
  --repl
```

Inspect metadata without loading all weights:

```sh
bin/gopherllm /path/to/model.gguf --inspect --list-metadata
```

Create an embedding:

```sh
bin/gopherllm /path/to/model.gguf --embed --prompt "semantic search query"
```

## Server

Start the API server with the embedded chat UI:

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models/lmstudio-community" \
  --model "model-name-or-file-fragment" \
  --serve 127.0.0.1:8080 \
  --chat
```

Open `http://127.0.0.1:8080/chat` for the browser UI.

Minimal OpenAI-compatible chat request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about Go."}],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

Streaming is supported on `/v1/chat/completions` by setting `"stream": true`.

## Make Targets

- `make run MODEL=... PROMPT='...'` builds and runs one prompt.
- `make repl MODEL=...` starts the REPL.
- `make serve MODEL=... CHAT=1` starts the HTTP server and chat UI.
- `make list-models` scans `MODEL_DIR`.
- `make inspect MODEL=...` prints model metadata summary.
- `make list-tensors MODEL=...` prints the tensor inventory.
- `make bench` runs Go microbenchmarks.
- `make bench-model MODEL=...` runs generation benchmark JSON.
- `make kernel-bench MODEL=...` benchmarks isolated model kernels.
- `make test`, `make vet`, and `make check` verify the codebase.
- `make cross-build` compiles release binaries for macOS, Linux, and Windows on
  `amd64` and `arm64`.

## Performance Notes

- Use `--threads <N>` to set both GopherLLM worker threads and `GOMAXPROCS`.
- Use `--temp 0 --top-k 1` for deterministic greedy output.
- Use `--min-p <F>` (e.g. `0.05`) for min-p nucleus sampling; `0` disables it.
- `--bench-json` and `--kernel-bench-json` are intended for repeatable performance
  comparisons.
- The runtime currently reports Metal as unavailable; inference runs through the Go
  and assembly CPU kernels.
- On x86-64, `DotF32`, the F32 vector ops, and the Q4_K/Q6_K matvecs use AVX2 + FMA
  kernels (auto-detected via CPUID). This gives roughly a 3x end-to-end decode
  speedup over the scalar path on Q4_K_M models. Set `GOPHERLLM_DISABLE_SIMD=1` to
  force the portable scalar kernels (useful for A/B benchmarking).
- Prompt processing (prefill) is batched: each weight is dequantized once per
  prompt chunk and reused across all prompt tokens, instead of re-running the
  quantized kernels once per token. Since prefill is compute-bound this is a large
  speedup (~4x measured on a Q4_K_M model). Set `GOPHERLLM_NO_BATCH_PREFILL=1` to
  fall back to the per-token path (A/B benchmarking / debugging).
- Set `GOPHERLLM_Q8_ACTIVATIONS=1` (x86-64, opt-in) to quantize activations to int8
  and run the Q4_K matvecs with `VPMADDUBSW` integer dot products. This is roughly
  1.15-1.2x faster on the Q4_K path at the cost of a small activation-quantization
  error (output direction stays within cosine 0.999 of the float path). Off by
  default so the float kernels remain bit-for-bit reproducible against the scalar
  reference.
- On ARM64, Q4_K and Q6_K matvecs use NEON block kernels, attention heads are spread
  across the worker pool at longer contexts, and matvec work is over-chunked so
  performance cores absorb efficiency-core stragglers.
- Set `GOPHERLLM_DISABLE_YARN=1` to skip YaRN RoPE scaling for models that declare
  it.
- See `OPTIMIZATION_LOG.md` for measured optimization attempts, including rejected
  Q6_K/NEON approaches that should not be retried without new evidence.

## Supported Architectures

The loader currently accepts GGUF files whose `general.architecture` is one of:

```text
llama, llama2, llama3, mistral, mistral3, ministral, mixtral, qwen2, gpt-oss,
gemma, gemma2, gemma4
```

Mistral-family instruct models (including Ministral) use the `[INST]…[/INST]`
chat format, the Tekken byte-level BPE pre-tokenizer, and YaRN RoPE context
scaling when the GGUF declares it.

Projector files such as `mmproj-*` are detected and excluded from text-model
selection.

## Development

Run the full local check:

```sh
make check
```

Run a focused benchmark:

```sh
go test -run '^$' -bench=BenchmarkMatvecQ4K -benchmem .
```

Local build artifacts are kept in `bin/` and `.cache/`, both ignored by git.

GitHub Actions runs `go test`, `go vet`, and `go build` on Linux, macOS, and
Windows, plus the `make cross-build` release matrix on Linux.
