# GopherLLM

GopherLLM is a local GGUF inference tool written in Go. It can run one-shot prompts,
interactive REPL sessions, embeddings, model inspection, benchmark runs, and an HTTP
server with OpenAI-compatible, Ollama-compatible, and built-in endpoints.

## Features

- Pure Go runtime with optional ARM64 assembly kernels.
- Memory-mapped GGUF loading for fast startup and lower copy pressure.
- Quantized matrix kernels for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, and MXFP4 tensors.
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
- `make run-prep MODEL=...` runs the prompt with `--prepare-quant`.
- `make build-metal` builds `bin/gopherllm-metal` with CGO and the `metal` tag.
- `make run-metal MODEL=...` runs with experimental `--metal` enabled.
- `make run-full MODEL=...` and `make run-full-prep MODEL=...` run 256-token
  prompt checks without and with `--prepare-quant`.
- `make run-full-metal MODEL=...` and `make run-full-metal-prep MODEL=...`
  run 256-token prompt checks with Metal enabled.
- `make repl MODEL=...` starts the REPL.
- `make serve MODEL=... CHAT=1` starts the HTTP server and chat UI.
- `make list-models` scans `MODEL_DIR`.
- `make inspect MODEL=...` prints model metadata summary.
- `make list-tensors MODEL=...` prints the tensor inventory.
- `make bench` runs Go microbenchmarks.
- `make bench-model MODEL=...` runs generation benchmark JSON.
- `make bench-model-prep MODEL=...` and `make compare-bench MODEL=...` benchmark
  the prepared quant path.
- `make bench-model-metal MODEL=...` benchmarks the experimental Metal path.
- `make kernel-bench MODEL=...` benchmarks isolated model kernels.
- `make kernel-bench-prep MODEL=...` and `make compare-kernel-bench MODEL=...`
  benchmark isolated kernels with prepared quant enabled.
- `make kernel-bench-metal MODEL=...` benchmarks isolated kernels with Metal
  enabled.
- `make test`, `make vet`, and `make check` verify the codebase.
- `make cross-build` compiles release binaries for macOS, Linux, and Windows on
  `amd64` and `arm64`.

## Performance Notes

- Use `--threads <N>` to set both GopherLLM worker threads and `GOMAXPROCS`.
- Use `--prepare-quant` when slower startup is acceptable; it precomputes Q4_K
  scale/min data and switches those rows to row-level NEON prepared kernels.
- Use `--temp 0 --top-k 1` for deterministic greedy output.
- `--bench-json` and `--kernel-bench-json` are intended for repeatable performance
  comparisons.
- Metal is available only in `bin/gopherllm-metal` builds made with
  `CGO_ENABLED=1 -tags metal`, and must be enabled with `--metal`. The current
  path is experimental and only offloads large Q6_K matvecs to a Metal compute
  kernel inspired by MLX's QMV/GEMV dispatch structure. It can improve isolated
  output-matvec timings, but full generation can still be slower because each
  token crosses the CPU/GPU boundary. Measure before using it for real runs;
  more of the decode graph needs to stay resident on the GPU for Metal to win
  consistently end to end.
- On ARM64, Q4_K and Q6_K matvecs use NEON block kernels, attention heads are spread
  across the worker pool at longer contexts, and matvec work is over-chunked so
  performance cores absorb efficiency-core stragglers.
- See `OPTIMIZATION_LOG.md` for measured optimization attempts, including rejected
  Q6_K/NEON approaches that should not be retried without new evidence.

## Supported Architectures

The loader currently accepts GGUF files whose `general.architecture` is one of:

```text
llama, llama2, llama3, mistral, mistral3, qwen2, gpt-oss, gemma, gemma2, gemma4
```

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
