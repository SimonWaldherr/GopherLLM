# GopherLLM

GopherLLM is a local GGUF inference tool written in Go. It can run one-shot prompts,
interactive REPL sessions, embeddings, model inspection, benchmark runs, and an HTTP
server with OpenAI-compatible, Ollama-compatible, and built-in endpoints.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Use as a Go Library](#use-as-a-go-library)
- [Build](#build)
- [CLI Usage](#cli-usage)
- [GGUF Analyzer](#gguf-analyzer)
- [Server](#server)
- [Tool Use / Agentic](#tool-use--agentic)
- [Benchmarking and Profiling](#benchmarking-and-profiling)
- [Make Targets](#make-targets)
- [Performance Notes](#performance-notes)
- [Supported Architectures](#supported-architectures)
- [Development](#development)

## Features

- Pure Go runtime with optional ARM64 (NEON) and x86-64 (AVX2 + FMA) assembly kernels.
- Memory-mapped GGUF loading for fast startup and lower copy pressure, on
  every platform (Unix `mmap`, Windows `CreateFileMapping`/`MapViewOfFile`):
  weights page in on demand and quantized tensors borrow the mapping
  zero-copy.
- Split/sharded GGUF loading: point at any one shard of a
  `<name>-00001-of-00005.gguf`-style download and every sibling is discovered
  and merged automatically (see [Performance Notes](#performance-notes)).
- Quantized matrix kernels for Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1,
  Q5_0, Q5_1, Q8_0, Q8_1, and MXFP4 tensors; F32/F16/BF16 loaded directly
  (BF16 covers QAT-derived and modern full-precision GGUFs).
- Temperature, top-k, top-p, and min-p sampling with a repetition penalty.
- OpenAI-compatible tool/function calling, with a native prompt format for
  Mistral-family models and a generic convention for everything else.
- Chain-of-thought extraction (`<think>` blocks, gpt-oss channels) into a
  separate `reasoning_content` field instead of leaving it in the answer text.
- Skills: point `--skills-dir` at a folder of `SKILL.md` files and the server
  resolves the model's `load_skill` calls itself, agentically, before replying.
- CLI generation, REPL mode, embeddings, metadata inspection, and tensor listing.
- HTTP API with `/generate`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/skills`, `/api/generate`, `/api/chat`, and `/api/embeddings`.
- Optional browser chat UI served from the embedded `web_ui` assets.
- Model discovery for LM Studio community model directories.

## Requirements

- Go 1.25 or newer.
- A GGUF text model. By default the tool scans:

```sh
~/.cache/lm-studio/models/lmstudio-community
```

That default is resolved in this order: the `--model-dir <path>` flag (highest
priority), then the `GOPHERLLM_MODEL_DIR` environment variable (with
`RUSTY_LLM_MODEL_DIR`, the project's pre-rename spelling, still honored as a
deprecated fallback), then the built-in default above. `MODEL_DIR` is a separate thing: it's a *Makefile*
variable (see [Make Targets](#make-targets)) that `make` targets use to fill in
`--model-dir` for you — it isn't read by the `gopherllm` binary itself, so
`MODEL_DIR=... bin/gopherllm ...` (without `make`) has no effect.

## Quickstart

```sh
make build                                    # -> bin/gopherllm
bin/gopherllm --model-dir /path/to/models --list-models
bin/gopherllm --model-dir /path/to/models --model "some-model" \
  --prompt "Explain local LLM inference in three sentences." --max-tokens 128
```

You can also pass an absolute `.gguf` path directly:

```sh
bin/gopherllm /path/to/model.gguf \
  --prompt "Explain local LLM inference in three sentences." \
  --max-tokens 128
```

Or, with `make` filling in the CLI flags for you:

```sh
make build
make list-models MODEL_DIR=/path/to/models
make run MODEL_DIR=/path/to/models MODEL="some-model" PROMPT="Explain local LLM inference in three sentences."
```

## Use as a Go Library

GopherLLM is an importable module — inference runs in-process, with no child
process and no HTTP round-trips:

```sh
go get github.com/SimonWaldherr/GopherLLM
```

```go
import gopherllm "github.com/SimonWaldherr/GopherLLM"

model, err := gopherllm.Open(ctx, "model.gguf")
if err != nil { ... }
defer model.Close()

// One-shot generation with functional options.
res, err := model.Generate(ctx, "Explain GGUF in one sentence.",
    gopherllm.WithMaxTokens(128), gopherllm.WithTemperature(0.7))
fmt.Println(res.Text)

// Streaming (ctx cancels cleanly between tokens).
model.Stream(ctx, []gopherllm.ChatMessage{gopherllm.UserMessage("hi")},
    func(delta string) error { fmt.Print(delta); return nil })

// Embeddings, tokenization, GGUF analysis:
emb, _ := model.Embed(ctx, "semantic search query")
ids := model.Tokenize("hello")
gopherllm.AnalyzeGGUF(model.GGUF(), model.Tokenizer()).WriteText(os.Stdout)
```

For applications that expose the model over HTTP themselves, the entire
OpenAI-/Ollama-compatible API mounts as a plain `http.Handler` — under any
router, prefix, or middleware stack:

```go
mux.Handle("/llm/", http.StripPrefix("/llm",
    model.HTTPHandler(gopherllm.HandlerOptions{Defaults: gopherllm.DefaultGenerationOptions()})))
```

The library never writes to stdout/stderr on its own; pass
`gopherllm.WithLogWriter(os.Stderr)` (or `HandlerOptions.LogWriter`) to opt
into diagnostics. Tool calling, reasoning extraction, and skills are available
via `WithTools`, `Result.ReasoningText`, and `RunAgenticChat` — see the godoc
and the runnable examples in `example_test.go`; `testdata/consumer` is a
complete external application using the API.

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

On sandboxed macOS shells, `/usr/bin/make` may print `xcrun_db-*` cache
warnings before the Makefile can set its build environment. Use the Command
Line Tools `make` directly if that happens:

```sh
/Library/Developer/CommandLineTools/usr/bin/make build-metal
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

Run a prompt against an exact GGUF file:

```sh
bin/gopherllm /path/to/model.gguf \
  --prompt "Explain local LLM inference in three sentences." \
  --max-tokens 128 \
  --temp 0.7
```

Start an interactive REPL:

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models/lmstudio-community" \
  --model "model-name-or-file-fragment" \
  --repl
```

Run with a [skill](#tool-use--agentic) available (one-shot or REPL alike):

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models/lmstudio-community" \
  --model "model-name-or-file-fragment" \
  --skills-dir ./skills \
  --prompt "How do I fill out a PDF form on the command line?"
```

Inspect metadata without loading all weights:

```sh
bin/gopherllm /path/to/model.gguf --inspect --list-metadata
```

Create an embedding:

```sh
bin/gopherllm /path/to/model.gguf --embed --prompt "semantic search query"
```

## GGUF Analyzer

Inspect any GGUF's structure without loading weights (instant, even on
multi-gigabyte files):

```sh
bin/gopherllm /path/to/model.gguf --analyze
```

Reports architecture/geometry, parameter count, effective bits per weight,
the quantization mix per tensor type, rope/sliding-window configuration,
tokenizer + detected chat-template family, KV-cache size estimates, and the
largest tensors.

Search the vocabulary:

```sh
bin/gopherllm /path/to/model.gguf --find-token "weather"
```

Explore embedding space — which tokens the model treats as related (this
loads the weights and scans the embedding table):

```sh
bin/gopherllm /path/to/model.gguf --token-neighbors king --neighbors 8
#  34567  "King"      cos=0.5807
#  12566  " king"     cos=0.5079
#  108083 "キング"     cos=0.3692
#  25776  "王"         cos=0.3416
```

The same features are available in the library as `AnalyzeGGUF`,
`SearchTokens`, and `Model.NearestTokens`.

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

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + loaded model id |
| POST | `/generate` | Native generation API (prompt or messages; accepts tools) |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (streaming, tools, reasoning) |
| POST | `/v1/completions` | OpenAI-compatible text completion |
| POST | `/v1/embeddings` | OpenAI-compatible embeddings |
| GET | `/v1/models` | OpenAI-compatible model listing (the loaded model) |
| GET | `/v1/skills` | Names + descriptions of configured skills |
| POST | `/api/generate` | Ollama-compatible generation |
| POST | `/api/chat` | Ollama-compatible chat (accepts tools) |
| POST | `/api/embeddings` | Ollama-compatible embeddings |
| GET | `/models` | Scan `--model-dir` and list all discovered GGUFs |
| POST | `/models/load` | Hot-swap the loaded model (`{"path": "..."}`) |
| GET | `/chat`, `/style.css`, `/script.js` | Embedded browser chat UI (with `--chat`) |

## Tool Use / Agentic

`/v1/chat/completions` (and the native `/generate` and Ollama-compatible
`/api/chat` endpoints) accept an OpenAI-shaped `tools` array. `/api/generate`
and `/v1/completions` don't (matching the real OpenAI/Ollama APIs, where tools
are chat-only), but skills (below) still apply there since those are a
server-side capability independent of any client-supplied `tools`:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}]
  }'
```

A model that decides to call the tool returns `finish_reason: "tool_calls"` and
a `message.tool_calls` array (`content` is `null` when the turn is only a tool
call). Continue the conversation by appending the assistant's tool-call
message and a `role: "tool"` message with the result:

```json
{"role": "assistant", "tool_calls": [{"id": "…", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Berlin\"}"}}]},
{"role": "tool", "tool_call_id": "…", "content": "{\"temperature_c\": 18, \"conditions\": \"sunny\"}"}
```

Rendering is native (`[AVAILABLE_TOOLS]`/`[TOOL_CALLS]`/`[TOOL_RESULTS]`,
verified directly against a real Ministral GGUF's `chat_template`) for
Mistral-family models, and a generic `<tool_call>{"name":...,"arguments":...}</tool_call>`
JSON convention for every other supported chat template. gpt-oss tool calling
is not yet implemented (only its reasoning channels are, see below).

Set `"tool_choice": "none"` to suppress tool offering (and skills, see below)
for a single request.

### Reasoning

Models that emit `<think>...</think>` chain-of-thought (DeepSeek-R1, QwQ,
etc.) have it split out of the answer and returned separately as
`reasoning_content` on the message (and as `delta.reasoning_content` when
streaming), rather than left mixed into the visible text. gpt-oss's
analysis/final channels are parsed the same way, though gpt-oss generation
currently still forces the final channel directly in the prompt — see the
comment on `renderGptOssMessages` for how to unlock full channel-based
reasoning once validated against a real gpt-oss GGUF.

### Skills

Point `--skills-dir` at a directory of skills, Claude-Agent-Skills style —
a name and one-line description are always visible to the model (via a
`load_skill` tool), and the full body is only loaded into context once the
model actually asks for it:

```text
skills/
  pdf-fill/SKILL.md
  git-review/SKILL.md
```

```markdown
---
name: pdf-fill
description: Fill out a PDF form given field values.
---
Full instructions the model receives once it loads this skill...
```

When skills are configured, every generation endpoint runs an agentic loop
server-side: if the model calls `load_skill`, the server resolves it
internally (feeding the skill body back as a tool result and letting the
model continue) before ever returning a response — the client never sees the
internal `load_skill` call. A `GET /v1/skills` endpoint lists the configured
skills' names and descriptions. Tool calls for anything else (i.e. tools the
*caller* supplied) are returned to the caller as usual, even with skills
configured. `--skills-dir` works the same way in one-shot/`--repl` CLI mode.

## Benchmarking and Profiling

Run synthetic Go microbenchmarks:

```sh
go test -run '^$' -bench=. -benchmem .
```

Run an end-to-end generation benchmark against a real GGUF:

```sh
bin/gopherllm /path/to/model.gguf \
  --prompt "Wer war Albert Einstein?" \
  --max-tokens 128 \
  --temp 0 \
  --bench --bench-json --bench-runs 3
```

Time individual model kernels for one transformer layer:

```sh
bin/gopherllm /path/to/model.gguf \
  --kernel-bench-json \
  --kernel-bench-runs 25 \
  --kernel-bench-layer 0
```

Capture a CPU profile during a real generation benchmark:

```sh
bin/gopherllm /path/to/model.gguf \
  --prompt "Wer war Albert Einstein?" \
  --max-tokens 128 \
  --temp 0 \
  --bench --bench-json --bench-runs 1 \
  --cpuprofile /tmp/gopherllm.prof
```

If your Go toolchain includes `pprof`, inspect it with:

```sh
go tool pprof -top bin/gopherllm /tmp/gopherllm.prof
```

For repeatable comparisons, keep the prompt, token count, sampler settings,
thread count, and model path fixed. The first run may include cache and warmup
effects, so prefer `--bench-runs 3` or more when comparing changes.

## Make Targets

- `make run MODEL=... PROMPT='...'` builds and runs one prompt.
- `make run-prep MODEL=...` runs the prompt with `--prepare-quant`.
- `make build-metal` builds `bin/gopherllm-metal` with CGO and the `metal` tag.
- `make run-metal MODEL=...` runs with experimental `--metal` enabled.
- `make run-full MODEL=...` and `make run-full-prep MODEL=...` run 256-token
  prompt checks without and with `--prepare-quant`.
- `make run-full-metal MODEL=...` and `make run-full-metal-prep MODEL=...`
  run 256-token prompt checks with Metal enabled.
- `make run ARGS='...'` runs the CLI with a fully custom argument list instead
  (bypasses `MODEL`/`PROMPT`/sampler variables entirely).
- `make repl MODEL=...` starts the REPL.
- `make serve MODEL=... CHAT=1` starts the HTTP server and chat UI.
- `make serve-metal MODEL=... CHAT=1 THREADS=8` starts the Metal server with
  prepared CPU fallback kernels enabled by default (`PREPARE_QUANT=0` disables
  preparation).
- `make list-models` scans `MODEL_DIR`.
- `make inspect MODEL=...` prints model metadata summary.
- `make list-tensors MODEL=...` prints the tensor inventory.
- `make bench` runs Go microbenchmarks.
- `make bench-model MODEL=...` runs generation benchmark JSON.
- `make bench-model-prep MODEL=...` and `make compare-bench MODEL=...` benchmark
  the prepared quant path.
- `make bench-model-metal MODEL=...` benchmarks the experimental Metal path.
- `make synonym-bench MODEL=...` / `make nato-bench MODEL=...` run fixed
  benchmark prompts useful for spotting output-quality regressions.
- `make kernel-bench MODEL=...` benchmarks isolated model kernels.
- `make kernel-bench-prep MODEL=...` and `make compare-kernel-bench MODEL=...`
  benchmark isolated kernels with prepared quant enabled.
- `make kernel-bench-metal MODEL=...` benchmarks isolated kernels with Metal
  enabled.
- `make test`, `make vet`, and `make check` verify the codebase.
- `make coverage` runs the test suite and prints per-function coverage; `make
  coverage-html` does the same and opens an HTML report.
- `make cross-build` compiles release binaries for macOS, Linux, and Windows on
  `amd64` and `arm64`.
- `run`, `repl`, and `serve` all accept `SKILLS_DIR=path/to/skills` to enable
  [skills](#tool-use--agentic); `run` and `repl` also accept `MIN_P`,
  `REPEAT_PENALTY`, and `SEED` alongside the existing `TEMP`/`TOP_P`/`TOP_K`.
- Run `make help` for the full target and variable list.

## Performance Notes

- Use `--threads <N>` to set both GopherLLM worker threads and `GOMAXPROCS`.
  Make targets expose the same setting as `THREADS=<N>`; 8 was fastest in the
  measured M2 Max setup, but should be re-benchmarked on each target Mac.
- Use `--prepare-quant` when slower startup is acceptable; it precomputes Q4_K
  scale/min data plus selected Q6_K scale data, then switches supported rows to
  prepared kernels.
- Use `--temp 0 --top-k 1` for deterministic greedy output.
- Use `--min-p <F>` (e.g. `0.05`) for min-p nucleus sampling; `0` disables it.
- `--bench-json` and `--kernel-bench-json` are intended for repeatable performance
  comparisons.
- Metal is available only in `bin/gopherllm-metal` builds made with
  `CGO_ENABLED=1 -tags metal`, and must be enabled with `--metal`. The selective
  path fuses mixed Q4_K/Q4_K/Q6_K Q/K/V projections into one command buffer and
  offloads Q4_K attention-output, Q4_K gate/up + SiLU + Q6_K FFN-down in one
  command buffer, and Q6_K vocabulary-output projections. GGUF files opened
  through mmap are exposed to Metal as shared no-copy weight buffers;
  byte-backed models retain the copying path for cgo safety. Prepared ARM64
  kernels remain as the fallback for small projections and Metal failures. The
  path remains experimental; use
  `--kernel-bench-json` and `--bench-json` on the target Mac before deployment.
- On x86-64 (AVX2 + FMA + F16C, auto-detected via CPUID), Q4_K, Q5_K, Q6_K,
  Q8_0, Q4_0, Q4_1, and MXFP4 matvecs default to int8-activation full-row kernels: the activation
  vector is quantized once per matvec to int8 with one scale per 256-element
  block (llama.cpp's Q8_K convention, `q8kQuantize`), and each weight row is
  processed by a single assembly call (`q4kDotQ8KRow` / `q5kDotQ8KRow` /
  `q6kDotQ8KRow` / `q8_0DotQ8KRow`) that decodes block scales in-register,
  dots 32 weights per `VPMADDUBSW` (Q8_0's own signed weights use the
  abs/sign-restore identity so the same unsigned-operand instruction applies),
  applies scales via `VPMADDWD`, and reduces horizontally once per row. Versus
  the previous per-block float kernels this is ~2.5x (Q4_K) to ~6x (Q6_K and
  Q8_0) per-row — and >20x for Q5_K, which previously had no SIMD fast path at
  all — and roughly 4x end-to-end decode on a Ministral 3B Q4_K_M. Set
  `GOPHERLLM_Q8_ACTIVATIONS=0` to force the exact float kernels
  (bit-reproducible against the scalar reference; the int8 path stays within
  cosine 0.999 of it — the same accuracy tradeoff llama.cpp makes by default).
  `GOPHERLLM_DISABLE_SIMD=1` still forces portable scalar
  kernels everywhere.
- Prompt processing (prefill) is batched. With the int8 path active, each raw
  quantized weight row is streamed from memory exactly once per prompt chunk and
  dotted against all prompt tokens' pre-quantized int8 activations in
  L2-resident row tiles (`matvecBatchQ8`) — no f32 dequantization pass at all.
  With `GOPHERLLM_Q8_ACTIVATIONS=0` the older dequantize-once-per-chunk f32 path
  runs instead. ARM64 reuses per-worker dequantization rows and dispatches one
  coarse batch range per worker to avoid allocation and scheduling overhead.
  Set `GOPHERLLM_NO_BATCH_PREFILL=1` to fall back to the per-token path (A/B
  benchmarking / debugging), or `GOPHERLLM_PREFILL_CHUNK=<N>` to tune the chunk
  size on the deployment machine.
- SwiGLU's `x*sigmoid(x)*up` runs through an AVX2 kernel with a Cephes-style
  expf polynomial (~1e-7 relative error) instead of per-element `math.Exp`.
- On ARM64, Q4_K and Q6_K matvecs use NEON block kernels, attention heads are
  spread across the worker pool at longer contexts, and single-token matvec work
  is over-chunked so performance cores absorb efficiency-core stragglers.
- Set `GOPHERLLM_DISABLE_YARN=1` to skip YaRN RoPE scaling for models that declare
  it.
- Split GGUFs (llama.cpp's `gguf-split` naming convention,
  `<name>-00001-of-00005.gguf`) are detected from any one shard's
  `split.count` metadata; every sibling is located next to it, and their
  tensor data is merged into one in-memory buffer before loading. This costs
  one full copy of the model's weights at load time — true zero-copy mmap
  borrowing only applies to single-file GGUFs — but needs no other opt-in.
- On x86-64 (F16C) the KV cache stores K/V rows as f16 by default: half the
  cache memory (double the context fits the reusable-workspace cap) and half
  the bytes attention streams per generated token, with rows converted
  in-register (`VCVTPH2PS`) inside the attention kernels. Greedy decode on
  the test model is bit-identical to the f32 cache; set `GOPHERLLM_KV_F16=0`
  to force the exact f32 cache. Attention itself is two-pass (independent
  score dots, then max-stabilized softmax weights and the weighted V
  accumulation), which measured ~1.15x over the previous online-softmax loop
  at 4k-16k context and uses the true score maximum for stability.
- After mmap'ing a single-file GGUF, every page is touched once up front
  across all worker threads (`prefaultPages`) before the model is reported
  loaded. A memory-mapped file only pages in on first touch, and a forward
  pass touches essentially every weight byte — without this, the *first*
  request after startup silently inherited that page-in cost (disk I/O, or on
  Windows, real-time antivirus scanning of each mapped page) inside its own
  TTFT instead of load time. For a one-shot CLI run this doesn't change total
  wall-clock; for the HTTP server and REPL cases it means every request,
  including the first, sees consistent latency instead of one random request
  eating a multi-second page-in tax. Set `GOPHERLLM_NO_PREFAULT=1` to restore
  pure lazy paging.

### Environment variables

Quick reference for the runtime toggles described above (unset by default;
details in the bullets they annotate):

| Variable | Effect |
|---|---|
| `GOPHERLLM_MODEL_DIR` | Default model directory when `--model-dir` is not given (`RUSTY_LLM_MODEL_DIR` remains a deprecated fallback) |
| `GOPHERLLM_DISABLE_SIMD` | Force portable scalar kernels (skip AVX2 detection) |
| `GOPHERLLM_NO_BATCH_PREFILL` | Per-token prefill instead of batched |
| `GOPHERLLM_PREFILL_CHUNK` | Override batched-prefill chunk size (`1`-`256`) |
| `GOPHERLLM_Q8_ACTIVATIONS` | `0` disables the default int8-activation Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q4_1/MXFP4 matvecs (x86-64) |
| `GOPHERLLM_NO_PREFAULT` | Skip the post-mmap page warm-up; restores pure lazy paging |
| `GOPHERLLM_KV_F16` | `0` stores the KV cache as exact f32 instead of f16 (x86-64) |
| `GOPHERLLM_METAL_ROWS_PER_GROUP` | Override Metal rows per threadgroup (`2`, `4`, `6`, or `8`; default `4`) |
| `GOPHERLLM_METAL_FUSED_FFN` | `0` disables Metal Gate/Up + SiLU + Down fusion |
| `GOPHERLLM_DISABLE_YARN` | Ignore declared YaRN RoPE scaling |

## Supported Architectures

The loader currently accepts GGUF files whose `general.architecture` is one of:

```text
llama, llama2, llama3, mistral, mistral3, ministral, mixtral, qwen2, qwen3,
gpt-oss, gemma, gemma2, gemma4
```

qwen3 (including the DeepSeek-R1 Qwen3 distills) adds per-head QK-norm on top
of the qwen2 graph; DeepSeek-R1 reasoning output is separated into
`reasoning_content` in both template conventions (self-opened `<think>` blocks
and the newer forced-open templates whose output begins mid-reasoning).
`deepseek2` (MLA attention) is not supported. Mistral-family models support
assistant-message prefill: a conversation ending in an assistant message
leaves the turn open so generation continues it.

Mistral-family instruct models (including Ministral) use the `[INST]…[/INST]`
chat format, the Tekken byte-level BPE pre-tokenizer, and YaRN RoPE context
scaling when the GGUF declares it.

Gemma-family support (`gemma`/`gemma2`/`gemma4`, including the Gemma QAT
GGUFs) is **experimental**: the dense Gemma graph is implemented — hardcoded
`sqrt(dim)` embedding scaling, GELU FFN, QK-norm, post-attention/post-FFN
norms, attention/final logit softcapping, the per-layer sliding-window map
(explicit `sliding_window_pattern` bool-array metadata or the known Gemma 2/4
interleave defaults), the `<start_of_turn>` chat template, and `<end_of_turn>`
as a stop token — but it has not been validated against real Gemma weights
yet, and the Gemma 4-specific mechanisms (p-RoPE frequency factors, per-layer
RoPE bases, cross-layer KV sharing, per-layer embeddings, the 26B MoE) are
still missing. The loader prints a warning. See
[docs/INFERENCE_NOTES.md](docs/INFERENCE_NOTES.md) for the architecture notes,
QAT specifics, and per-family recommended sampling settings (e.g. Gemma:
`--temp 1.0 --top-p 0.95 --top-k 64`).

Projector files such as `mmproj-*` are detected and excluded from text-model
selection.

## Development

### Project layout

| Area | Files |
|---|---|
| GGUF parsing + file mapping | `gguf.go`, `mmap.go` (Unix) / `mmap_windows.go` (Win32 file mapping) |
| Model loading + forward pass | `model.go`, `forward_batch.go` (batched prefill) |
| Compute kernels + worker pool | `simd.go`; assembly in `*_amd64.s` / `*_arm64.s` behind `dot_f32_*.go`, `vector_ops_*.go`, `quant_*.go`, `q4k_q8_*.go` dispatch shims |
| Tokenizers | `tokenizer.go` (SentencePiece + GPT-2/Tekken BPE) |
| Sampling | `sampling.go` |
| Generation orchestration + chat templates | `runtime.go` |
| Tool calling / reasoning / skills | `tools.go`, `extract.go`, `agent.go`, `skills.go` |
| Model discovery + selection | `catalog.go` |
| HTTP server | `server.go`, `web_ui/` |
| CLI | `cmd/gopherllm/main.go`, `lib.go` (package doc + version), `kernel_bench.go` |

A full architecture walkthrough — load path, inference data flow, kernel
dispatch tiers, and how to add a quant kernel / architecture / endpoint — is
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The same map, with more detail, is in the package comment in `doc.go`. Every
SIMD kernel has a portable Go scalar reference implementation, and
differential tests assert they agree — when touching a kernel, run the `Q4K`/
`Q6K`/`DotF32`/`VectorOps` test groups first. Model-behavior research notes
(Gemma 4 / QAT specifics, per-family sampling recommendations) live in
[docs/INFERENCE_NOTES.md](docs/INFERENCE_NOTES.md).

Run the full local check:

```sh
make check
```

Check test coverage:

```sh
make coverage      # per-function summary in the terminal
make coverage-html  # same, plus an interactive HTML report
```

Run a focused benchmark:

```sh
go test -run '^$' -bench=BenchmarkMatvecQ4K -benchmem .
```

Profile a real-model benchmark:

```sh
bin/gopherllm /path/to/model.gguf --prompt "test" --max-tokens 128 \
  --temp 0 --bench --bench-json --bench-runs 1 \
  --cpuprofile /tmp/gopherllm.prof
```

Local build artifacts are kept in `bin/` and `.cache/`, both ignored by git.

GitHub Actions runs `go test`, `go vet`, and `go build` on Linux, macOS, and
Windows, plus the `make cross-build` release matrix on Linux.
