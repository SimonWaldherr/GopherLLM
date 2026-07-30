# GopherLLM

[![DOI](https://zenodo.org/badge/1264366305.svg)](https://doi.org/10.5281/zenodo.21197831)

GopherLLM is a local GGUF inference tool written in Go. It can run one-shot prompts,
interactive REPL sessions, embeddings, model inspection, benchmark runs, and an HTTP
server with OpenAI-compatible, Ollama-compatible, and built-in endpoints.

**[Project website](https://simonwaldherr.github.io/GopherLLM/)** ·
**[Go package documentation](https://pkg.go.dev/github.com/SimonWaldherr/GopherLLM)**

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Dependency policy and layout](#dependency-policy-and-layout)
- [Quickstart](#quickstart)
- [Use as a Go Library](#use-as-a-go-library)
- [Build](#build)
- [CLI Usage](#cli-usage)
- [GGUF Analyzer](#gguf-analyzer)
- [Server](#server)
- [Tool Use / Agentic](#tool-use--agentic)
- [Auto Mode (hardware autotuning)](#auto-mode-hardware-autotuning)
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
- Quantized matrix kernels for TQ1_0, TQ2_0, Q1_0, Q2_0, Q2_K, Q3_K, Q4_K,
  Q5_K, Q6_K, Q8_K, IQ2_S, IQ3_S, IQ4_NL, IQ4_XS, Q4_0, Q4_1, Q5_0,
  Q5_1, Q8_0, Q8_1, and MXFP4 tensors; F32/F16/F64/BF16 load directly
  (BF16 covers QAT-derived and modern full-precision GGUFs).
- Temperature, top-k, top-p, and min-p sampling with a repetition penalty.
- OpenAI-compatible tool/function calling, with a native prompt format for
  Mistral-family models and a generic convention for everything else.
- Optional Wikipedia and Wikidata research tools, including bounded read-only
  SPARQL queries, resolved server-side into the model's answer.
- Chain-of-thought extraction (`<think>` blocks, gpt-oss channels) into a
  separate `reasoning_content` field instead of leaving it in the answer text.
- Skills: point `--skills-dir` at a folder of `SKILL.md` files and the server
  resolves the model's `load_skill` calls itself, agentically, before replying.
- CLI generation, REPL mode, embeddings, metadata inspection, and tensor listing.
- HTTP API with `/generate`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/skills`, `/api/generate`, `/api/chat`, and `/api/embeddings`.
- Optional browser chat UI served from the embedded `web_ui` assets, with
  persistent local conversations and a template-aware smart context window.
- Model discovery across the complete local LM Studio model library.
- Direct Hugging Face GGUF imports with cache reuse, split-model downloads,
  private/gated-model tokens, and revision selection.

## Requirements

- Go 1.25 or newer.
- A GGUF text model. By default the tool scans:

```sh
~/.cache/lm-studio/models
```

That default is resolved in this order: the `--model-dir <path>` flag (highest
priority), then the `GOPHERLLM_MODEL_DIR` environment variable (with
`RUSTY_LLM_MODEL_DIR`, the project's pre-rename spelling, still honored as a
deprecated fallback), then the built-in default above. `MODEL_DIR` is a separate thing: it's a *Makefile*
variable (see [Make Targets](#make-targets)) that `make` targets use to fill in
`--model-dir` for you — it isn't read by the `gopherllm` binary itself, so
`MODEL_DIR=... bin/gopherllm ...` (without `make`) has no effect.

## Dependency policy and layout

The checked-in Go module intentionally has no third-party dependencies: its
`go.mod` contains only this module and the Go version, and `make deps-check`
enforces that policy. The embedded chat UI is also self-contained; it does not
load packages, fonts, or scripts from a CDN.

The module-root Go files form the public `gopherllm` package, so core package
sources remain there to preserve the stable import path
`github.com/SimonWaldherr/GopherLLM`. Architecture-specific kernel dispatch and
assembly are consolidated into `kernels_<arch>.go` / `kernels_<arch>.s`
instead of being spread across one file per operation. Executable entry points
live in `cmd/`, HTTP code and UI assets in `server/`, generated tables and
other implementation details in `internal/`, and test-only fixtures—including
preserved profiling captures—in `testdata/`. Public-boundary and opt-in local
model tests live in `integration/`. Build output and local model/RAG data are
ignored and are not part of the repository.

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

### Hugging Face imports

Use an `hf:` selector to download a GGUF directly. Add the quantization after
the repository name when it contains more than one GGUF, and optionally add a
branch, tag, or commit after `@`:

```sh
bin/gopherllm hf:bartowski/Qwen3-4B-GGUF:Q4_K_M --repl
bin/gopherllm hf:bartowski/Qwen3-4B-GGUF:Q4_K_M@main --serve 127.0.0.1:8080 --chat
```

Explore a repository first when you do not know its available quantizations:

```sh
bin/gopherllm --hf-list bartowski/Qwen3-4B-GGUF
```

The output includes each selectable quantization, its total download size,
the number of GGUF shards, and a ready-to-run `hf:` selector. Downloads show
progress and resume from a saved partial blob after an interrupted transfer.
Split-model downloads use a bounded three-worker pool; pressing Ctrl-C cancels
repository requests and active transfers through Go contexts while retaining
the partial blobs for a later resume.

Downloads, including every shard of split GGUFs, use the shared Hugging Face
`blobs`/`refs`/`snapshots` cache under `$HF_HOME/hub` (or the platform cache
when `HF_HOME` is unset). Existing cached snapshots remain usable offline.
Set `HF_TOKEN` for gated or private repositories.

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

### Package layout

The root package is inference only. HTTP serving and the embedded web UI live
in the `server` subpackage, so importing GopherLLM to run a model does not pull
in `net/http`, `html/template`, or the web assets:

| Import | You get | Transitive deps |
|---|---|---|
| `github.com/SimonWaldherr/GopherLLM` | GGUF loading, generation, chat, embeddings, tokenizer, sampling, autotuning, skills/agent loop | 102 |
| `github.com/SimonWaldherr/GopherLLM/server` | the above **plus** the OpenAI-/Ollama-compatible HTTP API and the `/chat` web UI | 194 |

A minimal inference-only binary is ~3.4 MB against ~9.1 MB for the full CLI.
`TestInferencePackageStaysFreeOfServerDependencies` enforces the boundary
against the real dependency graph, so it cannot regress silently.

For applications that expose the model over HTTP themselves, the entire
OpenAI-/Ollama-compatible API mounts as a plain `http.Handler` — under any
router, prefix, or middleware stack:

```go
import "github.com/SimonWaldherr/GopherLLM/server"

mux.Handle("/llm/", http.StripPrefix("/llm",
    server.HandlerForModel(model, server.HandlerOptions{Defaults: gopherllm.DefaultGenerationOptions()})))
```

`server.NewHandler` returns a closeable `*server.Handler`. Hosts that enable
model hot-swapping should stop their HTTP server and then call
`handler.Close()` so the current chat and embedding GGUF mappings are released;
`server.Serve` does this automatically when it returns.

> **Moved in this release.** `gopherllm.Serve`, `gopherllm.NewHandler`,
> `gopherllm.HandlerOptions`/`ServeOptions` and the request/response types are
> now `server.*`, and `model.HTTPHandler(opts)` is now
> `server.HandlerForModel(model, opts)`. Inference APIs are unchanged.

The library never writes to stdout/stderr on its own; pass
`gopherllm.WithLogWriter(os.Stderr)` (or `HandlerOptions.LogWriter`) to opt
into diagnostics. Tool calling, reasoning extraction, and skills are available
via `WithTools`, `Result.ReasoningText`, and `RunAgenticChat` — see the godoc
and the runnable examples in `example_test.go`; `testdata/consumer` is a
complete external application using the API.

### Native Go research tools

Go hosts can choose the same bounded factual source tools without running a
separate process or exposing an HTTP endpoint. No source is enabled by default:

```go
tools := server.NewResearchTools(server.ResearchOptions{
    Wikimedia:     true,
    OpenStreetMap: true,
})
result, err := gopherllm.RunAgenticChatWithTools(
    model.Runner(), []gopherllm.ChatMessage{gopherllm.UserMessage("Where is the Brandenburg Gate?")},
    gopherllm.DefaultGenerationOptions(), nil, tools, nil,
)
```

`ResearchOptions` can enable Wikipedia/Wikidata and a bounded OpenStreetMap
place lookup independently. Every result carries a source URL and attribution.
For a self-hosted Nominatim-compatible service, set `OSMSearchURL` instead of
using the public endpoint.

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
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models" --list-models
```

Run a prompt against a selected model:

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models" \
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
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models" \
  --model "model-name-or-file-fragment" \
  --repl
```

Run with a [skill](#tool-use--agentic) available (one-shot or REPL alike):

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models" \
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
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models" \
  --model "model-name-or-file-fragment" \
  --serve 127.0.0.1:8080 \
  --chat
```

Open `http://127.0.0.1:8080/chat` for the browser UI.

The CLI remembers every successfully loaded **local** GGUF, whether it was
used for a one-shot prompt, REPL, benchmark, server startup, or selected
through the browser/API model picker. Later commands can omit the model
selector entirely; an explicit selector always wins. For example, restart a
server with:

```sh
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models" \
  --serve 127.0.0.1:8080 --chat
```

The remembered absolute path is stored as `last-model.json` below the
platform's per-user configuration directory (override with
`GOPHERLLM_MODEL_STATE_PATH`). Only the first run, a removed model, or an
unreadable state file can therefore start a server without weights. In that
case, choose a discovered GGUF in the browser's model picker or with
`POST /models/load`; the successful load immediately repairs the remembered
selection.

### OpenAI-compatible remote APIs

As a lower-priority alternative to a local GGUF, the server can proxy its chat
endpoint to an OpenAI-compatible API. This covers OpenAI and local compatible
servers such as Ollama, llama.cpp, and LM Studio. Configure it on the trusted
local server; the key is kept only in server memory and is never returned by
the configuration endpoint:

```sh
curl http://127.0.0.1:8080/remote \
  -H 'Content-Type: application/json' \
  -d '{"base_url":"http://127.0.0.1:11434","model":"llama3.2"}'
```

For OpenAI, use `{"base_url":"https://api.openai.com/v1","api_key":"…","model":"…"}`.
The configured remote becomes the target for `/v1/chat/completions`; use
`DELETE /remote` to switch back to the local model. `GET /remote/models`
lists models advertised by the remote service.

The chat UI is a local workspace rather than a thin request form: conversations,
drafts, per-chat instructions, sampling settings, and the selected appearance
are saved in the browser's IndexedDB. It supports chat search, rename/delete,
non-destructive edit and retry branches, local text-file insertion, and
JSON/Markdown export. Editing an earlier prompt or retrying an answer opens a
new local branch while retaining the original conversation for comparison.
Every assistant answer also has an under-text **Copy message** action and a
**Change model** action; changing the model creates a comparison branch and
automatically asks the same question again while preserving the original. The
**Model & chat** settings page presents discovered GGUFs as a searchable
two-column library with architecture, file size, context length, compatibility,
load progress, and the currently active model visible at a glance. Unsupported
or auxiliary GGUFs stay hidden unless requested, and the model name in the chat
header opens this picker directly.
Nothing is synced to a third party; exported archives are the portable backup
format. The UI assets use no-store and same-origin security headers, so start
the server on a trusted local address unless you add your own network security
in front of it.

The composer keeps the default path deliberately small: write a message,
attach files, and send. **Pro tools** reveals quick controls for context mode,
output length, and slash commands; the complete configuration remains in
Settings. The Settings dialog separates **Model & chat**, **Capabilities**,
**Generation**, and **Workspace** into keyboard-accessible tabs so common model
switching stays close while advanced sampling and storage controls do not
compete for attention. The model library and tab bar collapse cleanly on narrow
screens, and the mobile header wraps its actions below the chat title instead
of clipping them. Any file type can be attached. Text files up to 500 KB are
included as text in the model request; non-text files (images, audio, video,
PDFs, archives, and other binaries) stay local to the browser and are shown as
attachment cards. With the built-in text-only server, their filename, type, and
size are sent as metadata rather than pretending that binary content was
analysed.

### Smart context for long chats

The browser UI defaults to **Smart — recent complete turns**. It retains the
entire conversation in IndexedDB, but when sending a reply it asks the server
to select the newest complete turns that fit the loaded model's *actual* chat
template and token budget. Leading system instructions stay pinned; an
assistant tool call and its tool results stay with the user turn that caused
them. The latest turn is never cut mid-message: if it alone cannot fit after
reserving `max_tokens`, the request returns a clear error instead.

The Settings panel reports the exact prompt token count and how many earlier
messages remain saved locally. **Auto-compress — dense technical context**
first condenses ordinary user, system, and assistant text with conservative
terminology/abbreviation substitutions (for example, `application programming
interface` → `API` and `zum Beispiel` → `z. B.`). It uses the condensed form
only when the active model's tokenizer confirms that it is shorter, preserves
tool payloads and fenced code unchanged, then applies the same complete-turn
selection as Smart context. Choose **Full history — stop when full** for a
strict, untrimmed transcript. This is also the default for normal API clients;
the local extension is opt-in on `/v1/chat/completions`:

```json
{
  "gopherllm_context_mode": "autoCompress"
}
```

For a non-streaming recent-context request, the response includes
`X-GopherLLM-Context-*` headers (`Mode`, `Budget`, `Prompt-Tokens`,
input/retained/dropped message counts) and a `gopherllm_context` object. For
streaming requests, that object is carried by the terminal SSE choice instead,
so it always describes the final model call even after an internal skill/tool
loop. Allowed values are `recent`, `autoCompress`, and `full`.

### Model context cache

While a local server stays running, it retains one bounded **KV prefix cache**
for the most recently used rendered prompt. The browser still sends its normal
chat history, preserving the stateless OpenAI-compatible API, but the runner
compares the exact rendered token IDs and forwards only the changed suffix to
the model. Consecutive long-chat turns therefore avoid reprocessing their
unchanged context; edits and branches safely reuse only the unchanged prefix.

The cache reuses the normal generation workspace, grows geometrically for
follow-up turns, and is capped at 512 MiB rather than allocating one KV cache
per saved chat. It is memory-only and naturally cold after a server/model
restart (or when a context is too large to retain). Responses expose the
measured result as `gopherllm_cache` with `mode`, `hit`, `reused_tokens`, and
`prompt_tokens`; streaming responses carry it in the terminal SSE choice. The
browser UI shows the same information as cache warming or reuse, separately
from Smart Context's message-selection status.

The runner also retains an immutable, vocabulary-sized snapshot of the prompt
logits before sampling. An exactly repeated prompt can therefore reuse every
input token without re-running the final prompt token through the transformer.
Logit, generated-token, repeat-window, and streaming buffers are retained in
the bounded runner workspace to reduce allocation and garbage-collection work
across requests.

OpenAI-compatible completion responses also report the reused portion as
`usage.prompt_tokens_details.cached_tokens` (zero on a cold request).
When `stream_options.include_usage` is enabled, streaming chat completions emit
the standard final usage chunk with an empty `choices` array.

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
| GET | `/models` | Scan `--model-dir` and list discovered GGUFs, including each model's context length |
| POST | `/models/load` | Hot-swap to a supported GGUF discovered under `--model-dir` (`{"model": "<catalog-id>"}`; response includes the loaded context length) |
| POST | `/models/embed/load` | Load a compatible embedding GGUF for history RAG (`{"model": "<catalog-id>"}`); BERT, Nomic-BERT, and Granite Embedding models are supported |
| GET / POST / DELETE | `/remote` | Inspect, configure, or clear an OpenAI-compatible chat proxy (the API key is write-only) |
| GET | `/remote/models` | List models advertised by the configured remote API |
| GET | `/autotune` | Report Auto Mode status: whether a tuning is active this session, whether one is cached on disk for this model+machine, and the result either way |
| POST | `/autotune/run` | Run (or apply a cached) tuning for the loaded model, same effort levels as `--auto-effort` (`{"effort": "quick\|balanced\|thorough", "refresh": false}`) |
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

### Wikipedia and Wikidata research

The browser chat can opt into **Wikipedia & Wikidata research** from
**Options** (or the chat settings). The setting is saved per chat and defaults
to off: when enabled, only a search term, article title, Wikidata Q-ID, or
read-only SPARQL query requested by the model is sent to Wikimedia. The full
chat transcript is never sent to Wikimedia.

The server executes four bounded tools and feeds their JSON results back into
the same agentic turn, so the model can use the data in its final response:

- `wikipedia_search` uses Wikipedia's REST search endpoint.
- `wikipedia_summary` retrieves a concise article summary and canonical URL.
- `wikidata_entity` retrieves labels, descriptions, and a small set of claims
  for one Q-ID through the Wikidata Action API.
- `wikidata_sparql` runs read-only `SELECT`/`ASK` queries against the Wikidata
  Query Service; updates and `SERVICE` calls are rejected, responses are
  capped at 25 rows.

API clients enable the same integration per request with
`"gopherllm_wikimedia": true` on `/v1/chat/completions`, `/generate`,
`/api/chat`, or `/api/generate`. Set `tool_choice` to `none` to suppress it.
Results include their Wikimedia source URL or query endpoint; answers should
attribute factual claims to those results.

### OpenStreetMap place research

The browser chat can separately enable **OpenStreetMap place research** in
**Options → Tools & agents**. API clients use
`"gopherllm_openstreetmap": true` on the same chat and generation endpoints.
It is off by default. Only the bounded place query selected by the model is
sent to the configured Nominatim endpoint; the chat transcript is not sent.
Results include OpenStreetMap attribution and an object URL.

The default public Nominatim service is intentionally restricted to direct,
low-volume place lookups: GopherLLM enforces at most one request per second and
does not offer autocomplete or bulk geocoding. Do not send personal or
confidential data. Operators with a larger workload should configure their own
compatible endpoint with `server.HandlerOptions{OSMSearchURL: "..."}`. See the
[Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/).

### Privacy report

Local model inference does not make network requests and GopherLLM has no
telemetry. Inspect the complete contract with:

```sh
gopherllm --privacy
curl http://127.0.0.1:8080/privacy
```

The report names every feature that can send data externally, its destination,
and the limited data it may send. Hugging Face imports, remote model proxies,
and factual research sources remain opt-in.

## Auto Mode (hardware autotuning)

`--auto` measures **this model on this machine** at startup and runs it with the
fastest settings it can find, instead of trusting a default that was tuned on
somebody else's hardware:

```sh
bin/gopherllm /path/to/model.gguf --auto --prompt "Explain local inference."
```

```
Auto-tuning mistral3 on amd64+avx2+f16c (12 CPUs)...
  q8-activations on
  threads        12
  oversubscribe  on (was off)
  kv-cache-f16   on
  verify: oversubscribe wins, 145.2 -> 132.9 ms/token
Auto: calibrated in 7.9s
  threads=12 q8-activations=true kv-f16=true oversubscribe=true prefill-chunk=128
  decode 145.2 -> 132.9 ms/token (1.09x)
```

The result is **cached per model + hardware** under the user cache directory, so
only the first run pays for calibration. It applies to every mode — one-shot,
`--repl`, `--serve`, and `--bench`.

| Flag | Effect |
| --- | --- |
| `--auto` | Tune (or reuse a cached tuning) before generating |
| `--auto-effort quick` | Decode knobs only, ~8s. Prefill samples cost a whole chunk of prompt processing, so they are skipped here |
| `--auto-effort balanced` | Default. Adds prefill chunk tuning, ~1-2 min on a 3B model |
| `--auto-effort thorough` | More interleaved rounds and a 2048-token probe context; minutes, but the most reliable on a noisy machine |
| `--auto-refresh` | Re-measure and overwrite the cached result |
| `--auto-json` | Print the full result — including every candidate's median — and exit |

### From the web UI

Auto Mode isn't just a startup flag: the embedded chat UI (`--chat`) has a
"Performance" panel in Settings with an effort selector and a **Tune now**
button, backed by `GET /autotune` and `POST /autotune/run` (see the endpoint
table above). It reports whether a tuning is *active* this session versus
merely *cached* on disk from an earlier run — useful when the server was
started without `--auto` but a previous `--auto` run (or an earlier click of
Tune now) already measured this model on this machine. Triggering a tune
pauses generation until it finishes, same as the CLI: measurement and
generation share the same runner lock.

### Makefile workflow

The generation, REPL, server, and model-benchmark targets accept `AUTO=1`, so
the same cached tuning is used regardless of how the model is started:

```sh
# Fast first-pass calibration, then generate.
make run MODEL="my-model.gguf" AUTO=1 AUTO_EFFORT=quick

# Calibrate before exposing the local chat UI.
make serve MODEL="my-model.gguf" CHAT=1 AUTO=1
make serve-metal MODEL="my-model.gguf" CHAT=1 AUTO=1

# Inspect the cached result (or force a fresh calibration) as JSON and exit.
make autotune MODEL="my-model.gguf"
make autotune MODEL="my-model.gguf" AUTO_REFRESH=1
```

`AUTO_EFFORT` accepts `quick`, `balanced` (the default), or `thorough`.
`AUTO_REFRESH=1` bypasses the cache. `AUTO_JSON=1` can be used with any of
those Make targets, but it prints the tuning report and exits before generation
or serving; `make autotune` is the convenient report-only target. The
`run-auto`, `run-auto-metal`, `serve-auto`, and `serve-auto-metal` targets are
shortcuts for their corresponding command with `AUTO=1`.

What it tunes: thread count, the int8-activation matvec kernels, the f16 KV
cache, worker-dispatch oversubscription, and the prefill chunk size. Load-time
choices (`--metal`, `--prepare-quant`) are *not* tuned, because changing them
means reloading the weights; `--auto-json` reports whether they were active.
They are nevertheless part of the cache key, so a Metal or prepared-quant
run never reuses a CPU-only calibration.

### Why it is built the way it is

The measurement methodology is the substance here, not the knob list. Naively
timing candidate A and then candidate B produces garbage on any thermally
limited machine: this repo's dev laptop drops from ~4 GHz burst to ~1.2 GHz
sustained, so two runs of *identical* code can differ by 2-3x — far more than
any real tuning gain. The tuner therefore:

- **interleaves candidates in serpentine order** (`A B C`, then `C B A`). Plain
  round-robin is not enough: under a steady thermal ramp, whichever candidate is
  visited first each round is always measured at the coolest moment and wins
  systematically. Alternating direction cancels that gradient.
- **reports medians**, never means or minimums.
- **requires two independent hurdles** to change a setting: beat the incumbent's
  median by a margin *and* win a majority of individual rounds. A single lucky
  sample during a clock spike clears neither.
- **treats coordinate descent as a hypothesis, not an answer.** After the cheap
  per-knob exploration, one final interleaved sweep judges the starting config,
  the full proposed set, and each proposed change *in isolation*. So a set that
  only looked good because two knobs each got a lucky sample is rejected, while
  a single genuine win inside a losing set is still kept — and since the starting
  config is always a candidate, auto mode can never leave the model slower than
  it found it.
- **repeats probes until they are measurable.** A single forward pass through a
  small model times as exactly zero against the Windows clock's granularity,
  which would leave every candidate tied.

Expect run-to-run variation in *which* knobs it changes on a noisy machine —
that is the honest reflection of the hardware, and `--auto-effort thorough`
buys more rounds where it matters.

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

- `make build`, `make run`, `make repl`, and `make serve` auto-detect Metal:
  on macOS with Xcode Command Line Tools installed, they build with
  `CGO_ENABLED=1 -tags metal` and pass `--metal` for you (a real ~1.5-2x
  decode speedup and ~10x faster load from measurements on an M2 Max).
  Set `METAL=0` (e.g. `make build METAL=0`) to force the portable CPU-only
  build instead — useful for CI or a machine without Xcode. Cross-compiled
  binaries (`make cross-build`) are unaffected either way; they always use
  `CROSS_CGO_ENABLED` (default `0`) since Metal only exists on macOS.
- `make run MODEL=... PROMPT='...'` builds and runs one prompt.
- `make run-prep MODEL=...` runs the prompt with `--prepare-quant`.
- `make build-metal` builds `bin/gopherllm-metal` explicitly, regardless of
  the `METAL` auto-detection above (useful to keep both binaries around).
- `make run-metal MODEL=...` runs with experimental `--metal` enabled.
- `make run-auto MODEL=...` and `make run-auto-metal MODEL=...` tune (or reuse
  a cached tuning) before generating; set `AUTO_EFFORT=quick|balanced|thorough`
  to select calibration depth.
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
- `make serve-auto MODEL=... CHAT=1` and `make serve-auto-metal MODEL=...`
  perform startup tuning before accepting requests. Equivalently, add `AUTO=1`
  to `run`, `repl`, `serve`, or any model benchmark target.
- `make autotune MODEL=...` prints the cached or newly measured tuning result
  as JSON and exits; add `AUTO_REFRESH=1` to force a fresh measurement. Use
  `make autotune-metal MODEL=...` to report a Metal-enabled load.
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

- Prefer `--auto` over hand-tuning the flags below: it measures them on the
  actual machine and caches the result. See
  [Auto Mode](#auto-mode-hardware-autotuning).
- **Decode is at the memory-bandwidth roofline, and that bounds what any further
  kernel work can achieve.** Measured on the dev laptop (i7-10850H, DDR4-2933
  dual channel) with Ministral-3 3B Q4_K_M, which streams ~2.2 GB of weights per
  generated token:

  | Measurement | Throughput |
  | --- | --- |
  | `MatvecQ6KInto`, 330 MB DRAM-resident, 12 threads | ~22-25 GB/s |
  | Pure read of the same footprint, no weight decode | ~28-33 GB/s |
  | Same int8 row kernel on L2-resident weights, 12 threads | ~52-58 GB/s |

  So the kernels already run at ~75-80% of the achievable *streaming* rate,
  while having ~2.1x of idle compute capacity behind the memory wall. Two
  consequences worth knowing before optimizing:
  - Making the row kernels faster cannot help decode; they are already waiting
    on DRAM. Reducing *bytes per token* is the only lever.
  - Thread count barely matters once past ~6 threads: 12, 8, and 6 threads all
    measured within noise of each other (4 was clearly worse). This is why
    `--threads` is not the tuning knob it looks like.
- **Batching amortizes only ~1.7x, which is why speculative decoding does not
  pay here.** With a DRAM-resident weight, `matvecBatchQ8`'s cost per token falls
  from ~18 ms at p=1 to ~11 ms and then flattens — each extra position in a batch
  still costs ~0.6-0.7 of a full pass, because the int8 kernel re-decodes the
  weight row per token rather than register-blocking across tokens. A 3-position
  verification batch therefore costs ~2.4 passes, capping any speculative scheme
  at ~1.2x even with *perfect* draft acceptance. (An n-gram/prompt-lookup drafter
  was built and measured: it produced bit-identical output but ran at 0.81x,
  since real acceptance was ~39% on a 17% drafter hit rate.) Making batching
  genuinely cheap needs a kernel that dots one decoded weight row against N
  activation vectors; that would speed up prefill directly and only then make
  speculation viable.
- Use `--threads <N>` to set both GopherLLM worker threads and `GOMAXPROCS`.
  Make targets expose the same setting as `THREADS=<N>`; 8 was fastest in the
  measured M2 Max setup, but should be re-benchmarked on each target Mac.
- The short-context attention path avoids constructing an escaping per-layer
  worker closure. On the M2 Max development machine this reduced a tiny-model
  forward pass from 2,502 to 2,339 ns/op, allocations from 11 to 8, and a real
  0.11B F32 GGUF generation benchmark from 39.150 to 38.081 ms/op.
- Use `--prepare-quant` when slower startup is acceptable; it precomputes Q4_K
  scale/min data plus selected Q6_K scale data, then switches supported rows to
  prepared kernels.
- Use `--out-of-core` when a single-file GGUF, especially a large sparse-MoE
  model, does not fit comfortably in RAM or Apple unified memory. It keeps the
  model CPU-only, disables Metal and prepared-quant copies, leaves F16/F32/BF16
  matrices as mmap-backed scalar bytes, and does not prewarm the rank-3 expert
  banks (including fused `ffn_gate_up_exps` layouts). The operating system
  pages selected experts in on demand; this is not
  a hard RSS limit, so a cold expert can add SSD/page-fault latency and dense
  models that are far larger than RAM can still thrash. It intentionally rejects
  `--metal`, `--prepare-quant`, `--auto`, byte-backed loads, and split GGUFs
  (current split support merges shards into RAM). Library users can use
  `gopherllm.WithOutOfCore(true)` and optionally
  `WithMmapPrefault(gopherllm.MmapPrefaultNone)` for fully lazy paging.
- Use `--temp 0 --top-k 1` for deterministic greedy output.
- Use `--min-p <F>` (e.g. `0.05`) for min-p nucleus sampling; `0` disables it.
- `--bench-json` and `--kernel-bench-json` are intended for repeatable performance
  comparisons.
- Metal requires a build with `CGO_ENABLED=1 -tags metal` (the plain `make
  build`/`run`/`serve` targets do this automatically on macOS when Xcode
  Command Line Tools are present; `make build-metal` does it explicitly
  regardless of platform detection) and must be enabled with `--metal` at
  runtime (also automatic from the `make` targets above; pass it yourself for
  a manually built binary). The selective
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
  The same path now covers StableLM's tensor-selected sequential or
  parallel-residual LayerNorm block, dense Qwen3's per-head QK norm, EXAONE
  4's QK plus post-branch norms, OLMo 2/3's full-projection QK plus post-branch
  norms, and Phi-2's shared biased LayerNorm plus parallel exact-GELU branch.
  This moves all five families' prompt ingestion from per-token weight
  streaming onto the chunked path as well. Q4_0/Q8_0 rows also use the
  dequantize-once path, which is important for Stable Code and other legacy-Q4
  GGUFs on Apple Silicon.
  Set `GOPHERLLM_NO_BATCH_PREFILL=1` to fall back to the per-token path (A/B
  benchmarking / debugging), or `GOPHERLLM_PREFILL_CHUNK=<N>` to tune the chunk
  size on the deployment machine.
- SwiGLU's `x*sigmoid(x)*up` runs through an AVX2 kernel with a Cephes-style
  expf polynomial (~1e-7 relative error) instead of per-element `math.Exp`.
- On ARM64, Q4_K and Q6_K matvecs use NEON block kernels, attention heads are
  spread across the worker pool at longer contexts, and single-token matvec work
  is over-chunked so performance cores absorb efficiency-core stragglers.
  Apple Silicon also uses NEON FP16 conversion, dot, and accumulation kernels
  for the optional compact KV cache; at 4k context they made the f16 attention
  benchmark about 3.7x faster than the scalar path. Exact f32 remains the
  default there because it was still faster on the measured M2 Max.
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
  On non-amd64 systems the exact f32 cache remains the default; set
  `GOPHERLLM_KV_F16=1` to opt into the compact cache when memory capacity
  matters more than decode speed (for example, large Kimi contexts on a
  unified-memory Mac). Apple Silicon converts its rows with dedicated NEON
  FP16 kernels; other non-amd64 targets use the portable scalar fallback.
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
| `GOPHERLLM_KV_F16` | `0` stores the KV cache as exact f32 instead of the default f16 cache on fast x86-64; `1` opts into f16 on other targets (NEON-accelerated on Apple Silicon) to halve KV memory |
| `GOPHERLLM_METAL_ROWS_PER_GROUP` | Override Metal rows per threadgroup (`2`, `4`, `6`, or `8`; default `4`) |
| `GOPHERLLM_METAL_FUSED_FFN` | `0` disables Metal Gate/Up + SiLU + Down fusion |
| `GOPHERLLM_DISABLE_YARN` | Ignore declared YaRN RoPE scaling |

Settings chosen by `--auto` override the corresponding environment variables for
the rest of the process.

## Supported Architectures

The loader currently accepts GGUF files whose `general.architecture` is one of:

```text
llama, llama2, llama3, mistral, mistral3, ministral, mixtral, qwen2, qwen2moe, qwen3, qwen3moe,
qwen35, qwen35moe,
deepseek2, kimi_k2,
phi2, phi3, granite (dense), exaone, exaone4, smollm3, internlm2, stablelm,
olmo2, gpt-oss, gemma, gemma2, gemma3, gemma4,
nemotron_h, nemotron_h_moe, mamba2, bert, nomic-bert
```

The architecture value is only accepted when its execution graph and expected
tensor layout are implemented; an unknown GGUF is not treated as Llama merely
because some tensor names happen to match. The main coverage is:

| Family | Covered GGUF models / notes |
|---|---|
| Llama-style | Llama 2/3 text models, compatible `llama` exports, SmolLM3 3B (including its every-fourth-layer no-RoPE schedule) |
| Mistral | Mistral, Mistral Small/Devstral exports, Mistral 3, Ministral, and Mixtral |
| Qwen | Qwen2/2.5, QwQ, dense/sparse Qwen3 and Qwen3 Coder, plus experimental text-only Qwen3.5/3.6 hybrid exports |
| DeepSeek / Kimi | Modern DeepSeek-V2/V3 and Kimi K2 MLA layouts |
| Gemma | Gemma 1–3 and native dense/MoE/E2B Gemma 4 text graphs |
| Other decoders | Phi-2, Phi-3/3.5, dense Granite, EXAONE 3, EXAONE 4 1.2B/32B, OLMo 2/3, InternLM2, StableLM, GPT-OSS |
| Recurrent / hybrid | Mamba2 and Nemotron-H / Nemotron-H-MoE |
| Embeddings | BERT and Nomic-BERT (`/v1/embeddings`, not chat generation) |

Important upstream GGUF families that are **not implemented yet**:

| Missing family | Required work |
|---|---|
| Llama 4 | Its architecture-specific attention/normalization graph and model validation |
| Phi-MoE | Sparse expert routing and the architecture-specific expert graph |
| OLMoE | Its sparse expert router and expert execution graph |
| Command-R / Cohere2 | Their attention, normalization, and tokenizer/chat conventions |
| GLM4 / GLM4-MoE, MiniMax M2, LFM2 | Dedicated dense, sparse, or hybrid execution graphs |
| Falcon, GPT-NeoX/GPT-2, StarCoder2 | Non-Llama block layouts and their tokenizer conventions |
| Jamba, RWKV, Hyena-family hybrids | Recurrent/state-space cache and mixing kernels |
| Multimodal Qwen/Gemma/Llama models | Vision projector loading, visual-token injection, and multimodal positional encoding |
| Standalone MTP/assistant draft files (`deepseek4_mtp*`, `gemma4-assistant`) | A parent-model speculative-decoding runtime; these auxiliary GGUFs are not standalone chat models |

This gap list tracks architecture families, not every fine-tune name: a
fine-tune is supported when its GGUF declares one of the implemented
architectures and retains that architecture's tensor layout. The reference
catalog is llama.cpp's
[current GGUF architecture enum](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py);
adding a family generally requires both the
[hyperparameter/tensor loader and a matching computation graph](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md).

Sparse MoE is native for Mixtral-style GGUFs (including checkpoints that
declare `llama`), `qwen2moe`, and `qwen3moe`. The loader validates the router
and every `[input, output, expert]` tensor before loading; Mixtral/Qwen3 use
top-k-renormalized routing, while Qwen2-MoE preserves its full-router mass and
adds its gated shared expert. `gpt-oss` uses the same sparse foundation with
its expert/router biases, OAI-SwiGLU activation, learned attention sinks, and
alternating local/full-attention schedule. Sparse MoE prompt prefill stays
per-token so the decode graph and its quantized expert kernels are shared.

`qwen2` covers text-only Qwen2/Qwen2.5 GGUFs, including Qwen2.5-Coder,
Qwen2.5-Math, and QwQ checkpoints when they declare that architecture;
`qwen2moe` implements Qwen2-MoE's unnormalized selected-router mass and gated
shared expert. `qwen3` covers dense text-only Qwen3 (including Qwen3-based
DeepSeek-R1 distills) with mandatory per-head QK-norm. `qwen3moe` adds the
matching normalized sparse routing and QK-norm required by Qwen3-MoE, including
Qwen3-Coder GGUFs that declare `qwen3moe`. `qwen35` and `qwen35moe` have a
native experimental hybrid Gated-DeltaNet / periodic-attention path;
`qwen35moe` also uses the sparse-expert implementation and Qwen-style gated
shared experts (including Ornith-style GGUFs). The hybrid DeltaNet graph has
focused scalar-reference tests and text-only local-GGUF smoke coverage; full
cross-runtime logit parity remains pending, so the runtime emits an explicit
experimental warning. Vision families that require visual-feature injection
or multimodal MRoPE remain outside this text-generation scope: `qwen2vl`,
`qwen3vl`, `qwen3vlmoe`, and `qwen3next`. Qwen3.6 GGUFs with trailing MTP
draft layers load for ordinary generation; the draft layer is intentionally
skipped until speculative decoding is implemented.

`deepseek2` and `kimi_k2` provide a dedicated Multi-head Latent Attention
(MLA) path for Kimi K2 and compatible modern DeepSeek-V2/V3 GGUFs. It uses a
compressed KV cache, the split `attn_k_b`/`attn_v_b` MLA tensors, and the
sigmoid/noaux sparse router with its always-on shared expert. Group-limited
DeepSeek-V3 routing is native: it ranks the configured expert groups from
corrected sigmoid scores, chooses the final experts within those groups, then
mixes them using the original sigmoid probabilities. This path is currently
CPU-only (`--metal` is rejected); legacy fused `attn_kv_b` layouts and MLA
files without the compact modern metadata remain unsupported.

DeepSeek-R1 reasoning output is separated into `reasoning_content` in both
template conventions (self-opened `<think>` blocks and the newer forced-open
templates whose output begins mid-reasoning). Mistral-family models support
assistant-message prefill: a conversation ending in an assistant message
leaves the turn open so generation continues it.

Phi-3 (including the Phi-3.5 GGUFs that declare `phi3`), dense Granite,
EXAONE 3, and InternLM2 use GopherLLM's standard pre-norm RoPE/GQA/SwiGLU
decoder path. SmolLM3 shares the dense SwiGLU graph but uses interleaved RoPE
and deliberately omits RoPE in every fourth layer. EXAONE 4 has its own
post-norm block behavior: raw residual input feeds attention/FFN, Q and K are
RMS-normalized per head, branch projections are RMS-normalized before the
residual add, and the 32B model follows its three-local/one-global SWA/RoPE
schedule. Its `[|system|]` / `[|user|]` / `[|assistant|]` /
`[|endofturn|]` instruct protocol is rendered natively. StableLM adds
LayerNorm and learned norm biases. Its residual layout is selected from the
actual tensors rather than the sometimes-stale `use_parallel_residual`
metadata: checkpoints with `ffn_norm` (including Stable Code) use the
sequential attention-then-FFN graph, while variants without it feed attention
and FFN from one shared normalized input.
Sparse Granite MoE checkpoints remain intentionally rejected: their expert
router and expert tensors require the separate MoE execution graph.

`phi2` uses its native parallel block rather than the Phi-3 graph: one biased
mean/variance LayerNorm feeds both attention and the sequential, ungated
exact-GELU MLP; both branch outputs are added to the original residual.
Attention-output, FFN-up/down, output-norm, and vocabulary-output biases are
loaded and validated. The vocabulary bias is applied in both ordinary logits
and the allocation-saving greedy argmax path. Dense Phi-2 prompt ingestion
uses batched prefill; Phi-MoE remains a separate unsupported architecture.

`olmo2` covers both dense OLMo 2 and OLMo 3 GGUFs. The latter retains the
`olmo2` architecture label and adds a three-local/one-global sliding-attention
schedule. Q and K use one RMSNorm over their complete projections rather than
one norm per head; attention and FFN branches are normalized before their
residual adds. OLMo 3 local layers use a separate SWA frequency base with
ordinary unscaled RoPE, while global layers retain the checkpoint's normal
long-context scaling. Both decode and batched prefill select the matching
precomputed RoPE table per layer. Sparse `olmoe` remains unsupported.

Mistral-family instruct models (including Ministral) use the `[INST]…[/INST]`
chat format, the Tekken byte-level BPE pre-tokenizer, and YaRN RoPE context
scaling when the GGUF declares it.

`nemotron_h` and `nemotron_h_moe` are native hybrid Mamba-2 / attention
graphs. The dense variant (including NVIDIA Nemotron 3 Nano 4B) uses
`ffn_up → ReLU² → ffn_down`; the MoE variant retains its sparse router. Both
retain Mamba convolution and SSM state locally and do not rely on a llama.cpp
process. Prompt prefill is deliberately per-token for these architectures
because recurrent state makes the regular batched transformer prefill invalid.
The MoE variant also supports canonical `ffn_latent_down/up` projections and
the optional shared ReLU² expert.

Pure `mamba2` GGUFs (including the canonical 2.7B/7B family) run through a
native convolution/SSM recurrence with no fabricated attention cache. The
canonical `RMSNorm(y · SiLU(z))` gate and state update are shared with
Nemotron-H's Mamba-2 blocks; Mamba2 variants with an additional MLP branch are
rejected explicitly rather than being evaluated with an incompatible graph.

Gemma-family support (`gemma`/`gemma2`/`gemma3`/`gemma4`, including Gemma QAT
GGUFs) implements `sqrt(dim)` embedding scaling, GELU FFN, QK-norm,
post-attention/post-FFN norms, attention/final-logit softcapping and the
per-layer sliding-window map. Native **Gemma 4 12B dense**, **26B A4B MoE**
and **E2B** GGUFs use their real mixed local/global attention geometry: the
local/global RoPE bases and global proportional `rope_freqs.weight`, K-as-V
with V RMSNorm where declared, per-layer output scales, and the
`<|turn>…<turn|>` chat protocol are all executed. 26B additionally executes
its real shared-dense-plus-sparse GEGLU graph: scaled RMS router input, fused
expert gate/up banks, expert down scales, and the three branch/sum norms. E2B
executes the exact token-conditioned per-layer embedding residual (its global
pre-norm-scaled projection, gated exact-GELU residual) and its query-only shared-KV tail:
15 physical slots, with tail SWA blocks reading slot 13 and global blocks slot
14. Local 12B, 26B and E2B Q4 out-of-core decode smoke tests are green.

Multimodal `mmproj` files remain separate work. Gemma 1--3 and Gemma 4 retain
an experimental warning until cross-runtime logit-parity coverage is
available. A sensible Gemma sampling starting point is `--temp 1.0 --top-p
0.95 --top-k 64`.

Projector files such as `mmproj-*` are detected and excluded from text-model
selection.

`bert` and `nomic-bert` are encoder-only architectures: they are available as
embedding models for `/v1/embeddings` and history RAG, but intentionally cannot
be selected for chat generation. This covers BERT-format Granite Embedding
GGUFs as well as Nomic Embed GGUFs carrying `general.architecture = nomic-bert`.
GGUF tokenizers declaring `tokenizer.ggml.model = bert` use native WordPiece
normalization and greedy segmentation, including CLS/SEP boundaries and both
llama.cpp's phantom-space vocabulary layout and raw `##` continuation pieces.

## Development

### Project layout

| Area | Files |
|---|---|
| GGUF parsing + file mapping | `gguf.go`; public facade in `mmap.go`, platform backends in `internal/mmapfile/` |
| Model loading + forward pass | `model.go`, `forward_batch.go` (batched prefill) |
| Compute kernels + worker pool | `simd.go`; platform dispatch and assembly grouped in `kernels_*.go` / `kernels_*.s` |
| Generated inference tables | `internal/iqcodebook/` |
| Tokenizer normalization tables | `internal/wordpiece/` |
| Tokenizers | `tokenizer.go` (SentencePiece + GPT-2/Tekken BPE + BERT WordPiece) |
| Sampling | `sampling.go` |
| Generation orchestration + chat templates | `runtime.go` |
| Tool calling / reasoning / skills | `agent.go`, `extract.go`, `skills.go`; wire types and helpers in `internal/tooling/` |
| Model discovery + selection | `catalog.go` |
| HTTP server | `server/server.go`, `server/web_ui/` |
| CLI | `cmd/gopherllm/main.go`, `lib.go` (package doc + version), `kernel_bench.go` |

A full architecture walkthrough — load path, inference data flow, kernel
dispatch tiers, and how to add a quant kernel / architecture / endpoint — is
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

For a concise directory map and guidance on where new code belongs, see
[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

For a concise directory map and guidance on where new code belongs, see
[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

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

Validate every supported GGUF layout in a local model library without loading
all weights into RAM:

```sh
GOPHERLLM_MODEL_SMOKE_DIR="$HOME/.cache/lm-studio/models" \
  go test -run TestLocalGGUFLoadSmoke -v ./integration
```

For a slower end-to-end answer check of every supported text model below 5 GB,
first build the CLI and then run the opt-in sweep. Embedding GGUFs are
deliberately excluded from chat generation and remain covered by the loader
test plus `/v1/embeddings` tests.

```sh
go build -o /tmp/gopherllm-model-sweep ./cmd/gopherllm
GOPHERLLM_RUN_MODEL_SWEEP=1 \
GOPHERLLM_SWEEP_BINARY=/tmp/gopherllm-model-sweep \
GOPHERLLM_MODEL_DIR="$HOME/.cache/lm-studio/models" \
  go test -run TestSmallLocalModelsAnswerEinsteinPrompt -v ./integration
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
