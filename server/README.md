# GopherLLM demo application

This directory contains GopherLLM's optional demo application: the `server`
Go package, its OpenAI-/Ollama-compatible HTTP API, and the embedded browser
chat. It is built on the in-process `gopherllm` inference package; the core
package and command-line inference workflow are documented in the
[repository root](../README.md).

## Contents

- [Start locally](#start-locally)
- [Embed the handler](#embed-the-handler)
- [Configuration and deployment profiles](#configuration-and-deployment-profiles)
- [Remote OpenAI-compatible APIs](#remote-openai-compatible-apis)
- [Browser workspace](#browser-workspace)
- [Context and prefix cache](#context-and-prefix-cache)
- [HTTP API](#http-api)
- [Tools, skills, and research](#tools-skills-and-research)
- [Autotuning from the UI](#autotuning-from-the-ui)
- [Server Make targets](#server-make-targets)
- [Privacy](#privacy)

## Start locally

Build the CLI, load a local GGUF, and serve the embedded chat UI:

```sh
make build
bin/gopherllm --model-dir "$HOME/.cache/lm-studio/models" \
  --model "model-name-or-file-fragment" \
  --serve 127.0.0.1:8080 \
  --chat
```

Open <http://127.0.0.1:8080/chat>. The application is optional: importing the
root Go package or using the CLI for local generation does not start it.

The CLI remembers the successfully loaded local GGUF in `last-model.json`
below the platform configuration directory (override with
`GOPHERLLM_MODEL_STATE_PATH`). Subsequent server starts can omit the model
selector; an explicit selector always wins. If the saved model is unavailable,
choose a discovered model in the browser picker or call `POST /models/load`.

## Embed the handler

Applications that already own an HTTP server can mount the entire API under
their own router, path prefix, and middleware stack:

```go
import (
    "net/http"

    gopherllm "github.com/SimonWaldherr/GopherLLM"
    "github.com/SimonWaldherr/GopherLLM/server"
)

model, err := gopherllm.Open(ctx, "model.gguf")
if err != nil {
    return err
}
defer model.Close()

mux.Handle("/llm/", http.StripPrefix("/llm",
    server.HandlerForModel(model, server.HandlerOptions{
        Defaults: gopherllm.DefaultGenerationOptions(),
    }),
))
```

`server.NewHandler` returns a closeable `*server.Handler`. A host that enables
model hot-swapping should stop its HTTP server and call `handler.Close()` to
release the active chat and embedding model mappings; `server.Serve` does this
when it returns.

The server API moved out of the root package: use `server.NewHandler`,
`server.HandlerOptions`, `server.ServeOptions`, and
`server.HandlerForModel(model, opts)` instead of the former `gopherllm.*`
server symbols.

## Configuration and deployment profiles

The server portion of an explicit `--config` file lives under `server`:

```json
{
  "version": 1,
  "model_dir": "/path/to/models",
  "server": {
    "address": "127.0.0.1:8080",
    "chat": true,
    "max_connections": 16
  }
}
```

Choose an explicit deployment profile when exposing the app. The policy is
enforced by the HTTP server; a hidden browser control is never the only
protection.

| Profile | Intended use | Inference and control boundary |
| --- | --- | --- |
| `local` (default) | One person on one laptop | Listener must use a loopback address. The owner may select models, download, tune, and change settings. |
| `managed` | Shared server | Users can generate replies. Model changes, downloads, embedding-model loads, tuning, remote credentials, and agentic OS actions require the administrator token. |
| `browser` | Browser-hosted local inference | The app serves only the UI and WASM runtime. Each tab selects and runs its own GGUF through WASM/WebGPU; server inference and model controls are disabled. |

Local setup:

```sh
bin/gopherllm --model-dir /path/to/models --serve 127.0.0.1:8080 --chat \
  --deployment local
```

Managed setup (keep the token file readable only by the service account):

```sh
bin/gopherllm --model-dir /srv/gopherllm/models --model my-model \
  --serve 0.0.0.0:8080 --chat --deployment managed \
  --admin-token-file /etc/gopherllm/admin-token
```

Send the token as `X-GopherLLM-Admin-Token` or
`Authorization: Bearer ...`, or enter it in **Server administration** in the
browser for the current tab. It is not written to browser storage, URLs, logs,
or `--print-config` output. The built-in token protects operational controls,
not user identity or transport: use HTTPS and normal authentication at a
reverse proxy before exposing a managed server.

Browser-only setup:

```sh
make wasm-build
bin/gopherllm --serve 0.0.0.0:8080 --deployment browser --wasm-dir ./bin
```

This mode intentionally rejects a server-side `--model`: prompts, images, and
model bytes remain in the browser. WebGPU, camera, and screen capture require
HTTPS outside `localhost`.

## Remote OpenAI-compatible APIs

The local server can proxy `/v1/chat/completions` to an existing
OpenAI-compatible upstream. This keeps the chat UI/API available alongside
Ollama, llama.cpp, LM Studio, RustyLLM, or a hosted provider. It is a chat-only
proxy; the local engine remains independent of it.

Configure the upstream on a trusted server. The API key remains only in server
memory and is never returned by the configuration endpoint:

```sh
curl http://127.0.0.1:8080/remote \
  -H 'Content-Type: application/json' \
  -d '{"base_url":"http://127.0.0.1:11434","model":"llama3.2"}'
```

For OpenAI, use a body such as
`{"base_url":"https://api.openai.com/v1","api_key":"…","model":"…"}`.
`DELETE /remote` switches back to the local model and `GET /remote/models`
lists the upstream's advertised models.

## Browser workspace

The browser chat is a local workspace, not just a request form. Conversations,
drafts, per-chat instructions, sampling settings, and appearance are stored in
the browser's IndexedDB. It supports search, rename/delete, non-destructive
edit and retry branches, local text-file insertion, and JSON/Markdown export.
Changing a model creates a comparison branch and resubmits the same question.
Pinned chats stay above normal history and are preferred during bounded history
cleanup.

Nothing is synchronised to a third party by default. Assets use `no-store` and
same-origin security headers; run on a trusted local address unless a reverse
proxy adds network security. The embedded UI loads no packages, fonts, or
scripts by default. An operator can opt into a Mermaid renderer with
`?mermaid=jsdelivr` (or `unpkg`/`cdnjs`); only then is that chosen origin added
to the page's content-security policy.

**Model & chat** presents discovered GGUFs as a searchable library with
architecture, file size, context length, compatibility, load progress, and the
active model. Unsupported or auxiliary GGUFs stay hidden unless requested.
Settings separate model/chat, capabilities, generation, and workspace controls
without replacing manual sampling or context choices.

For a shared-device workspace, opt into server-backed history:

```sh
bin/gopherllm --chat-history ./gopherllm-chats.json.gz --serve 127.0.0.1:8080 --chat
```

Alternatively set `server.chat_history_path` in the JSON config and select
**GopherLLM server** under Settings → Workspace → Chat data. The server writes
a compact gzip file atomically with mode `0600`, caps the uncompressed workspace
at 64 MiB, and uses ETags to reject stale writes. It has no built-in user
authentication, so protect it before sharing.

The composer accepts any file type. Text files up to 500 KB are included in the
request; `.xlsx` and `.ods` are parsed locally into bounded tabular text.
Images, audio, video, PDFs, archives, and other binary files remain in the
browser as attachment cards unless a matching model path handles them; the
text-only server sends only their filename, type, and size. Batch mode accepts
JSON arrays/JSONL, CSV/TSV, Markdown chapters, text, `.xlsx`, and `.ods`.

With **Power commands**, `/goal` accepts `--rounds 2..8` and an optional
`--focus "…"`; `/review` audits the current conversation and `/plan` produces
an ordered implementation plan. Intermediate prompts are retained as ordinary
workspace history and passed to the model with an explicit untrusted-reference
boundary.

## Context and prefix cache

The chat defaults to **Smart — recent complete turns**. Full history stays in
IndexedDB, but each request selects the newest complete turns that fit the
loaded model's actual template and token budget. Leading system instructions
stay pinned, tool calls stay with the originating user turn, and the latest
turn is never cut mid-message.

**Auto-compress — dense technical context** conservatively shortens ordinary
prose only when the active tokenizer confirms a smaller result. Tool payloads
and fenced code remain unchanged. **Full history — stop when full** preserves
the strict behaviour used by normal API clients. The API can opt into the
local modes with:

```json
{"gopherllm_context_mode":"autoCompress"}
```

Allowed values are `recent`, `autoCompress`, and `full`. Non-streaming replies
include `X-GopherLLM-Context-*` headers and a `gopherllm_context` object;
streaming replies carry the final data in the terminal SSE choice.

While a local server runs, it retains one bounded KV prefix cache for the most
recent rendered prompt. Follow-up turns can process only the changed suffix;
edits and branches reuse only an exact unchanged prefix. The cache uses the
normal generation workspace, is capped at 512 MiB, and is memory-only. Responses
expose `gopherllm_cache` with `mode`, `hit`, `reused_tokens`, and
`prompt_tokens`; OpenAI-compatible usage also reports
`usage.prompt_tokens_details.cached_tokens`.

## HTTP API

Minimal OpenAI-compatible request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about Go."}],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

Set `"stream": true` for SSE streaming. The handler exposes:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness and loaded-model id |
| POST | `/generate` | Native generation API (prompt/messages, tools) |
| POST | `/v1/chat/completions` | OpenAI-compatible chat, streaming, tools, and reasoning |
| POST | `/v1/completions` | OpenAI-compatible text completion |
| POST | `/v1/embeddings` | OpenAI-compatible embeddings |
| GET | `/v1/models` | OpenAI-compatible loaded-model listing |
| GET | `/v1/skills` | Configured skill names and descriptions |
| POST | `/api/generate` | Ollama-compatible generation |
| POST | `/api/chat` | Ollama-compatible chat with tools |
| POST | `/api/embeddings` | Ollama-compatible embeddings |
| GET | `/models` | Discover GGUFs under `--model-dir` |
| POST | `/models/load` | Hot-swap a supported discovered GGUF |
| POST | `/models/embed/load` | Load an embedding GGUF for API embeddings/history search |
| GET / POST / DELETE | `/remote` | Inspect, configure, or clear an upstream chat proxy |
| GET | `/remote/models` | List upstream models |
| GET | `/autotune` | Report active/cached Auto Mode state |
| POST | `/autotune/run` | Run or apply Auto Mode tuning |
| GET | `/chat`, `/style.css`, `/script.js` | Browser UI (`--chat`) |
| GET | `/chat/storage` | Report server-history availability |
| GET / PUT / DELETE | `/chat/workspace` | Read, replace, or clear server workspace |
| POST | `/batch/parse` | Parse local `.xlsx`/`.ods` batch data |

## Tools, skills, and research

`/v1/chat/completions`, `/generate`, and `/api/chat` accept an OpenAI-shaped
`tools` array. A model that requests a tool returns
`finish_reason: "tool_calls"`; append that assistant message and a `role: "tool"` result to
the next request. Mistral-family models use their native tool format; other
templates use the generic `<tool_call>{...}</tool_call>` convention. Set
`"tool_choice":"none"` to disable offering tools for one request.

Models that emit `<think>...</think>` (and gpt-oss channels) return reasoning
separately as `reasoning_content`, including streaming deltas, rather than
leaving it mixed into visible text.

### Skills

Point `--skills-dir` at a directory of `SKILL.md` files:

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

The model sees each name/description through a `load_skill` tool and receives a
full skill body only if it asks for it. In the server application that loop is
resolved internally before a response is returned; callers still receive their
own non-skill tool calls as usual. `GET /v1/skills` lists configured skills.
`--skills-dir` also works in CLI one-shot and REPL mode.

### Wikimedia and OpenStreetMap

The browser can opt into **Wikipedia & Wikidata research** per chat; API
clients set `"gopherllm_wikimedia": true`. The model can use bounded
`wikipedia_search`, `wikipedia_summary`, `wikidata_entity`, and read-only
`wikidata_sparql` tools. Only the selected search term/title/Q-ID/query is sent
to Wikimedia, never the full chat transcript. SPARQL permits only `SELECT` or
`ASK`, rejects `SERVICE`/updates, and caps results at 25 rows.

**OpenStreetMap place research** is separately opt-in with
`"gopherllm_openstreetmap": true`. Only the bounded model-selected place query
is sent to the configured Nominatim endpoint. The public endpoint is limited to
direct low-volume lookups (at most one request per second); configure a
self-hosted compatible service with `server.HandlerOptions{OSMSearchURL: "..."}`
for larger workloads. See the [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/).

Go hosts can construct the same optional research tools directly:

```go
tools := server.NewResearchTools(server.ResearchOptions{
    Wikimedia:     true,
    OpenStreetMap: true,
})
result, err := gopherllm.RunAgenticChatWithTools(
    model.Runner(),
    []gopherllm.ChatMessage{gopherllm.UserMessage("Where is the Brandenburg Gate?")},
    gopherllm.DefaultGenerationOptions(), nil, tools, nil,
)
```

Every result carries a source URL and attribution. No research source is active
by default.

## Autotuning from the UI

With `--chat`, Settings contains a **Performance** panel with an effort
selector and **Tune now** button. It uses `GET /autotune` and
`POST /autotune/run`, and distinguishes a tuning active in this process from a
cached result from an earlier run. Tuning pauses generation while measurement
holds the shared runner lock. The underlying `--auto` modes and methodology are
documented in the [root README](../README.md#auto-mode-hardware-autotuning).

## Server Make targets

`make serve MODEL=... CHAT=1` starts the HTTP server and chat UI. On macOS,
`make serve` auto-detects Metal when Xcode Command Line Tools are present;
`METAL=0` forces the portable CPU build.

```sh
make serve MODEL="my-model.gguf" CHAT=1
make serve-metal MODEL="my-model.gguf" CHAT=1 THREADS=8
make serve-auto MODEL="my-model.gguf" CHAT=1
make serve-auto-metal MODEL="my-model.gguf" CHAT=1
```

`serve-metal` prepares CPU fallback kernels by default
(`PREPARE_QUANT=0` disables preparation). Add `AUTO=1` to tune before accepting
requests; `AUTO_EFFORT=quick|balanced|thorough` selects calibration depth.
`SKILLS_DIR=path/to/skills` enables skills for `serve`, as it does for the CLI
inference modes.

## Privacy

Local inference makes no network request and GopherLLM has no telemetry. Use:

```sh
gopherllm --privacy
curl http://127.0.0.1:8080/privacy
```

The report names every opt-in feature that can send data externally, its
destination, and the limited data it may send. Hugging Face imports, remote
proxies, and factual research sources remain opt-in.
