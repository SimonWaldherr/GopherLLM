# GopherLLM

[![DOI](https://zenodo.org/badge/1264366305.svg)](https://doi.org/10.5281/zenodo.21197831)

GopherLLM is a local GGUF inference engine written in Go. It loads and runs models
directly in the calling process, with no external runtime or child process
required. The Go package and CLI cover generation, streaming chat, embeddings,
tokenization, model inspection, compression, and benchmarks.

It is an independent implementation. GopherLLM does not wrap, bind to, or link
against llama.cpp or any other inference library: the GGUF parser, the
tokenizers, the quantized kernels, and the assembly are written for this
project. `go.mod` has no `require` block at all, and `make deps-check` keeps it
that way — see [Dependency policy and layout](#dependency-policy-and-layout).
The same reasoning has its own Rust sibling in
**[RustyLLM](https://github.com/SimonWaldherr/RustyLLM)**.

**[Project website](https://simonwaldherr.github.io/GopherLLM/)** ·
**[Go package documentation](https://pkg.go.dev/github.com/SimonWaldherr/GopherLLM)** ·
**[Demo application documentation](server/README.md)**

## Try it in five minutes

```sh
make build
bin/gopherllm --model-dir /path/to/your/models --serve --chat
```

That opens a chat server on `http://127.0.0.1:8080/chat`: loopback only, with
the optional features off, and a model picker for the GGUFs in that directory.
If you have no GGUF yet, `bin/gopherllm --hf-list <owner/repo>` lists the
variants in a Hugging Face repository and `hf:<owner>/<repo>:<quant>` in place
of the model path downloads one into the shared HF cache. See
[Serving and safety defaults](#serving-and-safety-defaults) for what "optional"
covers and how to turn things on.

## Contents

- [Try it in five minutes](#try-it-in-five-minutes)
- [Features](#features)
- [Requirements](#requirements)
- [Dependency policy and layout](#dependency-policy-and-layout)
- [Quickstart](#quickstart)
- [Use as a Go Library](#use-as-a-go-library)
- [Build](#build)
- [Serving and safety defaults](#serving-and-safety-defaults)
- [CLI Usage](#cli-usage)
- [GGUF Analyzer](#gguf-analyzer)
- [Model Compression](#model-compression)
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
- Chain-of-thought extraction (`<think>` blocks, gpt-oss channels) into a
  separate `reasoning_content` field instead of leaving it in the answer text.
- CLI generation, REPL mode, embeddings, metadata inspection, and tensor listing.
- Model discovery across the complete local LM Studio model library.
- Direct Hugging Face GGUF imports with cache reuse, split-model downloads,
  private/gated-model tokens, and revision selection.
- `--compress`: requantize any GGUF to Q8_0/Q4_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, writing
  a smaller, independently loadable file (see
  [Model Compression](#model-compression)).

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
enforces that policy.

The module-root Go files form the public `gopherllm` package, so core package
sources remain there to preserve the stable import path
`github.com/SimonWaldherr/GopherLLM`. Architecture-specific kernel dispatch and
assembly are consolidated into `kernels_<arch>.go` / `kernels_<arch>.s`
instead of being spread across one file per operation. Executable entry points
live in `cmd/`, generated tables and other implementation details in `internal/`,
and test-only fixtures—including preserved profiling captures—in `testdata/`.
Public-boundary and opt-in local model tests live in `integration/`. Build output
and local model/RAG data are ignored and are not part of the repository.

The optional demo application is documented separately in
[server/README.md](server/README.md).

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

### Reproducible configuration

For repeatable local runs or deployments, pass one explicit JSON file with
`--config`. Precedence is **built-in defaults and environment defaults → config
file → CLI flags**. The file is strict: unknown fields and unsupported schema
versions fail early instead of being silently ignored. It is never discovered
automatically, and it contains no access tokens.

```json
{
  "version": 1,
  "preset": "precise",
  "model": "hf:bartowski/Qwen3-4B-GGUF:Q4_K_M@main",
  "model_dir": "/path/to/models",
  "generation": {
    "max_tokens": 384,
    "temperature": 0.25,
    "top_p": 0.9,
    "stop": ["<|end_of_text|>"],
    "context_window_mode": "recent"
  },
  "runtime": {
    "threads": 8,
    "prepare_quant": true,
    "timeout": "2m"
  },
  "huggingface": {"offline": true}
}
```

```sh
bin/gopherllm --config ./gopherllm.json
bin/gopherllm --config ./gopherllm.json --temp 0.4 --prompt "Summarize this."
bin/gopherllm --config ./gopherllm.json --print-config
```

`balanced` is the default sampler preset; `precise`, `creative`, and
`deterministic` offer deliberate starting points. A preset is applied before
explicit generation values, so a config or CLI `--temp`/`--top-p` setting can
always fine-tune it. `--print-config` emits a reusable effective config and
never prints `HF_TOKEN` or a transient user prompt.

`context_window_mode` (or `--context-window`) is `full` by default, which
returns a context-limit error instead of dropping history. Choose `recent` to
retain complete recent turns, or `autoCompress` to compact ordinary prose
before applying the same turn-preserving selection.

### Hugging Face imports

Use an `hf:` selector to download a GGUF directly. Add the quantization after
the repository name when it contains more than one GGUF, and optionally add a
branch, tag, or commit after `@`:

```sh
bin/gopherllm hf:bartowski/Qwen3-4B-GGUF:Q4_K_M --repl
bin/gopherllm hf:bartowski/Qwen3-4B-GGUF:Q4_K_M@main \
  --prompt "Explain local inference." --max-tokens 128
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

For an auditable air-gapped run, add `--hf-offline` (including to `--hf-list`).
It makes no HTTP request and accepts only a complete cached snapshot for the
selected revision. GopherLLM also honors Hugging Face's shared
`HF_HUB_OFFLINE=1` setting, so Python tooling and GopherLLM follow the same
network policy:

```sh
bin/gopherllm hf:bartowski/Qwen3-4B-GGUF:Q4_K_M@main --hf-offline --repl
HF_HUB_OFFLINE=1 bin/gopherllm --hf-list bartowski/Qwen3-4B-GGUF@main
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

### Repeated Mistral chat prefixes

For a Ministral/Mistral application that repeatedly renders the same system
prompt and tool schema (especially with `PrepareChatContext`), opt into the
small, per-model render cache:

```go
runner := model.Runner()
runner.EnableMistralPromptPrefixCache(8 << 20) // 8 MiB; 0 selects this default
defer runner.DisableMistralPromptPrefixCache()

// Optional explicit boundary when rotating sensitive system instructions.
runner.ClearMistralPromptPrefixCache()
```

It retains only tokenized BOS/system/tool-prefix text; message content, image
embeddings, logits, and KV state are not stored by this cache. The normal
generation KV-prefix cache remains independent.

### Package layout

The root package is inference only. Its import closure excludes the optional
demo application and its UI assets:

| Import | You get | Transitive deps |
|---|---|---|
| `github.com/SimonWaldherr/GopherLLM` | GGUF loading, generation, chat, embeddings, tokenizer, sampling, autotuning, skills/agent loop | 90 |
| `github.com/SimonWaldherr/GopherLLM/huggingface` | Hub search, variant listing, and `owner/repo` → local GGUF resolution | 192 |

`TestInferencePackageStaysFreeOfServerDependencies` enforces that boundary
against the real dependency graph, so it cannot regress silently.

Those counts are all standard library — the module has no `require` block, no
`go.sum` and no vendor directory — but stdlib is not the same as free, since an
embedder compiles every package in the closure. The root package therefore
avoids `crypto/sha256` and `regexp`, each of which cost far more than its single
use was worth, and `TestInferencePackageStaysFreeOfTheCryptoAndRegexpTrees` pins
that.

The PNG and JPEG decoders are the remaining optional chunk. Vision needs them,
so they are a build-time knob rather than a removal:

```sh
go build -tags noimagedecoders ./...
```

That takes the root package from 90 transitive dependencies to 83, dropping
`image/png` and `image/jpeg` along with `compress/zlib`, `compress/flate`,
`hash/adler32` and `hash/crc32`. The public API is unchanged either way —
`image.Image` stays in `DecodeImageBytes`' and `PreprocessImagePixtral`'s
signatures — and so is the default build; what the tag costs is PNG and JPEG
input to the vision path. Because Go's image-format registry is process-global,
a program built with the tag can take back just the formats it wants with its
own `import _ "image/png"`.

Hub access is a separate import for the same reason: it needs `net/http`, which
drags in `crypto/tls` and the HTTP/2 stack. A program that only ever opens a
local file never pays for that, and one that wants
`hf:owner/repo` resolution opts in explicitly:

```go
path, err := huggingface.Resolve(ctx, "hf:owner/repo:model-Q4_K_M.gguf", os.Stderr, huggingface.DefaultOptions())
if err != nil {
    return err
}
model, err := gopherllm.Open(ctx, path)
```

Models already pulled with Ollama need no download at all — they are ordinary
GGUF blobs in a content-addressed store, so `DiscoverOllamaModels` reads them in
place, keyed by the same `name:tag` you would pass to `ollama run`:

```go
entries, err := gopherllm.DiscoverOllamaModelsDefault(os.Stderr)
```

`--list-models` includes them automatically. Set `OLLAMA_MODELS` when the store
is not at `~/.ollama/models`. This lets an embedder reuse existing local model
files without another download.

The library never writes to stdout/stderr on its own; pass
`gopherllm.WithLogWriter(os.Stderr)` to opt into diagnostics. Tool calling,
reasoning extraction, and agentic runs are available via `WithTools`,
`Result.ReasoningText`, and `RunAgenticChat`; see the godoc and runnable
examples in `example_test.go`. `testdata/consumer` is a complete external
application using the API.

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

## Serving and safety defaults

`--serve` starts the HTTP API; `--chat` adds the browser workspace at `/chat`.

```sh
bin/gopherllm --model-dir /path/to/models --serve --chat
```

Two defaults are deliberate.

**It listens on loopback.** `--serve` without an address binds
`127.0.0.1:8080`, so nothing outside the machine can reach it. Giving it a
network address is allowed — sharing a model with a phone or a second machine
is a reasonable thing to want — but it prints what that exposes, and a *local*
deployment reached that way switches its privileged routes off: model loading,
downloads, autotune, remote forwarding, and OS commands all answer 403, because
local mode has no token to check and "everyone on the subnet is an
administrator" is not a boundary. Use `--deployment managed` with an admin
token to keep those controls available on a shared server.

**Optional capabilities are off.** A server that nobody configured is a chat
and completions server plus the model catalog for `--model-dir`. Anything that
reaches the internet, rewrites process-wide state, or benchmarks the host has
to be asked for by name:

| `--enable` name  | Adds                                                    |
| ---------------- | ------------------------------------------------------- |
| `model-download` | `/models/search`, `/models/download` (Hugging Face)      |
| `autotune`       | `/autotune`, `/autotune/run`                             |
| `remote`         | `/remote` — forwards completions to another endpoint     |
| `web-lookup`     | Wikimedia and OpenStreetMap tools for chat requests      |
| `spreadsheet`    | `/batch/parse` for the batch runner                      |
| `all`            | Everything above (also spelled `--full`)                 |

```sh
bin/gopherllm --model-dir /path/to/models --serve --chat --enable model-download,autotune
```

A capability that is off is not registered at all, so its route answers 404
rather than presenting a permission check, and the Web UI drops the panels for
it instead of showing controls that cannot work. OS commands stay separate and
off unless `--os-commands` sets a policy. The same set is available to library
users through `server.Features` and `server.AllFeatures()`, and to config files
under `server.features`.

The Web UI itself opens in a **Simple** mode that shows the handful of settings
most people change; the **Advanced** toggle in the settings header reveals the
rest. That choice is per browser and changes nothing on the server.

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

Run with a local skill available (one-shot or REPL alike):

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

## Model Compression

Requantize a GGUF to a smaller format and write it to a new file:

```sh
bin/gopherllm /path/to/model.gguf --compress --compress-format Q4_K --compress-out smaller.gguf
```

Supported target formats: `Q8_0`, `Q4_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
`Q6_K`. Every eligible weight matrix (2-D or higher, row length divisible by
the target format's block size — 32 for Q8_0/Q4_0, 256 for K-quants) is
dequantized, if not already plain float, and requantized with
round-to-nearest; norm/bias vectors and any tensor that doesn't fit the target
block size are copied through unchanged. `Q3_K` reduces packed weight traffic
by about 24% relative to Q4_K (110 vs. 144 bytes per 256 weights); `Q2_K`
reduces it about 42% (84 vs. 144 bytes), but both are explicitly quality-for-
speed trade-offs and should be evaluated on representative prompts.
They currently use the CPU quantized kernels rather than the selective
Q4_K/Q6_K Metal fast path, so use `--bench-json` on the deployment machine
before choosing a low-bit artifact for GPU inference.
The result is an ordinary, independently loadable GGUF — no relationship to
the source file is retained, and the output is re-parsed and spot-checked
against the plan before `--compress` reports success.

The token embedding and output/LM-head tensors are floored at Q6_K rather
than following an aggressive target format down to e.g. Q4_0 — the same
convention llama.cpp's own `quantize` tool uses, since those two tensors are
disproportionately sensitive to quantization error. Pass `--compress-uniform`
to quantize them to the main target format like everything else instead.

`--compress-out` must name a different file than the source: the source
stays memory-mapped and readable throughout, so writing over it in place
would corrupt the mapping mid-read. Compress to a new path and rename it
afterward if you want to replace the original. A source that is one shard
of a split/sharded GGUF (`split.count` in its metadata) is rejected with a
clear error rather than silently compressing only that shard's tensors —
point `--compress` at a single merged file.

```sh
$ bin/gopherllm model-f16.gguf --compress --compress-format Q4_K --compress-out model-q4_k.gguf
Compressing model-f16.gguf -> model-q4_k.gguf (Q4_K)
requantize blk.0.attn_q.weight              F16 -> Q4_K (393216 bytes -> 110592 bytes)
...
total: 6871947264 bytes -> 2147024896 bytes (31.2%)
Wrote model-q4_k.gguf
```

This is plain round-to-nearest quantization today — no calibration data, no
weighted error minimization (unlike llama.cpp's `imatrix`-guided quantize).
It's the foundation a planned calibration-aware toolkit (GPTQ, AWQ,
SmoothQuant, and SparseGPT-style pruning) builds on, to get closer to
`llm-compressor`-style quality at low bit-widths; that work targets the
mainline dense-transformer attention/FFN tensors specifically and hasn't
landed yet.

The same feature is available in the library as `gopherllm.CompressModel`
and `gopherllm.ParseCompressFormat`.

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
only the first run pays for calibration. It applies to one-shot generation,
`--repl`, and `--bench`.

| Flag | Effect |
| --- | --- |
| `--auto` | Tune (or reuse a cached tuning) before generating |
| `--auto-effort quick` | Decode knobs only, ~8s. Prefill samples cost a whole chunk of prompt processing, so they are skipped here |
| `--auto-effort balanced` | Default. Adds prefill chunk tuning, ~1-2 min on a 3B model |
| `--auto-effort thorough` | More interleaved rounds and a 2048-token probe context; minutes, but the most reliable on a noisy machine |
| `--auto-refresh` | Re-measure and overwrite the cached result |
| `--auto-json` | Print the full result — including every candidate's median — and exit |

### Makefile workflow

The generation, REPL, and model-benchmark targets accept `AUTO=1`, so the same
cached tuning is used regardless of how the model is started:

```sh
# Fast first-pass calibration, then generate.
make run MODEL="my-model.gguf" AUTO=1 AUTO_EFFORT=quick

# Inspect the cached result (or force a fresh calibration) as JSON and exit.
make autotune MODEL="my-model.gguf"
make autotune MODEL="my-model.gguf" AUTO_REFRESH=1
```

`AUTO_EFFORT` accepts `quick`, `balanced` (the default), or `thorough`.
`AUTO_REFRESH=1` bypasses the cache. `AUTO_JSON=1` can be used with any of
those Make targets, but it prints the tuning report and exits before generation;
`make autotune` is the convenient report-only target. `run-auto` and
`run-auto-metal` are shortcuts for their corresponding command with `AUTO=1`.

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

- `make build`, `make run`, and `make repl` auto-detect Metal:
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
- `run` and `repl` accept `SKILLS_DIR=path/to/skills` to enable skills; they
  also accept `MIN_P`,
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
- Use `--out-of-core` when a GGUF, especially a large sparse-MoE
  model, does not fit comfortably in RAM or Apple unified memory. It keeps the
  model CPU-only, disables Metal and prepared-quant copies, leaves F16/F32/BF16
  matrices as mmap-backed scalar bytes, and does not prewarm the rank-3 expert
  banks (including fused `ffn_gate_up_exps` layouts). The operating system
  pages selected experts in on demand; this is not
  a hard RSS limit, so a cold expert can add SSD/page-fault latency and dense
  models that are far larger than RAM can still thrash.
  **Split (sharded) GGUFs are supported out-of-core and are the main reason to
  reach for it** — every checkpoint big enough to need demand paging ships as
  `-00001-of-000NN.gguf` shards. Each shard stays mapped independently and no
  merged copy is materialised, so peak memory tracks resident pages rather than
  model size; the default (non-out-of-core) split path still concatenates the
  shards into one buffer, which needs RAM for the whole model.
  It intentionally rejects `--metal`, `--prepare-quant`, `--auto`, and byte-backed loads.
  Library users can use
  `gopherllm.WithOutOfCore(true)` and optionally
  `WithMmapPrefault(gopherllm.MmapPrefaultNone)` for fully lazy paging.
- Use `--temp 0 --top-k 1` for deterministic greedy output.
- Use `--min-p <F>` (e.g. `0.05`) for min-p nucleus sampling; `0` disables it.
- `--bench-json` and `--kernel-bench-json` are intended for repeatable performance
  comparisons.
- Metal requires a build with `CGO_ENABLED=1 -tags metal` (the plain `make
  build`/`run` targets do this automatically on macOS when Xcode
  Command Line Tools are present; `make build-metal` does it explicitly
  regardless of platform detection) and must be enabled with `--metal` at
  runtime (also automatic from the `make` targets above; pass it yourself for
  a manually built binary). The selective
  path fuses sufficiently large mixed Q4_K/Q4_K/Q6_K Q/K/V projections into
  one command buffer and offloads large Q4_K projections, Q4_K gate/up + SiLU
  + Q6_K FFN-down in one command buffer, and Q6_K vocabulary-output
  projections. Narrow GQA K/V and 3K-5K-row attention-output projections stay
  on the faster CPU Q8_K/SIMD path; on the measured M2 Max, forcing those
  Ministral 3B/14B shapes through Metal was 35-65% slower per kernel.
  Deterministic decode
  (`--temp 0` or `--top-k 1`) also applies the bounded recent-token repeat
  penalty on-device before reducing the vocabulary output to its argmax. This
  keeps the default repeat penalty from forcing a 131k-logit readback and CPU
  scan on Ministral 3B. GGUF files opened
  through mmap are exposed to Metal as shared no-copy weight buffers;
  byte-backed models retain the copying path for cgo safety. Prepared ARM64
  kernels remain as the fallback for small projections and Metal failures. The
  path remains experimental; use
  `--kernel-bench-json` and `--bench-json` on the target Mac before deployment.
- On x86-64 (AVX2 + FMA + F16C, auto-detected via CPUID), Q4_K, Q5_K, Q6_K,
  Q8_0, Q4_0, Q4_1, MXFP4, Q2_K, and Q3_K matvecs default to int8-activation full-row kernels: the activation
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
  Q3_K is the newest of these and the one that gains most: its high bit is a
  per-*element* −4 bias rather than the per-sub-block constant the other
  K-quants carry, which is why it had no SIMD path for so long. It factors as
  `Σ(u + 4h)·a − 4·Σa`, keeping `u + 4h ≤ 7` inside `VPMADDUBSW`'s unsigned
  operand and costing one extra multiply-add against a vector of ones. Median
  of five interleaved runs on a 4096-column row: 54.8 µs scalar to 1.47 µs,
  about 37x.
- The same int8-activation kernels run on **every** arm64 target, not just
  Apple Silicon. They are hand-encoded `SDOT` (`FEAT_DotProd`), which is
  optional before ARMv8.4, so they used to be gated to `darwin && arm64` where
  the feature is guaranteed — leaving Graviton, Ampere, Snapdragon, Raspberry
  Pi 5 and Android on the portable scalar path for the hottest kernel in the
  engine. A startup probe (`/proc/self/auxv` `HWCAP_ASIMDDP` on Linux and
  Android, `IsProcessorFeaturePresent` on Windows, unconditional on Apple)
  now replaces that build gate. A CPU without dotprod keeps exactly its old
  behaviour: every per-format self-check reports unusable and each entry point
  returns its portable counterpart. The f16 KV-cache and grouped-GQA f16
  attention kernels were widened the same way, and need no probe at all —
  `FCVTL`/`FCVTN` are baseline ARMv8.0 conversion instructions, unlike
  `FEAT_FP16` arithmetic.
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
  is split into eight ranges per worker so performance cores absorb
  efficiency-core stragglers. This increased Ministral 3 14B decode from 6.67
  to 7.70 tok/s (+15%) in the local 48-token benchmark while remaining neutral
  within noise on Ministral 3 3B. The
  dispatch coordinator computes one shard itself instead of parking, and Q8
  activation scratch is returned without a per-projection closure allocation.
  The worker pool remains sized to the configured runtime even when a kernel
  exposes fewer independent shards, avoiding repeated 8/12-worker teardown and
  recreation between GQA and projection kernels. Four-query-head GQA (the
  32-Q/8-KV layout in Ministral 3B) has dedicated shared-row NEON dot and AXPY
  kernels: every K/V row is loaded once for all four queries. The isolated
  attention path measured about 2.1x faster at 4k and 1.9x at 32k context on an
  M2 Max. Decode dispatch retains the more parallel per-head path below 4k,
  where it won end-to-end, and switches to grouped GQA at 4k; the complete
  3B forward improved about 3-4% there. Batched prefill already has token-level
  parallelism and uses grouped GQA directly. Set `GOPHERLLM_NO_GROUPED_GQA=1`
  for an A/B rollback.
  Non-ARM64 targets (amd64 included) run the same grouped-GQA decode
  algorithm: the shared-row dot/AXPY step is composed from the portable,
  AVX2-backed `DotF32`/`AxpyF32` kernels instead of a hand-fused NEON kernel,
  but each K/V row is still read from the cache once per query-head group
  rather than once per head, so the same bandwidth saving applies. It reuses
  the 4k decode threshold above rather than a value re-tuned for amd64.
  Grouping is not limited to the 4:1 ratio or to the exact-f32 KV cache: any
  GQA/MQA layout with more than one query head per KV head goes through
  `attendHeadGroup`, which picks the dedicated 4-head kernel when it applies
  and a generic shared-row fallback (`onlineAttentionGroupEither`) for every
  other ratio, and the f16/int8-block KV cache tiers have their own grouped
  kernels (`onlineAttentionGroupF16`/`onlineAttentionGroupI8`) so switching
  `GOPHERLLM_KV_F16`/`GOPHERLLM_KV_I8` no longer silently falls back to the
  ungrouped per-head path. This widens the shared-row win to the many
  non-4:1 GQA/MQA architectures (Falcon, StarCoder2, ChatGLM/GLM4, Command-R,
  and others) and, since f16 is the default KV cache on fast x86-64, to the
  default configuration generally — previously the grouped path only ever
  ran with `GOPHERLLM_KV_F16=0` forcing the exact f32 cache.
  ARM64 also uses NEON FP16 conversion, dot, and accumulation kernels
  for the optional compact KV cache; at 4k context they made the f16 attention
  benchmark about 3.7x faster than the scalar path. Exact f32 remains the
  default there because it was still faster on the measured M2 Max. These were
  Apple-only until the `FCVTL`/`FCVTN` kernels were un-gated from `darwin`;
  they are baseline ARMv8.0, so every arm64 target gets them.
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
  unified-memory Mac). ARM64 converts its rows with dedicated NEON
  FP16 kernels; other non-amd64 targets use the portable scalar fallback.
- A third, more aggressive KV cache tier stores K/V rows as Q8_0 blocks
  (`GOPHERLLM_KV_I8=1`, off by default everywhere): one f16 scale plus 32
  signed int8 values per 32 elements, the same row format this project's
  `--compress` tool already writes for weight tensors. This is a
  **memory-capacity feature, not a speed feature** — be clear-eyed about the
  byte math before enabling it:

  | format | bytes/element | vs. f32 | vs. f16 |
  |---|---|---|---|
  | f32 | 4 | 1x | — |
  | f16 | 2 | 2x | 1x |
  | int8 (Q8_0) | 1.0625 | 3.76x | 1.88x |

  f16-over-f32 is a clean 2x, which is why it's the amd64 default; int8's
  *incremental* saving over the already-default f16 cache is much smaller
  (1.88x, not another 2x). At typical short chat context (dozens to a few
  hundred tokens) that difference is a small fraction of a percent of total
  per-token memory traffic — most of decode's cost is the weight matvecs, not
  the KV-cache scan — so expect **no measurable decode-speed change** there.
  The saving only starts displacing real DRAM traffic once resident context
  is large (roughly the same multi-thousand-token regime that already
  justifies this project's grouped-GQA threshold), where it can fit
  meaningfully more context in the same memory budget than f16 alone. Only
  enabled when every relevant dimension (`K`/`V` width and each individual
  head's width) is a multiple of 32 — Q8_0 rows are only addressable at block
  boundaries, so a model whose head dimension isn't 32-aligned automatically
  falls back to the f16 or f32 cache instead of silently corrupting attention.
  Scalar-only for now (no AVX2/NEON kernels yet); not part of `--auto`'s
  search.
- After mmap'ing a single-file GGUF, every page is touched once up front
  across all worker threads (`prefaultPages`) before the model is reported
  loaded. A memory-mapped file only pages in on first touch, and a forward
  pass touches essentially every weight byte — without this, the *first*
  request after startup silently inherited that page-in cost (disk I/O, or on
  Windows, real-time antivirus scanning of each mapped page) inside its own
  TTFT instead of load time. For a one-shot CLI run this doesn't change total
  wall-clock; for a REPL or another long-running inference process it makes
  first-use latency predictable instead of leaving one request to absorb a
  multi-second page-in tax. Set `GOPHERLLM_NO_PREFAULT=1` to restore pure lazy
  paging.

### Environment variables

Quick reference for the runtime toggles described above (unset by default;
details in the bullets they annotate):

| Variable | Effect |
|---|---|
| `GOPHERLLM_MODEL_DIR` | Default model directory when `--model-dir` is not given (`RUSTY_LLM_MODEL_DIR` remains a deprecated fallback) |
| `GOPHERLLM_DISABLE_SIMD` | Force portable scalar kernels (skip AVX2 detection on x86-64, `FEAT_DotProd` detection on arm64) |
| `GOPHERLLM_NO_BATCH_PREFILL` | Per-token prefill instead of batched |
| `GOPHERLLM_PREFILL_CHUNK` | Override batched-prefill chunk size (`1`-`256`) |
| `GOPHERLLM_NO_GROUPED_GQA` | Disable the grouped-GQA/MQA decode path (any ratio, any KV cache format) (A/B benchmarking and debugging) |
| `GOPHERLLM_Q8_ACTIVATIONS` | `0` disables the default int8-activation Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q4_1/MXFP4/Q2_K/Q3_K matvecs (x86-64, and arm64 with `FEAT_DotProd`) |
| `GOPHERLLM_NO_PREFAULT` | Skip the post-mmap page warm-up; restores pure lazy paging |
| `GOPHERLLM_KV_F16` | `0` stores the KV cache as exact f32 instead of the default f16 cache on fast x86-64; `1` opts into f16 on other targets (NEON-accelerated on all of arm64) to halve KV memory |
| `GOPHERLLM_KV_I8` | `1` opts into the Q8_0-block KV cache tier (off by default everywhere) — a memory-capacity option, not a speed one; see the KV cache section above |
| `GOPHERLLM_METAL_ROWS_PER_GROUP` | Override Metal rows per threadgroup (`2`, `4`, `6`, or `8`; default `4`, adaptively `6` for the fused Ministral-3B-sized FFN) |
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
phi2, phi3, granite (dense), granitemoe, exaone, exaone4, smollm3, internlm2, stablelm,
olmo2, gpt-oss, gemma, gemma2, gemma3, gemma4,
nemotron_h, nemotron_h_moe, mamba2, bert, nomic-bert,
gpt2, gptneox, gptj, bloom, mpt, falcon, starcoder, starcoder2, chatglm, glm4, command-r, minicpm
```

The architecture value is only accepted when its execution graph and expected
tensor layout are implemented; an unknown GGUF is not treated as Llama merely
because some tensor names happen to match. Before that check, though, the
loader resolves the label against the file itself (`ResolveArchitecture`), so
imperfect third-party conversions of supported architectures still load:

- **Spelling variants normalize.** Hugging Face class names
  (`Qwen2ForCausalLM`, `GPT2LMHeadModel`, `CohereForCausalLM`), case, and
  hyphen/underscore variants (`command_r`, `gpt_oss`, `deepseek-v2`) resolve to
  the canonical label. Nothing maps across graphs: `mamba` (Mamba-1) does not
  become `mamba2`, and OLMoE stays unsupported.
- **A missing `general.architecture` is detected, not assumed.** The loader
  finds the `<arch>.*` hyperparameter namespace actually present in the
  metadata (`block_count`, `embedding_length`, ...) and uses it; the historic
  blanket "assume llama" default only remains for files with no namespace at
  all.
- **A mislabeled file follows its hyperparameters.** When the declared label
  has no metadata namespace of its own (Kimi K2 conversions carrying
  `deepseek2.*` keys, `llama2`/`llama3` compatibility labels over `llama.*`),
  hyperparameters are read from the namespace that exists. When the label is
  entirely unknown but the namespace belongs to a supported architecture
  (e.g. a fine-tune label over stock `llama.*` metadata), the file loads as
  that architecture and the load log says so.
- **Vision projectors are diagnosed.** Passing an `mmproj`/CLIP GGUF as a text
  model produces an error pointing at `--mmproj` instead of a generic
  unsupported-architecture failure.

The main coverage is:

| Family | Covered GGUF models / notes |
|---|---|
| Llama-style | Llama 2/3 text models, compatible `llama` exports, SmolLM3 3B (including its every-fourth-layer no-RoPE schedule) |
| Mistral | Mistral, Mistral Small/Devstral exports, Mistral 3, Ministral, and Mixtral |
| Qwen | Qwen2/2.5, QwQ, dense/sparse Qwen3 and Qwen3 Coder, plus experimental text-only Qwen3.5/3.6/3.8 hybrid exports |
| DeepSeek / Kimi | Modern DeepSeek-V2/V3 and Kimi K2 MLA layouts |
| Gemma | Gemma 1–3 and native dense/MoE/E2B Gemma 4 text graphs |
| Other decoders | Phi-2, Phi-3/3.5, dense/sparse Granite (GraniteMoE), EXAONE 3, EXAONE 4 1.2B/32B, OLMo 2/3, InternLM2, StableLM, GPT-OSS |
| Classic / vendor decoders | GPT-2, GPT-NeoX, GPT-J, BLOOM (ALiBi), MPT (ALiBi), Falcon, StarCoder/StarCoder2, ChatGLM, GLM4 (dense), Command-R, MiniCPM |
| Recurrent / hybrid | Mamba2 and Nemotron-H / Nemotron-H-MoE |
| Embeddings | BERT and Nomic-BERT (embedding generation only; not chat generation) |

Important upstream GGUF families that are **not implemented yet**:

| Missing family | Required work |
|---|---|
| Llama 4 | Its architecture-specific attention/normalization graph and model validation |
| Phi-MoE | Sparse expert routing and the architecture-specific expert graph |
| OLMoE | Its sparse expert router and expert execution graph |
| Cohere2 | Its attention, normalization, and tokenizer/chat conventions (Command-R itself is implemented) |
| GLM4-MoE, MiniMax M2, LFM2 | Dedicated sparse or hybrid execution graphs (dense GLM4 is implemented) |
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
`qwen3vl`, `qwen3vlmoe`, and `qwen3next`. Qwen3.6/3.8 GGUFs with one trailing
MTP/NextN draft layer load that layer into its own KV cache. It is opt-in via
`--mtp-draft-tokens N` (or `WithMTPDraftTokens(N)`) and requires deterministic
generation (`--temp 0` or `--top-k 1`): every candidate is verified by the
target before it is emitted, so the result stays exact. The current hybrid
DeltaNet verifier is serial, so MTP is disabled by default and should be
benchmarked per machine rather than assumed to increase tokens/s.

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

Upstage SOLAR's `### System:` / `### User:` / `### Assistant:` template is
detected as its own family (`alpaca-chat`) and rendered exactly, including the
asymmetric spacing the original uses — system and user turns are followed by a
blank line, assistant turns are not. It uses no special tokens, so without
explicit detection it fell through to the generic `User: `/`Assistant: `
fallback, which is close enough to look correct and different enough to be out
of distribution. This covers every SOLAR derivative, including
SauerkrautLM-SOLAR-Instruct, plus the wider class of Alpaca/Orca-style
community fine-tunes that share the format.

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

`bert` and `nomic-bert` are encoder-only architectures: they are available for
embedding generation but intentionally cannot be selected for chat generation.
This covers BERT-format Granite Embedding GGUFs as well as Nomic Embed GGUFs
carrying `general.architecture = nomic-bert`.
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
| CLI | `cmd/gopherllm/main.go`, `lib.go` (package doc + version), `kernel_bench.go` |

A full architecture walkthrough — load path, inference data flow, kernel
dispatch tiers, and how to add a quant kernel or architecture — is
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
test plus embedding tests.

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
