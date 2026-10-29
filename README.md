# GopherLLM

[![DOI](https://zenodo.org/badge/1264366305.svg)](https://doi.org/10.5281/zenodo.21197831)

**GopherLLM has its own inference engine, written in Go with its own
architecture-specific assembly kernels.** It loads and runs GGUF models
directly in the calling process, with no external runtime or child process
required. The Go package and CLI cover generation, streaming chat, embeddings,
tokenization, model inspection, compression, and benchmarks.

**GopherLLM is closer to llama.cpp in scope than to LM Studio or Ollama:**
the core of the project is the inference engine itself, exposed as an embeddable
library and CLI. Its chat UI, server, and model-management features build on
that engine. The comparison describes its role, not a dependency on llama.cpp
or a claim of feature or performance parity.

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

## Classify a CSV with Laya

Ask one question for every record; receive the original CSV with one extra column.
GopherLLM downloads the weights, loads the model once, and writes each result to stdout:

```sh
bin/gopherllm --laya-model hf:convaiinnovations/laya \
  --laya-subfolder multilingual --classify-csv examples/laya/tickets.csv \
  --csv-column text --instruction 'Welcher Bereich ist zuständig?' \
  --criteria '["Abrechnung","Technik","Vertrieb"]' \
  --result-column category > classified.csv
```

Use `--classify-csv -` for stdin and Unix pipes. For a browser upload, start the
same model with `--serve 127.0.0.1:8080` instead of the CSV options, then open
[CSV classification](http://127.0.0.1:8080/classify).
[Full guide: downloads, CSV/TSV, Go integration and HTTP API](docs/laya.md).

For previews, editable results, uncertainty filters and CSV export, use the
standalone [CSV Workbench](cmd/csv-workbench/README.md). It embeds its browser UI
and optionally runs Ministral for task suggestions and row explanations.

## Contents

- [Try it in five minutes](#try-it-in-five-minutes)
- [Classify a CSV with Laya](#classify-a-csv-with-laya)
- [Features](#features)
- [Tested models](#tested-models)
- [Native Laya classification](docs/laya.md)
- [Native speech and image generation](docs/generation.md)
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
- [iOS / iPhone](#ios--iphone)
- [Bindings for Rust, Python, and C](#bindings-for-rust-python-and-c)

## Features

- Pure Go runtime with optional ARM64 (NEON) and x86-64 (SSE2, AVX2 + FMA) assembly kernels.
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
- Model discovery across the complete local LM Studio model library; discovered
  GGUFs run through GopherLLM's own inference engine.
- Direct Hugging Face GGUF imports with cache reuse, split-model downloads,
  private/gated-model tokens, and revision selection.
- `quantize` / `--compress`: convert local GGUF weights to Q8_0/Q4_0/Q2_K/Q3_K/
  Q4_K/Q5_K/Q6_K or mixed Q4_K_M/Q5_K_M presets, writing an independently loadable file (see
  [Model Compression](#model-compression)).
- Native Laya and compatible ModernBERT/mmBERT decision checkpoints: Hugging
  Face downloads, `choice`/`score`/`noul` classification, and a Jev-style
  `/v1/systemone` API, CSV stdin/stdout filters and browser uploads, with no Python runtime ([workflow and examples](docs/laya.md)).
- Native YOLOv8 and YOLO11 object detection from safetensors or Ultralytics
  `.pt` files, without an ONNX, PyTorch or Python runtime (see
  [Object detection (YOLO)](#object-detection-yolo)).
- Native German MMS/VITS speech synthesis to WAV, Qwen-Image-2.1/Turbo and
  experimental FLUX 3 Image Klein text-to-image generation to PNG (including
  the Web UI with `--image-model-dir`), and unconditional CIFAR-10 diffusion.
  Image editing and video generation are not yet implemented
  ([setup, limitations and examples](docs/generation.md)).

## Tested models

Recorded checkpoint tests as of **2026-10-28**. Each entry links to its evidence;
the final table identifies component and synthetic-reference coverage. The
broader [architecture list](#supported-architectures) describes implementation
support rather than a successful run of every pretrained checkpoint. Results
apply to the recorded formats, backends and workloads.

### Text generation

| Model | Tested weights | Upstream | Recorded validation |
| --- | --- | --- | --- |
| Ministral 3 3B Instruct 2512 | Q4_K_M | Mistral AI · [@mistralai](https://github.com/mistralai) | [Complete Metal chat requests and token/state parity](benchmarks/metal/ministral-decode.md) |
| Ministral 3 14B Reasoning 2512 | Q4_K_M | Mistral AI · [@mistralai](https://github.com/mistralai) | [Prefill/decode checks](docs/performance.md#metal) and [fixed-input decoder parity through 415 positions](benchmarks/metal/decode-paging.md#attention-through-512-positions-2026-10-25) |
| Qwen 4B | Q8_0 | Qwen / Alibaba · [@QwenLM](https://github.com/QwenLM) | [Complete serial/prompt-lookup generation comparisons](#speculative-decoding); the historical record does not identify the checkpoint revision |
| Qwen3.5-4B with native MTP | Q4_K_M, prithivMLmods conversion | Qwen / Alibaba · [@QwenLM](https://github.com/QwenLM) | [Paired complete generation with identical tokens, using the linked pinned GGUF](#speculative-decoding) |
| Qwen3.8-27B | Q4_K_M, `qwen35` architecture | Qwen / Alibaba · [@QwenLM](https://github.com/QwenLM) | [Complete prefill/decode requests and live-state parity](benchmarks/metal/qwen-worker-norm.md#complete-qwen38-27b-q4_k_m-model) |
| Ornith-1.0-35B | Q4_K_M | [Ornith / DeepReinforce](https://huggingface.co/ornith-ai/Ornith-1.0-35B) | [64-token generation comparisons and all 40 routed FFNs checked with CPU/Metal loading](#supported-architectures) |
| Gemma 4 E2B-it | Q4_K_M | Google DeepMind · [@google-deepmind](https://github.com/google-deepmind) | [Complete generation, output/state parity and packed lookup-table checks](benchmarks/metal/model-priorities-2026-10.md) |
| Gemma 4 26B-A4B-it | Q4_K_M | Google DeepMind · [@google-deepmind](https://github.com/google-deepmind) | [Complete generation and fixed-input vocabulary-selection parity](benchmarks/metal/model-priorities-2026-10.md) |

### Embeddings, classification, speech and images

| Model | Task | Upstream | Recorded validation |
| --- | --- | --- | --- |
| Granite Embedding 278m Multilingual, Q4_K_M | Embeddings / retrieval | IBM Granite · [@ibm-granite](https://github.com/ibm-granite) | [Source-GGUF/Core ML comparisons and a multilingual retrieval fixture](benchmarks/coreml/README.md#m2-max-results-2026-10-24) |
| Laya English, `convaiinnovations/laya` | Structured classification | Convai Innovations · [@NandhaKishorM](https://github.com/NandhaKishorM) | [Pretrained tokenizer/decision outputs compared with the upstream CPU reference](docs/laya.md#supported-checkpoints-and-validation) |
| Laya Multilingual, `convaiinnovations/laya` subfolder `multilingual` | Multilingual classification | Convai Innovations · [@NandhaKishorM](https://github.com/NandhaKishorM) | [Download → native German classification, upstream output comparison and offline cache reuse](docs/laya.md#supported-checkpoints-and-validation) |
| Parakeet-TDT-0.6B-v3, Q8_0 | Speech-to-text | NVIDIA · [@NVIDIA](https://github.com/NVIDIA) | [Complete transcription of a 3.61-second clip; repeated text, token and encoder-output parity](benchmarks/encoders/speech-frontend.md#complete-parakeet-transcription) |
| `facebook/mms-tts-deu`, F32 | German text-to-speech | Meta / FAIR · [@facebookresearch](https://github.com/facebookresearch) | [Full-checkpoint waveform comparison with PyTorch and native WAV generation](benchmarks/generative/native-tts.md) |
| `google/ddpm-cifar10-32`, F32 | Unconditional 32×32 image generation | Google checkpoint; [DDPM authors](https://github.com/hojonathanho/diffusion) | [Full-checkpoint UNet and complete DDIM sampling compared with Diffusers](benchmarks/generative/native-diffusion.md) |
| Qwen-Image-2.1-Turbo, Q8_0 matrices | Text-to-image | Qwen / Alibaba · [@QwenLM](https://github.com/QwenLM) | [Pretrained 64×64 and 256×256 PNG generation, visual inspection and recorded hashes](benchmarks/generative/qwen-image21.md#pretrained-turbo-generation) |
| FLUX 3 Image Klein distilled, Q8_0 matrices | Text-to-image | Black Forest Labs · [@black-forest-labs](https://github.com/black-forest-labs) | [Pretrained 256×256 generation, 16 steps, seed 42: 103.21 s including loading, 6.89 GiB maximum RSS](benchmarks/generative/flux3.md#pretrained-generation-2026-10-28) |

The timing and RSS above describe one M2 Max run, not a speed guarantee or a
swap-free memory budget. The linked reports retain workload, hardware and
paging limitations. FLUX's [generated image](benchmarks/generative/flux3-teapot-256.png)
and [raw measurements](benchmarks/generative/flux3-pretrained-256-m2-max.json)
are included; its downloaded weights remain outside Git.

### Component and reference coverage

| Model / graph | Upstream | Scope of successful checks |
| --- | --- | --- |
| Voxtral Mini 4B Realtime 2602, Q6_K GGUF | Mistral AI · [@mistralai](https://github.com/mistralai) | [Streaming encoder layer with real checkpoint weights](benchmarks/encoders/laya-voxtral.md), plus frontend/decoder fixtures; these measurements do not establish full-transcription accuracy or latency |
| Qwen3-VL-4B-Instruct | Qwen / Alibaba · [@QwenLM](https://github.com/QwenLM) | [First 32 text layers and eight intermediate states exercised inside pretrained FLUX generation](benchmarks/generative/flux3.md); no standalone vision-language evaluation |
| Qwen-Image-2.1 base | Qwen / Alibaba · [@QwenLM](https://github.com/QwenLM) | [Shared component references and base scheduler checks](benchmarks/generative/qwen-image21.md#independent-numerical-tests); no pretrained base image run recorded |
| FLUX 3 Image Klein Base | Black Forest Labs · [@black-forest-labs](https://github.com/black-forest-labs) | [Synthetic official-reference DiT and complete CFG pipelines](benchmarks/generative/flux3.md#checks-completed); no pretrained base generation recorded |
| YOLOv8 / YOLO11 | Ultralytics · [@ultralytics](https://github.com/ultralytics) | [Synthetic checkpoint loading, forward/detection and numerical-reference tests](internal/vision/yolo/yolo_model_test.go); [`.pt` loading tests](internal/vision/yolo/torch_checkpoint_test.go) cover the supported tensor layouts |

## Requirements

- Go 1.25 or newer.
- A GGUF model for text generation; other modalities use the checkpoints
  described in their guides. By default the text-model tool scans:

```sh
~/.cache/lm-studio/models
```

This is only a default location for model files. LM Studio and Ollama are not
required; GopherLLM loads the GGUF files and performs inference itself.

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

The module-root Go files form the public `gopherllm` facade, preserving the
stable import path `github.com/SimonWaldherr/GopherLLM`. Model loading,
inference, kernels, assembly and their tests live in `internal/engine/`.
Type aliases retain the engine's method sets and avoid copying model state;
forwarding functions are regenerated with `go generate .` after API changes.
Architecture-specific dispatch and assembly stay together there. Executable
entry points live in `cmd/`, the embedding surface for other languages in
`mobile/` and `bindings/`, implementation details in `internal/`, and
test-only fixtures in `testdata/`. Public-boundary and opt-in local model tests
live in `integration/`. Build output and local model/RAG data are ignored and
are not part of the repository. [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
has the full directory map.

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
retain complete recent turns, or `autoCompress` to conservatively shorten older
plain assistant prose only when the original prompt does not fit. Compression
preserves user/system messages, the newest complete turn, tool/image payloads
and formatted content; the same turn-preserving selection follows if needed.

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
`blobs`/`refs`/`snapshots` cache under `$HF_HUB_CACHE`, or `$HF_HOME/hub` (or the platform cache
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

### Text, audio, vision and other model artifacts

In the Web UI's **Download a model** section, choose a **Task** and a
**Format**. GGUF mode offers quantization variants; ambiguous names (for
example a model and vision projector both using F16) use exact file selectors.
The filename and projector role are displayed alongside the quantization.
**All files / formats**, **Safetensors** and **ONNX** enable repository file
selection, including nested weights, tokenizer files, processor configuration,
weight indexes and other assets. Select every required shard and companion
file; the client does not guess which of several architectures or exports you
intend to use. These selections are saved in the HF cache, preserving directory
structure. They are not automatically loaded into the GGUF model library.

The public Go API provides the same model-independent operations:

```go
opts := huggingface.DefaultOptions()
models, err := huggingface.SearchModels(ctx, huggingface.SearchOptions{
    Query: "voxtral", PipelineTag: "automatic-speech-recognition",
    // Format: "safetensors", // omit to search every format
}, opts)
if err != nil { return err }
_ = models

manifest, err := huggingface.Inspect(ctx, "owner/repository@main", opts)
if err != nil { return err }
_ = manifest // present Files and select exact paths from this inventory

files, err := huggingface.DownloadFiles(ctx, "owner/repository@main",
    []string{"config.json", "tokenizer.json", "model.safetensors"},
    os.Stderr, opts)
if err != nil { return err }
_ = files // local paths; use an appropriate runtime separately
```

`Inspect` returns file size, format and a filename-based role hint. Hub
`pipeline_tag` and library metadata describe intended use, not runtime
compatibility. Downloading Safetensors, ONNX, speech or vision artifacts does
not add execution support for their architectures. Repository code is never
executed. Existing `Search`, `Variants`, `Resolve` and `ResolveFiles` retain
their GGUF-specific contracts.

Transfers pin to the commit reported by the Hub, share bounded download
workers, validate resume ranges and publish the revision reference after the
requested set completes. Mirrors that omit the commit header retain
revision-based resolution. Offline `Inspect` lists only locally available
files; offline `DownloadFiles` requires every explicitly selected file. The
client coordinates concurrent writers to identical blobs within one process;
this is not a cross-process cache lock.

For API consumers, `/models/search` accepts `task` and `format` (`gguf`,
`all`, `safetensors`, `onnx`; default `gguf`).
`/models/download/variants?ref=owner/repo&artifacts=1` returns the file inventory.
POST `/models/download` with `{"ref":"owner/repo","files":["config.json"]}`
uses the existing NDJSON progress stream and finishes with `artifacts: true`
and local `files`. These operations retain the existing administrator and
`model-download` feature controls.

The integration belongs in GopherLLM's opt-in `huggingface` package so tinyRAG
and other applications can reuse it without implementing another downloader.
The root inference package continues to have no HTTP dependency.
Hub task/filter semantics follow the [Hugging Face API documentation](https://huggingface.co/docs/hub/api).

### Object detection (YOLO)

GopherLLM runs Ultralytics **YOLOv8** and **YOLO11** detection networks
natively in Go (Accelerate BLAS on macOS), with no ONNX, PyTorch or Python.
No weights are bundled. They come from one of three places:

- A **stock name** is fetched once from Hugging Face and then reused from the
  HF cache, including offline. `yolo11n` … `yolo11x` come from Ultralytics'
  own [`Ultralytics/YOLO11`](https://huggingface.co/Ultralytics/YOLO11), and
  `yolov8n` … `yolov8x` from
  [`lmz/candle-yolo-v8`](https://huggingface.co/lmz/candle-yolo-v8). Both are
  pinned to a fixed commit.
- An **Ultralytics `.pt` file** (`yolo11n.pt`, `yolov8s.pt`, or your own
  `best.pt` from training) loads directly.
- A **safetensors file** loads directly, in candle or Ultralytics tensor
  naming, as F32, F16 or BF16.

```sh
go run ./cmd/yolo-detect --image street.jpg --out boxes.png    # yolo11n, the default
go run ./cmd/yolo-detect --model yolov8s --image street.jpg --json
go run ./cmd/yolo-detect --model runs/detect/train/weights/best.pt --image part.jpg
go run ./cmd/yolo-detect --model hf:owner/repo:best.safetensors --image x.jpg --offline
```

The generation, model size, class count and head layout are read from the
checkpoint itself. A custom-trained model therefore loads the same way as a
stock one, and a `.pt` file brings its own class names. For safetensors
files, pass `--labels` with one name per line; 80-class models default to the
COCO names.

Images are letterboxed with the same bilinear resize and 114-grey padding
Ultralytics uses, so confidences match the reference implementation. On an
M-series Mac, one 640×640 image takes about 60 ms with `yolov8n` or `yolo11n`
using Accelerate, and about 150 ms in pure Go (`CGO_ENABLED=0`).

**`.pt` safety.** A `.pt` file is a Python pickle, and `pickle.load` on an
untrusted file is remote code execution. GopherLLM never runs pickle code. It
interprets the opcodes as data and rebuilds only tensors, dicts and lists;
every class or function the file names stays an inert value. See
`internal/vision/yolo/torch_checkpoint.go`.

From Go, the root package stays offline. `YOLOCheckpointReference` only builds
the pinned `hf:` reference for a stock name; downloading is left to the
`huggingface` package, as `cmd/yolo-detect` shows:

```go
model, err := gopherllm.LoadYOLO("yolo11n.pt") // or a .safetensors file
if err != nil { return err }
dets, err := model.Detect(img, gopherllm.DefaultYOLOConfig())
// dets[i].Label, .Confidence, .Box (source-image pixels); model.Version
```

A loaded model is safe for concurrent `Detect` calls. The server exposes the
same detector as `GET /models/detection` and `POST /v1/vision/detections`
(see [server/README.md](server/README.md)). Pose, segmentation, OBB and
YOLO26 are not supported yet.

**License:** the Ultralytics weights (including the `lmz/candle-yolo-v8`
copies) are licensed AGPL-3.0. GopherLLM does not redistribute them; whoever
downloads and uses them is bound by that license.

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

// Don't have a .gguf yet? huggingface.Resolve turns an "owner/repo" Hugging
// Face reference into a local path (see "Hugging Face imports" below); Open
// itself only understands local paths, and its error says so when a path
// looks like an unresolved Hub reference instead of a typo'd local file.

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

### Apple Neural Engine embeddings

The optional `coreml` package runs a separate embedding encoder with CPU and
Neural Engine permission, leaving the GPU available to the chat model. It
loads one compiled model and reuses its input arrays and output backing.
It does not allocate a chat KV cache or load a second GGUF Runner.
Build your application with `CGO_ENABLED=1 go build -tags 'metal coreml'`.
The backend requires macOS 13/iOS 16 or newer; other builds return
`coreml.ErrUnavailable`.

Export a local **BERT embedding GGUF** once, using Python 3.12 and the optional
conversion dependencies. The exporter uses static shapes, FP16, BC1S tensors,
1×1 convolutions and separate attention heads following
[Apple's ANE transformer guidance](https://machinelearning.apple.com/research/neural-engine-transformers).
It supports mean and CLS pooling; Nomic/RoPE and chat decoder graphs are not
supported by this exporter. FP16 exports can be larger than their Q4 sources.

```sh
python3.12 -m venv .venv-coreml
.venv-coreml/bin/python -m pip install 'coremltools>=9,<10' 'gguf>=0.19,<0.20'
.venv-coreml/bin/python scripts/export_coreml_embeddings.py encoder.gguf encoder.mlpackage \
  --sequence-length 128 --compile-to encoder.mlmodelc
```

Keep the compiled `.mlmodelc` at a stable path and keep the embedder open across
requests, so loading and specialization stay outside the prediction path.
For document indexing, `--batch-size 2`, `4` or `8` exports a fixed batch that
shares one set of encoder weights across several texts. `EmbedBatch` groups
inputs automatically, reuses bounded scratch and discards unused tail outputs.
The default is one text per prediction, suitable for individual search queries.
Larger exports can improve bulk throughput but also increase scratch and
single-query latency; compare them on the target Mac before choosing one.
For CLS-pooling encoders, `--prune-cls` omits the unused final-layer query and
feed-forward positions while retaining keys and values for every input token.
Optional `--weight-compression palette8` stores only convolution weights in
256-entry palettes. Vocabulary embeddings, biases and normalization parameters
remain FP16, as does model computation. Compression slightly changes vectors;
use a new `EmbedderID` and check retrieval on the intended corpus. The default
keeps FP16 weights and the complete final layer.
Use the exact source checkpoint's tokenizer, including its boundary tokens:

```go
import (
    "github.com/SimonWaldherr/GopherLLM/coreml"
    "github.com/SimonWaldherr/GopherLLM/rag"
)

tok, err := coreml.LoadTokenizer("encoder.gguf") // metadata only; no retained weights
if err != nil { return err }
encoder, err := coreml.OpenEmbedder("encoder.mlmodelc", tok, coreml.Options{PadToken: tok.PadToken})
if err != nil { return err }
defer encoder.Close()
chunks := rag.Chunker{Tokenizer: tok.Tokenizer,
    MaxRunes: encoder.Info().SequenceLength - len(tok.Encode("")), OverlapRunes: 8}
index := rag.New(rag.Options{Embedder: encoder, EmbedderID: "encoder-coreml-fp16-s128-v1", Chunker: chunks})
if err := index.Add(ctx, rag.Doc{ID: "note", Text: "The processor executes the program's instructions."}); err != nil {
    return err
}
// index.Search uses this encoder without disturbing a chat Runner.
```

`server.HandlerOptions` and `ServeOptions` accept the same caller-owned encoder
through `RAGEmbedder`, `RAGEmbedderID` and `RAGChunker`. Close it after the
handler stops. Change the ID when the model, tokenizer or embedding conventions
change. Inputs exceeding the model's fixed sequence length fail explicitly;
reserve tokens for boundaries and any document/query prefixes when chunking.

`CPUAndNeuralEngine` permits CPU fallback. `coreml.Inspect(path, units)` reports
the preferred devices in Core ML's compute plan on macOS 14.4/iOS 17.4 or newer;
these counts describe planned placement, not observed utilization or time share.
On the M2 Max, CLS pruning plus palette8 reduced warm Granite Embedding 278m
single-query latency from 3.18 to 1.75–1.77 ms at a 64-token capacity.
Minimum source-GGUF cosine was 0.9998955; the small multilingual retrieval
fixture retained every source Top-1 hit and Top-3 neighbour set. These encoder
measurements exclude model loading and do not establish faster chat decoding.
The [benchmark methodology and results](benchmarks/coreml/README.md) include
CPU/ANE comparisons, batch sizes, compression, memory limits and raw data.
M4 throughput has not been measured.

To repeat the encoder parity, compute-plan and warm-latency checks with the
exported model:

```sh
GOPHERLLM_COREML_MODEL="$PWD/encoder.mlmodelc" GOPHERLLM_COREML_GGUF="$PWD/encoder.gguf" \
  CGO_ENABLED=1 go test -tags 'metal coreml' ./coreml -run '^TestCoreMLConfiguredEmbeddings$' -count=1 -v
```

To compare exports with different batch sizes, set `GOPHERLLM_COREML_BATCH_MODELS`
to their colon-separated paths and run `TestCoreMLBatchComparison` instead.
It checks source-GGUF parity and reports warm latency for 1, 4, 17 and 32 texts,
loading one native model at a time. Repeat in reverse model order when other
applications are active.

### Core ML prefill for Qwen and Ministral

An optional [FFN exporter and runtime](benchmarks/coreml/ffn.md) can execute
selected dense Qwen/Ministral SwiGLU blocks through Core ML during ordinary
prefill. It reuses native buffers and checks source hashes, embedded graph
metadata and numerical parity before attaching an export. Complete Metal
prefill chunks retain priority; decode, speculative verification and Qwen MTP
prefill retain their current arithmetic. Build with `-tags 'metal coreml'` and
enable exports explicitly through `Runner.SetCoreMLFFN`.

`CoreMLFFNOptions{AutoTune: true, LogWriter: os.Stderr}` optionally compares
CPU-only and CPU/Neural Engine execution for each export at attachment time.
Each candidate must pass the source quality probe to be eligible. When both
are valid, CPU-only replaces the default after two measurements show at least
10% lower latency. Calibrate on an otherwise idle machine. This adds startup
work and keeps the GPU available to Metal.

This adds separately compiled weights and is not a RAM-saving setting.
Benchmark the actual prompt/model before enabling it; layer timings do not
establish an end-to-end generation speedup.

### Tools with a Go function

`gopherllm.NewTool` derives a tool's JSON Schema from a plain Go struct by
reflection, so a function is a tool with no schema string to keep in sync:

```go
type WeatherArgs struct {
    City string `json:"city" desc:"City name, e.g. Hamburg"`
    Unit string `json:"unit,omitempty" desc:"Temperature unit" enum:"c,f"`
}

weather := gopherllm.NewTool("get_weather", "Current weather for a city.",
    func(ctx context.Context, a WeatherArgs) (string, error) {
        return fmt.Sprintf("22 degrees in %s", a.City), nil
    })
weather.Trusted = true // first-party data, skip the injection-warning wrapper

// RunAgenticChatWithTools predates the GenOption veneer and takes a plain
// GenerationOptions; ApplyGenOptions builds one the same way Model's own
// methods do, ctx included — that's what makes the loop's per-call tool
// timeouts and generation itself actually cancel with ctx, easy to forget
// since every other Model method takes ctx as a plain first argument instead.
options := gopherllm.ApplyGenOptions(ctx, gopherllm.WithTemperature(0.2))

result, err := gopherllm.RunAgenticChatWithTools(model.Runner(),
    []gopherllm.ChatMessage{gopherllm.UserMessage("Weather in Hamburg?")},
    options, nil, []gopherllm.AgenticTool{weather},
    func(s string) bool { fmt.Print(s); return true })
```

Prefer a `*Model` you already own, tools with retrieval, or citations? The
[`agent` package](#retrieval-over-your-own-documents) wraps this same loop
into `Agent.Ask`/`Chat` with a validated `ToolSet` and deduplicated sources.

The agent loop that executes a tool call bounds it with a per-call timeout
(default 30s), recovers a panic as a failed call instead of crashing the
process, caps and marks a truncated result before it reaches the model, and
replays an identical `(name, arguments)` call from an earlier round instead of
re-executing it — all opt-out-able per `AgenticTool` field
(`gopherllm.NoToolTimeout`, `gopherllm.NoToolResultLimit`). See `AgenticTool`'s
doc comment for the full contract.

> **Compile-time note:** `AgenticTool` gained the `Timeout`, `MaxResultBytes`
> and `Trusted` fields above. Every construction site in this repository
> already used field names (`AgenticTool{Definition: ..., Execute: ...}`), so
> this is a no-op here — but an unkeyed literal (`AgenticTool{def, exec}`)
> outside this repo will now fail to compile. Add the field names.

[`examples/tool-calling`](examples/tool-calling) is this pattern as a
complete, runnable program (`go run ./examples/tool-calling -model
/path/to/model.gguf`) with a real arithmetic tool instead of a fake string, so
the output is self-verifying.

### Retrieval over your own documents

`rag.Index` is a hybrid BM25 + optional-vector index — a chunker, an inverted
keyword index, and (when you supply an `Embedder`) a cosine-similarity scan —
with no database and no third-party dependency:

```go
import "github.com/SimonWaldherr/GopherLLM/rag"

ix := rag.New(rag.Options{}) // pure BM25; add Embedder: model for hybrid search
if err := ix.Add(ctx, rag.Doc{ID: "handbook", Title: "Employee Handbook", Text: handbookText}); err != nil {
    return err
}
hits, err := ix.Search(ctx, "return window for GX-1180", rag.Query{TopK: 3})
for _, h := range hits {
    fmt.Printf("%s (score %.2f): %s\n", h.Title, h.Score, h.Text)
}
```

BM25 alone already wins the queries people actually type at their own
documents — part numbers, error codes, invoice ids, surnames — which a
chat model's mean-pooled hidden states (not a trained embedding head) tend to
lose; see `rag.Options.Embedder`'s doc comment before wiring one up.
`Index.Save`/`rag.Load` persist a built index so a second run skips
re-embedding. The bundled server adds runtime ingestion for pasted text,
multi-file and ZIP uploads, and public web URLs (with clean full-article
imports for Wikipedia), plus an interactive retrieval preview; see
[Knowledge base search](server/README.md#knowledge-base-search-rag).
See the [rag package documentation](https://pkg.go.dev/github.com/SimonWaldherr/GopherLLM/rag)
for chunking, ranking weights, and directory ingestion via `Index.AddFS`.

### Speculative decoding

Greedy generation can propose a short continuation and let the target model
verify it in one block. A block contains 2–8 target inputs, including the
already-emitted seed, so it accepts at most seven draft candidates. An entirely
accepted block also supplies the next target prediction. Stop tokens, stop
strings, repetition penalties and cancellation retain their ordinary semantics.

Prompt lookup proposes known continuations of repeated 4–64-token suffixes in
the most recent 8192 prompt/output tokens. Enable it explicitly with
`--prompt-lookup-tokens 7`, `WithPromptLookupTokens(7)`, or the JSON configuration
field `generation.prompt_lookup_tokens`. Values are 0–7; zero disables it.
For example:

```sh
bin/gopherllm model.gguf --temp 0 --repeat-penalty 1 --prompt-lookup-tokens 7 \
  --prompt 'Continue this repeated code pattern...'
```

```go
res, err := model.Generate(ctx, prompt,
    gopherllm.WithTemperature(0), gopherllm.WithRepeatPenalty(1),
    gopherllm.WithPromptLookupTokens(7))
```

These copying examples disable repetition penalties because intentional reuse
is the task. Repetition penalties remain supported; keep the application's
sampling settings unchanged when comparing performance.

Qwen GGUFs with a loaded NextN head can also use `--mtp-draft-tokens N` or
`WithMTPDraftTokens(N)`. Both methods require deterministic sampling
(`--temp 0` or `--top-k 1`) and are incompatible with JSON-object constrained
output. When both are enabled, a prompt-lookup hit takes precedence; a miss
falls back to MTP. Every proposed token is checked against the target's own
greedy choice. Both options are off by default.

Each request starts at the second available window, usually three candidates,
and adjusts among 1/3/7 drafts plus an exact smaller user cap. Including the seed,
the regular windows fill 2/4/8 target positions. Selection uses acceptance and
elapsed draft, verification and reconciliation cost per emitted token.
Measurements are discarded after generation. A slow four-position block with
at least 85% draft acceptance can test one previously unmeasured larger window
before shrinking; its measured cost decides whether to retain it. Two completely
rejected blocks shrink the window early, independently of cold kernel timings.
If the one-candidate window
accepts fewer than half its proposals or costs over 10% more than measured
serial decoding, drafting pauses for twelve serial steps before retrying.
Eligible dense Metal transformers share unpacked weights across the short
block's positions in registers, with the body, vocabulary projection and
position-specific token reductions in one command buffer. Dense CPU vocabulary
projections reuse weight rows across four to eight positions. Qwen hybrids batch
target projections while keeping DeltaNet updates chronological. On rejection,
up to 32 MiB of retained projection inputs avoid repeating the large target
projections when restoring the accepted recurrent prefix. The MTP head batches
catch-up from known target states, including prefill. Catch-up computes only
embedding/hidden mixing and K/V: attention output, Q, O, FFN and vocabulary
projections are unused because the next boundary comes from the target model.
Eligible dense Metal hybrids batch FFNs, CPU-Q8 DeltaNet input/output
projections and direct Q6_K QKV projections.
Single-token and batched FFNs use the same GPU reduction order; attention and
recurrent updates retain their chronological CPU/Metal schedule. Contracting
Q6_K FFN down projections can be fused from 2048 rows, while standalone matvec
thresholds stay unchanged.
This dense path supports F32, F16 and I8 KV. MoE, out-of-core and unsupported
accelerator-backed Qwen graphs retain serial MTP. Quantized CPU Qwen projections
also retain it when Q8 activations are disabled.
For drafting, sufficiently large prepared Q/O and fused FFN matrices can stay
on Metal. The MTP input mixing and K/V remain on the CPU so drafting and cache
catch-up keep the same K/V arithmetic. WebGPU projection views retain full-body
catch-up.
Set `GOPHERLLM_NO_BLOCK_VERIFY=1` to compare against serial target verification.

`GenerationResult.Stats.Speculation` reports `Blocks`, `DraftedTokens`,
`AcceptedTokens`, `DiscardedInputs`, `DraftTime`, `VerificationTime` and
`ReconcileTime`. Discarded inputs are tentative target rows excluded from the
resident prefix, including work undone on cancellation. The field is nil when
drafting is disabled; serial fallback leaves `Blocks` at zero. Measure total
decode time and accepted tokens on the intended model and hardware before
assuming a throughput gain.

On M2 Max with eight CPU threads and Metal, a 128-token record-copying task
produced exactly the same token IDs with serial decoding and prompt lookup.
Five measured rounds alternated execution order after one warmup per mode;
prompt caches were cleared between requests. Greedy sampling used repeat
penalty 1. Median times were:

| Model | Serial decode | Lookup decode | Serial total | Lookup total |
|---|---:|---:|---:|---:|
| Ministral-3B Q4_K_M | 2.196 s | 1.427 s | 2.955 s | 2.183 s |
| Qwen-4B Q8_0 | 2.427 s | 2.140 s | 3.160 s | 2.888 s |

These gains depend on reusable text. A code-editing task with few accepted
proposals remained slower, so speculation remains opt-in. The record-copying
results measure prompt lookup; neither checkpoint contains a native MTP head.

The same 128-token copying task also measured native MTP on
[Qwen3.5-4B-MTP Q4_K_M](https://huggingface.co/prithivMLmods/Qwen3.5-4B-MTP-GGUF/blob/dd65086bdcdd7a8f242a2e54cfe11caf8cd51097/Qwen3.5-4B.Q4_K_M.gguf).
With `--mtp-draft-tokens 7`, five alternating pairs after warmup produced
identical tokens. Median decode time fell from 6.628 s to 6.107 s (7.9% less
time), and total request time from 13.540 s to 12.847 s (5.1% less time).
Decode was faster in all five pairs; total time was faster in four. Both modes
used the same default Metal FFN kernels, eight CPU threads and cleared prompt
caches. This is workload-dependent: separate three-pair code-editing runs with
one or three MTP candidates took about 3% and 9% longer to decode than serial.

`TestSpeculationConfiguredGenerationAB` repeats the paired copy/code comparison
when `GOPHERLLM_SPEC_AB_MODEL` points to a local GGUF. Set
`GOPHERLLM_SPEC_AB_METAL=1`, `GOPHERLLM_SPEC_AB_MAX_TOKENS=128` and build with
`-tags metal`; `GOPHERLLM_SPEC_AB_MODES` selects `lookup7`, `mtp1`, `mtp3`, `mtp7` or
combinations such as `lookup7_mtp3`. `GOPHERLLM_SPEC_AB_SAMPLES=5` and
`GOPHERLLM_SPEC_AB_WARMUPS=1` select the rounds above;
`GOPHERLLM_SPEC_AB_WORKLOAD=copy` selects the copying task;
`GOPHERLLM_SPEC_AB_JSON` saves samples and medians. The separate
`TestSpeculationConfiguredFullMTPBounded` accepts a complete Qwen hybrid model
through `GOPHERLLM_SPEC_AB_RAW_MODEL` for a bounded MTP parity/performance check.

`TestSpeculativeConfiguredModelParity` provides a shorter parity check when
`GOPHERLLM_VALIDATE_MODEL` points to a local dense GGUF; set
`GOPHERLLM_VALIDATE_METAL=1` and build with `-tags metal` to require the GPU.

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

The separate opt-in `EnableMistralKVPrefixCache` snapshot cache does not trigger
another static-prefix render while disabled. Its invalidation epoch is captured
before prompt rendering, so a concurrent clear cannot be undone by that
request's later snapshot store.

### Reusing conversations

A Runner reuses exact token prefixes in its live KV workspace. For standard
attention-only models and Gemma, it also retains up to four displaced text
conversations, capped at 128 MiB of KV snapshots and token IDs, so returning to
an earlier chat can skip its already processed prefix. Historical snapshots
require at least 32 matching tokens and compatible KV storage. Recurrent models,
MLA, image-bearing requests, and out-of-core execution keep their existing paths.

```go
runner.SetConversationCacheLimit(64 << 20) // optional smaller history budget
runner.SetConversationCacheLimit(0)       // disable history; keep live prefix reuse
runner.ClearConversationCache()          // clear history and resident reuse metadata
```

The history budget is additional to the live workspace.

### Package layout

The root package is inference only. Its import closure excludes the optional
demo application and its UI assets:

| Import | You get |
|---|---|
| `github.com/SimonWaldherr/GopherLLM` | GGUF loading, generation, chat, embeddings, tokenizer, sampling, autotuning, skills/agent loop |
| `github.com/SimonWaldherr/GopherLLM/rag` | Document chunking, hybrid BM25 + vector retrieval |
| `github.com/SimonWaldherr/GopherLLM/huggingface` | Model/task search, artifact inspection/download, and local GGUF resolution |

`TestInferencePackageStaysFreeOfServerDependencies` enforces that boundary
against the real dependency graph, so it cannot regress silently.

Dependencies are standard-library packages and packages within this module;
there is no `require` block, `go.sum` or vendor directory. The exact count
depends on the Go version and build tags. Stdlib is not the same as free, since an
embedder compiles every package in the closure. The root package therefore
avoids `crypto/sha256` and `regexp`, each of which cost far more than its single
use was worth, and `TestInferencePackageStaysFreeOfTheCryptoAndRegexpTrees` pins
that.

The PNG and JPEG decoders are the remaining optional chunk. Vision needs them,
so they are a build-time knob rather than a removal:

```sh
go build -tags noimagedecoders ./...
```

That drops
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

Quantize a local GGUF with positional arguments, like `llama-quantize`:

```sh
bin/gopherllm quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
# Optional: limit CPU workers
bin/gopherllm quantize model-f16.gguf model-Q5_K_M.gguf Q5_K_M --threads 8
```

Supported targets: `Q8_0`, `Q4_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q4_K_M`,
`Q5_K`, `Q5_K_M`, and `Q6_K`. Float sources (F32, F16, BF16, F64) and
supported quantized sources can be converted. Existing command-line flags
remain available:

```sh
bin/gopherllm model-f16.gguf --compress --compress-format Q4_K_M --compress-out model-Q4_K_M.gguf
```

`Q4_K_M` and `Q5_K_M` are mixed tensor presets, rather than aliases for a
single GGML type. They use Q4_K or Q5_K for most matrices and Q6_K for
selected attention-value and FFN-down projections and the output head.
A separate input embedding keeps the base type; a tied embedding receives
the output-head upgrade. Small tensors, normalization weights, expert
routers, and recurrent convolution kernels retain their source precision.
The planner also accounts for MoE/hybrid layouts and uses Q5_0, Q5_1, Q8_0,
or F16 when a matrix's row width cannot fit the selected block layout.
This tensor policy follows [llama.cpp's mixed presets](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp).

The base targets retain the previous policy: quantize eligible matrices
with compatible row widths, keep norm/bias vectors unchanged, and floor
both token embeddings and output heads at Q6_K when the target is lower.
Use `--uniform` with the subcommand, or `--compress-uniform` with the legacy
flags, to disable that floor. Uniform mode cannot be combined with a mixed
preset; choose its base type (`Q4_K` or `Q5_K`) instead.

Conversion streams row chunks with an 8 MiB shared float/output scratch
budget, plus the writer buffer and one encoded row per active worker. A
single oversized row is the minimum chunk. Entire MoE expert banks are
never expanded into float32 at once. The source stays memory-mapped;
normal OS page-cache usage is additional to this scratch budget.

Output is a normal, independently loadable GGUF, with updated file-type and
quantization-version metadata and preserved tokenizer/model metadata. A
sibling temporary file is flushed, closed, and its tensor descriptors and
byte ranges verified before replacing the destination. Conversion errors
or cancellation preserve an existing destination and remove the temporary
file. Input and output must identify different files, including through
symlinks or hardlinks. Split GGUF shards must be merged first.

The encoders use plain round-to-nearest (RTN), without calibration data or
weighted error optimization. Matching a llama.cpp preset's tensor policy
does **not** promise identical bytes, perplexity, or model quality. Evaluate
the converted model on representative tasks before choosing a low-bit
artifact. Q3_K and Q2_K reduce packed weight traffic relative to Q4_K by
about 24% and 42%, respectively, with a larger quality trade-off. They use
the CPU quantized kernels rather than the selective Q4_K/Q6_K Metal path;
measure `--bench-json` on the deployment machine.

The library API is `CompressModel(source, output, options)`.
`ParseCompressOptions("Q4_K_M")` retains the mixed preset in
`CompressOptions.Preset`; the older `ParseCompressFormat` returns only its
base GGML type. Set `CompressOptions.Context` for cancellation.

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
go test -run '^$' -bench=. -benchmem ./internal/engine
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
  `CGO_ENABLED=1 -tags metal` and pass `--metal` for you. Throughput depends
  on the model and workload; see the [measurement records](#performance-notes).
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

Use `--auto` to measure supported settings for the loaded model on the target
machine. Keep the model quantization, prompt, generated token count, sampler,
thread count and KV format fixed when comparing versions. Report loading,
prefill and decoding separately; kernel speedups do not establish generation
throughput. See [Benchmarking and Profiling](#benchmarking-and-profiling).

- **Apple GPU:** build with `CGO_ENABLED=1 -tags metal` and enable `--metal`.
  Make targets select it automatically on macOS with Command Line Tools.
  Eligible dense models keep the complete decoder on the GPU; other models
  use selective offload. Auto mode can compare CPU and Metal when neither is
  explicitly selected.
- **Memory and paging:** `--prefault core` is the CLI default. It warms backbone
  tensors while leaving sparse expert banks and untied packed embeddings
  demand-paged. Avoid `--prepare-quant` when extra tables would exhaust RAM.
  For models that do not fit comfortably, `--out-of-core` uses CPU inference
  and maps GGUF shards separately. It does not impose a hard RSS limit.
- **Repeated requests:** prefix caching and prepared prompts reduce repeated
  prompt work. Speculative decoding remains opt-in; its benefit depends on
  acceptance and verification cost. See [Speculative decoding](#speculative-decoding)
  and [Reusing conversations](#reusing-conversations).
- **Core ML:** embeddings and dense prefill FFNs are optional integrations.
  Exported FFNs retain the original weights and require additional memory.
  They do not accelerate autoregressive decoding. See
  [Core ML prefill](#core-ml-prefill-for-qwen-and-ministral).

[Performance and memory reference](docs/performance.md) documents backend
limits, kernel behavior and historical measurements. For recent complete
experiments, use the recorded methodology and raw data:

| Workload | Measurement record |
|---|---|
| Ministral 3B Metal decode, M2 Max | [Paired requests and decode steps](benchmarks/metal/ministral-decode.md) |
| Ministral/Voxtral Tekken tokenization | [Splitter timing and allocation measurements](benchmarks/maintenance/tekken.md) |
| Voxtral offline convolution stem, M2 Max | [Accelerate timing and scratch usage](benchmarks/encoders/laya-voxtral.md#offline-voxtral-convolution-tiles-2026-10-29) |
| Core ML embeddings, M2 Max | [CPU/Neural Engine comparisons](benchmarks/coreml/README.md) |
| Core ML FFNs, Qwen and Ministral | [Validation, memory and timing](benchmarks/coreml/ffn.md) |

### M2 Max, M4 and M4 Pro Macs

Thread topology, memory bandwidth and GPU size differ between these Macs.
Auto mode measures settings on the current machine and includes its hardware
identity in the cache key. The [hardware and backend reference](docs/performance.md#m2-max-m4-and-m4-pro-macs)
records the differences. M2 Max results do not establish M4/M4 Pro performance;
the SME experiment still requires validation on supported hardware.

### Environment variables

Quick reference for runtime toggles (unset by default). Backend constraints
and measurements are in the [performance reference](docs/performance.md).

| Variable | Effect |
|---|---|
| `GOPHERLLM_MODEL_DIR` | Default model directory when `--model-dir` is not given (`RUSTY_LLM_MODEL_DIR` remains a deprecated fallback) |
| `GOPHERLLM_DISABLE_SIMD` | Disable SSE2/AVX2 on x86-64, ARM64 Q8 activation quantization, and ARM64 `FEAT_DotProd` detection |
| `GOPHERLLM_NO_BATCH_PREFILL` | Per-token prefill instead of batched |
| `GOPHERLLM_NO_BLOCK_VERIFY` | `1` disables speculative block verification for serial A/B comparison; native MTP retains its serial verifier |
| `GOPHERLLM_NO_MTP_BATCH` | `1` restores per-token MTP catch-up for comparison |
| `GOPHERLLM_MTP_METAL_BODY` | `0` keeps the complete native MTP draft body on the CPU; the shared vocabulary head remains on its configured backend |
| `GOPHERLLM_PREFILL_CHUNK` | Override batched-prefill chunk size (`1`-`256`) |
| `GOPHERLLM_NO_GROUPED_GQA` | Disable the grouped-GQA/MQA decode path (any ratio, any KV cache format) (A/B benchmarking and debugging) |
| `GOPHERLLM_Q8_ACTIVATIONS` | `0` disables the default int8-activation Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/Q4_1/MXFP4/Q2_K/Q3_K matvecs (x86-64, and arm64 with `FEAT_DotProd`) |
| `GOPHERLLM_NO_PREFAULT` | Skip the post-mmap page warm-up; restores pure lazy paging |
| `GOPHERLLM_KV_F16` | `0` stores the KV cache as exact f32 instead of the default f16 cache on fast x86-64; `1` opts into f16 on other targets (NEON-accelerated on all of arm64) to halve KV memory |
| `GOPHERLLM_KV_I8` | `1` opts into the Q8_0-block KV cache tier (off by default everywhere) — a memory-capacity option, not a speed one; see [KV cache and paging](docs/performance.md#kv-cache-and-paging) |
| `GOPHERLLM_METAL_ROWS_PER_GROUP` | Override legacy Metal rows per threadgroup (`2`, `4`, `6`, or `8`; default `4`, adaptively `6` for Ministral-3B FFNs); new Q8_0 and coalesced prefill kernels use four rows |
| `GOPHERLLM_METAL_COALESCED` | `0` disables the four-token Q4_K/Q6_K prefill fallback; also set `GOPHERLLM_METAL_MATRIX=0` to restore the legacy prefill kernels |
| `GOPHERLLM_METAL_MATRIX` | `0` disables SIMD-group matrix prefill and restores the four-token kernels; decode is unchanged |
| `GOPHERLLM_METAL_VERIFY_VECTOR` | `0` disables register-shared 2–8-position matvecs, including short prefill batches; larger batches and serial decode are unchanged |
| `GOPHERLLM_METAL_QWEN_FFN_VECTOR` | `0` restores the legacy Qwen FFN reduction in both single-token decode and block verification |
| `GOPHERLLM_NO_MOE_DOWN_BATCH` | `1` restores separate worker dispatches for selected experts' down projections |
| `GOPHERLLM_METAL_GEMMA` | `0` disables fused GeGLU, vectorized Gemma output and native Metal batch prefill (enabled by default) |
| `GOPHERLLM_METAL_DENSE_DECODE` | `0` disables complete dense Metal decode for eligible Ministral/Qwen models (enabled by default) |
| `GOPHERLLM_METAL_WHOLE_PREFILL` | `0` disables complete GPU-resident dense prefill chunks (enabled by default for eligible shapes) |
| `GOPHERLLM_METAL_DENSE_VECTOR` | `0` uses legacy matvec kernels within complete Metal decode for comparison |
| `GOPHERLLM_METAL_DENSE_ROWS_PER_GROUP` | Override dense Metal SIMD groups per threadgroup (`2`, `4`, or `8`; uncalibrated default `4`, auto mode measures all three) |
| `GOPHERLLM_METAL_DENSE_CONCURRENT` | `0` serializes all dispatches within complete Metal decode |
| `GOPHERLLM_METAL_DENSE_PARALLEL_ARGMAX` | `0` restores single-threadgroup vocabulary selection in dense Metal greedy decode for A/B measurements |
| `GOPHERLLM_METAL_DENSE_SHARED_KV` | `0` uses staging copies instead of pinned shared F32 KV buffers |
| `GOPHERLLM_METAL_FUSED_FFN` | `0` disables hybrid-path Metal Gate/Up + SiLU + Down fusion; disable dense decode above to select that path |
| `GOPHERLLM_DISABLE_YARN` | Ignore declared YaRN RoPE scaling |

Settings chosen by `--auto` apply process-wide. Explicit dense Metal row counts
and `GOPHERLLM_METAL_DENSE_VECTOR=0` take precedence over schedule calibration;
other calibrated settings override their corresponding environment defaults.

## Supported Architectures

The loader currently accepts GGUF files whose `general.architecture` is one of
(also available at runtime via `gopherllm.SupportedArchitectures()`, which
backs the same check `ArchitectureSupported` makes — useful to check a file
before ever calling `Open`, e.g. to grey it out in a picker):

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
alternating local/full-attention schedule. These sparse graphs share the
per-token decode kernels; Qwen3.5-MoE also has the grouped prefill path below.

The common sparse graph and native Gemma 4 A4B coalesce eligible selected
experts' gate/up and down projections into one worker dispatch per stage.
Down inputs remain distinct per expert, and the weighted sum retains router
order, biases, activation functions and Gemma's branch norms. The down path
supports F32, Q4_K, Q5_K, Q6_K, Q8_0 and MXFP4, provided gate/up batching is
available. It reuses existing CPU SIMD kernels and bounds additional down-stage
scratch to 8 MiB and 32 active experts; other layouts retain per-expert decode.
Set `GOPHERLLM_NO_MOE_DOWN_BATCH=1` to disable down batching.

Gate/up dispatch tasks are pooled and cleared after use, so they retain no
model mappings or request buffers. On M2 Max with Go 1.25.3, eight threads,
eight active experts and 2048×512 projections, the warmed
`BenchmarkExpertGateUpBatch` went from five allocations per Q4_K/Q6_K call
to zero (F32: three to zero). Compatible CPU shared experts also combine
gate/up into one dispatch and reuse activation preprocessing; direct GPU and
prepared projections keep their existing kernels. These changes apply to
serial prefill and decoding, including the CPU expert work in Metal models.
`TestMoEConfiguredSharedParity` compares every shared-expert layer of a local
Qwen3.5-MoE checkpoint with the original separate projections. Alternating
timing runs encountered competing builds in all 32 host-load samples, so the
allocation reduction is verified but no additional throughput or paging gain
is claimed from those runs.

Down projections prepare all selected experts' activation blocks in one pass,
using one temporary Q8/scales buffer pair instead of one pair per expert.
Block scales and offset sums remain independent, and expert outputs retain
their original reduction order. The grouped prefill path also quantizes
directly into its existing workspace, avoiding the temporary-buffer copy.
`TestMoEConfiguredDownParity` checks all routed FFNs against separately
dispatched down projections; Ornith's 40 layers match exactly with both CPU
and Metal loading. `BenchmarkExpertDownInputs` isolates input preparation
from matrix multiplication and worker dispatch. On M2 Max, Go 1.25.3, one
thread and eight 512-wide expert inputs, four alternating pairs measured
median preparation times of 2.503 µs → 2.306 µs for Q4_K and 2.810 µs →
2.612 µs for Q6_K (7.9% and 7.1% less time). No competing compiler/test
process appeared in 32 host-load samples. These are preparation timings,
not full-model token throughput. ARM64 SIMD and portable tests pass;
AMD64 fallback tests also pass under Rosetta; the AMD64 Q8 tests are skipped
there because the required CPU features are unavailable.

Bias-free SwiGLU experts with eight-lane-aligned hidden rows also share one
activation call across the slab. Biases, GPT-OSS activations and scalar tails
retain per-expert evaluation. `BenchmarkMoEActivation` measured 3.477 µs
per-expert versus 3.426 µs batched for eight 512-wide experts on M2 Max
(Go 1.25.3, one thread, four alternating pairs, no competing compiler/test
process in 39 load samples). This is a small activation-stage saving of
1.5%; the 32-expert case showed no clear benefit. The arithmetic and
reduction order are unchanged, with bit-exact output checks.

An experimental Qwen3.5-MoE/Ornith prefill groups up to 32 tokens by selected
expert, streams that expert's gate/up/down rows across its matching tokens,
distributes work by each group's token count, and retains the original router
order when summing each token's result. It reuses the CPU Q8
activation kernels, including ARM's four-position Q4_K/Q6_K row kernels.
Attention and DeltaNet stay chronological and retain their CPU/Metal choices.
Additional grouped-expert scratch is limited to 16 MiB per decode workspace;
expert banks stay mapped and are never copied to Metal or expanded to floats.
This path requires Q8 activations, F32 KV, Q4_K/Q6_K gate/up, and a supported
quantized down projection. Compressed KV, out-of-core, unsupported layouts
and MTP-enabled sparse prefill keep their established schedule.
Enable it explicitly with `GOPHERLLM_MOE_PREFILL=1`;
`GOPHERLLM_NO_MOE_PREFILL=1` disables it again for comparison. It stays off by
default: isolated-layer savings have not translated into a repeatable
full-model latency improvement under memory pressure.

Metal loading without `--prepare-quant` skips the separate token embedding
and Qwen3.5-MoE projection pairs that always execute in fused CPU kernels.
Tied output heads and used GPU projections retain their accelerator resources.
The CLI defaults to `--prefault core`, warming backbone tensors without eagerly reading every
expert bank; `--prefault all` restores full warm-up, and `none` leaves all
weights demand-paged. The library's zero-value load option still means `all`;
`--out-of-core` remains CPU-only. Avoid `--prepare-quant` when minimizing RAM
use. The operating system can still evict mapped pages under memory pressure.

On macOS, `TestMoEConfiguredPrefillParity` compares complete 64-token
responses with grouping on/off, clears conversation caches for each request,
and reports prefill/decode times, process-local major/minor faults and scratch
bytes. On an M2 Max with 32 GB, three alternating pairs after a warmup
(45 prompt tokens, 64 generated tokens, eight threads, Q8 activations,
F32 KV, no prepared weights, prefault=none) reproduced identical output IDs
within each backend. A follow-up with other builds paused recorded median
CPU prefill of 10.206 s serial versus 11.316 s grouped; complete-request
latency was 23.435 s versus 25.363 s, and median major faults were 722,232
versus 745,751. This remains an observation under memory pressure, with no
continuous host-load monitoring for the CPU series. A fresh Metal series
also passed output parity, but host-load monitoring captured other Go
compilers or test binaries in 25 of 39 samples; those timings are unsuitable
for a speedup claim. These results do not justify enabling grouped prefill
by default or claiming a decode gain.
Use the same `GOPHERLLM_MOE_MODEL` and optional
`GOPHERLLM_MOE_METAL=1` settings as the generation benchmark below.

With Ornith-1.0-35B Q4_K_M on M2 Max, eight CPU threads and Q8 activations,
three alternating warm pairs generated the same 64 tokens. Median CPU-only
decode fell from 3.951 s to 3.240 s, and total request time from 5.338 s to
4.339 s. Prefix caches were cleared before every request and both modes used
the same greedy sampling. Later CPU/Metal comparisons under memory pressure
retained output parity but had substantially higher absolute times; the
21.2 GB checkpoint's working set matters on a 32 GB Mac.
To repeat this full-model comparison, set
`GOPHERLLM_MOE_MODEL` to a local MoE GGUF and run
`go test -run '^TestMoEConfiguredGenerationParity$' -v ./internal/engine`; add `-tags metal`
and `GOPHERLLM_MOE_METAL=1` to test Metal. `BenchmarkMoEConfiguredLayer`
isolates one real sparse layer with synthetic inputs, rather than reporting
whole-model throughput.

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
experimental warning. Dense CPU hybrids batch attention/DeltaNet and FFN
projections in chunks of up to 32 tokens while preserving the existing chronological
convolution/DeltaNet recurrence and causal attention. MTP prefill uses the same
batches and catches its separate draft head up with every target hidden state.
Dense Metal hybrids preserve their decode arithmetic while batching supported
projections and FFNs, including with F16/I8 KV. Other accelerator-backed
layouts and out-of-core models retain serial prefill; grouped sparse prefill
has its separate opt-in above. With active MTP, quantized CPU-only projections
retain serial prefill when Q8 activations are disabled.
`GOPHERLLM_NO_BATCH_PREFILL=1` disables hybrid batching.
CPU float projection kernels can reduce sums differently from serial decode;
F16/I8 cache rounding can therefore produce different stored bytes.
On the M2 Max, the small quantized test graph measured 1.9–2.4× faster prefill.
A separate check using the first four of a real 27B model's 64 layers produced
identical logits, convolution and recurrent state for 8/32-token prompts.
This partial-graph check is not a throughput measurement of the complete 27B model.

ARM64 DeltaNet fuses decay/recall and update/output into two passes per row,
preserving the previous dot-product reduction and rounding. Four-tap convolution,
SiLU and output gating use NEON, and the recurrent worker task is reused without
per-step allocations. Other architectures and unsupported dimensions retain the
existing kernels. This includes Qwen3.8-27B GGUFs declaring `qwen35`.

Metal prefill prepares independent DeltaNet inputs across up to 32 positions
before chronological state updates. CPU projections retain Q8 row arithmetic;
prepared and float-only inputs keep their token path. Direct Q6_K QKV retains
the scalar GPU reduction with one submission/readback per block. Weight storage
is unchanged. Dense F16/I8 KV uses the same schedule with token-local cache
codecs. Tests compare state, cache bytes, replay and continuation bit-for-bit.

For memory-constrained Macs, the Qwen prefix cache reuses the unchanged live
DeltaNet state rather than retaining a second copy. Qwen3.8-27B keeps 149.625 MiB
of recurrent state instead of 299.25 MiB; KV growth transfers that allocation.
Serial requests release unused MTP KV and speculative verification scratch.
Mapped, unprepared MTP weights register their Metal resources only when used.
`--prefault core` skips the optional NextN block and the separate packed input
embedding: 251.1 + 682.0 MiB of avoided eager reads in this GGUF. Tied output
embeddings remain warm; physical residency remains under OS control.
Metal borrowing failures retain the CPU path without silently copying a matrix.
Explicit `--prefault all` and prepared loading retain their existing policies.

`TestQwen35ConfiguredMemory` uses the same model environment variables as the
comparison below. On a 32-GB M2 Max, Qwen3.8-27B Q4_K_M with Metal, Q8 activations,
no prepared weights and no prefault used 380.8 → 232.1 MiB of retained Go heap
after GC for a 16-token prompt plus four generated tokens, with identical token
digests. This excludes mapped weights and driver allocations; it is not total
model residency or a paging speedup measurement. Leave `--prepare-quant` off
when conserving memory: its
expanded scale tables are unused by Q8 activation kernels.

Component benchmarks on a 32-GB M2 Max with Go 1.25.3 measured these medians
over four alternating pairs. Parallel CPU work used eight threads.

| Component | Previous | Optimized |
| --- | ---: | ---: |
| DeltaNet step, 48 heads × 128, excluding projections | 389.8 µs | 124.3 µs |
| Four-tap convolution, 10240 channels, single-threaded | 47.5 µs | 14.9 µs |
| Metal Q6_K QKV, 10240×5120 matrix, 32 positions | 37.45 ms | 13.30 ms |
| Recurrent mixer including input/output projections, 32 positions | 72.49 ms | 34.60 ms |

The projection/mixer fixtures retain one layer with repeated encoded row patterns;
both mixer modes use the same batched output projection. These measurements
exclude FFN, periodic full attention and vocabulary work and do not establish
complete-model throughput.

For a complete local-model comparison, set `GOPHERLLM_QWEN_PERF_MODEL` to the
GGUF path and run `TestQwen35ConfiguredSerialPerformance`; add
`GOPHERLLM_QWEN_PERF_METAL=1` with a `-tags metal` test build for Metal. Each
run uses 16 raw prompt tokens, a 16-token output cap, cleared conversation
caches, one warm-up and three samples, logging token digests and separate
prefill/decode times. A complete Qwen3.8-27B Q4_K_M check completed four requests
per build: all eight emitted 16 tokens with identical token digests. Paging and
concurrent compiler/simulator activity prevented a reliable timing comparison;
no full-model speedup is claimed.

Vision families that require visual-feature injection
or multimodal MRoPE remain outside this text-generation scope: `qwen2vl`,
`qwen3vl`, `qwen3vlmoe`, and `qwen3next`. Qwen3.6/3.8 GGUFs with one trailing
MTP/NextN draft layer load that layer into its own KV cache. It is opt-in via
`--mtp-draft-tokens N` (or `WithMTPDraftTokens(N)`) and requires deterministic
generation (`--temp 0` or `--top-k 1`): every candidate is verified by the
target before it is emitted. Eligible dense CPU hybrids verify short target
blocks, restore rejected recurrent suffixes and retain an exact accepted-prefix
checkpoint. Unsupported graphs use serial verification. MTP remains off by
default; see [Speculative decoding](#speculative-decoding) for prompt lookup,
adaptive windows, diagnostics and measurement counters.

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
| Public API | `api_generated.go` (type aliases and forwarding functions), `doc.go` (package overview and version); implementations in `internal/engine/api.go` |
| GGUF parsing + file mapping | `internal/engine/gguf.go` over `internal/formats/gguf/`; `internal/engine/mmap.go` over `internal/mmapfile/` |
| Model loading + forward pass | `internal/engine/runner.go`, `internal/engine/model_*.go`, per-family files (`internal/engine/gemma4.go`, `internal/engine/qwen35.go`, `internal/engine/moe.go`, …), `internal/engine/forward_batch.go` (batched prefill) |
| Compute kernels + worker pool | `internal/engine/simd_*.go`, `internal/engine/quant_*.go`; platform dispatch and assembly in `internal/engine/kernels_*.go` / `internal/engine/kernels_*.s` |
| Tokenizers | `internal/tokenizer/` (SentencePiece, GPT-2/Tekken BPE, WordPiece), tables in `internal/wordpiece/` |
| Sampling | `internal/engine/sampling.go` |
| Generation orchestration + chat templates | `internal/engine/runtime.go`, `internal/engine/chat_render_*.go` |
| Tool calling / reasoning / skills | `internal/engine/agent.go`, `internal/engine/extract.go`, `internal/engine/skills.go`; wire types in `internal/tooling/` |
| Model discovery + selection | `internal/engine/catalog.go`, `internal/engine/catalog_ollama.go` |
| Embedding from other languages | `mobile/`, `bindings/` (C ABI, Swift, Rust, Python) |
| CLI | `cmd/gopherllm/`, `internal/engine/kernel_bench.go` |

[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) maps every directory,
shows how the language bindings layer on each other, and says where new code
belongs. Every SIMD kernel has a portable Go scalar reference implementation,
and differential tests assert they agree — when touching a kernel, run the
`Q4K`/`Q6K`/`DotF32`/`VectorOps` test groups first.

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
go test -run '^$' -bench=BenchmarkMatvecQ4K -benchmem ./internal/engine
```

Profile a real-model benchmark:

```sh
bin/gopherllm /path/to/model.gguf --prompt "test" --max-tokens 128 \
  --temp 0 --bench --bench-json --bench-runs 1 \
  --cpuprofile /tmp/gopherllm.prof
```

Local build artifacts are kept in `bin/` and `.cache/`, both ignored by git.

## iOS / iPhone

GopherLLM runs in iOS and macOS apps through a Swift package
(`bindings/swift`) with an async/await API: load a GGUF, stream or await chat
replies, cancel, count tokens, and inspect a model file before loading it.
Its binary part is an XCFramework built from the C ABI with only Go and Xcode:

```sh
make xcframework    # bindings/swift/GopherLLMCore.xcframework
make swift-test     # the package's tests against a synthetic model
```

Inference runs on the device with ARM64/NEON kernels and an opt-in Metal GPU
path; nothing goes over the network. [iOS integration](docs/ios.md) covers
adding the package to an app, getting models onto the device, memory
planning and entitlements, and the SwiftUI demo in `examples/ios`.

GitHub Actions runs `go test`, `go vet`, and `go build` on Linux, macOS, and
Windows, the `make cross-build` release matrix, the C ABI test on Linux and
macOS, and the XCFramework, Swift tests and iOS demo build on macOS.

## Bindings for Rust, Python, and C

Swift (above), Rust and Python all embed GopherLLM in-process through one
small C ABI (`bindings/c/include/gopherllm.h`) over the `mobile.Engine`
surface — no server process, no HTTP round trip. The C ABI loads a GGUF,
generates, chats or streams a completion, counts tokens and inspects models;
the Rust and Python packages currently wrap its load, generate, stream and
model-info calls.

```sh
make capi-build   # or: ./scripts/build-capi.sh
```

builds `build/capi/libgopherllm.{dylib,so,dll}` plus `gopherllm.h`. Then:

```sh
# Rust — bindings/rust/gopherllm (see its README for details)
cd bindings/rust/gopherllm && cargo run --example generate -- model.gguf "Hello!"

# Python — bindings/python (see its README for details)
cd bindings/python && pip install -e . && python3 examples/generate.py model.gguf "Hello!"
```

`bindings/c/shim` is a thin, hand-written cgo layer (no new business logic —
everything forwards to the already-tested `mobile.Engine`); it is kept
behind a `capi` build tag so `go build ./...` never needs a C toolchain to
find it, and CGO_ENABLED=1 with the tag is required to build it. If you only
need to call GopherLLM's OpenAI-compatible HTTP API from Rust, Python, or
anywhere else, the [Quickstart](#quickstart) server needs none of this —
these bindings are specifically for embedding the engine in-process.
