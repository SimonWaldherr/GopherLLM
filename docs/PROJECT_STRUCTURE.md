# Project structure

GopherLLM keeps its public Go package at the repository root so applications
keep importing `github.com/SimonWaldherr/GopherLLM`. The root holds the public
API and the tightly coupled inference runtime; everything that stands on its
own lives in a subpackage, and implementation details below `internal/`.

## Directory map

| Location | Responsibility |
|---|---|
| `*.go`, `*.s` (root) | Public API (`api.go`, `doc.go`) and the inference runtime: model loading (`model_*.go`, `runner.go`), forward passes per family (`gemma4.go`, `qwen35.go`, `moe.go`, …), batched prefill (`forward_batch.go`), kernels and worker pool (`simd_*.go`, `quant_*.go`, platform dispatch in `kernels_*.go` / `kernels_*.s`), sampling, chat templates (`chat_render_*.go`), tool calling and reasoning extraction, speech (`voxtral_*.go`, `parakeet_*.go`), vision (`pixtral_*.go`), Laya classification (`laya_*.go`), Metal offload (`metal_*_darwin.go`) |
| `mobile/` | The language-neutral embedding surface: strings and JSON in, strings and JSON out |
| `bindings/c/` | C ABI over `mobile` (`include/gopherllm.h`, cgo shim, end-to-end C test) |
| `bindings/swift/` | Swift package over the C ABI for iOS and macOS apps |
| `bindings/rust/`, `bindings/python/` | Rust crate and Python package over the C ABI |
| `server/` | OpenAI-/Ollama-compatible HTTP API, streaming, embedded chat UI |
| `rag/`, `agent/` | Hybrid BM25/vector retrieval; Model + retrieval + tools with citations |
| `agentos/` | Sandboxed command execution for agent tools |
| `huggingface/` | Hub search, download and `hf:` model resolution (kept out of the root package so it does not pull in `net/http`) |
| `voiceweb/` | Browser audio-capture worklet shared by web UIs |
| `cmd/` | Executables: the `gopherllm` CLI, wasm build, evaluation, speech and YOLO tools, the synthetic test model writer, Hestia |
| `examples/` | Small runnable programs, including the SwiftUI demo in `examples/ios/` |
| `internal/formats/` | GGUF and safetensors container parsing and writing |
| `internal/tokenizer/`, `internal/wordpiece/` | Tokenizers and their normalization tables |
| `internal/metal/`, `internal/layablas/`, `internal/voxtralblas/`, `internal/webgpu/` | GPU and Accelerate backends |
| `internal/mmapfile/`, `internal/numeric/`, `internal/threads/` | File mapping backends, float16/fast-math helpers, the shared worker count |
| `internal/vision/`, `internal/huggingface/`, `internal/jsonconstraint/`, `internal/tooling/` | Image preprocessing and YOLO, hub client, JSON-constrained decoding, tool-call wire types |
| `internal/testmodel/`, `internal/yolotest/` | Deterministic fixtures written on demand, so no binary model is checked in |
| `integration/` | Black-box tests of the public module, consumer-module and opt-in local-model tests |
| `testdata/` | Small fixtures and the external-consumer module |
| `scripts/` | Build and test scripts for the bindings (C ABI, XCFramework) and fixture generators |
| `docs/` | Guides and the GitHub Pages site |

## Embedding layers

Every non-Go language goes through the same three layers, so a feature added
to `mobile` reaches all of them the same way:

```
Swift app ──► bindings/swift (GopherLLMEngine, async/await)
Rust      ──► bindings/rust ─┐
Python    ──► bindings/python┼─► bindings/c (gopherllm.h, cgo) ──► mobile.Engine ──► gopherllm.Model
C / C++   ───────────────────┘
```

- `mobile` owns behavior: option parsing and validation, concurrency, result
  shapes. It is tested with a real (synthetic) model in Go.
- `bindings/c/shim` only converts types. `gopherllm.h` is the contract; the
  Rust, Python and Swift declarations mirror it.
- Language packages add idioms (async/await, iterators, context managers) and
  nothing else.

To expose something new: add it to `mobile` with a test, forward it in the shim
and declare it in `gopherllm.h`, extend `bindings/c/test/engine_test.c`, then
surface it in the language packages that need it.

## Placement rules

- Keep an exported inference API, or code that needs package-private `Runner`
  or weight internals, in the module root.
- Put implementation details that can stand on their own under `internal/`.
- Keep network access (`net/http`) out of the root package; the dependency
  tests in `integration/` enforce the root package's import closure.
- Put black-box tests that import the public module under `integration/`.
- Generate fixtures in code (`internal/testmodel`) or with a script; keep model
  files, build products and local caches out of the repository.
