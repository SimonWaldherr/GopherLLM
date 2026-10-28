# Project structure

GopherLLM keeps its public Go package at the repository root so applications
keep importing `github.com/SimonWaldherr/GopherLLM`. The root contains a public
facade; the tightly coupled inference runtime lives in `internal/engine/`.
Independent implementation details have their own packages below `internal/`.

## Directory map

| Location | Responsibility |
|---|---|
| `doc.go`, `api_generated.go`, `example_test.go` (root) | Public package documentation, generated type aliases and forwarding functions, public Go examples |
| `internal/engine/` | Model loading, forward passes, prefill, decoding, sampling, chat/tool handling, speech and image pipelines; tightly coupled kernels, assembly, platform adapters and their tests |
| `internal/cmd/genapi/` | Generates the public facade from exported engine declarations; CI checks it with `go run ./internal/cmd/genapi -check` |
| `mobile/` | The language-neutral embedding surface: strings and JSON in, strings and JSON out |
| `bindings/c/` | C ABI over `mobile` (`include/gopherllm.h`, cgo shim, end-to-end C test) |
| `bindings/swift/` | Swift package over the C ABI for iOS and macOS apps |
| `bindings/rust/`, `bindings/python/` | Rust crate and Python package over the C ABI |
| `server/` | OpenAI-/Ollama-compatible HTTP API, streaming, embedded chat UI |
| `rag/`, `agent/` | Hybrid BM25/vector retrieval; Model + retrieval + tools with citations |
| `coreml/`, `internal/coreml/` | Optional embedding adapter and native Core ML runtime; `internal/engine/coreml_ffn*.go` attaches experimental prefill FFNs (`coreml` build tag) |
| `agentos/` | Sandboxed command execution for agent tools |
| `huggingface/` | Hub search, download and `hf:` model resolution (kept out of the root package so it does not pull in `net/http`) |
| `voiceweb/` | Browser audio-capture worklet shared by web UIs |
| `cmd/` | Executables: the `gopherllm` CLI, wasm build, evaluation, speech and YOLO tools, the synthetic test model writer, Hestia |
| `examples/` | Small runnable programs, including the SwiftUI demo in `examples/ios/` |
| `internal/formats/` | GGUF and safetensors container parsing and writing |
| `internal/tokenizer/`, `internal/wordpiece/` | Tokenizers and their normalization tables |
| `internal/metal/`, `internal/layablas/`, `internal/voxtralblas/`, `internal/webgpu/` | GPU and Accelerate backends |
| `internal/cpuinfo/` | CPU instruction-set probes, performance-core counts and hardware identity for kernel dispatch and autotuning |
| `internal/mmapfile/`, `internal/numeric/`, `internal/threads/` | File mapping backends, float16/fast-math helpers, the shared worker count |
| `internal/vision/`, `internal/huggingface/`, `internal/jsonconstraint/`, `internal/tooling/` | Image preprocessing and YOLO, hub client, JSON-constrained decoding, tool-call wire types |
| `internal/testmodel/`, `internal/yolotest/` | Deterministic fixtures written on demand, so no binary model is checked in |
| `integration/` | Black-box tests of the public module, consumer-module and opt-in local-model tests |
| `testdata/` | Small fixtures and the external-consumer module |
| `internal/engine/{tts,diffusion,qwen_image21,flux3}_*.go` | Native MMS/VITS speech synthesis, CIFAR-10 diffusion, Qwen-Image-2.1 and FLUX text-to-image generation; commands in `cmd/tts-synthesize` and `cmd/image-generate` |
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

- Keep inference implementations and code that needs package-private `Runner`
  or weight internals in `internal/engine/`, with their tests. After changing
  exported declarations, run `go generate .` from the module root to refresh
  the public facade. Type aliases preserve method sets; forwarding functions
  keep the public entry points callable as functions.
- Aliased runtime types are defined in `internal/engine/`, so reflection reports
  that package path. Re-exported error sentinels retain their identity; callers
  should compare them with `errors.Is` rather than reassigning package variables.
- Put implementation details that can stand on their own under `internal/`.
- Keep network access (`net/http`) out of the root package; the dependency
  tests in `integration/` enforce the root package's import closure.
- Put black-box tests that import the public module under `integration/`.
- Generate fixtures in code (`internal/testmodel`) or with a script; keep model
  files, build products and local caches out of the repository.

Run engine tests or microbenchmarks explicitly with `go test ./internal/engine`;
`go test ./...` still includes the engine, public examples and integration tests.
The `go doc` command does not expand methods behind the public type aliases;
from a checkout, use `go doc ./internal/engine.Model.Generate` for method
documentation, replacing `Model.Generate` with the desired type and method.
Shared reference fixtures remain under the module's `testdata/` so the Python
reference generators and Go tests use the same files.
