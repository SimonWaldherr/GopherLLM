# GopherLLM C ABI

A small, stable C ABI over `mobile.Engine` for embedding GopherLLM in-process
from any language with a C FFI. The Swift package (`bindings/swift`), the Rust
crate (`bindings/rust`) and the Python package (`bindings/python`) are all
built on it.

- `include/gopherllm.h` — the public header and the contract of every
  function: ownership, the `error_out` convention, threading, and the JSON
  shapes of options, messages and results.
- `shim/` — the cgo implementation. It only converts types and forwards to
  `mobile.Engine`; behavior and its tests live in `mobile/`.
- `test/engine_test.c` — an end-to-end test through the header against the
  static library and a synthetic model.

## Build

From the repository root:

```sh
make capi-build     # build/capi/libgopherllm.{dylib,so,dll} + gopherllm.h
make capi-test      # build the static library and run test/engine_test.c
make xcframework    # static libraries for iOS/macOS, packed for Swift
```

Building directly needs cgo and the `capi` tag, which keeps this package out
of the module's default `go build ./...` (CI's Windows runner has no C
toolchain):

```sh
go build -tags capi -buildmode=c-shared  -o libgopherllm.so ./bindings/c/shim
go build -tags capi -buildmode=c-archive -o libgopherllm.a  ./bindings/c/shim
```

Include `gopherllm.h`. Go also writes a `libgopherllm.h` next to the library,
which declares the same functions with Go's type names.

## API in short

- `gopherllm_engine_new` / `gopherllm_engine_free` — one handle per model.
- `gopherllm_load` / `gopherllm_unload` / `gopherllm_is_loaded` /
  `gopherllm_model_name` / `gopherllm_info_json` / `gopherllm_count_tokens`.
- `gopherllm_generate` (one prompt) and `gopherllm_chat` (a message history),
  blocking; `gopherllm_generate_stream` / `gopherllm_chat_stream` call
  `on_delta` per piece of text, then exactly one of `on_complete` /
  `on_error`.
- `gopherllm_cancel` — interrupts a running call from another thread.
- `gopherllm_inspect_model` (header only, no weights loaded),
  `gopherllm_runtime_info_json`, `gopherllm_version`.
- `gopherllm_free_string` — every returned `char *` is freed exactly once
  with this, not with libc `free`.
