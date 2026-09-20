# GopherLLM C ABI

A small, stable C ABI over `mobile.Engine` — the same gomobile-friendly
surface the Swift/Obj-C binding uses — for embedding GopherLLM in-process
from any language with a C FFI. The Rust crate (`bindings/rust`) and Python
package (`bindings/python`) are both built directly on top of this.

## Build

```sh
../../scripts/build-capi.sh
```

produces, per platform, into `build/capi/` at the repo root:

- `libgopherllm.dylib` / `libgopherllm.so` / `gopherllm.dll`
- `libgopherllm.h` — the generated header (includes `callbacks.h`)
- `callbacks.h` — the streaming callback typedefs, needed alongside the
  generated header at compile time

Building directly with `go build` needs the `capi` tag (kept out of the
module's default `go build ./...` deliberately — see `shim/main.go`'s doc
comment for why):

```sh
go build -tags capi -buildmode=c-shared -o libgopherllm.dylib ./shim
```

## API

See `shim/main.go`'s doc comments for the authoritative contract of each
function (ownership, NULL conventions, threading). In short:

- `gopherllm_engine_new` / `gopherllm_engine_free` — one opaque handle per
  model.
- `gopherllm_load` / `gopherllm_unload` / `gopherllm_is_loaded` /
  `gopherllm_model_name` / `gopherllm_info_json`.
- `gopherllm_generate` — blocking, returns text or an error.
- `gopherllm_generate_stream` — blocking, invokes `onDelta` per increment
  then exactly one of `onComplete`/`onError`.
- `gopherllm_cancel` — interrupts an in-flight call from another thread.
- `gopherllm_free_string` — every `char*` this API returns must be freed
  exactly once with this.

Every returned `char*` is heap-allocated by Go (`C.CString`) and must be
freed with `gopherllm_free_string`, not libc `free` directly (they happen to
be compatible today, but treat the allocator as GopherLLM's, not libc's).
