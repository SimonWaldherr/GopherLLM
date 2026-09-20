# GopherLLM bindings

In-process embedding of GopherLLM from languages other than Go, all built on
the same small surface (`mobile.Engine`): load a GGUF, generate or stream a
completion, read basic model info.

- **Swift / Objective-C** — via `gomobile` directly against `mobile/` at the
  repo root (no C ABI involved). See the top-level README's
  [iOS / iPhone](../README.md#ios--iphone) section and `docs/ios.md`.
- **[c](c/)** — the C ABI (`libgopherllm` + headers) everything below is
  built on. Build with `../scripts/build-capi.sh`.
- **[rust](rust/gopherllm/)** — a safe Rust crate over the C ABI.
- **[python](python/)** — a `ctypes`-based Python package over the C ABI.

See the top-level README's
[Bindings for Rust, Python, and C](../README.md#bindings-for-rust-python-and-c)
section for the quickest way to get started, or each subdirectory's own
README for details.
