# GopherLLM bindings

In-process embedding of GopherLLM from languages other than Go. All of them
sit on one C ABI over `mobile.Engine` (see
[docs/PROJECT_STRUCTURE.md](../docs/PROJECT_STRUCTURE.md#embedding-layers)):

- **[c](c/)** — the C ABI itself: `include/gopherllm.h`, the cgo shim, and an
  end-to-end test (`make capi-test`).
- **[swift](swift/)** — a Swift package for iOS and macOS apps, over an
  XCFramework built with `make xcframework`. See [docs/ios.md](../docs/ios.md).
- **[rust](rust/gopherllm/)** — a safe Rust crate over the shared library
  (`make capi-build`).
- **[python](python/)** — a `ctypes`-based Python package over the shared
  library.
