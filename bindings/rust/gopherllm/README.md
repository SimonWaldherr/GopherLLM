# gopherllm

Safe Rust bindings for [GopherLLM](https://github.com/SimonWaldherr/GopherLLM),
a pure-Go GGUF inference engine, over its C ABI (`bindings/c/shim` in the
GopherLLM repo). Mirrors the same small surface the Swift/Obj-C binding
(`mobile.Engine`) uses: load a GGUF, generate or stream a completion, read
basic model info — all in-process, no server to run.

## Setup

Build the C library once, from the GopherLLM repo root:

```sh
./scripts/build-capi.sh
```

This crate's `build.rs` finds it automatically at `<repo>/build/capi`. To
link a library built or installed elsewhere, set `GOPHERLLM_LIB_DIR` to its
directory.

## Usage

```rust
use gopherllm::{Engine, GenerationOptions, LoadOptions};

let engine = Engine::new();
engine.load("model.gguf", &LoadOptions::default())?;

// Non-streaming:
let text = engine.generate("Hello!", &GenerationOptions { max_tokens: Some(64), ..Default::default() })?;

// Streaming:
let result = engine.generate_stream(
    "Hello!",
    &GenerationOptions { max_tokens: Some(64), ..Default::default() },
    |delta| print!("{delta}"),
)?;
println!("\n{} tokens, finish_reason={}", result.generated_tokens, result.finish_reason);
# Ok::<(), gopherllm::Error>(())
```

Run the example (from this directory):

```sh
cargo run --example generate -- /path/to/model.gguf "Hello!"
```

## Testing

`tests/engine_test.rs` generates its own tiny, deterministic fixture model
on the fly via `go run ../../../cmd/gopherllm-synth-testmodel` — no binary
GGUF is checked into the repo. Run with:

```sh
cargo test
```

## Notes

- `Engine` is `Send + Sync`: calls are serialized on the Go side
  (`mobile.Engine`'s own mutex), so concurrent use from multiple threads is
  safe, if pointless (they simply queue).
- The FFI declarations in `src/ffi.rs` are hand-written against
  `bindings/c/shim`'s small, stable surface rather than `bindgen`-generated,
  so building this crate needs no `libclang`.
