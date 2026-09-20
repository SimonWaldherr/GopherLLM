# gopherllm

Python bindings for [GopherLLM](https://github.com/SimonWaldherr/GopherLLM),
a pure-Go GGUF inference engine, over its C ABI (`bindings/c/shim` in the
GopherLLM repo) via `ctypes` — no compiled extension, no server to run.
Mirrors the same small surface the Swift/Obj-C binding (`mobile.Engine`)
uses: load a GGUF, generate or stream a completion, read basic model info.

## Setup

Build the C library once, from the GopherLLM repo root:

```sh
./scripts/build-capi.sh
```

Then install this package (from this directory):

```sh
pip install -e .
```

`gopherllm` finds the built library automatically at `<repo>/build/capi`
when imported from a checkout. To point at a library built or installed
elsewhere, set `GOPHERLLM_LIB_DIR` (a directory) or `GOPHERLLM_LIBRARY_PATH`
(the library file itself).

## Usage

```python
from gopherllm import Engine, GenerationOptions

with Engine() as engine:
    engine.load("model.gguf")

    # Non-streaming:
    print(engine.generate("Hello!", GenerationOptions(max_tokens=64)))

    # Streaming, as an iterator:
    for delta in engine.stream("Hello!", GenerationOptions(max_tokens=64)):
        print(delta, end="", flush=True)

    # Streaming, with the final result (finish_reason, generated_tokens):
    result = engine.generate_stream("Hello!", on_delta=print)
```

Run the example:

```sh
python3 examples/generate.py /path/to/model.gguf "Hello!"
```

## Testing

`tests/test_engine.py` (via `conftest.py`) generates its own tiny,
deterministic fixture model on the fly with
`go run ../../cmd/gopherllm-synth-testmodel` — no binary GGUF is checked
into the repo. Run with:

```sh
pip install pytest
python3 -m pytest tests/
```

## Notes

- `Engine.stream()` runs generation on a background thread so it can yield
  deltas to a plain `for` loop as they arrive; `generate_stream()` blocks the
  calling thread instead but also gives you the final `GenerateResult`
  (`finish_reason`, `generated_tokens`).
- `Engine` supports the context-manager protocol (`with Engine() as e:`) to
  guarantee the underlying model is released deterministically; `close()` is
  also called from `__del__` as a backstop, and is idempotent either way.
- Invalid UTF-8 from the model is replaced (`errors="replace"`), matching
  the Rust binding's `to_string_lossy()` and Swift's `String(cString:)`.
