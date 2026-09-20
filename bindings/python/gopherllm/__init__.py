"""Safe Python bindings for GopherLLM (https://github.com/SimonWaldherr/GopherLLM),
a pure-Go GGUF inference engine, over its C ABI (bindings/c/shim in the
GopherLLM repo). Mirrors the same small surface the Swift/Obj-C binding
(mobile.Engine) uses: load a GGUF, generate or stream a completion, read
basic model info.

    from gopherllm import Engine, GenerationOptions

    with Engine() as engine:
        engine.load("model.gguf")
        print(engine.generate("Hello!", GenerationOptions(max_tokens=64)))
"""

from __future__ import annotations

import ctypes
import json
import queue
import threading
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, List, Optional

from . import _ffi

__all__ = [
    "Engine",
    "LoadOptions",
    "GenerationOptions",
    "ModelInfo",
    "GenerateResult",
    "GopherLLMError",
]

__version__ = "0.1.0"


class GopherLLMError(RuntimeError):
    """Raised for any error GopherLLM itself reported (a bad model path, an
    already-loaded engine, an invalid option, a generation failure, ...).
    The message is whatever GopherLLM reported; it is not further
    categorized."""


def _options_json(options: object) -> bytes:
    data = {k: v for k, v in asdict(options).items() if v not in (None, [])}
    return json.dumps(data).encode("utf-8")


@dataclass
class LoadOptions:
    """Options for Engine.load. Every field is optional and falls back to
    GopherLLM's own default when omitted (None)."""

    threads: Optional[int] = None
    prepare_quantized: Optional[bool] = None
    out_of_core: Optional[bool] = None
    prefault: Optional[str] = None  # "none" | "core" | "all"
    metal: Optional[bool] = None


@dataclass
class GenerationOptions:
    """Options for Engine.generate/generate_stream/stream."""

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None
    system_prompt: Optional[str] = None
    stop: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelInfo:
    """Stable, inexpensive model metadata; see Engine.info()."""

    name: str
    file_size_bytes: int
    architecture: str
    context_length: int
    vocab_size: int
    mapped: bool
    metal_available: bool


@dataclass(frozen=True)
class GenerateResult:
    """The result of a completed Engine.generate_stream() call."""

    text: str
    finish_reason: str
    generated_tokens: int


_STREAM_DONE = object()


class Engine:
    """One loaded (or loadable) model. Mirrors mobile.Engine, the same
    gomobile-friendly API the Swift/Obj-C binding uses: independent
    instances are unrelated, and calls that touch model memory are
    serialized on the Go side, so it is safe (if pointless) to call an
    Engine from multiple threads concurrently -- they simply queue.

    Use as a context manager to guarantee the underlying handle (and any
    loaded model) is released deterministically rather than whenever the
    garbage collector gets to it:

        with Engine() as engine:
            engine.load("model.gguf")
            ...
    """

    def __init__(self) -> None:
        self._handle = _ffi.lib().gopherllm_engine_new()
        self._closed = False

    def close(self) -> None:
        """Releases the engine (unmapping any loaded model). Idempotent."""
        if not self._closed:
            _ffi.lib().gopherllm_engine_free(self._handle)
            self._closed = True

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort backstop for a caller that forgets close()/`with`: a
        # loaded model should not outlive every reference to it for the rest
        # of the process. close() is idempotent, so this never double-frees.
        try:
            self.close()
        except Exception:
            pass

    def _check_open(self) -> None:
        if self._closed:
            raise GopherLLMError("engine is closed")

    def load(self, path: str, options: Optional[LoadOptions] = None) -> None:
        """Loads a GGUF model file. Raises if a model is already loaded
        (call unload() first) or if the file cannot be opened."""
        self._check_open()
        err = _ffi.lib().gopherllm_load(self._handle, path.encode("utf-8"), _options_json(options or LoadOptions()))
        if err:
            raise GopherLLMError(_ffi.take_string(err))

    def unload(self) -> None:
        """Releases the current model, if any. The engine itself remains
        usable for a subsequent load()."""
        self._check_open()
        err = _ffi.lib().gopherllm_unload(self._handle)
        if err:
            raise GopherLLMError(_ffi.take_string(err))

    def is_loaded(self) -> bool:
        if self._closed:
            return False
        return bool(_ffi.lib().gopherllm_is_loaded(self._handle))

    def model_name(self) -> str:
        """The loaded model's display name, or "" if nothing is loaded."""
        self._check_open()
        return _ffi.take_string(_ffi.lib().gopherllm_model_name(self._handle))

    def info(self) -> ModelInfo:
        self._check_open()
        raw = _ffi.take_string(_ffi.lib().gopherllm_info_json(self._handle)) or "{}"
        data = json.loads(raw)
        return ModelInfo(
            name=data.get("name", ""),
            file_size_bytes=data.get("file_size_bytes", 0),
            architecture=data.get("architecture", ""),
            context_length=data.get("context_length", 0),
            vocab_size=data.get("vocab_size", 0),
            mapped=data.get("mapped", False),
            metal_available=data.get("metal_available", False),
        )

    def cancel(self) -> None:
        """Interrupts an in-flight generate/generate_stream/stream call, if
        any. Safe to call at any time, including with nothing running."""
        if not self._closed:
            _ffi.lib().gopherllm_cancel(self._handle)

    def generate(self, prompt: str, options: Optional[GenerationOptions] = None) -> str:
        """Runs a non-streaming chat completion and returns the generated
        text."""
        self._check_open()
        error_out = ctypes.c_void_p()
        text_ptr = _ffi.lib().gopherllm_generate(
            self._handle,
            prompt.encode("utf-8"),
            _options_json(options or GenerationOptions()),
            ctypes.byref(error_out),
        )
        if error_out.value:
            raise GopherLLMError(_ffi.take_string(error_out.value))
        return _ffi.take_string(text_ptr)

    def generate_stream(
        self,
        prompt: str,
        options: Optional[GenerationOptions] = None,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> GenerateResult:
        """Runs a streaming chat completion, calling on_delta once per text
        increment as it is generated, then returns the final result. Blocks
        the calling thread until generation completes, fails, or is
        cancel()ed -- see Engine.stream() for a generator that runs this on a
        background thread instead."""
        self._check_open()
        options_json = _options_json(options or GenerationOptions())
        outcome: dict = {}

        @_ffi.DeltaCB
        def _on_delta(text: bytes, _userdata: int) -> None:
            if on_delta is not None:
                on_delta((text or b"").decode("utf-8", errors="replace"))

        @_ffi.CompleteCB
        def _on_complete(text: bytes, _userdata: int) -> None:
            outcome["json"] = (text or b"{}").decode("utf-8", errors="replace")

        @_ffi.ErrorCB
        def _on_error(text: bytes, _userdata: int) -> None:
            outcome["error"] = (text or b"unknown error").decode("utf-8", errors="replace")

        call_err = _ffi.lib().gopherllm_generate_stream(
            self._handle,
            prompt.encode("utf-8"),
            options_json,
            _on_delta,
            _on_complete,
            _on_error,
            None,
        )
        if call_err:
            raise GopherLLMError(_ffi.take_string(call_err))
        if "error" in outcome:
            raise GopherLLMError(outcome["error"])
        data = json.loads(outcome.get("json", "{}"))
        return GenerateResult(
            text=data.get("text", ""),
            finish_reason=data.get("finish_reason", ""),
            generated_tokens=data.get("generated_tokens", 0),
        )

    def stream(self, prompt: str, options: Optional[GenerationOptions] = None) -> Iterator[str]:
        """Generator convenience over generate_stream: runs generation on a
        background thread and yields each delta as it arrives, so a plain
        `for` loop can consume output while it is still being generated:

            for delta in engine.stream("Hello!"):
                print(delta, end="", flush=True)

        The final GenerateResult is not accessible this way (use
        generate_stream directly for finish_reason/generated_tokens).
        Abandoning the loop early (break) leaves the underlying generation
        running to completion in the background; call cancel() first if that
        matters.
        """
        q: "queue.Queue[object]" = queue.Queue()
        outcome: dict = {}

        def worker() -> None:
            try:
                self.generate_stream(prompt, options, on_delta=q.put)
            except GopherLLMError as e:
                outcome["error"] = e
            finally:
                q.put(_STREAM_DONE)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is _STREAM_DONE:
                break
            yield item  # type: ignore[misc]
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
