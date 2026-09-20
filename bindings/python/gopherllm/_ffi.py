"""Raw ctypes bindings to libgopherllm (bindings/c/shim in the GopherLLM
repo). Mirrors bindings/c/shim/main.go's exported C signatures one-to-one;
see that file for the authoritative doc comments on each function's
contract (ownership, NULL conventions, threading). Hand-written rather than
generated: the C surface is small (13 functions) and changes rarely.
"""

from __future__ import annotations

import ctypes
import os
import platform

DeltaCB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
CompleteCB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
ErrorCB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


def _library_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libgopherllm.dylib"
    if system == "Windows":
        return "gopherllm.dll"
    return "libgopherllm.so"


def _candidate_dirs() -> list[str]:
    dirs = []
    env_file = os.environ.get("GOPHERLLM_LIBRARY_PATH")
    if env_file:
        # A caller may point this directly at the library file rather than
        # its directory; both are accepted for convenience.
        dirs.append(os.path.dirname(env_file) or ".")
    env_dir = os.environ.get("GOPHERLLM_LIB_DIR")
    if env_dir:
        dirs.append(env_dir)
    # Bundled next to this package, for an installed wheel that ships the
    # library (see bindings/python/README.md for how to build one).
    dirs.append(os.path.dirname(os.path.abspath(__file__)))
    # <repo>/build/capi, scripts/build-capi.sh's default output, for running
    # straight out of a checkout without installing anything.
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.normpath(os.path.join(here, "..", "..", "..", "build", "capi")))
    return dirs


def _load_library() -> ctypes.CDLL:
    filename = _library_filename()
    explicit = os.environ.get("GOPHERLLM_LIBRARY_PATH")
    if explicit and os.path.isfile(explicit):
        return ctypes.CDLL(explicit)
    tried = []
    for d in _candidate_dirs():
        path = os.path.join(d, filename)
        tried.append(path)
        if os.path.isfile(path):
            return ctypes.CDLL(path)
    raise OSError(
        f"could not find {filename}. Build it first with "
        f"scripts/build-capi.sh from the GopherLLM repo root, then either run "
        f"from a checkout (searched {tried}) or set GOPHERLLM_LIB_DIR / "
        f"GOPHERLLM_LIBRARY_PATH to point at it."
    )


_lib = _load_library()

_lib.gopherllm_engine_new.argtypes = []
_lib.gopherllm_engine_new.restype = ctypes.c_size_t

_lib.gopherllm_engine_free.argtypes = [ctypes.c_size_t]
_lib.gopherllm_engine_free.restype = None

_lib.gopherllm_load.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p]
_lib.gopherllm_load.restype = ctypes.c_void_p

_lib.gopherllm_unload.argtypes = [ctypes.c_size_t]
_lib.gopherllm_unload.restype = ctypes.c_void_p

_lib.gopherllm_is_loaded.argtypes = [ctypes.c_size_t]
_lib.gopherllm_is_loaded.restype = ctypes.c_int

_lib.gopherllm_model_name.argtypes = [ctypes.c_size_t]
_lib.gopherllm_model_name.restype = ctypes.c_void_p

_lib.gopherllm_info_json.argtypes = [ctypes.c_size_t]
_lib.gopherllm_info_json.restype = ctypes.c_void_p

_lib.gopherllm_cancel.argtypes = [ctypes.c_size_t]
_lib.gopherllm_cancel.restype = None

_lib.gopherllm_generate.argtypes = [
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_void_p),
]
_lib.gopherllm_generate.restype = ctypes.c_void_p

_lib.gopherllm_generate_stream.argtypes = [
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_char_p,
    DeltaCB,
    CompleteCB,
    ErrorCB,
    ctypes.c_void_p,
]
_lib.gopherllm_generate_stream.restype = ctypes.c_void_p

_lib.gopherllm_free_string.argtypes = [ctypes.c_void_p]
_lib.gopherllm_free_string.restype = None


def take_string(ptr: int | None) -> str:
    """Decodes and frees a `char*` GopherLLM returned (every returned string
    must be freed exactly once, per bindings/c/shim's convention); "" for
    NULL. Invalid UTF-8 is replaced rather than raised, matching the Rust
    binding's to_string_lossy() and Swift's String(cString:) behavior."""
    if not ptr:
        return ""
    try:
        raw = ctypes.cast(ptr, ctypes.c_char_p).value or b""
        return raw.decode("utf-8", errors="replace")
    finally:
        _lib.gopherllm_free_string(ptr)


def lib() -> ctypes.CDLL:
    return _lib
