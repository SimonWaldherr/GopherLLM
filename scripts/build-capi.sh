#!/usr/bin/env bash
# Builds GopherLLM's C ABI shared library (bindings/c/shim) for the host
# platform: libgopherllm.{dylib,so} plus its headers (libgopherllm.h,
# callbacks.h) into build/capi/. The Rust crate (bindings/rust) and Python
# package (bindings/python) both link against this output -- see their
# READMEs for how each locates it.
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="$root/build/capi"
mkdir -p "$out"

ldflags=""
case "$(uname -s)" in
  Darwin)
    lib="libgopherllm.dylib"
    # Go's c-shared output otherwise records its own bare filename as the
    # install name (LC_ID_DYLIB), which dyld only resolves via the standard
    # library search paths -- an rpath entry a consumer adds (as the Rust
    # crate's build.rs does) is silently ignored unless the reference starts
    # with @rpath/. This is the standard fix for a relocatable macOS dylib.
    ldflags="-ldflags=-extldflags=-Wl,-install_name,@rpath/$lib"
    ;;
  Linux) lib="libgopherllm.so" ;;
  MINGW*|MSYS*|CYGWIN*) lib="gopherllm.dll" ;;
  *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

echo "Building $lib ..."
(cd "$root" && GO111MODULE=on CGO_ENABLED=1 go build -tags capi -buildmode=c-shared ${ldflags:+"$ldflags"} -o "$out/$lib" ./bindings/c/shim)
cp "$root/bindings/c/shim/callbacks.h" "$out/callbacks.h"

echo "Built:"
echo "  $out/$lib"
echo "  $out/libgopherllm.h"
echo "  $out/callbacks.h"
