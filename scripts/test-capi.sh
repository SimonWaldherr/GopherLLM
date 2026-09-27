#!/usr/bin/env bash
# End-to-end test of the C ABI (bindings/c): builds the static library the
# XCFramework also uses, writes internal/testmodel's GGUF, then compiles and
# runs bindings/c/test/engine_test.c against both. Needs Go with cgo and a C
# compiler (cc).
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/gopherllm-capi-test.XXXXXX")
trap 'rm -rf "$work"' EXIT

cd "$root"
CGO_ENABLED=1 go build -tags capi -buildmode=c-archive -o "$work/libgopherllm.a" ./bindings/c/shim
go run ./cmd/gopherllm-synth-testmodel -out "$work/tiny.gguf"

case "$(uname -s)" in
  Darwin) libs=(-framework Accelerate -framework Foundation) ;;
  *) libs=(-lpthread -lm -ldl) ;;
esac
${CC:-cc} -std=c11 -Wall -Wextra -Werror -I bindings/c/include \
  bindings/c/test/engine_test.c "$work/libgopherllm.a" "${libs[@]}" -o "$work/engine_test"
"$work/engine_test" "$work/tiny.gguf"
