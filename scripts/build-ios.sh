#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "iOS bindings require macOS, Xcode, and gomobile." >&2; exit 1
fi
command -v xcrun >/dev/null || { echo "Xcode command-line tools are required (xcrun not found)." >&2; exit 1; }
gomobile_bin="${GOMOBILE_BIN:-$(command -v gomobile 2>/dev/null || true)}"
if [[ -z "$gomobile_bin" && -x "$(go env GOPATH)/bin/gomobile" ]]; then gomobile_bin="$(go env GOPATH)/bin/gomobile"; fi
[[ -n "$gomobile_bin" ]] || { echo "gomobile not found. Install it with: go install golang.org/x/mobile/cmd/gomobile@${GOMOBILE_VERSION:-latest}; gomobile init" >&2; exit 1; }
xcrun --sdk iphoneos --show-sdk-path >/dev/null

root=$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="$root/build/GopherLLM.xcframework"
mkdir -p "$root/build"
rm -rf "$out"

# Newer gomobile releases require gobind as a Go tool dependency. The root
# module intentionally has no tool/runtime dependencies, so create a tiny
# disposable driver module which replaces the root with this checkout.
tmp=$(mktemp -d "${TMPDIR:-/tmp}/gopherllm-ios.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
cd "$tmp"
go mod init gopherllm-ios-bind >/dev/null
go mod edit -require=github.com/SimonWaldherr/GopherLLM@v0.0.0
go mod edit -replace=github.com/SimonWaldherr/GopherLLM="$root"
go get -tool "golang.org/x/mobile/cmd/gobind@${GOMOBILE_VERSION:-latest}" >/dev/null
"$gomobile_bin" bind -v -tags metal -target=ios,iossimulator -iosversion=17.0 -trimpath -o "$out" github.com/SimonWaldherr/GopherLLM/mobile
[[ -f "$out/Info.plist" ]] || { echo "gomobile did not produce an XCFramework" >&2; exit 1; }
plutil -lint "$out/Info.plist"
echo "Created $out"
plutil -p "$out/Info.plist" | sed -n '/AvailableLibraries/,/]/p'
