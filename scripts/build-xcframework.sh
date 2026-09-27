#!/usr/bin/env bash
# Builds GopherLLMCore.xcframework: the C ABI (bindings/c) as a static
# library for iOS devices, the iOS simulator and macOS, with gopherllm.h
# exposed as the Clang module GopherLLMCore. The Swift package in
# bindings/swift wraps it; apps normally depend on that package rather than
# on the XCFramework directly. No gomobile or other tool beyond Go and Xcode.
#
# Environment:
#   XCFRAMEWORK_OUT          output path (default bindings/swift/GopherLLMCore.xcframework)
#   PLATFORMS                any of "ios ios-simulator macos" (default: all three)
#   IOS_DEPLOYMENT_TARGET    default 17.0
#   MACOS_DEPLOYMENT_TARGET  default 14.0
#   METAL                    1 (default) compiles the Metal GPU kernels in; 0 leaves them out
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "building the XCFramework requires macOS with Xcode" >&2
  exit 1
fi
command -v xcrun >/dev/null || { echo "Xcode command-line tools are required (xcrun not found)" >&2; exit 1; }

root=$(CDPATH= cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="${XCFRAMEWORK_OUT:-$root/bindings/swift/GopherLLMCore.xcframework}"
platforms="${PLATFORMS:-ios ios-simulator macos}"
ios_min="${IOS_DEPLOYMENT_TARGET:-17.0}"
macos_min="${MACOS_DEPLOYMENT_TARGET:-14.0}"
tags="capi"
[[ "${METAL:-1}" == "1" ]] && tags="capi metal"

work=$(mktemp -d "${TMPDIR:-/tmp}/gopherllm-xcframework.XXXXXX")
trap 'rm -rf "$work"' EXIT

# build_slice <dir> <GOOS> <GOARCH> <sdk> <clang target triple>
build_slice() {
  local dir="$work/$1" goos=$2 goarch=$3 sdk=$4 target=$5 sysroot
  sysroot=$(xcrun --sdk "$sdk" --show-sdk-path)
  mkdir -p "$dir"
  echo "building $1 ($target)"
  (cd "$root" && CGO_ENABLED=1 GOOS="$goos" GOARCH="$goarch" \
    CC="$(xcrun --sdk "$sdk" --find clang)" \
    CGO_CFLAGS="-O2 -target $target -isysroot $sysroot" \
    CGO_LDFLAGS="-target $target -isysroot $sysroot" \
    go build -trimpath -tags "$tags" -buildmode=c-archive -o "$dir/libgopherllm.a" ./bindings/c/shim)
}

# The slices are independent and Go's build cache is safe for concurrent
# use, so build them in parallel and fail if any one fails.
slices=()
for platform in $platforms; do
  case "$platform" in
    ios) slices+=("ios-arm64 ios arm64 iphoneos arm64-apple-ios$ios_min") ;;
    ios-simulator)
      slices+=("sim-arm64 ios arm64 iphonesimulator arm64-apple-ios$ios_min-simulator")
      slices+=("sim-x86_64 ios amd64 iphonesimulator x86_64-apple-ios$ios_min-simulator") ;;
    macos)
      slices+=("macos-arm64 darwin arm64 macosx arm64-apple-macos$macos_min")
      slices+=("macos-x86_64 darwin amd64 macosx x86_64-apple-macos$macos_min") ;;
    *) echo "unknown platform $platform (want ios, ios-simulator or macos)" >&2; exit 1 ;;
  esac
done
pids=()
for slice in "${slices[@]}"; do
  # shellcheck disable=SC2086 # the slice spec is deliberately word-split
  build_slice $slice & pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[[ $fail == 0 ]] || { echo "a slice failed to build" >&2; exit 1; }

# The headers live in a directory named after the module: Clang finds
# <search path>/GopherLLMCore/module.modulemap, and the module map cannot
# collide with another XCFramework's in the app's shared include directory.
headers="$work/headers"
mkdir -p "$headers/GopherLLMCore"
cp "$root/bindings/c/include/gopherllm.h" "$headers/GopherLLMCore/"
cat > "$headers/GopherLLMCore/module.modulemap" <<'MODULEMAP'
module GopherLLMCore {
    header "gopherllm.h"
    link framework "Accelerate"
    link framework "Foundation"
    link framework "Metal"
    export *
}
MODULEMAP

args=()
add_library() { args+=(-library "$1" -headers "$headers"); }
for platform in $platforms; do
  case "$platform" in
    ios) add_library "$work/ios-arm64/libgopherllm.a" ;;
    ios-simulator)
      mkdir -p "$work/sim"
      lipo -create "$work/sim-arm64/libgopherllm.a" "$work/sim-x86_64/libgopherllm.a" -output "$work/sim/libgopherllm.a"
      add_library "$work/sim/libgopherllm.a" ;;
    macos)
      mkdir -p "$work/macos"
      lipo -create "$work/macos-arm64/libgopherllm.a" "$work/macos-x86_64/libgopherllm.a" -output "$work/macos/libgopherllm.a"
      add_library "$work/macos/libgopherllm.a" ;;
  esac
done

rm -rf "$out"
mkdir -p "$(dirname "$out")"
xcodebuild -create-xcframework "${args[@]}" -output "$out" >/dev/null
echo "created $out"
plutil -p "$out/Info.plist" | grep -E 'LibraryIdentifier|SupportedPlatform"|SupportedPlatformVariant' || true
