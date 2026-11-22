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
#   MACOS_ARCHS              "arm64 x86_64" by default; "arm64" for Apple Silicon only
#   METAL                    1 (default) compiles the Metal GPU kernels in; 0 leaves them out
#   WITH_HESTIA              1 embeds Hestia in a macOS-only build
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
macos_archs="${MACOS_ARCHS:-arm64 x86_64}"
tags="capi"
[[ "${METAL:-1}" == "1" ]] && tags="capi metal"
work=$(mktemp -d "${TMPDIR:-/tmp}/gopherllm-xcframework.XXXXXX")
hestia_stage=""
trap 'rm -rf "$work"; if [[ -n "$hestia_stage" ]]; then rm -rf "$hestia_stage"; fi' EXIT
go_dir="$root"
go_target=./bindings/c/shim
if [[ "${WITH_HESTIA:-0}" == 1 ]]; then
  [[ "$platforms" == macos ]] || { echo "WITH_HESTIA=1 requires PLATFORMS=macos" >&2; exit 1; }
  go_dir="$root/cmd/hestia"
  hestia_stage=$(mktemp -d "$go_dir/.native-build.XXXXXX")
  mkdir -p "$hestia_stage/shim" "$hestia_stage/include"
  cp "$root"/bindings/c/shim/*.go "$go_dir/macos/bridge.go" "$hestia_stage/shim/"
  cp "$root/bindings/c/include/gopherllm.h" "$go_dir/macos/hestia.h" "$hestia_stage/include/"
  go_target="./${hestia_stage##*/}/shim"
  tags="$tags hestiacapi"
fi

# build_slice <dir> <GOOS> <GOARCH> <sdk> <clang target triple>
build_slice() {
  local dir="$work/$1" goos=$2 goarch=$3 sdk=$4 target=$5 sysroot cflags
  sysroot=$(xcrun --sdk "$sdk" --show-sdk-path)
  cflags="-O2 -target $target -isysroot \"$sysroot\""
  mkdir -p "$dir"
  echo "building $1 ($target)"
  (cd "$go_dir" && CGO_ENABLED=1 GOOS="$goos" GOARCH="$goarch" \
    CC="$(xcrun --sdk "$sdk" --find clang)" \
    CGO_CFLAGS="$cflags" \
    CGO_LDFLAGS="-target $target -isysroot \"$sysroot\"" \
    go build -trimpath -tags "$tags" -buildmode=c-archive -o "$dir/libgopherllm.a" "$go_target")
}

# The slices are independent and Go's build cache is safe for concurrent
# use, so build them in parallel and fail if any one fails.
slices=()
macos_libraries=()
for platform in $platforms; do
  case "$platform" in
    ios) slices+=("ios-arm64 ios arm64 iphoneos arm64-apple-ios$ios_min") ;;
    ios-simulator)
      slices+=("sim-arm64 ios arm64 iphonesimulator arm64-apple-ios$ios_min-simulator")
      slices+=("sim-x86_64 ios amd64 iphonesimulator x86_64-apple-ios$ios_min-simulator") ;;
    macos)
      for arch in $macos_archs; do
        case "$arch" in
          arm64) goarch=arm64 ;;
          x86_64) goarch=amd64 ;;
          *) echo "unknown macOS architecture $arch (want arm64 or x86_64)" >&2; exit 1 ;;
        esac
        slices+=("macos-$arch darwin $goarch macosx $arch-apple-macos$macos_min")
        macos_libraries+=("$work/macos-$arch/libgopherllm.a")
      done
      [[ ${#macos_libraries[@]} != 0 ]] || { echo "no macOS architectures selected" >&2; exit 1; } ;;
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
{
cat <<'MODULEMAP'
module GopherLLMCore {
    header "gopherllm.h"
MODULEMAP
if [[ -n "$hestia_stage" ]]; then
  cp "$go_dir/macos/hestia.h" "$headers/GopherLLMCore/"
  printf '    header "hestia.h"\n'
fi
cat <<'MODULEMAP'
    link framework "Accelerate"
    link framework "Foundation"
    link framework "Metal"
    link framework "Security"
    export *
}
MODULEMAP
} > "$headers/GopherLLMCore/module.modulemap"

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
      if [[ ${#macos_libraries[@]} == 1 ]]; then
        add_library "${macos_libraries[0]}"
      else
        mkdir -p "$work/macos"
        lipo -create "${macos_libraries[@]}" -output "$work/macos/libgopherllm.a"
        add_library "$work/macos/libgopherllm.a"
      fi ;;
  esac
done

rm -rf "$out"
mkdir -p "$(dirname "$out")"
xcodebuild -create-xcframework "${args[@]}" -output "$out" >/dev/null
echo "created $out"
plutil -p "$out/Info.plist" | grep -E 'LibraryIdentifier|SupportedPlatform"|SupportedPlatformVariant' || true
