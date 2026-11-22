#!/usr/bin/env bash
set -euo pipefail
app_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(CDPATH= cd -- "$app_dir/../.." && pwd)
package="$app_dir/LinkedGopherLLM"
out="${HESTIA_BUILD_DIR:-$app_dir/build}"
action="${1:-build}"
shift || true

case "$action" in
  prepare|build|archive) ;;
  *) echo "usage: $0 [prepare|build|archive] [xcodebuild options...]" >&2; exit 2 ;;
esac
if [[ "$action" == archive && -z "${DEVELOPMENT_TEAM:-}" ]]; then
  echo "Set DEVELOPMENT_TEAM to the Apple Developer team for the App Store archive." >&2
  exit 2
fi

mkdir -p "$package" "$out"
WITH_HESTIA=1 PLATFORMS=macos MACOS_ARCHS=arm64 XCFRAMEWORK_OUT="$package/GopherLLMCore.xcframework" \
  bash "$root/scripts/build-xcframework.sh"
cp "$root/bindings/swift/Package.swift" "$package/"
rm -rf "$package/Sources" "$package/Tests" "$package/Notices"
cp -R "$root/bindings/swift/Sources" "$root/bindings/swift/Tests" "$package/"
mkdir -p "$package/Notices"
cp "$root/LICENSE" "$package/Notices/GopherLLM.txt"
# Preserve the accompanying notices for the Go modules linked into Hestia.
(cd "$root/cmd/hestia" && go list -m -f '{{.Path}} {{.Dir}}' all) > "$out/modules.txt"
while read -r module dir; do
  [[ -n "$dir" && -d "$dir" ]] || continue
  slug="${module//\//_}"
  for notice in "$dir"/LICENSE* "$dir"/NOTICE* "$dir"/COPYING*; do
    [[ -f "$notice" ]] || continue
    cp "$notice" "$package/Notices/$slug-${notice##*/}"
  done
done < "$out/modules.txt"

if [[ "$action" == prepare ]]; then exit 0; fi
settings=(-project "$app_dir/HestiaMac.xcodeproj" -scheme Hestia -derivedDataPath "$out/DerivedData")
if [[ "$action" == archive ]]; then
  xcodebuild "${settings[@]}" -configuration Release -archivePath "$out/Hestia.xcarchive" \
    DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" -allowProvisioningUpdates "$@" archive
else
  xcodebuild "${settings[@]}" -configuration Debug "$@" build
fi
