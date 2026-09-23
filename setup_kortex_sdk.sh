#!/usr/bin/env bash
# Download + stage the KORTEX C++ SDK for THIS machine's architecture.
#
# The SDK is not redistributable in-repo and is arch-specific, so it is fetched
# per machine. Everything else (Dockerfiles, scripts) is versioned, so this is
# the only per-host step. Prints the staged path on stdout.
#
#   ./setup_kortex_sdk.sh            # -> $HOME/kortex_api_2.8.0_<arch>
#   KORTEX_DEST=/opt/... ./setup_kortex_sdk.sh
set -euo pipefail

BASE=https://artifactory.kinovaapps.com/artifactory/generic-public/kortex/API/2.8.0
case "$(uname -m)" in
  # 2.8.0 is the last line with TCPTransport; see README for why not 3.x.
  aarch64|arm64) ZIP=linux_aarch64_gcc_7.4.zip; SUFFIX=aarch64 ;;
  x86_64|amd64)  ZIP=linux_x86-64_gcc_5.4.zip;  SUFFIX=x86-64  ;;   # no 7.4 build exists for x86
  *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

DEST="${KORTEX_DEST:-$HOME/kortex_api_2.8.0_$SUFFIX}"
echo "arch $(uname -m) -> $ZIP" >&2
rm -rf "$DEST"; mkdir -p "$DEST"
tmp=$(mktemp -t kortex.XXXXXX).zip
curl -fsSL --max-time 300 -o "$tmp" "$BASE/$ZIP"
unzip -q "$tmp" -d "$DEST"; rm -f "$tmp"

# The core's CMake guards on exactly this header, so fail loudly here instead.
[ -f "$DEST/include/client/RouterClient.h" ] || { echo "missing include/client/RouterClient.h" >&2; exit 1; }
[ -f "$DEST/lib/release/libKortexApiCpp.a" ] || { echo "missing lib/release/libKortexApiCpp.a" >&2; exit 1; }
echo "staged $(du -sh "$DEST" | cut -f1) at $DEST" >&2
echo "$DEST"
