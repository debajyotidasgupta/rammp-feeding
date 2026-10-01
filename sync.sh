#!/usr/bin/env bash
# Push this tree to another machine. The Mac is the source of truth; remote
# copies are replicas (--delete), so edit here, never there.
#
#   ./sync.sh                # -> triphammer:~/rammp
#   ./sync.sh cascadilla     # any tailnet host
#
# Excluded, because they are per-machine and must NOT be replicated:
#   .venv/                     arm64 vs x86-64 python wheels
#   docker/vendor/kortex_api_* the staged KORTEX SDK (arch-specific; `make sdk`)
#   out/                       everything generated (renders, videos, caches, debug images): regenerable
#   config/local.yaml          that machine's own settings (and excluded = never deleted by --delete)
#
#   SYNC_USER=abra SYNC_DEST=cornell/rammp ./sync.sh 192.168.1.11     # the shared site Jetson (README)
set -euo pipefail
cd "$(dirname "$0")"

HOST="${1:-triphammer}"
USER="${SYNC_USER:-debajyoti}"
DEST="${SYNC_DEST:-rammp}"

echo "syncing -> $USER@$HOST:$DEST/"
rsync -az --delete \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'docker/vendor/kortex_api_*' \
  --exclude '/out/' \
  --exclude 'sim/assets/smplx/' \
  --exclude 'sim/assets/human/' \
  --exclude 'config/local.yaml' \
  ./ "$USER@$HOST:$DEST/"

echo "remote status:"
ssh "$USER@$HOST" "cd $DEST && make status" 2>&1 | sed 's/^/  /'
