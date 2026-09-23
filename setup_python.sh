#!/usr/bin/env bash
# Create the local venv and install the Kortex Python API.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
"$PYTHON" -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo
echo "Done. Verify against the arm with:"
echo "  ./.venv/bin/python -m kinova.scripts.check_connection"
