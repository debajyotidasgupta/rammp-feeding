#!/usr/bin/env bash
# The run record end to end, without the arm: feeding/lib.sh's session / bite / stage / events / console log and
# the arm-record pull, filled from saved runs and the fake-arm harness, then read back by feeding/review.py.
#   tests/record_smoke.sh [PERCEIVE_RUN] [DELIVER_RUN]      # ON THE JETSON, from the repo root
# Writes only into a scratch runs directory (removed at the end unless KEEP=1).
set -uo pipefail
cd "$(dirname "$0")/.."
P="${1:-$HOME/rammp_runs/20260924_202544}"; D="${2:-$HOME/rammp_runs/deliver_20260924_194800}"
SCRATCH="$(mktemp -d /tmp/rammp_smoke.XXXX)"
export RAMMP_SET="deployment.runs_dir=$SCRATCH"
unset RAMMP_SESSION RAMMP_BITE RAMMP_TEE RAMMP_RECORD
(
  source feeding/lib.sh
  session_begin smoke
  bite_begin
  ( stage_begin perceive; echo "perceive: copying $P"; cp "$P"/{colour.jpg,depth.npy,detection.json,detection.jpg,target.json,target.jpg} "$STAGE_DIR/" 2>/dev/null
    PYTHONPATH=. "$PY" -c "import numpy as np; from feeding.viz import depth_png; depth_png(np.load('$STAGE_DIR/depth.npy'), '$STAGE_DIR/depth.png')"; exit 0 )
  ( stage_begin acquire; echo "acquire: the fake arm, recording in the driver container"
    docker exec -e PYTHONDONTWRITEBYTECODE=1 -e RAMMP_RECORD=$ARM_REC -w /work "$DRIVER" python3 tests/arm_trace.py /tmp/smoke_trace.json 2>&1 | tail -1
    exit 0 )           # stage_begin's trap pulls the arm record into acquire/arm/
  ( stage_begin deliver; mkdir -p "$STAGE_DIR/look_1"; cp "$D"/{colour.jpg,depth.npy,face.json,face_upright.jpg} "$STAGE_DIR/look_1/" 2>/dev/null
    echo "deliver: a saved look"; exit 0 )
  ev session_end ', "why": "smoke", "bites": 1'
)
S="$(ls -d "$SCRATCH"/*_smoke | head -1)"
echo; echo "== the record"; (cd "$S" && find . -maxdepth 3 | sort | head -40; echo "   ... $(find . -type f | wc -l) files")
echo; echo "== console.log (head)"; head -4 "$S/console.log"
echo; echo "== review"; PYTHONPATH=. "$(python3 -m settings get deployment.venvs.face)/bin/python" -m feeding.review "$S" --write
ls -la "$S/bite_01/contact_sheet.jpg" "$S/bite_01/summary.json"
[ -n "${KEEP:-}" ] && echo "kept: $S" || rm -rf "$SCRATCH"
