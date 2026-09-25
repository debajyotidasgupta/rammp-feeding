#!/usr/bin/env bash
# Food acquisition from food_detect_start, frozen 2026-09-23. Runs ON THE JETSON.
#
#   feeding/acquire.sh                      # over -> plunge (one smooth move to the floor) -> scoop
#   STOP_AFTER=over|plunge feeding/acquire.sh
#   STOP_AFTER=hover10 feeding/acquire.sh   # the stepped route, for aim checks: hover 30 -> hover 10 (-> punch)
#   START_AT=punch|plunge|scoop feeding/acquire.sh
#
# Uses the latest `make perceive` run for the target. Every depth reading is taken
# FROM WHERE THE ARM IS before the next move: at a 30 mm hover the food was
# 4.7-6.7 mm closer than the overlook reading predicted, every time today.
set -euo pipefail
cd "$(dirname "$0")/.."
# the NEWEST perceive run (named by date; face/deliver runs are not), and it must have a target: an empty-bowl
# run leaves none, and falling back to an older run would aim at food that is gone
RUN="${RUN:-$(ls -d "$HOME"/rammp_runs/2*/ | tail -1)}"
[ -f "$RUN/target.json" ] || { echo "the latest perceive run ($RUN) has no target -- $( [ -f "$RUN/no_food" ] && echo 'the bowl is empty' || echo 'run make perceive')"; exit 3; }
PY="$HOME/venvs/la/bin/python"
STOP_AFTER="${STOP_AFTER:-}"
START_AT="${START_AT:-over}"
ARM="docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work -w /work tcp_rec python3 -u -m kinova.scripts.acquire"

measure() {   # $1 = label; depth from the vision container, analysed on the host
  docker cp feeding/depth_probe.py kinova_vision:/tmp/depth_probe.py >/dev/null
  docker exec kinova_vision bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
    cd /tmp && python3 depth_probe.py --out /tmp/acq_$1 --frames 30" >/dev/null 2>&1
  docker cp kinova_vision:/tmp/acq_$1.npy "$RUN/acq_$1.npy" >/dev/null
  PYTHONPATH=. "$PY" -m feeding.measure_under --depth "$RUN/acq_$1.npy" --out "$RUN/under_$1.json" --guess "${2:-0.30}"
  docker cp "$RUN/under_$1.json" tcp_rec:/tmp/under.json >/dev/null
}
STAGES="over hover30 hover10 punch plunge scoop"
case " $STAGES " in *" $START_AT "*) ;; *) echo "START_AT must be one of: $STAGES"; exit 2 ;; esac
started=0
run() { [ "$1" = "$START_AT" ] && started=1; [ "$started" = 1 ]; }   # $1 = stage: run it?
stop_here() { [ "$STOP_AFTER" = "$1" ] && { echo; echo "stopped after $1 (STOP_AFTER)"; exit 0; } || true; }

echo "target from: $RUN"
docker cp "$RUN/target.json" tcp_rec:/tmp/target.json >/dev/null

# Default: OVER -> PLUNGE (one smooth move to the floor, aimed from the overlook reading) -> SCOOP.
# STOP_AFTER=hover30|hover10 (or START_AT at one of them) takes the old stepped route, for aim checks.
STEPPED=""; case "$STOP_AFTER $START_AT" in *hover30*|*hover10*|*punch*) STEPPED=1 ;; esac
if run over;    then echo "[1/3] OVER";          $ARM over 2>&1 | grep -v Deprecation;   stop_here over; fi
if [ -n "$STEPPED" ]; then
  if run hover30; then echo "  down to 30 mm";   measure over 0.41;  $ARM descend --to-above 0.030 2>&1 | grep -v Deprecation; stop_here hover30; fi
  if run hover10; then echo "  down to 10 mm";   measure hover30 0.29; $ARM descend --to-above 0.010 2>&1 | grep -v Deprecation; stop_here hover10; fi
  if run punch;   then echo "  PUNCH";           measure hover10 0.275; $ARM punch 2>&1 | grep -v Deprecation; stop_here punch; fi
elif run plunge; then
  if [ "$STOP_AFTER" = plunge ]; then
    echo "[2/3] PLUNGE";  measure over 0.41; $ARM plunge 2>&1 | grep -v Deprecation; stop_here plunge
  else                    # plunge and scoop as one planned sequence: no pause at the bottom
    echo "[2/3] PLUNGE + SCOOP"; measure over 0.41; $ARM plunge --then-scoop 2>&1 | grep -v Deprecation
    SCOOPED=1
  fi
fi
if [ -z "${SCOOPED:-}" ] && run scoop; then echo "[3/3] SCOOP"; $ARM scoop 2>&1 | grep -v Deprecation; fi
docker cp tcp_rec:/tmp/scoop_log.json "$RUN/scoop_log.json" >/dev/null 2>&1 || true
echo; echo "logs: $RUN"
