#!/usr/bin/env bash
# Food acquisition from food_detect_start, frozen 2026-09-23. Runs ON THE JETSON.
#
#   feeding/acquire.sh                      # over -> plunge (one smooth move to the floor) -> scoop
#   STOP_AFTER=over|plunge feeding/acquire.sh
#   STOP_AFTER=hover10 feeding/acquire.sh   # the stepped route, for aim checks: hover 30 -> hover 10 (-> punch)
#   START_AT=punch|plunge|scoop feeding/acquire.sh
#
# Uses the current bite's perceive/ for the target (the latest `make perceive`; RUN=<dir> for another), and
# records into the bite's acquire/ (feeding/lib.sh). Every depth reading is taken
# FROM WHERE THE ARM IS before the next move: at a 30 mm hover the food was
# 4.7-6.7 mm closer than the overlook reading predicted, every time today.
set -euo pipefail
cd "$(dirname "$0")/.."
source feeding/lib.sh
bite_attach; session_begin acquire; stage_begin acquire
# this bite's perceive run, and it must have a target: an empty-bowl run leaves none, and falling back to an
# older run would aim at food that is gone
RUN="${RUN:-$RAMMP_BITE/perceive}"
ACQ="$STAGE_DIR"
[ -f "$RUN/target.json" ] || { echo "the latest perceive run ($RUN) has no target -- $( [ -f "$RUN/no_food" ] && echo 'the bowl is empty' || echo 'run make perceive')"; exit 3; }
STOP_AFTER="${STOP_AFTER:-}"
START_AT="${START_AT:-over}"
ARM="$(arm_cmd acquire)"
G_OVER="$(S acquisition.over_depth_guess_m)"          # the floor measurement's starting depth, per position
G30="$(S acquisition.stepped.depth_guess_m.hover30)"; G10="$(S acquisition.stepped.depth_guess_m.hover10)"
H30="$(S acquisition.stepped.hover_m.hover30)"; H10="$(S acquisition.stepped.hover_m.hover10)"

measure() {   # $1 = label; depth from the vision container, analysed on the host (called right after a move)
  if ! PYTHONPATH=. "$PY" -m feeding.depth_grab "$ACQ/acq_$1.npy" --since "$(date +%s.%N)" >/dev/null 2>&1; then    # resident: ~0.5 s
    docker cp feeding/depth_probe.py "$VISION":/tmp/depth_probe.py >/dev/null
    docker exec "$VISION" bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
      cd /tmp && python3 depth_probe.py --out /tmp/acq_$1 --frames $FRAMES" >/dev/null 2>&1
    docker cp "$VISION":/tmp/acq_$1.npy "$ACQ/acq_$1.npy" >/dev/null 2>&1 || true
  fi
  if ! PYTHONPATH=. "$PY" -m feeding.measure_under --depth "$ACQ/acq_$1.npy" --out "$ACQ/under_$1.json" --guess "${2:-0.30}" 2>"$ACQ/measure_$1.err"; then
    # no depth under the tip: fill from Depth Anything 3 (needs a colour frame from here) and measure again
    echo "  $(tail -1 "$ACQ/measure_$1.err") -- trying the Depth Anything 3 fallback"
    ev depth_fallback ", \"at\": \"$1\""
    PYTHONPATH=. "$MPY" -m feeding.grab_frame "$ACQ/acq_$1_colour.jpg" >/dev/null 2>&1
    PYTHONPATH=.:$HOME/da3src/src "$PY" -m feeding.depth_fill "$ACQ" --colour "acq_$1_colour.jpg" --depth "acq_$1.npy" | sed -u 's/^/  /'
    PYTHONPATH=. "$PY" -m feeding.measure_under --depth "$ACQ/acq_$1.npy" --out "$ACQ/under_$1.json" --guess "${2:-0.30}"
  fi
  docker cp "$ACQ/under_$1.json" "$DRIVER":/tmp/under.json >/dev/null
}
STAGES="over hover30 hover10 punch plunge scoop"
case " $STAGES " in *" $START_AT "*) ;; *) echo "START_AT must be one of: $STAGES"; exit 2 ;; esac
started=0
run() { [ "$1" = "$START_AT" ] && started=1; [ "$started" = 1 ]; }   # $1 = stage: run it?
stop_here() { [ "$STOP_AFTER" = "$1" ] && { echo; echo "stopped after $1 (STOP_AFTER)"; exit 0; } || true; }

echo "target from: $RUN"
docker cp "$RUN/target.json" "$DRIVER":/tmp/target.json >/dev/null

# Default: OVER -> PLUNGE (one smooth move to the floor, aimed from the overlook reading) -> SCOOP.
# STOP_AFTER=hover30|hover10 (or START_AT at one of them) takes the old stepped route, for aim checks.
STEPPED=""; case "$STOP_AFTER $START_AT" in *hover30*|*hover10*|*punch*) STEPPED=1 ;; esac
PICKED=""
if [ -z "$STEPPED" ] && [ -z "$STOP_AFTER" ] && [ -n "$(S features.pick_in_one_program)" ] && run over; then
  # the default: OVER + re-measure + plunge + scoop in ONE arm program, no pauses between them
  echo "$(date +%T) [1-3/3] OVER + PLUNGE + SCOOP"
  set +e; $ARM pick 2>&1 | grep --line-buffered -v Deprecation; rc=${PIPESTATUS[0]}; set -e
  docker cp "$DRIVER":/tmp/under_over.json "$ACQ/under_over.json" >/dev/null 2>&1 || true
  if [ "$rc" = 6 ]; then           # no sensor depth under the tip: measure with the DA3 fill, then plunge
    echo "$(date +%T) [2/3] PLUNGE + SCOOP (depth fallback)"; measure over "$G_OVER"; $ARM plunge --then-scoop 2>&1 | grep --line-buffered -v Deprecation
  elif [ "$rc" != 0 ]; then exit "$rc"; fi
  PICKED=1; SCOOPED=1
fi
if [ -z "$PICKED" ] && run over; then echo "$(date +%T) [1/3] OVER"; $ARM over 2>&1 | grep --line-buffered -v Deprecation; stop_here over; fi
if [ -n "$PICKED" ]; then :
elif [ -n "$STEPPED" ]; then
  if run hover30; then echo "  down to 30 mm";   measure over "$G_OVER";  $ARM descend --to-above "$H30" 2>&1 | grep --line-buffered -v Deprecation; stop_here hover30; fi
  if run hover10; then echo "  down to 10 mm";   measure hover30 "$G30"; $ARM descend --to-above "$H10" 2>&1 | grep --line-buffered -v Deprecation; stop_here hover10; fi
  if run punch;   then echo "  PUNCH";           measure hover10 "$G10"; $ARM punch 2>&1 | grep --line-buffered -v Deprecation; stop_here punch; fi
elif run plunge; then
  if [ "$STOP_AFTER" = plunge ]; then
    echo "$(date +%T) [2/3] PLUNGE";  measure over "$G_OVER"; $ARM plunge 2>&1 | grep --line-buffered -v Deprecation; stop_here plunge
  else                    # plunge and scoop as one planned sequence: no pause at the bottom
    echo "$(date +%T) [2/3] PLUNGE + SCOOP"; measure over "$G_OVER"; $ARM plunge --then-scoop 2>&1 | grep --line-buffered -v Deprecation
    SCOOPED=1
  fi
fi
if [ -z "${SCOOPED:-}" ] && run scoop; then echo "$(date +%T) [3/3] SCOOP"; $ARM scoop 2>&1 | grep --line-buffered -v Deprecation; fi
docker cp "$DRIVER":/tmp/scoop_log.json "$ACQ/scoop_log.json" >/dev/null 2>&1 || true
echo; echo "logs: $ACQ"
