#!/usr/bin/env bash
# The feeding loop, from feeding_start. Runs ON THE JETSON.
#
#   feeding/feed.sh            # detect -> [perceive -> acquire -> deliver] until the bowl is empty or the
#                              # user declines (no open mouth in 60 s) -> back to feeding_start
#   BITES=1 feeding/feed.sh    # one bite only (a test run)
#   FROM_DETECT=1 feeding/feed.sh   # already at food_detect_start (e.g. resuming)
#
# Every stage keeps its own checks; this only chains them without waiting between them. The whole session is
# recorded in one directory (feeding/lib.sh): `python -m feeding.review <it>` afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."
source feeding/lib.sh
session_begin feed
BITES="${BITES:-99}"
echo "record: $RAMMP_SESSION"

moves() {   # the stage-1 motions, recorded into the session's moves/
  STAGE=moves; STAGE_DIR="$RAMMP_SESSION/moves"; mkdir -p "$STAGE_DIR"
  ev move_start ", \"what\": \"$1\""
  $(arm_cmd sequence) "$1"; local rc=$?
  arm_pull; ev move_end ", \"what\": \"$1\", \"rc\": $rc"; STAGE=feed; STAGE_DIR=
  return $rc
}
home() { echo; echo "== back to feeding_start"; moves home; }
finish() {  # $1 = why
  ev session_end ", \"why\": \"$1\", \"bites\": $n"
  "$MPY" -m feeding.review "$RAMMP_SESSION" --brief 2>/dev/null || true
}
# every model loaded and warm BEFORE the first bite, and kept loaded throughout
if [ -n "$(S features.resident_models)" ]; then make -s models || { echo "models not up -- not starting"; exit 1; }; fi
n=0
# the arm as the calibration assumes it (tool, faults, kinematics, limits, servers): nothing moves if not
$(arm_cmd preflight) || { echo "preflight failed -- not starting (make preflight shows why)"; finish preflight; exit 1; }
# the bowl and the table from the fixed scene camera (when there is one): food_detect_start is placed over the bowl
if [ -n "$(S profiles.scene_camera)" ]; then
  PYTHONPATH=. "$PY" -m feeding.locate_scene --out "$RAMMP_SESSION/scene" \
    || { echo "the scene camera found no reachable bowl -- not starting (see $RAMMP_SESSION/scene)"; finish no_bowl; exit 1; }
fi
if [ -z "${FROM_DETECT:-}" ]; then moves detect || { echo "could not reach food_detect_start"; finish no_detect; exit 1; }; fi
while [ "$n" -lt "$BITES" ]; do
  bite_begin
  echo; echo "== bite $((n + 1))  $(date +%T)   ($RAMMP_BITE)"; t0=$(date +%s); ev bite_start ", \"bite\": $((n + 1))"
  feeding/stage1.sh; rc=$?                       # scripts, not make: make turns every exit code into 2
  [ $rc = 3 ] && { echo "the bowl is empty"; ev bite_end ", \"result\": \"bowl empty\""; home; finish "bowl empty"; exit 0; }
  [ $rc = 5 ] && { echo "food is LEFT in the bowl but no pickable piece was detected -- stopping (see $RAMMP_BITE/perceive)"
                   ev bite_end ", \"result\": \"food left, none pickable\""; home; finish "food left, none pickable"; exit 5; }
  [ $rc != 0 ] && { echo "perceive failed (exit $rc) -- stopping at food_detect_start"; ev bite_end ", \"result\": \"perceive failed\", \"rc\": $rc"; finish "perceive failed"; exit 1; }
  STOP_AFTER= START_AT=over feeding/acquire.sh || { echo "acquire stopped -- arm left where it stopped; check it"; ev bite_end ", \"result\": \"acquire stopped\""; finish "acquire stopped"; exit 1; }
  STOP_AFTER= START_AT=swing feeding/deliver.sh; rc=$?          # WAIT_OPEN: from config/ unless set
  [ $rc = 7 ] && { echo "no open mouth: the user declined -- stopping"; ev bite_end ", \"result\": \"declined\""; home; finish declined; exit 0; }
  [ $rc = 8 ] && { echo "the approach stopped short of the mouth (retracted) -- stopping"; ev bite_end ", \"result\": \"approach stopped\""; home; finish "approach stopped"; exit 1; }
  [ $rc != 0 ] && { echo "deliver stopped (exit $rc) -- arm left where it stopped; check it"; ev bite_end ", \"result\": \"deliver stopped\", \"rc\": $rc"; finish "deliver stopped"; exit 1; }
  n=$((n + 1)); echo "== bite $n done in $(( $(date +%s) - t0 )) s"
  ev bite_end ", \"result\": \"eaten\", \"seconds\": $(( $(date +%s) - t0 ))"
  ( "$MPY" -m feeding.review "$RAMMP_BITE" --write >/dev/null 2>&1 & )   # the bite's summary + contact sheet, off the critical path
done
home
finish "bites done"
