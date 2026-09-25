#!/usr/bin/env bash
# The feeding loop, from feeding_start. Runs ON THE JETSON. Built 2026-09-24.
#
#   feeding/feed.sh            # detect -> [perceive -> acquire -> deliver] until the bowl is empty or the
#                              # user declines (no open mouth in 60 s) -> back to feeding_start
#   BITES=1 feeding/feed.sh    # one bite only (a test run)
#   FROM_DETECT=1 feeding/feed.sh   # already at food_detect_start (e.g. resuming)
#
# Every stage keeps its own checks; this only chains them without waiting between them.
set -uo pipefail
cd "$(dirname "$0")/.."
BITES="${BITES:-99}"
home() { echo; echo "== back to feeding_start"; make -s start; }
# the detection model stays loaded between bites (loading it costs 13.6 s a time)
make -s detector
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8765/ping | grep -q ok && break; sleep 1; done
if [ -z "${FROM_DETECT:-}" ]; then make -s detect || { echo "could not reach food_detect_start"; exit 1; }; fi
n=0
while [ "$n" -lt "$BITES" ]; do
  echo; echo "== bite $((n + 1))  $(date +%T)"; t0=$(date +%s)
  feeding/stage1.sh; rc=$?                       # scripts, not make: make turns every exit code into 2
  [ $rc = 3 ] && { echo "the bowl is empty"; home; exit 0; }
  [ $rc != 0 ] && { echo "perceive failed (exit $rc) -- stopping at food_detect_start"; exit 1; }
  STOP_AFTER= START_AT=over feeding/acquire.sh || { echo "acquire stopped -- arm left where it stopped; check it"; exit 1; }
  STOP_AFTER= START_AT=swing WAIT_OPEN=1 feeding/deliver.sh; rc=$?
  [ $rc = 7 ] && { echo "no open mouth: the user declined -- stopping"; home; exit 0; }
  [ $rc != 0 ] && { echo "deliver stopped (exit $rc) -- arm left where it stopped; check it"; exit 1; }
  n=$((n + 1)); echo "== bite $n done in $(( $(date +%s) - t0 )) s"
done
home
