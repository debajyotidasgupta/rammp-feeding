#!/usr/bin/env bash
# Everything that runs without the arm, the Jetson or a GPU (the Mac, ~1 min):
#   the unit tests (settings, camera mount, the scene camera's geometry, preflight, the computed overlook, the
#   planner's world), and the whole bite on real kinematics (tests/site_check.py) for every camera combination,
#   with the stand-in planner (tests/stub_planner.py). The real cuRobo: tests/site_check.py --planner curobo on a
#   CUDA box (README: the planner).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
fail=0
echo "== unit tests";  $PY -W ignore tests/unit_test.py 2>&1 | tail -3 || fail=1
for cams in "realsense_d405 orbbec_gemini336l" "realsense_d405 none" "kinova_wrist orbbec_gemini336l" "kinova_wrist none"; do
  set -- $cams
  echo "== site check: wrist $1, scene camera $2"
  $PY tests/site_check.py --camera "$1" --scene-camera "$2" 2>&1 | tail -1 | tee /dev/stderr | grep -q "GOES THROUGH" || fail=1
done
[ $fail = 0 ] && echo "OFFLINE OK" || { echo "OFFLINE FAILED"; exit 1; }
