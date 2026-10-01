#!/usr/bin/env bash
# Everything that runs without the arm, the Jetson or the models (the Mac, ~20 s):
#   the unit tests (settings, camera mount, the scene camera's geometry, preflight, the computed overlook),
#   the golden arm trace (the stages against the fake controller: must be IDENTICAL),
#   the site check (the whole bite on real kinematics for tests/sites/site_20260930.yaml).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
fail=0
echo "== unit tests";  $PY -W ignore tests/unit_test.py 2>&1 | tail -3 || fail=1
echo "== golden trace"; $PY tests/arm_trace.py "$(mktemp -d)/t.json.gz" --compare tests/golden/arm_trace.json.gz 2>&1 | tail -1 | tee /dev/stderr | grep -q IDENTICAL || fail=1
echo "== site check";  $PY tests/site_check.py 2>&1 | tail -1 | tee /dev/stderr | grep -q "GOES THROUGH" || fail=1
[ $fail = 0 ] && echo "OFFLINE OK" || { echo "OFFLINE FAILED"; exit 1; }
