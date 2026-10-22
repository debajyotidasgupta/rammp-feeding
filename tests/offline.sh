#!/usr/bin/env bash
# Everything that runs without the arm or ROS (any machine with numpy, scipy, Pillow, PyYAML): the unit tests and the
# whole bite against tests/fake_robot.py, for several piece and user positions. The ROS wiring: tests/test_ros.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
fail=0
echo "== unit tests"; $PY -W ignore tests/unit_test.py 2>&1 | tail -1 | grep -q OK || { $PY -W ignore tests/unit_test.py; fail=1; }
for a in "" "--piece 0.03,0" "--piece -0.03,0.02" "--piece 0,-0.035" "--user-offset 0.1,0,0" "--user-offset -0.25,0.05,0.05" \
         "--user-offset 0,-0.12,-0.05" "--user-offset -0.4,0,0 --expect 8"; do
  printf "== bite %-38s " "$a"
  $PY tests/test_bite.py $a 2>&1 | tail -1 | tee /dev/stderr | grep -qE "GOES THROUGH|AS EXPECTED" || fail=1
done 2>&1
[ $fail = 0 ] && echo "OFFLINE OK" || { echo "OFFLINE FAILED"; exit 1; }
