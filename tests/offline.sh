#!/usr/bin/env bash
# Everything that runs without the arm or ROS (any machine with numpy, scipy, Pillow, PyYAML): the unit tests, the fork
# finder on real frames, and the whole bite against tests/fake_robot.py for several piece and user positions, on both
# test sites (tests/fixture.py: level, and tilt30). The ROS wiring: tests/test_ros.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
fail=0
echo "== unit tests"; $PY -W ignore tests/unit_test.py 2>&1 | tail -1 | grep -q OK || { $PY -W ignore tests/unit_test.py; fail=1; }
echo "== the fork finder on real end-on frames"; $PY -W ignore tests/unit_fork_finder.py 2>&1 | tail -1 | grep -q OK || { $PY -W ignore tests/unit_fork_finder.py; fail=1; }
for a in "" "--piece 0.03,0" "--piece -0.03,0.02" "--piece 0,-0.035" "--user-offset 0.08,0,0" "--user-offset -0.06,0.04,0.04" \
         "--user-offset 0,-0.07,-0.05" "--user-offset -0.25,0.05,0.05" "--user-offset -0.6,0,0" "--user-offset -1.0,0,0 --expect 8"; do
  printf "== bite %-38s " "$a"
  $PY tests/test_bite.py $a 2>&1 | tail -1 | tee /dev/stderr | grep -qE "GOES THROUGH|AS EXPECTED" || fail=1
done 2>&1
for a in "" "--piece 0.012,0.01" "--user-offset 0,-0.04,0.03"; do   # the arm leaning 30 deg, the user to its right
  printf "== tilt30 bite %-31s " "$a"
  RAMMP_TEST_SITE=tilt30 $PY tests/test_bite.py $a 2>&1 | tail -1 | tee /dev/stderr | grep -qE "GOES THROUGH|AS EXPECTED" || fail=1
done 2>&1
[ $fail = 0 ] && echo "OFFLINE OK" || { echo "OFFLINE FAILED"; exit 1; }
