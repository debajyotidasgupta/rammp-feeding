#!/usr/bin/env python3
"""The acquisition steps, one at a time. Each is run and verified on its own.

    python -m kinova.scripts.acquire over     # uses /tmp/target.json (make over copies it in)

  over   translate at the overlook height until the utensil tip's vertical line
         passes through the target food (locate_target's tool_goal). Refuses
         unless the arm is at food_detect_start and the move is small and level.
         Verified 2026-09-23: predicted berry position (668,286), observed
         (670,282) -- 5 px, ~1.5 mm.

  punch, scoop: planned and IK-checked by feeding/acquire_plan.py, not yet run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from kinova.connection import kinova_connection
from kinova.scripts.safe_move import MoveError, fold, goto_cartesian, joints, tilt_deg, tool_pose

CAL = Path(__file__).resolve().parent.parent.parent / "calibration"
MAX_OVER_M = 0.08        # OVER is a small correction, never a transit


def over(conn, target_file: str) -> None:
    det = json.loads((CAL / "food_detect_pose.json").read_text())
    off = max(abs(fold(a - b)) for a, b in zip(joints(conn), det["joints_deg"]))
    if off > 1.0:
        raise MoveError("not at food_detect_start (%.1f deg away) -- run: make detect" % off)
    goal = json.loads(Path(target_file).read_text())["tool_goal"]
    p = tool_pose(conn)
    d = math.hypot(goal[0] - p[0], goal[1] - p[1])
    if abs(goal[2] - p[2]) > 0.002 or d > MAX_OVER_M:
        raise MoveError("target is not a small level move (%.0f mm across, %.0f mm up) -- "
                        "re-run make perceive" % (1000 * d, 1000 * (goal[2] - p[2])))
    goto_cartesian(conn, goal, tuple(p[3:]), speed=0.02, label="OVER")
    p = tool_pose(conn)
    print("  tool (%.4f %.4f %.4f)  utensil %.2f deg off vertical" % (p[0], p[1], p[2], tilt_deg(conn)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["over"])
    ap.add_argument("--target", default="/tmp/target.json")
    a = ap.parse_args()
    with kinova_connection() as conn:
        try:
            over(conn, a.target)
        except MoveError as e:
            print("\n  STOPPED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
