#!/usr/bin/env python3
"""Move the arm to one of the named, measured poses.

    python -m kinova.scripts.feeding_start                    # feeding_start
    python -m kinova.scripts.feeding_start --pose food_detect # over the bowl

Poses live in calibration/<name>_pose.json and are measured on this
installation with kinova.scripts.capture_home:

    home         feeding_start    -- parked, arm swung clear, fork at the floor
    food_detect  food_detect_start -- above the bowl, camera straight down and
                                      rolled to a multiple of 90 deg

Every sequence begins at feeding_start, and on deployment the arm is always
near whichever pose it is going to, so this is a direct joint move -- no
staging, no path search.

The one subtlety kept: Kortex reports joint angles in [0, 360) and joints
1/3/5/7 are continuous, so a target near the wrap must be compared on
(-180, 180]. Without that, "go to -174" from +170 reads as a 344 degree journey
instead of a 16 degree one.

NEVER touches the gripper -- the 2F-85 is holding the fork. No gripper RPC is
imported in this file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kortex_api.autogen.messages import Base_pb2

from kinova.connection import kinova_connection


def fold(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def main() -> int:
    ap = argparse.ArgumentParser(prog="goto pose")
    ap.add_argument("--ip", default=None)
    ap.add_argument("--pose", default="home",
                    help="name: resolves calibration/<name>_pose.json")
    ap.add_argument("--home", default=None, help="explicit path, overrides --pose")
    ap.add_argument("--speed", type=float, default=10.0, help="deg/s per joint")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    path = Path(a.home) if a.home else Path("calibration") / ("%s_pose.json" % a.pose)
    if not path.exists():
        print("no such pose: %s" % path)
        return 2
    rec = json.loads(path.read_text())
    target = [fold(v) for v in rec["joints_deg"]]

    with kinova_connection(ip=a.ip) as conn:
        fb = conn.base_cyclic.RefreshFeedback()
        q0 = [fold(act.position) for act in fb.actuators]
        delta = [fold(t - c) for t, c in zip(target, q0)]
        worst = max(abs(d) for d in delta)

        print("-> %s   (captured %s)" % (rec["label"], rec["captured"]))
        print("           " + "".join("%9s" % ("J%d" % (i + 1)) for i in range(7)))
        print("  current: " + "".join("%9.3f" % v for v in q0))
        print("  target : " + "".join("%9.3f" % v for v in target))
        print("  delta  : " + "".join("%9.3f" % v for v in delta))
        print("  move   : %.1f deg at %.0f deg/s -> ~%.1f s" % (worst, a.speed, worst / a.speed))

        if a.dry_run:
            print("\ndry run, nothing commanded")
            return 0
        if worst < 0.05:
            print("\nalready at %s" % rec["label"])
            return 0

        cja = Base_pb2.ConstrainedJointAngles()
        for i, v in enumerate(target):
            ja = cja.joint_angles.joint_angles.add()
            ja.joint_identifier = i
            ja.value = v
        cja.constraint.type = Base_pb2.JOINT_CONSTRAINT_SPEED
        cja.constraint.value = a.speed

        print("\n  moving...", flush=True)
        conn.base.PlayJointTrajectory(cja)

        deadline = time.time() + max(20.0, 4.0 * worst / a.speed)
        while time.time() < deadline:
            time.sleep(0.1)
            q = [fold(x.position) for x in conn.base_cyclic.RefreshFeedback().actuators]
            if all(abs(fold(t - c)) < 0.05 for t, c in zip(target, q)):
                break

        fb = conn.base_cyclic.RefreshFeedback()
        q1 = [fold(x.position) for x in fb.actuators]
        err = [fold(t - c) for t, c in zip(target, q1)]
        print("  reached: " + "".join("%9.3f" % v for v in q1))
        print("  error  : " + "".join("%9.3f" % v for v in err))
        print("  faults : a=0x%08X b=0x%08X"
              % (fb.base.fault_bank_a, fb.base.fault_bank_b))

        if max(abs(e) for e in err) > 0.2:
            print("\nFAILED: did not reach %s" % rec["label"])
            return 1
        print("\nat %s" % rec["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
