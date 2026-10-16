#!/usr/bin/env python3
"""Where is the arm, right now.

    python -m kinova.scripts.arm_state [--watch]

Read-only. Reports the joint vector, the tool point, which way the utensil is
pointing, and how far the arm is from the feeding start pose. `--watch` repeats
until interrupted, which is the useful form while someone is teleoperating.

The gripper is REPORTED but never commanded -- the 2F-85 is holding the utensil.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from kinova.connection import kinova_connection
from kinova.scripts.safe_move import approach_axis, fold, joints, tool_pose

HOME_JSON = Path(__file__).resolve().parent.parent.parent / "calibration" / "home_pose.json"


def report(conn, home) -> None:
    fb = conn.base_cyclic.RefreshFeedback()
    q = [fold(a.position) for a in fb.actuators]
    tau = [a.torque for a in fb.actuators]
    b = fb.base
    ax = approach_axis(b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, -ax[2]))))

    print("           " + "".join("%9s" % ("J%d" % (i + 1)) for i in range(7)))
    print("  joints : " + "".join("%9.3f" % v for v in q))
    print("  torque : " + "".join("%9.3f" % v for v in tau))
    if home:
        err = [fold(t - c) for t, c in zip(home, q)]
        print("  vs home: " + "".join("%9.3f" % v for v in err)
              + "   max %.3f deg" % max(abs(e) for e in err))
    print("  tool   : (%.4f %.4f %.4f) m   rpy (%.2f %.2f %.2f) deg"
          % (b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
             b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z))
    print("  utensil: (%.3f %.3f %.3f), %.1f deg below horizontal"
          % (ax[0], ax[1], ax[2], tilt))
    try:
        g = fb.interconnect.gripper_feedback.motor[0].position
        print("  gripper: %.1f%% closed  (holding the utensil; never commanded here)" % g)
    except (AttributeError, IndexError):
        pass
    print("  faults : a=0x%08X b=0x%08X" % (b.fault_bank_a, b.fault_bank_b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=None)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--joints", action="store_true", help="print only the joints (deg) and the tool pose (m, deg): "
                                                           "two comma-separated lines")
    a = ap.parse_args()
    if a.joints:
        with kinova_connection(ip=a.ip) as conn:
            print(",".join("%.4f" % v for v in joints(conn)))
            print(",".join("%.5f" % v for v in tool_pose(conn)))
        return 0

    home = None
    if HOME_JSON.exists():
        home = [fold(v) for v in json.loads(HOME_JSON.read_text())["joints_deg"]]

    with kinova_connection(ip=a.ip) as conn:
        if not a.watch:
            report(conn, home)
            return 0
        try:
            while True:
                report(conn, home)
                print()
                time.sleep(1.0 / a.hz)
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
