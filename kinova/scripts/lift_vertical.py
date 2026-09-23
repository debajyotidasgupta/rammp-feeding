#!/usr/bin/env python3
"""Raise the tool straight up, holding x, y and orientation.

    python -m kinova.scripts.lift_vertical --dz 0.45          # go up 450 mm
    python -m kinova.scripts.lift_vertical --to-limit         # go up until it stalls

Stage 1 of finding the goal. The wrist camera has to look down on the whole
workspace before the arm swings from the start pose to face it, and it has to
get there without sweeping anything -- so the tool goes UP first, on a pure
vertical, and only then does the base rotate.

"Vertical" here is a Cartesian constraint, not a joint move: x, y and the tool
orientation are all held at their current values and only z changes, so every
joint participates. The arm starts folded (J4 near -139), so this unfolds it.

Climbs in steps and watches the achieved z after each one. Near the top of the
reach the arm goes singular and simply stops tracking; that shows up as a step
that does not move, so progress -- not a nominal target -- decides when to stop.

NEVER touches the gripper.
"""

from __future__ import annotations

import argparse
import math
import time

from kortex_api.autogen.messages import Base_pb2

from kinova.connection import kinova_connection


def fold(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def pose_of(conn):
    b = conn.base_cyclic.RefreshFeedback().base
    return [b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
            b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=None)
    ap.add_argument("--dz", type=float, default=0.45, help="metres to climb")
    ap.add_argument("--to-limit", action="store_true",
                    help="keep climbing until the arm stops making progress")
    ap.add_argument("--step", type=float, default=0.05, help="metres per step")
    ap.add_argument("--speed", type=float, default=0.04, help="m/s")
    ap.add_argument("--stall", type=float, default=0.5,
                    help="fraction of a step that counts as progress")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with kinova_connection(ip=a.ip) as conn:
        p0 = pose_of(conn)
        q0 = [fold(x.position) for x in conn.base_cyclic.RefreshFeedback().actuators]
        print("  start tool : (%.4f %.4f %.4f) m" % tuple(p0[:3]))
        print("  start rpy  : (%.2f %.2f %.2f) deg" % tuple(p0[3:]))
        print("  start joints: " + "".join("%8.2f" % v for v in q0))
        target = "to the reach limit" if a.to_limit else "%.0f mm" % (1000 * a.dz)
        print("  climbing %s in %.0f mm steps at %.0f mm/s, holding x, y, rpy"
              % (target, 1000 * a.step, 1000 * a.speed))
        if a.dry_run:
            print("\ndry run, nothing commanded")
            return 0

        z = p0[2]
        n_steps = 200 if a.to_limit else max(1, math.ceil(a.dz / a.step))
        print()
        for i in range(n_steps):
            want = z + a.step if a.to_limit else min(p0[2] + a.dz, z + a.step)
            cp = Base_pb2.ConstrainedPose()
            t = cp.target_pose
            t.x, t.y, t.z = p0[0], p0[1], want
            t.theta_x, t.theta_y, t.theta_z = p0[3], p0[4], p0[5]
            cp.constraint.speed.translation = a.speed
            cp.constraint.speed.orientation = 15.0
            conn.base.PlayCartesianTrajectory(cp)
            time.sleep(a.step / a.speed + 1.2)

            p = pose_of(conn)
            gained = p[2] - z
            drift = math.hypot(p[0] - p0[0], p[1] - p0[1])
            print("    step %2d  z %.4f -> %.4f  (+%5.1f mm)  lateral drift %5.1f mm"
                  % (i + 1, z, p[2], 1000 * gained, 1000 * drift))
            z = p[2]
            if gained < a.stall * a.step:
                print("    stalled -- at the reach limit for this x, y and orientation")
                break
            if not a.to_limit and z >= p0[2] + a.dz - 0.002:
                break

        p = pose_of(conn)
        q = [fold(x.position) for x in conn.base_cyclic.RefreshFeedback().actuators]
        fb = conn.base_cyclic.RefreshFeedback()
        print()
        print("  end tool   : (%.4f %.4f %.4f) m   climbed %.1f mm"
              % (p[0], p[1], p[2], 1000 * (p[2] - p0[2])))
        print("  end rpy    : (%.2f %.2f %.2f) deg" % tuple(p[3:]))
        print("  end joints : " + "".join("%8.2f" % v for v in q))
        print("  lateral    : %.1f mm from where it started"
              % (1000 * math.hypot(p[0] - p0[0], p[1] - p0[1])))
        print("  faults     : a=0x%08X b=0x%08X"
              % (fb.base.fault_bank_a, fb.base.fault_bank_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
