#!/usr/bin/env python3
"""Trace a small cross in the x-y plane with the tool, at fixed z and orientation.

Intended as a visible, low-risk motion for checking the camera feeds: it only
translates laterally, so clearance below the tool never changes.

    python -m kinova.scripts.move_xy [--step 0.05] [--speed 0.05]
"""
from __future__ import annotations

import argparse
import time

from kortex_api.autogen.messages import Base_pb2

from kinova.connection import DEFAULT_IP, ConnectionConfig, KinovaConnection


def wait_settled(conn, label, timeout=25.0):
    last, still, t0 = None, 0, time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.15)
        fb = conn.base_cyclic.RefreshFeedback()
        if fb.base.fault_bank_a | fb.base.fault_bank_b:
            conn.base.Stop()
            raise RuntimeError(
                "FAULT during %s -> STOPPED (a=0x%08X b=0x%08X)"
                % (label, fb.base.fault_bank_a, fb.base.fault_bank_b))
        b = fb.base
        cur = (b.tool_pose_x, b.tool_pose_y, b.tool_pose_z)
        if last and max(abs(c - l) for c, l in zip(cur, last)) < 0.0005:
            still += 1
            if still >= 5:
                return cur
        else:
            still = 0
        last = cur
    return last


def main() -> int:
    p = argparse.ArgumentParser(description="Small x/y cross with the tool")
    p.add_argument("--ip", default=DEFAULT_IP)
    p.add_argument("--step", type=float, default=0.05, help="metres")
    p.add_argument("--speed", type=float, default=0.05, help="m/s")
    a = p.parse_args()

    with KinovaConnection(ConnectionConfig(ip=a.ip)) as conn:
        b = conn.base_cyclic.RefreshFeedback().base
        X, Y, Z = b.tool_pose_x, b.tool_pose_y, b.tool_pose_z
        RX, RY, RZ = b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z
        print("  start  x=%.3f y=%.3f z=%.3f   (z + orientation held fixed)" % (X, Y, Z), flush=True)

        def goto(dx, dy, label):
            cp = Base_pb2.ConstrainedPose()
            t = cp.target_pose
            t.x, t.y, t.z = X + dx, Y + dy, Z
            t.theta_x, t.theta_y, t.theta_z = RX, RY, RZ
            cp.constraint.speed.translation = a.speed
            cp.constraint.speed.orientation = 15.0
            conn.base.PlayCartesianTrajectory(cp)
            got = wait_settled(conn, label)
            print("    %-7s -> x=%.3f y=%.3f z=%.3f" % (label, got[0], got[1], got[2]), flush=True)

        for label, dx, dy in [("+x", +a.step, 0.0), ("-x", -a.step, 0.0), ("centre", 0.0, 0.0),
                              ("+y", 0.0, +a.step), ("-y", 0.0, -a.step), ("centre", 0.0, 0.0)]:
            goto(dx, dy, label)

        fb = conn.base_cyclic.RefreshFeedback()
        print("  end    x=%.3f y=%.3f z=%.3f   faults a=0x%08X b=0x%08X" % (
            fb.base.tool_pose_x, fb.base.tool_pose_y, fb.base.tool_pose_z,
            fb.base.fault_bank_a, fb.base.fault_bank_b), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
