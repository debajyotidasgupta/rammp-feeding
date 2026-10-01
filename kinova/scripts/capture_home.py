#!/usr/bin/env python3
"""Capture the arm's present configuration as the canonical feeding start pose.

Read-only: it commands no motion and never touches the gripper (the utensil is
held in a closed 2F-85; opening it drops the tool).

Every feeding sequence begins from this pose: a fixed constant, captured once, not perceived per run.

Averages over a short window: holding under gravity, the encoders dither by a few millidegrees.

    python -m kinova.scripts.capture_home [--seconds 2] [--out calibration/home_pose.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from kinova.connection import kinova_connection


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--hz", type=float, default=40.0)
    ap.add_argument("--out", default="calibration/home_pose.json")
    ap.add_argument("--label", default="feeding_start",
                    help="name this pose is referred to by in the stack")
    a = ap.parse_args()

    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient

    samples: list[dict] = []
    with kinova_connection(ip=a.ip) as conn:
        bc = BaseCyclicClient(conn._tcp_router)
        tc = ControlConfigClient(conn._tcp_router).GetToolConfiguration()
        tool_cfg = dict(
            transform_m=[round(tc.tool_transform.x, 5), round(tc.tool_transform.y, 5),
                         round(tc.tool_transform.z, 5)],
            rotation_deg=[round(tc.tool_transform.theta_x, 3),
                          round(tc.tool_transform.theta_y, 3),
                          round(tc.tool_transform.theta_z, 3)],
            mass_kg=round(tc.tool_mass, 4),
            mass_centre_m=[round(tc.tool_mass_center.x, 5), round(tc.tool_mass_center.y, 5),
                           round(tc.tool_mass_center.z, 5)],
        )
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            fb = bc.RefreshFeedback()
            samples.append(dict(
                q=[act.position for act in fb.actuators],
                tau=[act.torque for act in fb.actuators],
                tool=[fb.base.tool_pose_x, fb.base.tool_pose_y, fb.base.tool_pose_z,
                      fb.base.tool_pose_theta_x, fb.base.tool_pose_theta_y,
                      fb.base.tool_pose_theta_z],
            ))
            time.sleep(1.0 / a.hz)

    if not samples:
        print("no feedback received", file=sys.stderr)
        return 1

    n_j = len(samples[0]["q"])

    def col(key, i):
        return [s[key][i] for s in samples]

    # joint angles come in [0, 360): fold to (-180, 180] before averaging, or a joint near 0/360 averages to 180
    def fold(d):
        return (d + 180.0) % 360.0 - 180.0

    q = [round(statistics.fmean(fold(v) for v in col("q", i)), 3) for i in range(n_j)]
    q_spread = [round(max(map(fold, col("q", i))) - min(map(fold, col("q", i))), 4)
                for i in range(n_j)]
    tau = [round(statistics.fmean(col("tau", i)), 3) for i in range(n_j)]
    tool = [round(statistics.fmean(col("tool", i)), 4) for i in range(6)]

    rec = dict(
        label=a.label,
        captured=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        n_samples=len(samples),
        joints_deg=q,
        joints_spread_deg=q_spread,
        holding_torque_nm=tau,
        tool_pose=dict(x_m=tool[0], y_m=tool[1], z_m=tool[2],
                       theta_x_deg=tool[3], theta_y_deg=tool[4],
                       theta_z_deg=tool[5]),
        tool_config=tool_cfg,
        note=("tool_pose is the CONFIGURED TOOL frame, not the flange: the arm "
              "carries a 120 mm z-offset for the 2F-85. The utensil adds more. "
              "Measured camera-to-tine-tip is ~105 mm, so the reported point "
              "lands within a few cm of the real tip -- good enough to report "
              "and to stand off by, NOT good enough for the 4 mm flesh/bowl "
              "window, which is terminated on torque instead."),
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")

    print("pose '%s'  (%d samples, %.1f s)" % (a.label, len(samples), a.seconds))
    print("  joints deg : " + "  ".join("%8.3f" % v for v in q))
    print("  spread deg : " + "  ".join("%8.4f" % v for v in q_spread))
    print("  torque N.m : " + "  ".join("%8.3f" % v for v in tau))
    print("  tool  xyz : %.4f %.4f %.4f m" % tuple(tool[:3]))
    print("  tool  rpy : %.2f %.2f %.2f deg" % tuple(tool[3:]))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
