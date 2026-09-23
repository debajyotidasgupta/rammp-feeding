#!/usr/bin/env python3
"""Pivot calibration: where are the fork's tine tips, relative to the flange?

The fork is a custom part held in a 2F-85 and appears in no URDF. `/ee_state`
reports the flange pose from `models/gen3_7dof_2f85.urdf`, which models the
gripper and knows nothing about a fork. Every number the skewer uses lives at
the tine tips -- insertion depth, the plate floor, the mouth standoff -- so an
unmeasured tool offset is a systematic error in all of them.

Method. Touch ONE fixed physical point with the tine tips from N different
orientations. For each pose the tip must land on the same world point:

    R_i @ p_tool + t_i = p_fixed        for every i

which is linear in the six unknowns (p_tool, p_fixed):

    [ R_i  -I ] [ p_tool ; p_fixed ] = -t_i

Stack the poses and solve least squares. The residual is the honest accuracy
estimate: it is how far the "same point" actually moved.

Orientations must differ substantially. Translating the arm while keeping its
orientation leaves p_tool unobservable and the solve will look converged and be
wrong -- the same failure mode as hand-eye calibration from pure translation.

    # with the driver running, jog the arm so the tines touch a fixed point, then:
    python -m feeding.calibrate_tcp record      # appends the current pose
    python -m feeding.calibrate_tcp solve       # solves from what has been recorded
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

STORE = pathlib.Path("calibration/tcp_poses.json")
MIN_POSES = 4


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def solve(poses: list[dict]) -> dict:
    """Least-squares tool offset and the pivot point, with residuals."""
    A, b = [], []
    for p in poses:
        R = quat_to_R(*p["quat"])
        A.append(np.hstack([R, -np.eye(3)]))
        b.append(-np.asarray(p["pos"], float))
    A = np.vstack(A)
    b = np.concatenate(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_tool, p_fixed = sol[:3], sol[3:]

    tips = np.array([quat_to_R(*p["quat"]) @ p_tool + np.asarray(p["pos"], float)
                     for p in poses])
    resid = np.linalg.norm(tips - p_fixed, axis=1)

    # How much did orientation actually vary? Without spread, the fit is
    # meaningless however small the residual looks.
    Rs = [quat_to_R(*p["quat"]) for p in poses]
    spread = max(np.degrees(np.arccos(np.clip(
        (np.trace(Ri.T @ Rj) - 1) / 2, -1, 1)))
        for Ri in Rs for Rj in Rs)

    return dict(tool_offset_m=p_tool.tolist(), pivot_point_m=p_fixed.tolist(),
                residual_mm=dict(mean=float(resid.mean() * 1000),
                                 max=float(resid.max() * 1000)),
                orientation_spread_deg=float(spread), n_poses=len(poses))


def read_ee_state() -> dict:
    """Read one /ee_state message. Requires a sourced ROS 2 environment."""
    import subprocess
    out = subprocess.run(
        ["ros2", "topic", "echo", "--once", "/ee_state"],
        capture_output=True, text=True, timeout=20).stdout
    def grab(after: str, keys: list[str]) -> list[float]:
        seg = out.split(after, 1)[1]
        return [float(seg.split(k + ":", 1)[1].split("\n", 1)[0]) for k in keys]
    return dict(pos=grab("position:", ["x", "y", "z"]),
                quat=grab("orientation:", ["x", "y", "z", "w"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["record", "solve", "clear", "show"])
    a = ap.parse_args()
    STORE.parent.mkdir(parents=True, exist_ok=True)
    poses = json.loads(STORE.read_text()) if STORE.exists() else []

    if a.action == "clear":
        STORE.unlink(missing_ok=True)
        print("cleared")
        return 0
    if a.action == "record":
        p = read_ee_state()
        poses.append(p)
        STORE.write_text(json.dumps(poses, indent=2))
        print("recorded pose %d: pos %s" % (len(poses), np.round(p["pos"], 4)))
        if len(poses) < MIN_POSES:
            print("  need at least %d, with WIDELY DIFFERENT orientations" % MIN_POSES)
        return 0
    if a.action == "show":
        for i, p in enumerate(poses):
            print("  %2d pos %s quat %s" % (i, np.round(p["pos"], 4), np.round(p["quat"], 3)))
        return 0

    if len(poses) < MIN_POSES:
        print("only %d poses; need >= %d" % (len(poses), MIN_POSES))
        return 1
    r = solve(poses)
    t = r["tool_offset_m"]
    print("fork tip relative to the flange frame:")
    print("  x %+.4f  y %+.4f  z %+.4f  m   (|p| = %.1f mm)"
          % (t[0], t[1], t[2], np.linalg.norm(t) * 1000))
    print("  residual: mean %.2f mm, max %.2f mm over %d poses"
          % (r["residual_mm"]["mean"], r["residual_mm"]["max"], r["n_poses"]))
    print("  orientation spread: %.0f deg" % r["orientation_spread_deg"])
    if r["orientation_spread_deg"] < 45:
        print("  *** spread is too small -- the offset is not observable. Re-take"
              " with genuinely different wrist orientations, not just positions.")
    if r["residual_mm"]["max"] > 3.0:
        print("  *** residual too large to trust for skewering 15 mm pieces.")
    (STORE.parent / "tcp_result.json").write_text(json.dumps(r, indent=2))
    print("  wrote calibration/tcp_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
