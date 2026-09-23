#!/usr/bin/env python3
"""The named arm poses.

FEEDING_START is measured on this installation and checked in
(calibration/home_pose.json). Every feeding sequence begins and ends there.
Re-measure it with `python -m kinova.scripts.capture_home`; go to it with
`python -m kinova.scripts.feeding_start`.

The pose is safe at rest by construction: joint 1 at -90 swings the arm
clockwise, clear of the user on their right, and the tool points ~81 degrees
below horizontal, so the fork aims at the floor rather than at anyone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HOME_POSE_JSON = Path(__file__).resolve().parent.parent / "calibration" / "home_pose.json"


def load_start_pose(path=None) -> dict:
    p = Path(path) if path else HOME_POSE_JSON
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing. Capture it from the parked arm with:\n"
            "    python -m kinova.scripts.capture_home")
    return json.loads(p.read_text())


def approach_axis(theta_x, theta_y, theta_z):
    """Tool +z (the direction the fork points) in base coords, from Kortex's
    fixed-frame XYZ Euler angles: R = Rz @ Ry @ Rx, third column."""
    tx, ty, tz = (math.radians(v) for v in (theta_x, theta_y, theta_z))
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    return (cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx)


def main() -> int:
    rec = load_start_pose()
    t = rec["tool_pose"]
    ax = approach_axis(t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"])
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, -ax[2]))))
    print("%s  (captured %s)" % (rec["label"], rec["captured"]))
    print("  joints deg : " + "  ".join("%8.3f" % v for v in rec["joints_deg"]))
    print("  tool point : %.4f %.4f %.4f m" % (t["x_m"], t["y_m"], t["z_m"]))
    print("  fork points: %.1f deg below horizontal" % tilt)
    print("  tool xform : %s m, %.3f kg"
          % (rec["tool_config"]["transform_m"], rec["tool_config"]["mass_kg"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
