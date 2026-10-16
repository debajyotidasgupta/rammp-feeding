#!/usr/bin/env python3
"""The moves between the parked pose and the bowl.

    python -m kinova.scripts.sequence detect [--plan]   # -> food_detect_start (over the bowl)
    python -m kinova.scripts.sequence home              # -> feeding_start (parked)
    python -m kinova.scripts.sequence direct [--speed]  # -> feeding_start by one joint move, no planner

detect and home are one collision-free cuRobo path each, from wherever the arm is (kinova/curobo_client.py), then
a slow settle onto the measured pose. direct is the recovery when the planner is down: one joint move, every joint
the short way round, for a workspace the operator has checked is clear.

feeding_start is calibration/home_pose.json; food_detect_start is computed over the bowl (kinova/scene.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import runlog
from kinova import curobo_client, scene
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import MAX_TILT_DEG, MoveError, fold, goto, joints, pitch_sum, tilt_deg, tool_pose
from settings import cfg

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

log = runlog.get("sequence")

HOME = Path(__file__).resolve().parent.parent.parent / "calibration" / "home_pose.json"


def home_joints() -> list[float]:
    if not HOME.exists():
        raise SystemExit("missing %s -- make capture POSE=feeding_start" % HOME)
    return [float(v) for v in json.loads(HOME.read_text())["joints_deg"]]


def report(conn: KinovaConnection, what: str) -> None:
    """Where the arm is: joints, tool position, utensil tilt, pitch sum."""
    q, p, tilt = joints(conn), tool_pose(conn), tilt_deg(conn)
    log.info("  %-18s joints " % what + "".join("%8.2f" % v for v in q))
    log.info("  %-18s tool (%.4f %.4f %.4f)  utensil %.2f deg off vertical  J2+J4+J6 %.1f",
             "", p[0], p[1], p[2], tilt, pitch_sum(q))
    runlog.event("at", pose_name=what, joints=q, tool=p, tilt_deg=tilt)


def detect(conn: KinovaConnection, plan: bool = False) -> None:
    """-> food_detect_start, over the bowl where the workspace says it is."""
    ws = scene.workspace()
    if ws["age_s"] > float(cfg.workspace.scene_max_age_s):   # a new bite: the bowl may have moved since
        raise MoveError("the scene camera's reading is %.0f min old -- make locate-scene first" % (ws["age_s"] / 60))
    qd = scene.overlook_joints(conn)              # a bowl out of reach refuses here, before anything moves
    t = scene.overlook_pose()["tool_pose"]
    log.info("  bowl at (%.3f, %.3f) m, table z %.3f (%s); food_detect_start tool (%.3f %.3f %.3f)",
             ws["bowl_xy"][0], ws["bowl_xy"][1], ws["table_z"], ws["source"], t["x_m"], t["y_m"], t["z_m"])
    report(conn, "start")
    curobo_client.travel(conn, qd, "to food_detect_start", go=not plan)
    if not plan:
        report(conn, "at food_detect")


def home(conn: KinovaConnection) -> None:
    """-> feeding_start."""
    report(conn, "start")
    curobo_client.travel(conn, home_joints(), "to feeding_start")
    report(conn, "at feeding_start")


def direct(conn: KinovaConnection, speed: float) -> None:
    """-> feeding_start by one joint move (no planner). The utensil's tilt only falls along it (J2+J4+J6 goes
    linearly to straight down), so the tilt guard allows where it starts."""
    q, goal = joints(conn), home_joints()
    travel = max(abs(fold(t - c)) for t, c in zip(goal, q))
    tilt0 = tilt_deg(conn)
    max_tilt = max(MAX_TILT_DEG, tilt0 + 3.0)
    log.info("direct to feeding_start: %.0f deg on the furthest joint at %.0f deg/s (~%.0f s); utensil starts "
             "%.0f deg off vertical (allowed %.0f)", travel, speed, travel / max(speed, 1e-3), tilt0, max_tilt)
    report(conn, "start")
    goto(conn, goal, speed=speed, max_tilt=max_tilt, label="direct to feeding_start")
    report(conn, "at feeding_start")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detect", "home", "direct"])
    ap.add_argument("--plan", action="store_true", help="detect: plan and check, move nothing")
    ap.add_argument("--speed", type=float, default=float(cfg.motion.direct_dps), help="direct: deg/s, furthest joint")
    a = ap.parse_args()
    runlog.setup("sequence")
    runlog.event("start", command=a.what)
    with kinova_connection() as conn:
        try:
            if a.what == "detect":
                detect(conn, a.plan)
            elif a.what == "direct":
                direct(conn, a.speed)
            else:
                home(conn)
        except MoveError as e:
            log.error("\n  STOPPED: %s", e)
            runlog.event("stopped", command=a.what, reason=str(e))
            return 1
    runlog.event("done", command=a.what)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
