#!/usr/bin/env python3
"""The frozen stage-1 motions: feeding_start <-> food_detect_start.

    python -m kinova.scripts.sequence detect     # feeding_start -> food_detect_start
    python -m kinova.scripts.sequence home       # retrace back to feeding_start

Three measured poses, all in calibration/:

    home_pose.json         feeding_start      parked, arm swung clear of the user,
                                              utensil straight down
    lift_pose.json         top of the lift    feeding_start's xy and orientation,
                                              LIFT_Z up -- recorded on the first
                                              `detect`, re-used by `home`
    food_detect_pose.json  food_detect_start  over the bowl, camera straight down,
                                              rolled so the bowl is framed

detect:  feeding_start --[one smooth vertical Cartesian lift]--> top of lift
                       --[joint move, via its midpoint]------> food_detect_start
home:    exactly that, backwards, then a final joint snap onto feeding_start

Why this shape:

  * VERTICAL FIRST. feeding_start sits BELOW table height, parked beside it. A
    straight joint move from there to the overlook can drag the utensil across
    the table edge. Going straight up first puts the whole swing ~300 mm above
    the bowl.
  * ONE SMOOTH LIFT, not a staircase of Cartesian steps.
  * LIFT_Z BELOW THE REACH LIMIT. At this xy the lift stalls at ~0.565 m, where
    J6 is pinned at its -120.3 limit -- and any correction from there that
    pushes J6 further is silently ignored. 0.545 keeps margin.
  * THE SWING GOES VIA ITS MIDPOINT. J1 turns ~92 deg and J7 ~104 deg. Two
    half-moves make the direction of travel explicit instead of trusting the
    controller to pick the short way round on a continuous joint. The short way
    for J1 goes round the BACK, away from the table.
  * THE UTENSIL STAYS DOWN THROUGHOUT. Every waypoint has J2+J4+J6 ~ -180, and
    a joint interpolation between two such poses keeps the sum between them.
    safe_move enforces it, and stops on any torque step (contact).
  * home IS A RETRACE. It uses the same recorded waypoints in reverse rather
    than any new path, and refuses to start from anywhere it does not
    recognise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import runlog
from kinova import scene
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import (Joints, MoveError, fold, goto, goto_cartesian, joints,
                                      pitch_sum, tilt_deg, tool_pose)
from settings import cfg

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

log = runlog.get("sequence")

CAL = Path(__file__).resolve().parent.parent.parent / "calibration"
HOME = CAL / "home_pose.json"
LIFT = CAL / "lift_pose.json"
DETECT = cfg.path("bowl.overlook_pose")           # the selected bowl's food_detect_start, as captured
                                                  # (moved with the bowl / table: kinova/scene.py)

LIFT_Z = float(cfg.motion.sequence.lift_z_m)      # m, tool z at the top of the lift (stall is ~0.565)
NEAR_DEG = float(cfg.motion.near_pose_deg)        # within this of a pose's joints = "at" that pose
LIFT_SPEED = float(cfg.motion.sequence.lift_mps)
DESCEND_SPEED = float(cfg.motion.sequence.descend_mps)   # into the parking pose: slower, it brakes near the table
SWING_SPEED = float(cfg.motion.sequence.swing_dps)
SETTLE_DPS = float(cfg.motion.settle_dps)         # the final settle / snap onto a measured pose
KNOWN_DEG = float(cfg.motion.known_pose_deg)      # within this of a pose = one `home` knows how to retrace from


def load(p: Path) -> dict:
    """A measured pose file (calibration/*.json)."""
    if not p.exists():
        raise SystemExit("missing %s -- see this module's docstring for how it is made" % p)
    return json.loads(p.read_text())


def dist_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Largest single-joint difference, deg (the short way round)."""
    return max(abs(fold(x - y)) for x, y in zip(a, b))


def midpoint(a: Sequence[float], b: Sequence[float]) -> Joints:
    """Joint-space midpoint along the SHORT way round each joint."""
    return [fold(x + fold(y - x) / 2.0) for x, y in zip(a, b)]


def record_lift(conn: KinovaConnection) -> dict:
    """Save where the arm is now as the top of the lift (lift_pose.json)."""
    q = joints(conn)
    p = tool_pose(conn)
    rec = dict(label="lift_top", joints_deg=[round(v, 3) for v in q],
               tool_pose=dict(x_m=round(p[0], 4), y_m=round(p[1], 4), z_m=round(p[2], 4),
                              theta_x_deg=round(p[3], 3), theta_y_deg=round(p[4], 3),
                              theta_z_deg=round(p[5], 3)),
               note="Recorded at the top of the vertical lift from feeding_start. "
                    "`sequence home` swings back to exactly these joints before "
                    "descending, so the return retraces the outbound path.")
    LIFT.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def report(conn: KinovaConnection, what: str) -> None:
    """Print where the arm is: joints, tool position, utensil tilt, pitch sum."""
    q = joints(conn)
    p = tool_pose(conn)
    tilt = tilt_deg(conn)
    log.info("  %-18s joints " % what + "".join("%8.2f" % v for v in q))
    log.info("  %-18s tool (%.4f %.4f %.4f)  utensil %.2f deg off vertical  J2+J4+J6 %.1f",
             "", p[0], p[1], p[2], tilt, pitch_sum(q))
    runlog.event("at", pose_name=what, joints=q, tool=p, tilt_deg=tilt)


def detect(conn: KinovaConnection, rerecord: bool) -> None:
    """feeding_start -> food_detect_start: the vertical lift, then the swing via its midpoint."""
    home = load(HOME)
    clearance = LIFT_Z - scene.table_height()
    if clearance < float(cfg.motion.sequence.min_lift_above_table_m):
        raise MoveError("the lift top (z %.3f) is only %.0f mm above the table (z %.3f): the swing to the bowl "
                        "crosses the table -- not moving" % (LIFT_Z, 1000 * clearance, scene.table_height()))
    q = joints(conn)
    if dist_deg(q, home["joints_deg"]) > NEAR_DEG:
        raise MoveError("not at feeding_start (%.1f deg away) -- run `home` first"
                        % dist_deg(q, home["joints_deg"]))
    qd = scene.overlook_joints(conn)          # before anything moves: a bowl out of reach refuses here
    report(conn, "at feeding_start")
    h = home["tool_pose"]
    xy = (h["x_m"], h["y_m"])
    rpy = (h["theta_x_deg"], h["theta_y_deg"], h["theta_z_deg"])
    goto_cartesian(conn, (xy[0], xy[1], LIFT_Z), rpy, speed=LIFT_SPEED, label="lift (smooth)")
    if rerecord or not LIFT.exists():
        record_lift(conn)
        log.info("    recorded %s", LIFT.name)
    lift = load(LIFT)["joints_deg"]
    mid = midpoint(lift, qd)
    goto(conn, mid, speed=SWING_SPEED, label="swing, first half")
    goto(conn, qd, speed=SWING_SPEED, label="swing, second half")
    report(conn, "at food_detect")


def home_(conn: KinovaConnection) -> None:
    """Back to feeding_start by retracing detect(), from food_detect_start or the top of the lift."""
    home, lift = load(HOME), load(LIFT)
    qd = scene.overlook_joints(conn)
    q = joints(conn)
    if dist_deg(q, home["joints_deg"]) <= NEAR_DEG:
        goto(conn, home["joints_deg"], speed=SETTLE_DPS, label="settle on feeding_start")
        report(conn, "at feeding_start")
        return
    d_det = dist_deg(q, qd)
    d_lift = dist_deg(q, lift["joints_deg"])
    if min(d_det, d_lift) > KNOWN_DEG:
        raise MoveError("the arm is %.1f deg from food_detect_start and %.1f deg from the "
                        "top of the lift -- not a pose this sequence knows how to retrace "
                        "from. Bring it near one of them by hand." % (d_det, d_lift))
    report(conn, "start")
    if d_det < d_lift:
        goto(conn, qd, speed=SETTLE_DPS, label="settle on food_detect")
        mid = midpoint(qd, lift["joints_deg"])
        goto(conn, mid, speed=SWING_SPEED, label="swing back, first half")
    goto(conn, lift["joints_deg"], speed=SWING_SPEED, label="swing back, second half")
    h = home["tool_pose"]
    goto_cartesian(conn, (h["x_m"], h["y_m"], h["z_m"]),
                   (h["theta_x_deg"], h["theta_y_deg"], h["theta_z_deg"]),
                   speed=DESCEND_SPEED, label="descend (smooth)")
    goto(conn, home["joints_deg"], speed=SETTLE_DPS, label="snap onto feeding_start")
    report(conn, "at feeding_start")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detect", "home"])
    ap.add_argument("--rerecord-lift", action="store_true",
                    help="overwrite lift_pose.json with where this lift ends")
    a = ap.parse_args()
    runlog.setup("sequence")
    runlog.event("start", command=a.what)
    with kinova_connection() as conn:
        try:
            if a.what == "detect":
                detect(conn, a.rerecord_lift)
            else:
                home_(conn)
        except MoveError as e:
            log.error("\n  STOPPED: %s", e)
            runlog.event("stopped", command=a.what, reason=str(e))
            return 1
    runlog.event("done", command=a.what)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
