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
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

import runlog
from kinova import scene
from kinova.motion import _ik, fk_tip
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import (MAX_TILT_DEG, Joints, MoveError, fold, goto, goto_cartesian, joints,
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

LIFT_Z = float(cfg.motion.sequence.lift_z_m)      # m, tool z at the top of the lift (stall is ~0.565); computed
                                                  # overlook: scene.lift_z() instead (no higher than the overlook)
STATE_DIR = Path(os.environ.get("RAMMP_STATE_DIR", "/tmp"))
LIFT_REACHED = STATE_DIR / "lift_reached.json"    # a computed overlook: where the last lift ended (home retraces it)
NEAR_DEG = float(cfg.motion.near_pose_deg)        # within this of a pose's joints = "at" that pose
LIFT_SPEED = float(cfg.motion.sequence.lift_mps)
DESCEND_SPEED = float(cfg.motion.sequence.descend_mps)   # into the parking pose: slower, it brakes near the table
SWING_SPEED = float(cfg.motion.sequence.swing_dps)
SWING_TORQUE_NM = float(cfg.guards.torque_step_nm.sequence_swing)
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


def lift_joints(conn: KinovaConnection, top: float) -> Joints:
    """The joints at the top of the lift: where the last lift ended (computed overlook), else the captured lift."""
    if scene.computed():
        if LIFT_REACHED.exists():
            rec = json.loads(LIFT_REACHED.read_text())
            if abs(float(rec["z_m"]) - top) < 0.005:
                return [float(v) for v in rec["joints_deg"]]
        h = load(HOME)["tool_pose"]                  # not recorded yet: the controller's IK, seeded with the capture
        q = _ik(conn, (h["x_m"], h["y_m"], top), (h["theta_x_deg"], h["theta_y_deg"], h["theta_z_deg"]),
                load(LIFT)["joints_deg"])
        if q is None:
            raise MoveError("the lift top z %.3f above feeding_start is not reachable" % top)
        return list(q)
    return load(LIFT)["joints_deg"]


def check_path(conn: KinovaConnection, legs: Sequence[tuple[Joints, Joints]], top: float) -> tuple[float, float]:
    """(highest, lowest) utensil tip z along these joint legs (the controller's FK, sampled). Refuses a path that
    rises more than motion.sequence.max_path_above_lift_m over where it starts, or takes the tip below
    scene.tip_floor() anywhere (above the bowl's top over the bowl, above the table elsewhere)."""
    tips = [fk_tip(conn, [x + fold(y - x) * s for x, y in zip(a, b)]) for a, b in legs for s in np.linspace(0.0, 1.0, 13)]
    zs = [float(t[2]) for t in tips]
    peak, low = max(zs), min(zs)
    rise = float(cfg.motion.sequence.max_path_above_lift_m)
    if peak > zs[0] + rise:
        raise MoveError("the swing would lift the utensil tip to z %.3f, %.0f mm above where it starts (the lift top, "
                        "tool z %.3f) -- more than %.0f mm: not moving" % (peak, 1000 * (peak - zs[0]), top, 1000 * rise))
    ws = scene.workspace()
    for t in tips:
        if t[2] < scene.tip_floor(t[0], t[1]) - 1e-6:
            raise MoveError("the swing would bring the utensil tip down to z %.3f at (%.3f, %.3f): %.0f mm above the "
                            "table, %.0f mm above the bowl's top -- below the floor there (z %.3f): not moving"
                            % (t[2], t[0], t[1], 1000 * (t[2] - ws["table_z"]), 1000 * (t[2] - ws["bowl_top_z"]),
                               scene.tip_floor(t[0], t[1])))
    return peak, low


def detect(conn: KinovaConnection, rerecord: bool, plan: bool = False) -> None:
    """feeding_start -> food_detect_start: the vertical lift, then the swing via its midpoint. With a computed
    overlook (kinova/scene.py) the lift goes no higher than the overlook, and the swing's peak is checked first."""
    home = load(HOME)
    top = scene.lift_z()
    clearance = top - scene.table_height()
    if clearance < float(cfg.motion.sequence.min_lift_above_table_m):
        raise MoveError("the lift top (z %.3f) is only %.0f mm above the table (z %.3f): the swing to the bowl "
                        "crosses the table -- not moving" % (top, 1000 * clearance, scene.table_height()))
    q = joints(conn)
    if dist_deg(q, home["joints_deg"]) > NEAR_DEG and not plan:
        raise MoveError("not at feeding_start (%.1f deg away) -- run `home` first"
                        % dist_deg(q, home["joints_deg"]))
    qd = scene.overlook_joints(conn)          # before anything moves: a bowl out of reach refuses here
    if scene.computed():
        ws = scene.workspace()
        lift = lift_joints(conn, top)
        peak, low = check_path(conn, [(lift, midpoint(lift, qd)), (midpoint(lift, qd), qd)], top)
        t = scene.overlook_pose()["tool_pose"]
        log.info("  overlook over the bowl at (%.3f, %.3f) m, table z %.3f (%s)", ws["bowl_xy"][0], ws["bowl_xy"][1],
                 ws["table_z"], ws["source"])
        log.info("  food_detect_start: tool (%.3f %.3f %.3f), joints %s", t["x_m"], t["y_m"], t["z_m"],
                 " ".join("%.1f" % v for v in qd))
        log.info("  lift to tool z %.3f (the captured lift top: %.3f); on the swing the tip stays z %.3f .. %.3f "
                 "(%.0f .. %.0f mm above the table; its lowest %.0f mm above the bowl's top)", top, LIFT_Z, low, peak,
                 1000 * (low - ws["table_z"]), 1000 * (peak - ws["table_z"]), 1000 * (low - ws["bowl_top_z"]))
    if plan:
        log.info("  PLAN only: nothing moved")
        return
    report(conn, "at feeding_start")
    h = home["tool_pose"]
    xy = (h["x_m"], h["y_m"])
    rpy = (h["theta_x_deg"], h["theta_y_deg"], h["theta_z_deg"])
    goto_cartesian(conn, (xy[0], xy[1], top), rpy, speed=LIFT_SPEED, label="lift (smooth)")
    if scene.computed():
        LIFT_REACHED.write_text(json.dumps(dict(z_m=top, joints_deg=joints(conn))))
    elif rerecord or not LIFT.exists():
        record_lift(conn)
        log.info("    recorded %s", LIFT.name)
    lift = lift_joints(conn, top)
    mid = midpoint(lift, qd)
    goto(conn, mid, speed=SWING_SPEED, torque_limit=SWING_TORQUE_NM, label="swing, first half")
    goto(conn, qd, speed=SWING_SPEED, torque_limit=SWING_TORQUE_NM, label="swing, second half")
    report(conn, "at food_detect")


def home_(conn: KinovaConnection) -> None:
    """Back to feeding_start by retracing detect(), from food_detect_start or the top of the lift."""
    home, lift = load(HOME), dict(joints_deg=lift_joints(conn, scene.lift_z()))
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
        goto(conn, mid, speed=SWING_SPEED, torque_limit=SWING_TORQUE_NM, label="swing back, first half")
    goto(conn, lift["joints_deg"], speed=SWING_SPEED, torque_limit=SWING_TORQUE_NM, label="swing back, second half")
    h = home["tool_pose"]
    goto_cartesian(conn, (h["x_m"], h["y_m"], h["z_m"]),
                   (h["theta_x_deg"], h["theta_y_deg"], h["theta_z_deg"]),
                   speed=DESCEND_SPEED, label="descend (smooth)")
    goto(conn, home["joints_deg"], speed=SETTLE_DPS, label="snap onto feeding_start")
    report(conn, "at feeding_start")


def direct(conn: KinovaConnection, speed: float) -> None:
    """Straight to feeding_start from anywhere: one joint move, every joint the short way round.

    For a workspace the operator has checked is clear (it does not route round the table like `home`). The usual
    guards hold: joint limits, the utensil ending down, a torque step stops it."""
    home = load(HOME)
    q = joints(conn)
    travel = max(abs(fold(t - c)) for t, c in zip(home["joints_deg"], q))
    # From a hand-placed pose the utensil can start tipped past the usual limit. The move takes J2+J4+J6 linearly
    # to straight down, so the tilt only falls from here (twin check, 2026-09-30: 29 -> 0.7 deg, monotone): allow
    # where it starts, not more.
    tilt0 = tilt_deg(conn)
    max_tilt = max(MAX_TILT_DEG, tilt0 + 3.0)
    log.info("direct to feeding_start: %.0f deg on the furthest joint at %.0f deg/s (~%.0f s); utensil starts "
             "%.0f deg off vertical (allowed %.0f)", travel, speed, travel / max(speed, 1e-3), tilt0, max_tilt)
    report(conn, "start")
    goto(conn, home["joints_deg"], speed=speed, max_tilt=max_tilt, label="direct to feeding_start")
    report(conn, "at feeding_start")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["detect", "home", "direct"])
    ap.add_argument("--plan", action="store_true", help="detect: compute and check, move nothing")
    ap.add_argument("--speed", type=float, default=float(cfg.motion.sequence.direct_dps),
                    help="direct: the furthest joint's average speed, deg/s")
    ap.add_argument("--rerecord-lift", action="store_true",
                    help="overwrite lift_pose.json with where this lift ends")
    a = ap.parse_args()
    runlog.setup("sequence")
    runlog.event("start", command=a.what)
    with kinova_connection() as conn:
        try:
            if a.what == "detect":
                detect(conn, a.rerecord_lift, a.plan)
            elif a.what == "direct":
                direct(conn, a.speed)
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
