#!/usr/bin/env python3
"""Food delivery, after acquire: swing to the user, present at the mouth, wait for the bite, retract.

    python -m kinova.scripts.deliver swing     [--validate-only]
    python -m kinova.scripts.deliver approach  --face face.json --colour colour.jpg [--allow-nose] [--validate-only]
    python -m kinova.scripts.deliver sidestep  [--step 0.05]
    python -m kinova.scripts.deliver bite      [--timeout 60]
    python -m kinova.scripts.deliver retract   [--validate-only]

Runs in the driver container; feeding/deliver.sh orchestrates it with the mouth detection on the host.

swing     from the scoop end to the user, one cuRobo path carrying the piece (swing_goal: where it ends).
approach  from a fresh detection made at this pose: a straight line, the utensil's orientation held, to the tip
          delivery.stop_m short of the mouth and delivery.aim_dz_m below it. Exit 3 when the mouth is not usable
          (sidestep and look again), 5 when its depth is not plausibly a face at the utensil.
sidestep  horizontally away from the user (-Y), orientation held, when the food hides the mouth.
bite      read-only: the tool-force estimate against a baseline at rest, above delivery.bite_n for bite_hold_s.
retract   straight back from the mouth to the swing's end, then one cuRobo path to food_detect_start.

Exit codes: 0 done, 1 stopped (nothing further moved), 3 mouth not usable (sidestep and look again), 4 no bite,
5 the depth is not plausibly a face at the fork (look again from here).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

import runlog
from kinova import scene
from kinova.connection import kinova_connection
from settings import cfg
from feeding.camera import COLOUR_SIZE, CXC, CYC, FXC, FYC, cam_to_tool
from kinova.motion import R_from_kortex, Rz, _ik, axis, dps_for, homotopy, margin, max_step, run_path, tip_in_tool
from kinova import curobo_client
from kinova.scripts.safe_move import Joints, MoveError, goto_cartesian, joints, tool_pose

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

log = runlog.get("deliver")

DL, G = cfg.delivery, cfg.guards.torque_step_nm
LIFT_M = float(DL.lift_m)
RETRACT_TRIES = tuple(float(x) for x in DL.retract_tries_m)
SWING_DPS = float(DL.swing_dps)
DETECT_SIDESTEP_M = float(DL.detect_sidestep_m) if cfg.features.detect_from_the_side else 0.0
STOP_M = float(DL.stop_m)
AIM_DZ = float(DL.aim_dz_m)
APPROACH_MPS = float(DL.approach_mps)
MAX_APPROACH_M = float(DL.max_approach_m)
MAX_SHIFT_S = float(DL.detection_max_age_s)
NOSE_TO_MOUTH = float(DL.nose_to_mouth)
MOUTH_BEHIND_NOSE = float(DL.mouth_behind_nose_m)
BITE_N = float(DL.bite_n)
BITE_HOLD_S = float(DL.bite_hold_s)
RETRACT_MPS = float(DL.retract.back_off_mps)
RETRACT_HOME_DPS = float(DL.retract.home_dps)
RETRACT_TORQUE_NM = float(G.retract)
SWING_TORQUE_NM = float(G.swing)
APPROACH_TORQUE_NM = float(G.approach)
UP_W, UP_H = COLOUR_SIZE[1], COLOUR_SIZE[0]       # the upright (rotated) image the face detector sees
MC = DL.mouth_check
LEVEL_TOL = float(DL.level_tol)                   # |utensil axis z| below this = level
POINTING_COS = float(DL.pointing_tol_cos)         # the utensil points along a direction: cos of the angle off it
ARRIVED_M = float(DL.arrived_m)                   # closer than this to a goal = already there
BITE_BASELINE_N = int(DL.bite_baseline_samples)

STATE_DIR = Path(os.environ.get("RAMMP_STATE_DIR", "/tmp"))   # what the steps hand each other, across bites
STATE = STATE_DIR / "deliver_state.json"          # the swing's end: the retract backs off to it
GOAL_CACHE = STATE_DIR / "deliver_swing_goal.json"   # the scoop always ends on the scoop end: its goal is reused


# ------------------------------------------------------------------ swing

def swing_tip(tipS: np.ndarray, axS: np.ndarray, retract_m: float, u: float) -> np.ndarray:
    """The utensil tip at progress u of the turn toward the user: +90 deg about the base's z, retracting."""
    return Rz(90.0 * u) @ (tipS - retract_m * (1 - (1 - u) ** 2) * axS)


def swing_goal(conn: KinovaConnection, q0: Joints, T0: np.ndarray, R0: np.ndarray) -> Joints:
    """Where the swing ends, the arm's shape included: from the scoop end, LIFT_M up, turned 90 deg toward the
    user (the utensil level, food side up) retracted along itself as far as the joints allow, then
    DETECT_SIDESTEP_M to the user's right (the food out of the camera's view of the mouth). Continuous IK along
    that geometry -- the joints only, cuRobo plans the way there. Cached for the next bite."""
    try:
        c = json.loads(GOAL_CACHE.read_text())
        if max_step(c["q_start"], q0) <= float(DL.goal_cache_tol_deg):
            return c["goal"]
    except (OSError, ValueError, KeyError):
        pass
    lift, err = homotopy(conn, q0, lambda s: (T0 + np.array([0, 0, LIFT_M * s]), R0), 35)
    if lift is None:
        raise MoveError("the %.0f mm lift is not reachable (%s) -- nothing moved" % (1000 * LIFT_M, err))
    tip0, ax0, tip_t = T0 + np.array([0, 0, LIFT_M]) + R0 @ tip_in_tool(), axis(R0), tip_in_tool()
    for retract_m in RETRACT_TRIES:
        def at(s: float, r: float = retract_m) -> tuple[np.ndarray, np.ndarray]:
            return swing_tip(tip0, ax0, r, s) - Rz(90.0 * s) @ R0 @ tip_t, Rz(90.0 * s) @ R0
        turn, err = homotopy(conn, lift[-1], at, 60)
        if turn and min(margin(q) for q in turn) >= min(3.0, margin(lift[-1]) - 0.5):
            break
    else:
        raise MoveError("no swing toward the user from here -- nothing moved")
    Ts, Rs = at(1.0)
    side, _ = homotopy(conn, turn[-1], lambda s: (Ts - np.array([0, DETECT_SIDESTEP_M * s, 0]), Rs), 8)
    goal = (side or turn)[-1]
    log.info("    swing goal: lift %.0f mm, turn retracting %.0f mm, %.0f mm to the user's right", 1000 * LIFT_M,
             1000 * retract_m, 1000 * DETECT_SIDESTEP_M if side else 0)
    GOAL_CACHE.write_text(json.dumps(dict(q_start=q0, goal=goal)))
    return goal


def swing(conn: KinovaConnection, validate_only: bool = False) -> None:
    """From the scoop end to the user, carrying the piece: one cuRobo path to swing_goal()."""
    p, q0 = tool_pose(conn), joints(conn)
    R0 = R_from_kortex(*p[3:])
    if axis(R0)[1] < POINTING_COS:
        raise MoveError("utensil is not pointing at the user (+Y) -- run from the end of the scoop")
    goal = swing_goal(conn, q0, np.array(p[:3]), R0)
    curobo_client.travel(conn, goal, "swing to the user", carrying=True, go=not validate_only, dps=SWING_DPS,
                         torque_limit=SWING_TORQUE_NM, max_tilt=95.0)
    if not validate_only:
        end = tool_pose(conn)
        STATE.write_text(json.dumps(dict(end_tool=list(end[:3]))))
        runlog.event("swing", end_tool=end)


# ------------------------------------------------------------------ present

def _mouth(f: dict, allow_nose: bool = False) -> tuple[list[float] | None, str]:
    """Mouth in camera coordinates and where it came from: the plausible mouth box, else (allowed) the nose
    estimate. (None, why) when the mouth is not usable and the nose estimate is not allowed."""
    n = f.get("nose")
    if not n:
        raise MoveError("no nose detected")
    # the nose's box is what the checks below need; its depth can be missing (the food hides it from the depth
    # sensor, which sits 27 mm beside the colour camera) -- then only the nose-based fallback is unavailable
    x1, y1, x2, y2 = n["box_upright"]
    ncx, ncy, nw, nh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if nw * nh > float(MC.max_box_frac) * UP_W * UP_H:
        raise MoveError("nose box is %.0f%% of the frame -- junk" % (100 * nw * nh / (UP_W * UP_H)))
    why = _mouth_box_problem(f.get("mouth"), n, ncx, ncy, nw, nh)
    if why is None:
        m = f["mouth"]
        return m["cam_xyz"], "mouth box (%s)" % m.get("prompt", "")
    if not allow_nose:
        return None, why
    if not n.get("depth_m"):
        raise MoveError("mouth not usable (%s) and the nose has no depth -- cannot estimate it" % why)
    u, v = ncy + NOSE_TO_MOUTH * nh, UP_W - 1 - ncx          # upright -> original colour pixel
    z = n["depth_m"] + MOUTH_BEHIND_NOSE
    return [(u - CXC) * z / FXC, (v - CYC) * z / FYC, z], "nose estimate (%s)" % why


def _mouth_box_problem(m: dict | None, n: dict, ncx: float, ncy: float, nw: float, nh: float) -> str | None:
    """Why the mouth box is not anatomically plausible relative to the nose, or None if it is."""
    if not m or not m.get("depth_m"):
        return "no mouth box"
    a1, b1, a2, b2 = m["box_upright"]
    mcx, mcy = (a1 + a2) / 2, (b1 + b2) / 2
    lo, hi = MC.below_nose_heights
    if (a2 - a1) * (b2 - b1) > float(MC.max_box_frac) * UP_W * UP_H:
        return "mouth box covers %.0f%% of the frame" % (100 * (a2 - a1) * (b2 - b1) / (UP_W * UP_H))
    if not float(lo) * nh < mcy - ncy < float(hi) * nh:
        return "mouth box %.2f nose heights below the nose" % ((mcy - ncy) / nh)
    if abs(mcx - ncx) > float(MC.max_sideways_nose_widths) * nw:
        return "mouth box %.2f nose widths sideways" % (abs(mcx - ncx) / nw)
    if n.get("depth_m") and abs(m["depth_m"] - n["depth_m"]) > float(MC.max_depth_diff_m):
        return "mouth depth %.0f mm off the nose's" % (1000 * abs(m["depth_m"] - n["depth_m"]))
    return None


def _keep_mouth(mouth: np.ndarray) -> None:
    """The measured mouth, for the planner's world (the user is an obstacle in it)."""
    try:
        runs = Path(os.path.expanduser(str(cfg.deployment.runs_dir)))
        (runs / "mouth_state.json").write_text(json.dumps(dict(mouth_base=[float(v) for v in mouth], t=time.time())))
        curobo_client.refresh_world()
    except (OSError, MoveError) as e:
        log.info("    (the mouth not kept for the planner: %s)", e)


def _depth_implausible(f: dict, mouth_z: float) -> str | None:
    """Why the mouth's depth cannot be a face at the fork (a stale or misregistered depth frame), or None."""
    lo, hi = (float(x) for x in MC.depth_range_m)
    if not lo <= mouth_z <= hi:
        return "mouth depth %.0f mm is outside %.0f-%.0f mm" % (1000 * mouth_z, 1000 * lo, 1000 * hi)
    face_z = (f.get("face") or {}).get("depth_m")
    if face_z and abs(mouth_z - face_z) > float(MC.max_face_diff_m):
        return "mouth depth %.0f mm is %.0f mm off the face's %.0f" % (1000 * mouth_z, 1000 * abs(mouth_z - face_z),
                                                                        1000 * face_z)
    return None


def approach(conn: KinovaConnection, face_path: str, colour_path: str, allow_nose: bool = False,
             validate_only: bool = False) -> int:
    """Straight in to STOP_M short of the mouth, from a fresh detection made at this pose. 3 = mouth not usable."""
    if os.path.getmtime(face_path) < os.path.getmtime(colour_path):
        raise MoveError("the detection is older than its capture -- not moving")
    if time.time() - os.path.getmtime(face_path) > MAX_SHIFT_S:
        raise MoveError("the detection is more than %.0f s old -- not moving" % MAX_SHIFT_S)
    f = json.loads(Path(face_path).read_text())
    cam, source = _mouth(f, allow_nose=allow_nose)
    if cam is None:
        log.info("    mouth not usable (%s) -- sidestep and look again", source)
        runlog.event("mouth_not_usable", why=source)
        return 3
    bad = _depth_implausible(f, cam[2])
    if bad:
        log.info("    %s -- not approaching; look again", bad)
        runlog.event("depth_implausible", why=bad)
        return 5
    p = tool_pose(conn)
    q0 = joints(conn)
    T0 = np.array(p[:3])
    R = R_from_kortex(*p[3:])
    ax = axis(R)
    if abs(ax[2]) > LEVEL_TOL:
        raise MoveError("utensil is not level -- run the swing first")
    mouth = T0 + R @ cam_to_tool(cam)
    _keep_mouth(mouth)
    stop = STOP_M
    if f.get("mouth", {}).get("depth_filled"):
        stop += float(DL.stop_extra_if_depth_filled_m)   # the fallback's local error can reach ~60 mm: stand off
        log.info("    the mouth's depth is the Depth Anything 3 fallback: stopping %.0f mm short instead of %.0f",
                 1000 * stop, 1000 * STOP_M)
    goal_tip = mouth - stop * ax + np.array([0, 0, AIM_DZ])
    Tg = goal_tip - R @ tip_in_tool()
    L = np.linalg.norm(Tg - T0)
    log.info("    mouth (%.3f %.3f %.3f) from the %s", *mouth, source)
    log.info("    stop: utensil tip (%.3f %.3f %.3f), %.0f mm short and %.0f mm below; %.0f mm to go",
             *goal_tip, 1000 * stop, -1000 * AIM_DZ, 1000 * L)
    runlog.event("approach_goal", mouth_base=mouth, mouth_cam=cam, source=source, stop_m=stop, goal_tip=goal_tip,
                 distance_m=float(L), face=face_path)
    if L > MAX_APPROACH_M:
        raise MoveError("an approach of %.0f mm is too long -- not moving" % (1000 * L))
    if L < ARRIVED_M:
        log.info("    already there")
        return 0
    path, err = homotopy(conn, q0, lambda s: (T0 + (Tg - T0) * s, R), 40)
    if path is None:
        raise MoveError("approach " + err + " -- not moving")
    run_path(conn, path, dps_for(path, L, APPROACH_MPS), "approach at %.0f mm/s" % (1000 * APPROACH_MPS),
             not validate_only, torque_limit=APPROACH_TORQUE_NM)
    if not validate_only:
        tip = np.array(tool_pose(conn)[:3]) + R @ tip_in_tool()
        log.info("    utensil tip %.0f mm from the detected mouth", 1000 * np.linalg.norm(mouth - tip))
        runlog.event("presented", tip=tip, tip_to_mouth_m=float(np.linalg.norm(mouth - tip)))
    return 0


def sidestep(conn: KinovaConnection, step_m: float) -> None:
    """Horizontally away from the user (-Y) by step_m, orientation held -- when the food hides the mouth."""
    p = tool_pose(conn)
    q0 = joints(conn)
    if not all(_ik(conn, (p[0], p[1] - step_m * k / 5, p[2]), tuple(p[3:]), q0) for k in range(1, 6)):
        raise MoveError("the sidestep is not reachable -- not moving")
    goto_cartesian(conn, (p[0], p[1] - step_m, p[2]), tuple(p[3:]), speed=float(DL.sidestep_mps),
                   label="sidestep -Y %.0f mm" % (1000 * step_m), max_tilt=95.0)


# ------------------------------------------------------------------ bite

def bite(conn: KinovaConnection, timeout_s: float) -> int:
    """Wait (read-only) for the bite: tool force above BITE_N for BITE_HOLD_S. 4 = no bite within timeout_s."""
    tel = runlog.Telemetry("bite")

    def F() -> np.ndarray:
        fb = conn.base_cyclic.RefreshFeedback()
        tel.add(fb)
        b = fb.base
        return np.array([b.tool_external_wrench_force_x, b.tool_external_wrench_force_y, b.tool_external_wrench_force_z])
    base = np.median([F() for _ in range(BITE_BASELINE_N)], 0)      # ~1 s at rest
    log.info("    waiting up to %.0f s for the bite (> %.1f N for %.2f s)", timeout_s, BITE_N, BITE_HOLD_S)
    t0 = time.time()
    above = None
    peak = 0.0
    while time.time() - t0 < timeout_s:
        f = np.linalg.norm(F() - base)
        t = time.time() - t0
        peak = max(peak, f)
        if f <= BITE_N:
            above = None
        elif above is None:
            above = t
        elif t - above >= BITE_HOLD_S:
            break
    else:
        log.info("    no bite in %.0f s", timeout_s)
        tel.save(outcome="no bite", baseline_n=base, threshold_n=BITE_N, peak_n=peak)
        runlog.event("no_bite", timeout_s=timeout_s, peak_n=peak)
        return 4
    log.info("    BITE at %.1f s (%.1f N)", above, peak)
    tail = []
    ts = time.time()
    n = BITE_BASELINE_N
    while time.time() - ts < float(DL.bite_settle_s):
        tail.append(F() - base)
        if len(tail) > n and max(np.linalg.norm(x) for x in tail[-n:]) < BITE_N:
            break
    rest = np.linalg.norm(np.median(tail[-n:], 0))
    log.info("    settled; resting shift %.2f N (the food's weight gone: ~0.2-0.4 N)", rest)
    tel.save(outcome="bite", baseline_n=base, threshold_n=BITE_N, peak_n=peak, bite_at_s=above, resting_shift_n=rest)
    runlog.event("bite", at_s=above, peak_n=peak, resting_shift_n=rest)
    return 0


# ------------------------------------------------------------------ retract

def retract(conn: KinovaConnection, validate_only: bool = False) -> None:
    """Away from the mouth straight back along the utensil to where the swing ended (never stopped on torque:
    moving away is the safe direction, and stopping could leave the food in the mouth), then one cuRobo path to
    food_detect_start."""
    go = not validate_only
    p, q = tool_pose(conn), joints(conn)
    T0, R0 = np.array(p[:3]), R_from_kortex(*p[3:])
    if abs(axis(R0)[2]) <= LEVEL_TOL:                  # at the user
        if not STATE.exists():
            raise MoveError("no swing recorded (%s) -- not guessing the way back from the user" % STATE)
        Te = np.array(json.loads(STATE.read_text())["end_tool"])
        L = float(np.linalg.norm(Te - T0))
        if L > float(DL.retract.max_from_swing_end_m):
            raise MoveError("%.0f mm from the swing's end -- not backing off from here" % (1000 * L))
        if L > ARRIVED_M:
            path, err = homotopy(conn, q, lambda s: (T0 + (Te - T0) * s, R0), 40)
            if path is None:
                raise MoveError("back off " + err + " -- nothing moved")
            run_path(conn, path, dps_for(path, L, RETRACT_MPS), "back off %.0f mm" % (1000 * L), go, torque_stop=False)
    curobo_client.travel(conn, scene.overlook_joints(conn), "back to food_detect_start", go=go, dps=RETRACT_HOME_DPS,
                         torque_limit=RETRACT_TORQUE_NM)
    if go:
        STATE.unlink(missing_ok=True)
        runlog.event("retracted")


# ------------------------------------------------------------------ command line

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", metavar="step", choices=["swing", "approach", "sidestep", "bite", "retract"])
    ap.add_argument("--validate-only", action="store_true", help="plan and let the controller validate; do not move")
    ap.add_argument("--face", help="approach: detect_mouth.py output, made at the current pose")
    ap.add_argument("--colour", help="approach: the capture that detection was made from")
    ap.add_argument("--allow-nose", action="store_true", help="approach: estimate the mouth from the nose")
    ap.add_argument("--step", type=float, default=float(DL.sidestep_m), help="sidestep: metres")
    ap.add_argument("--timeout", type=float, default=float(DL.bite_timeout_s), help="bite: seconds")
    a = ap.parse_args()
    log_ = runlog.setup("deliver")
    steps: dict[str, Callable[[KinovaConnection], Any]] = dict(
        swing=lambda c: swing(c, a.validate_only),
        approach=lambda c: approach(c, a.face, a.colour, a.allow_nose, a.validate_only),
        sidestep=lambda c: sidestep(c, a.step),
        bite=lambda c: bite(c, a.timeout),
        retract=lambda c: retract(c, a.validate_only),
    )
    runlog.event("start", command=a.command, args=vars(a))
    with kinova_connection() as conn:
        try:
            rc = steps[a.command](conn)
        except MoveError as e:
            log_.error("\n  STOPPED: %s", e)
            runlog.event("stopped", command=a.command, reason=str(e), rc=1)
            return 1
        except Exception as e:  # noqa: BLE001 -- a controller error mid-sequence: stop, say where
            log_.error("\n  STOPPED on a controller error: %s", " ".join(str(e).split())[:300])
            runlog.event("stopped", command=a.command, reason="controller error: %s" % e, rc=1)
            return 1
    runlog.event("done", command=a.command, rc=rc or 0)
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
