#!/usr/bin/env python3
"""Food delivery, after acquire: swing to the user, present at the mouth, wait for the bite, retract.

    python -m kinova.scripts.deliver swing     [--validate-only]
    python -m kinova.scripts.deliver approach  --face face.json --colour colour.jpg [--allow-nose] [--validate-only]
    python -m kinova.scripts.deliver sidestep  [--step 0.05]
    python -m kinova.scripts.deliver bite      [--timeout 60]
    python -m kinova.scripts.deliver retract   [--validate-only]

Runs in the driver container. `feeding/deliver.sh` orchestrates it with the mouth detection on the host.
Built 2026-09-24 with the user, one leg at a time; each choice below is theirs unless it says otherwise.

swing     from the scoop end: lift 175 mm (so the swing clears the bowl -- at +60 mm it touched it, and the
          user's head sat at the top of the image), then J1 -90 toward the user WHILE retracting the utensil
          along its own axis (in parallel, never one then the other), as far as the arm allows (120 mm). The
          utensil stays level, food side up: a 90 deg camera roll was tried and rejected -- it turns the food
          sideways for the bite. The image is rolled 90 deg instead, and rotated upright in software.
approach  (detected from 50 mm to the user's right -- the swing ends there) straight line, utensil orientation held,
          to the utensil tip 80 mm short of the mouth (80, not 50:
          safe for any cube-sized food piece) and 10 mm lower than the detection (it felt nearer the nose).
          Computed ONLY from a detection made at this pose, newer than its capture, under 2 min old. A
          stale second file once let an approach run to an old reading. Mouth: the mouth box when it is
          anatomically plausible (the detector has returned a full-frame "mouth" with the mouth in plain
          view); else exit 3 so the caller sidesteps and looks again; the nose estimate only when allowed.
sidestep  horizontally AWAY from the user (-Y), orientation held, when the food hides the mouth.
bite      read-only: the controller's tool-force estimate against a baseline at rest, above 1 N for 0.1 s
          (user's threshold; one recorded bite peaked at 5.7 N, non-bite bumps reached 1.0-1.1 N).
retract   (fast: no food, moving away) back off along the utensil to the swing's end pose (does NOT stop on a torque step: moving away
          from the face is the safe direction, and stopping could leave the food in the mouth), reverse the
          swing, go down to the scoop end + 80 mm steering the arm back into the scoop's own shape, then the
          proven joint move to food_detect_start. Without the steering the joint move dips the tip to the
          bowl floor (222 mm); with it, never below ~340 mm.

Exit codes: 0 done, 1 stopped (nothing further moved), 3 mouth not usable (sidestep and look again), 4 no bite.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

import runlog
from kinova import scene
from kinova.connection import kinova_connection
from settings import cfg
from feeding.camera import CAM_IN_TOOL, CXC, CYC, FXC, FYC
from kinova.motion import (R_from_kortex, Rz, _ik, axis, dps_for, fk_tip, homotopy, margin, max_step, run_path,
                           tip_in_tool)
from kinova.scripts.safe_move import Joints, MoveError, fold, goto, goto_cartesian, joints, tool_pose

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

log = runlog.get("deliver")

DL, G = cfg.delivery, cfg.guards.torque_step_nm
LIFT_M = float(DL.lift_m)                         # above the scoop end, before the swing (60 mm touched the bowl)
RETRACT_TRIES = tuple(float(x) for x in DL.retract_tries_m)
SWING_DPS = float(DL.swing_dps)
DETECT_SIDESTEP_M = float(DL.detect_sidestep_m) if cfg.features.detect_from_the_side else 0.0
STOP_M = float(DL.stop_m)                         # utensil tip short of the mouth
AIM_DZ = float(DL.aim_dz_m)                       # below the detected mouth
APPROACH_MPS = float(DL.approach_mps)
MAX_APPROACH_M = float(DL.max_approach_m)
MAX_SHIFT_S = float(DL.detection_max_age_s)       # detection must be this fresh
NOSE_TO_MOUTH = float(DL.nose_to_mouth)
MOUTH_BEHIND_NOSE = float(DL.mouth_behind_nose_m)
BITE_N = float(DL.bite_n)
BITE_HOLD_S = float(DL.bite_hold_s)
# retract carries no food and moves away from the user: fast
RETRACT_MPS = float(DL.retract.back_off_mps)
RETRACT_SWING_DPS = float(DL.retract.swing_dps)
RETRACT_DOWN_DPS = float(DL.retract.down_dps)
RETRACT_HOME_DPS = float(DL.retract.home_dps)
RETRACT_TORQUE_NM = float(G.retract)
SWING_TORQUE_NM = float(G.swing)
APPROACH_TORQUE_NM = float(G.approach)
MIN_TIP_Z = scene.min_tip_z()                     # the return's joint move keeps the utensil tip above this (table-relative)
UP_W, UP_H = 720, 1280         # upright (rotated) image

STATE_DIR = Path(os.environ.get("RAMMP_STATE_DIR", "/tmp"))   # what the steps hand each other, across bites
STATE = STATE_DIR / "deliver_state.json"
# The scoop always ends on the calibrated scoop_end pose, so the swing -- and the retract's reverse swing and
# descent -- are the same joint paths every time: planned once, then reused (planning cost ~10 s of stillness)
SWING_CACHE = STATE_DIR / "deliver_swing_plan.json"
RETRACT_CACHE = STATE_DIR / "deliver_retract_plan.json"


# ------------------------------------------------------------------ what the steps hand each other

@dataclass
class DeliveryState:
    """Written by the swing, read by the retract (STATE): how to retrace the way back."""
    start_tool: list[float]        # tool xyz at the top of the lift, where the swing started
    start_rpy: list[float]         # tool orientation there (the scoop end's)
    retract_m: float               # how far the swing retracted the utensil along its axis
    end_tool: list[float]          # tool xyz where the swing ended (the back-off returns here)
    q80: Joints                    # the arm's shape 80 mm up the lift: the descent steers back into it
    scoop_end_z: float             # tool z at the scoop end

    def save(self) -> None:
        STATE.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls) -> "DeliveryState":
        if not STATE.exists():
            raise MoveError("no swing recorded (%s) -- cannot retrace" % STATE)
        return cls(**json.loads(STATE.read_text()))


def _cached(cache: Path, q_start: Sequence[float], tol: float = 1.0) -> dict | None:
    """A cached plan, if it was planned from (within tol deg of) q_start."""
    try:
        c = json.loads(cache.read_text())
    except (OSError, ValueError):
        return None
    if max_step(c["q_start"], q_start) > tol:
        return None
    return c


@dataclass
class SwingPlan:
    """The lift + swing + step-right joint path, cached (SWING_CACHE) for the next bite."""
    q_start: Joints
    path: list[Joints]
    retract_m: float
    q80: Joints

    def save(self) -> None:
        SWING_CACHE.write_text(json.dumps(asdict(self)))

    @classmethod
    def cached(cls, q_start: Joints) -> "SwingPlan | None":
        c = _cached(SWING_CACHE, q_start)
        return cls(**c) if c else None


@dataclass
class RetractPlan:
    """The reverse swing and the descent, cached (RETRACT_CACHE) for the next bite."""
    q_start: Joints
    path2: list[Joints]            # reverse swing
    path3: list[Joints]            # down to the scoop end + 80 mm
    lo: float                      # lowest utensil-tip z on the final joint move home

    def save(self) -> None:
        RETRACT_CACHE.write_text(json.dumps(asdict(self)))

    @classmethod
    def cached(cls, q_start: Joints) -> "RetractPlan | None":
        c = _cached(RETRACT_CACHE, q_start, tol=1.5)
        return cls(**c) if c else None


def swing_tip(tipS: np.ndarray, axS: np.ndarray, retract_m: float, u: float) -> np.ndarray:
    """The swing's utensil-tip path at progress u: +90 deg about world z (== J1 -90), retraction front-loaded."""
    return Rz(90.0 * u) @ (tipS - retract_m * (1 - (1 - u) ** 2) * axS)


# ------------------------------------------------------------------ swing

def swing(conn: KinovaConnection, validate_only: bool = False) -> None:
    """Lift, swing (retracting), and step to the user's right -- ONE continuous validated trajectory, no
    stops between them (user, 2026-09-24: 'painfully slow ... these need not be waited for')."""
    p = tool_pose(conn)
    q0 = joints(conn)
    R0 = R_from_kortex(*p[3:])
    if axis(R0)[1] < 0.99:
        raise MoveError("utensil is not pointing at the user (+Y) -- run from the end of the scoop")
    # The swing turns about the BASE axis: it was built and validated for scoops ending on the base's
    # -Y side (y -0.04 .. -0.10). From a bowl in front of the user (+Y) the same turn would sweep the
    # utensil through where the user's face is. Refuse anything outside the validated region.
    y_lo, y_hi = cfg.bowl.swing_start_y_range_m
    if not y_lo < p[1] < y_hi:
        raise MoveError("scoop ended at y %+.3f, outside the region the swing was validated for (%.2f..%.2f): "
                        "from here the swing could sweep into the user -- not moving" % (p[1], y_lo, y_hi))
    plan = SwingPlan.cached(q0)
    if plan:
        plan.path = [q0] + plan.path[1:]
        log.info("    lift + swing + step right: planned before, reused (joint margin >= %.1f deg)",
                 min(margin(q) for q in plan.path))
    else:
        plan = _plan_swing(conn, q0, np.array(p[:3]), R0)
        plan.save()
    _run_swing(conn, plan, p, validate_only)


def _plan_swing(conn: KinovaConnection, q0: Joints, T0: np.ndarray, R0: np.ndarray) -> SwingPlan:
    # 1. lift: a steered joint path -- from a scoop that ends nearer the base, the controller's own straight
    #    lift runs a joint out at +70 mm; steering the redundancy reaches 175
    n_lift = 35
    lift, err = homotopy(conn, q0, lambda s: (T0 + np.array([0, 0, LIFT_M * s]), R0), n_lift)
    if lift is None:
        raise MoveError("the %.0f mm lift is not reachable (%s) -- nothing moved" % (1000 * LIFT_M, err))
    q80 = lift[round(n_lift * 0.080 / LIFT_M)]                  # the return's arm shape, for retract
    # 2. swing toward the user while retracting along the utensil, as far as the joints allow
    tip0 = T0 + np.array([0, 0, LIFT_M]) + R0 @ tip_in_tool()
    ax0 = axis(R0)
    best = None
    for retract_m in RETRACT_TRIES:
        def at(s: float, r: float = retract_m) -> tuple[np.ndarray, np.ndarray]:
            return swing_tip(tip0, ax0, r, s) - Rz(90.0 * s) @ R0 @ tip_in_tool(), Rz(90.0 * s) @ R0
        path, err = homotopy(conn, lift[-1], at, 60)
        if path and min(margin(q) for q in path) < min(3.0, margin(lift[-1]) - 0.5):
            err = "joint-limit margin only %.1f deg" % min(margin(q) for q in path)
            path = None
        if path:
            best = (retract_m, path)
            break
    if best is None:
        raise MoveError("no swing toward the user from here -- nothing moved")
    retract_m, swing_path = best
    # 3. to the user's right, so the food is not between the camera and the mouth
    Ts, Rs = (swing_tip(tip0, ax0, retract_m, 1.0) - Rz(90.0) @ R0 @ tip_in_tool(), Rz(90.0) @ R0)
    side, err = homotopy(conn, swing_path[-1], lambda s: (Ts - np.array([0, DETECT_SIDESTEP_M * s, 0]), Rs), 8)
    if side is None:
        log.info("    (the %.0f mm step to the user's right is not reachable -- detecting from the swing's end)",
                 1000 * DETECT_SIDESTEP_M)
        side = [swing_path[-1]]
    path = lift + swing_path[1:] + side[1:]
    log.info("    lift %.0f mm + swing retracting %.0f mm + %.0f mm to the user's right: one trajectory, joint margin >= %.1f deg",
             1000 * LIFT_M, 1000 * retract_m, 1000 * DETECT_SIDESTEP_M if len(side) > 1 else 0,
             min(margin(q) for q in path))
    return SwingPlan(q_start=q0, path=path, retract_m=retract_m, q80=q80)


def _run_swing(conn: KinovaConnection, plan: SwingPlan, start: Sequence[float], validate_only: bool) -> None:
    run_path(conn, plan.path, SWING_DPS, "lift + swing + step right", not validate_only, torque_limit=SWING_TORQUE_NM)
    if validate_only:
        return
    end = tool_pose(conn)          # the retract's back-off returns here
    st = DeliveryState(start_tool=[start[0], start[1], start[2] + LIFT_M], start_rpy=list(start[3:]),
                       retract_m=plan.retract_m, end_tool=list(end[:3]), q80=plan.q80, scoop_end_z=start[2])
    st.save()
    runlog.event("swing", retract_m=plan.retract_m, end_tool=end)


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
    if nw * nh > 0.03 * UP_W * UP_H:
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
    if (a2 - a1) * (b2 - b1) > 0.03 * UP_W * UP_H:
        return "mouth box covers %.0f%% of the frame" % (100 * (a2 - a1) * (b2 - b1) / (UP_W * UP_H))
    if not 0.4 * nh < mcy - ncy < 1.5 * nh:
        return "mouth box %.2f nose heights below the nose" % ((mcy - ncy) / nh)
    if abs(mcx - ncx) > nw:
        return "mouth box %.2f nose widths sideways" % (abs(mcx - ncx) / nw)
    if n.get("depth_m") and abs(m["depth_m"] - n["depth_m"]) > 0.030:
        return "mouth depth %.0f mm off the nose's" % (1000 * abs(m["depth_m"] - n["depth_m"]))
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
    p = tool_pose(conn)
    q0 = joints(conn)
    T0 = np.array(p[:3])
    R = R_from_kortex(*p[3:])
    ax = axis(R)
    if abs(ax[2]) > 0.1:
        raise MoveError("utensil is not level -- run the swing first")
    mouth = T0 + R @ (CAM_IN_TOOL + np.array([-cam[0], -cam[1], cam[2]]))
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
    if L < 0.003:
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
    goto_cartesian(conn, (p[0], p[1] - step_m, p[2]), tuple(p[3:]), speed=0.02,
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
    base = np.median([F() for _ in range(40)], 0)                    # ~1 s at rest
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
    while time.time() - ts < 10:
        tail.append(F() - base)
        if len(tail) > 40 and max(np.linalg.norm(x) for x in tail[-40:]) < BITE_N:
            break
    rest = np.linalg.norm(np.median(tail[-40:], 0))
    log.info("    settled; resting shift %.2f N (the food's weight gone: ~0.2-0.4 N)", rest)
    tel.save(outcome="bite", baseline_n=base, threshold_n=BITE_N, peak_n=peak, bite_at_s=above, resting_shift_n=rest)
    runlog.event("bite", at_s=above, peak_n=peak, resting_shift_n=rest)
    return 0


# ------------------------------------------------------------------ retract

def retract(conn: KinovaConnection, validate_only: bool = False) -> None:
    """Back along the way the swing came, then down and home to food_detect_start (see the module docstring)."""
    st = DeliveryState.load()
    go = not validate_only
    p = tool_pose(conn)
    q0 = joints(conn)
    T0 = np.array(p[:3])
    R0 = R_from_kortex(*p[3:])
    if abs(axis(R0)[2]) > 0.1:
        raise MoveError("utensil is not level -- not at the user")
    RS = R_from_kortex(*st.start_rpy)
    axS = axis(RS)
    TS = np.array(st.start_tool)
    # resuming after legs 1-2 (utensil already turned back, at the swing's start): straight to leg 3.
    # Leg 1 from here would "back off" toward the swing's END, i.e. toward the user.
    resumed = axis(R0) @ axS > 0.99 and np.linalg.norm(T0 - TS) < 0.020
    if not resumed and axis(R0) @ axS > 0.5:
        raise MoveError("utensil is neither at the user nor back at the swing's start -- not guessing a path")
    Te = np.array(st.end_tool)
    L = 0.0 if resumed else np.linalg.norm(Te - T0)
    if L > 0.400:
        raise MoveError("%.0f mm from the swing's end -- not retracing from here" % (1000 * L))
    # 1. back off along the utensil, to where the swing ended
    if resumed:
        log.info("    legs 1-2 already done (at the swing's start): going home")
        q1 = q0
    else:
        q1 = _back_off(conn, q0, T0, R0, Te, L, go)
    # 2. reverse the swing
    rc = RetractPlan.cached(q1)
    if resumed:
        q2, path2 = q0, None
    else:
        q2, path2 = _reverse_swing(conn, st, q1, T0, R0, Te, L, RS, TS, rc, go)
    # 3. down to the scoop end + 80 mm in the scoop's own arm shape, then the proven joint move
    _down_and_home(conn, st, q1, q2, path2, RS, TS, rc, resumed, go)


def _back_off(conn: KinovaConnection, q0: Joints, T0: np.ndarray, R0: np.ndarray, Te: np.ndarray, L: float,
              go: bool) -> Joints:
    """Leg 1: straight back along the utensil to the swing's end. Never stops on torque (moving away)."""
    if L <= 0.003:
        return q0
    path, err = homotopy(conn, q0, lambda s: (T0 + (Te - T0) * s, R0), 40)
    if path is None:
        raise MoveError("back off " + err + " -- nothing moved")
    run_path(conn, path, dps_for(path, L, RETRACT_MPS), "back off %.0f mm" % (1000 * L), go, torque_stop=False)
    return joints(conn) if go else path[-1]


def _reverse_swing(conn: KinovaConnection, st: DeliveryState, q1: Joints, T0: np.ndarray, R0: np.ndarray,
                   Te: np.ndarray, L: float, RS: np.ndarray, TS: np.ndarray, rc: RetractPlan | None,
                   go: bool) -> tuple[Joints, list[Joints]]:
    """Leg 2: the swing backwards, from wherever along it the arm is (u0), absorbing any offset e."""
    tipv_t = tip_in_tool()
    tipS = TS + RS @ tipv_t
    axS = axis(RS)
    R1 = R0 if not go else R_from_kortex(*tool_pose(conn)[3:])
    ax1 = axis(R1)
    th = math.degrees(math.atan2(np.cross(axS, ax1)[2], axS @ ax1))
    u0 = th / 90.0
    tip1 = (Te if L > 0.003 else T0) + R1 @ tipv_t
    e = tip1 - swing_tip(tipS, axS, st.retract_m, u0)

    def at2(s: float) -> tuple[np.ndarray, np.ndarray]:
        u = u0 * (1 - s)
        R = Rz(90.0 * u) @ RS
        return swing_tip(tipS, axS, st.retract_m, u) + (1 - s) * e - R @ tipv_t, R
    if rc:
        path2 = [q1] + rc.path2[1:]
        log.info("    reverse swing + descent: planned before, reused")
        run_path(conn, path2, RETRACT_SWING_DPS, "reverse swing", go, from_here=go or L <= 0.003,
                 torque_limit=RETRACT_TORQUE_NM)
    else:
        path2, err = homotopy(conn, q1, at2, 60)
        if path2 is None:
            raise MoveError("reverse swing " + err)
        run_path(conn, path2, RETRACT_SWING_DPS, "reverse swing %.0f deg" % th, go, from_here=go or L <= 0.003,
                 torque_limit=RETRACT_TORQUE_NM)
    return (joints(conn) if go else path2[-1]), path2


def _down_and_home(conn: KinovaConnection, st: DeliveryState, q1: Joints, q2: Joints, path2: list[Joints] | None,
                   RS: np.ndarray, TS: np.ndarray, rc: RetractPlan | None, resumed: bool, go: bool) -> None:
    """Leg 3: down to the scoop end + 80 mm steering into the lift's arm shape, then the joint move home --
    refused if that joint move would take the utensil tip below MIN_TIP_Z."""
    Ta = TS
    Td = np.array([TS[0], TS[1], st.scoop_end_z + 0.080])
    qd = scene.overlook_joints(conn)
    if rc and not resumed:
        path3, lo = [q2] + rc.path3[1:], rc.lo
    else:
        path3, err = homotopy(conn, q2, lambda s: (Ta + (Td - Ta) * s, RS), 40, toward=st.q80)
        if path3 is None:
            raise MoveError("descent " + err)
        lo = min(fk_tip(conn, [fold(x + fold(y - x) * k / 20) for x, y in zip(path3[-1], qd)])[2] for k in range(21))
        if not resumed and go:
            RetractPlan(q_start=q1, path2=path2, path3=path3, lo=lo).save()
    log.info("    return: utensil tip never below %.0f mm", 1000 * lo)
    if lo < MIN_TIP_Z:
        raise MoveError("the return would dip the utensil tip to %.0f mm -- stopped" % (1000 * lo))
    run_path(conn, path3, RETRACT_DOWN_DPS, "down to scoop end + 80 mm", go, from_here=go or resumed,
             torque_limit=RETRACT_TORQUE_NM)
    if not go:
        return
    try:
        goto(conn, qd, speed=RETRACT_HOME_DPS, label="-> food_detect_start", max_tilt=95.0)
    except MoveError as e:
        # braking at the end of this faster move reads as a torque step (J2 -7.8 -> -4.8 N.m on
        # 2026-09-24, nothing touched): if the arm is already there, it arrived
        off = max_step(joints(conn), qd)
        if "torque" not in str(e) or off > 0.2:
            raise
        log.info("    -> food_detect_start: arrived (%.2f deg off); the torque step was the braking", off)
    STATE.unlink()
    runlog.event("retracted", lowest_tip_z_m=lo)


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
