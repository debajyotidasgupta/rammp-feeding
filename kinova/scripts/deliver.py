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
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

import runlog
from kinova import scene
from kinova.connection import kinova_connection
from settings import cfg
from feeding.camera import COLOUR_SIZE, CXC, CYC, FXC, FYC, cam_to_tool
from kinova.frames import R_from_kortex, Rz, axis, kortex_from_R
from kinova.motion import MAX_STEP_DEG, _ik, dps_for, fk_tip, homotopy, margin, max_step, run_path, steered_guess, tip_in_tool
from kinova import curobo_client
from kinova.workspace import mouth_now
from kinova.scripts.safe_move import Joints, MoveError, TorqueGuard, fold, goto_cartesian, joints, tool_pose

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

log = runlog.get("deliver")

DL, G = cfg.delivery, cfg.guards.torque_step_nm
LIFT_M = float(DL.lift_m)
RETRACT_TRIES = tuple(float(x) for x in DL.retract_tries_m)
SWING_DPS = float(DL.swing_dps)
STANDOFF_M = None if DL.swing_standoff_m is None else float(DL.swing_standoff_m)
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
BACK_OFF_M = None if DL.retract.back_off_m is None else float(DL.retract.back_off_m)
RETRACT_TORQUE_NM = float(G.retract)
SWING_TORQUE_NM = float(G.swing)
APPROACH_TORQUE_NM = float(G.approach)
UP_W, UP_H = COLOUR_SIZE[1], COLOUR_SIZE[0]       # the upright (rotated) image the face detector sees
MC = DL.mouth_check
TR = DL.track
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
        if max_step(c["q_start"], q0) <= float(DL.goal_cache_tol_deg) and np.allclose(c["mouth"], mouth_now(), atol=0.005):
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
    mouth = np.array(mouth_now())
    if STANDOFF_M is not None:                       # on toward the mouth: the tip STANDOFF_M short of it, on its line
        Tm = mouth - STANDOFF_M * axis(Rs) + np.array([0, 0, AIM_DZ]) - Rs @ tip_t
        near, err = homotopy(conn, turn[-1], lambda s: (Ts + (Tm - Ts) * s, Rs), 30)
        if near is None:
            raise MoveError("no swing to %.0f mm short of the mouth (%s) -- nothing moved" % (1000 * STANDOFF_M, err))
        turn, Ts = near, Tm
    side, _ = homotopy(conn, turn[-1], lambda s: (Ts - np.array([0, DETECT_SIDESTEP_M * s, 0]), Rs), 8)
    goal = (side or turn)[-1]
    log.info("    swing goal: lift %.0f mm, turn retracting %.0f mm,%s %.0f mm to the user's right", 1000 * LIFT_M,
             1000 * retract_m, "" if STANDOFF_M is None else " on to %.0f mm short of the mouth," % (1000 * STANDOFF_M),
             1000 * DETECT_SIDESTEP_M if side else 0)
    GOAL_CACHE.write_text(json.dumps(dict(q_start=q0, goal=goal, mouth=[float(v) for v in mouth])))
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
    if f.get("depth_from_size"):
        log.info("    the mouth's range is only from the face's size (no depth on it) -- not approaching; look again")
        runlog.event("depth_implausible", why="range from the face's size")
        return 5
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
    # the utensil turned toward the mouth on the way, at most delivery.track.max_turn_deg (a far or off-axis user is
    # reached with the wrist turned a little, not the arm stretched along a fixed heading); stop_m short along it
    tip0 = T0 + R @ tip_in_tool()
    dh = np.array([mouth[0] - tip0[0], mouth[1] - tip0[1], 0.0])
    yaw = float(np.degrees(np.arctan2(ax[0] * dh[1] - ax[1] * dh[0], ax[0] * dh[0] + ax[1] * dh[1])))
    yaw = float(np.clip(yaw, -float(TR.max_turn_deg), float(TR.max_turn_deg)))
    Rg = Rz(yaw) @ R
    goal_tip = mouth - stop * axis(Rg) + np.array([0, 0, AIM_DZ])
    Tg = goal_tip - Rg @ tip_in_tool()
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
    n = max(40, int(np.ceil(L / float(DL.approach_step_m))))      # fine steps: a long approach stays continuous
    path, err = homotopy(conn, q0, lambda s: (T0 + (Tg - T0) * s, Rz(yaw * s) @ R), n)
    if path is None:
        raise MoveError("approach " + err + " -- not moving")
    run_path(conn, path, dps_for(path, L, APPROACH_MPS), "approach at %.0f mm/s" % (1000 * APPROACH_MPS),
             not validate_only, torque_limit=APPROACH_TORQUE_NM)
    if not validate_only:
        pn = tool_pose(conn)
        tip = np.array(pn[:3]) + R_from_kortex(*pn[3:]) @ tip_in_tool()
        log.info("    utensil tip %.0f mm from the detected mouth", 1000 * np.linalg.norm(mouth - tip))
        runlog.event("presented", tip=tip, tip_to_mouth_m=float(np.linalg.norm(mouth - tip)))
    return 0


# ------------------------------------------------------------------ present: tracking

def _track(start: bool = False) -> dict:
    """The face server's latest tracked face (POST: start tracking)."""
    req = urllib.request.Request("http://127.0.0.1:%d/track" % int(cfg.deployment.servers.face_port),
                                 data=b"{}" if start else None, method="POST" if start else "GET")
    with urllib.request.urlopen(req, timeout=float(TR.poll_timeout_s)) as r:
        return json.loads(r.read())


class _Watchdog:
    """kinova/scripts/twist_watchdog.py: stops the arm from its own session if this loop stops beating."""

    def __init__(self):
        self.p = subprocess.Popen([sys.executable, "-m", "kinova.scripts.twist_watchdog", str(float(TR.watchdog_s))],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        if (self.p.stdout.readline() or "").strip() != "ready":
            self.p.kill()
            raise MoveError("the watchdog did not start -- not moving")

    def beat(self) -> None:
        self.p.stdin.write("b\n"); self.p.stdin.flush()

    def close(self) -> None:
        try:
            self.p.stdin.write("done\n"); self.p.stdin.flush(); self.p.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()


def _joint_speeds(conn: KinovaConnection, dps) -> None:
    from kortex_api.autogen.messages import Base_pb2
    cmd = Base_pb2.JointSpeeds()
    for i, w in enumerate(dps):
        js = cmd.joint_speeds.add()
        js.joint_identifier, js.value = i, float(w)
    conn.base.SendJointSpeedsCommand(cmd)


def _step(conn: KinovaConnection, q, T, R, v, wz: float) -> tuple[list[float], str | None]:
    """Joint speeds (deg/s) carrying the tool by v (m/s) and turning it wz (deg/s) about the vertical, through our
    own IK (limit-steered, from the joints now) for the pose TR.lookahead_s ahead -- so the arm keeps its shape --
    and why not, if that pose is out of reach, a jump, or within plan_min_margin_deg of a joint limit."""
    tau = float(TR.lookahead_s)
    if float(np.linalg.norm(v)) < 1e-4 and abs(wz) < 1e-3:
        return [0.0] * 7, None
    qn = _ik(conn, T + np.asarray(v) * tau, kortex_from_R(Rz(wz * tau) @ R), steered_guess(q))
    if qn is None:
        return [0.0] * 7, "the next pose is out of reach"
    if max_step(qn, q) > MAX_STEP_DEG:
        return [0.0] * 7, "a joint jump of %.0f deg" % max_step(qn, q)
    if margin(qn) < float(cfg.motion.plan_min_margin_deg):
        return [0.0] * 7, "it would come within %.0f deg of a joint limit" % margin(qn)
    w = np.array([fold(b - a) for a, b in zip(q, qn)]) / tau
    top = float(np.abs(w).max())
    if top > float(TR.max_joint_dps):
        w *= float(TR.max_joint_dps) / top                 # slower, the same direction
    return [float(x) for x in w], None


def track(conn: KinovaConnection, validate_only: bool = False) -> int:
    """Present: the utensil tip in one continuous motion to STOP_M short of the mouth, steered live by the wrist
    camera (the face server's /track, every frame) -- wherever the user is in its view and in reach, following the
    head as it moves. At TR.rate_hz, a tool velocity toward the goal -- driven as JOINT speeds from our own IK
    (kinova.motion: continuous, limit-steered; a pose that nears a limit is refused before it is sent): up to far_mps, easing to near_mps within
    slow_from_m of the mouth, accelerations limited (every slowing too); the utensil kept level and turned toward
    the mouth, at most max_turn_deg from where the swing pointed it (a big turn is a big wrist rotation near the
    face; the tip goes stop_m short of the mouth either way). Between the tracker's frames it keeps on toward the last measured mouth (fixed in the base frame;
    a frame is placed with the arm's pose at that frame's time); no fresh face for hold_s: it eases to a hold. The
    stop is set only from depth on the lips: while the range is from the face's size, the tip holds size_hold_m
    short. Stops (Base.Stop): arrived; the face lost past lost_timeout_s (3); a torque step, a joint near its limit,
    the utensil tilting, more than max_travel_m or timeout_s (refused, 1). A watchdog process stops the arm if this
    loop dies (joint speed commands, like twists, run until replaced). 0 = presented."""
    if validate_only:
        _track(start=True)
        log.info("    validate-only: the face tracker answers; the tracked approach is not simulated here")
        return 0
    _track(start=True)
    dog = _Watchdog() if TR.watchdog else None
    hist: deque = deque(maxlen=200)                      # (time, tool position, R): the camera's pose at a frame's time
    guard = TorqueGuard(limit=APPROACH_TORQUE_NM)
    dt, v = 1.0 / float(TR.rate_hz), np.zeros(3)
    t0 = seen = time.time()
    mouth, source, p0, why = None, "", None, None
    cycles = held = 0                                    # control cycles, and those commanding no motion
    q_start = joints(conn)
    tel: dict[str, list] = dict(t=[], q=[], tool=[], v=[], mouth=[])   # every cycle, for the run record
    ax0 = None                                           # the utensil's heading at the start: turns stay within max_turn_deg
    progress: deque = deque()                            # (time, tool position, commanded speed): a stall -- told
                                                          # to move and not moving -- is the edge of reach
    try:
        while True:
            now = time.time()
            fb = conn.base_cyclic.RefreshFeedback()
            b = fb.base
            T = np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z])
            R = R_from_kortex(b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z)
            hist.append((now, T, R))
            tel["t"].append(now); tel["tool"].append(list(T))
            tel["q"].append([fold(a.position) for a in fb.actuators]); tel["v"].append(list(v))
            tel["mouth"].append(list(mouth) if mouth is not None else [np.nan] * 3)
            p0 = T if p0 is None else p0
            if dog:
                dog.beat()
            hit = guard.update([a.torque for a in fb.actuators])
            q = [fold(a.position) for a in fb.actuators]
            ax = axis(R)
            if hit:
                why = "joint %d torque step" % (hit[0] + 1)
            elif margin(q) < float(cfg.motion.plan_min_margin_deg) - 1.0:          # a backstop: _step refuses first
                why = "a joint within %.0f deg of its limit" % margin(q)
            elif abs(ax[2]) > float(TR.max_tilt_cos):
                why = "the utensil tilted (axis z %.2f)" % ax[2]
            elif np.linalg.norm(T - p0) > float(TR.max_travel_m):
                why = "moved %.0f mm without arriving" % (1000 * np.linalg.norm(T - p0))
            elif now - t0 > float(TR.timeout_s):
                why = "no arrival in %.0f s" % float(TR.timeout_s)
            if why:
                raise MoveError("tracking stopped: " + why)
            try:
                f = _track()
            except OSError:
                f = {}
            if f.get("face") and now - float(f["t"]) < float(TR.frame_max_age_s):
                tf, Tf, Rf = min(hist, key=lambda h: abs(h[0] - float(f["t"])))     # the camera when the frame was taken
                mouth, source, seen = Tf + Rf @ cam_to_tool(f["cam_xyz"]), f["source"], now
            elif now - seen > float(TR.lost_timeout_s):
                conn.base.Stop()
                log.info("    the face lost for %.1f s -- stopped" % (now - seen))
                runlog.event("track_lost", tool=list(T))
                return 3
            elif mouth is None:                                # no face yet: keep still, keep looking
                _joint_speeds(conn, [0.0] * 7)
                time.sleep(max(0.0, dt - (time.time() - now)))
                continue
            hold = now - seen > float(TR.hold_s)               # between frames: on toward the last measured mouth
            tip = T + R @ tip_in_tool()
            d = mouth - tip
            dh = np.array([d[0], d[1], 0.0]) / max(1e-9, np.hypot(d[0], d[1]))
            stop = STOP_M if source == "depth" else max(STOP_M, float(TR.size_hold_m))
            e = mouth - stop * dh + np.array([0, 0, AIM_DZ]) - tip
            dist, to_mouth = float(np.linalg.norm(e)), float(np.linalg.norm(d))
            if source == "depth" and dist < float(TR.arrive_m):
                break
            progress.append((now, T, float(np.linalg.norm(v))))
            if now - progress[0][0] > float(TR.stall_s):
                moved = float(np.linalg.norm(T - progress[0][1]))
                asked = float(np.mean([x[2] for x in progress])) * (now - progress[0][0])
                if asked > 3 * float(TR.stall_m) and moved < float(TR.stall_m):
                    raise MoveError("tracking stopped: told to move %.0f mm in %.1f s, moved %.0f -- the edge of "
                                    "reach? come closer" % (1000 * asked, now - progress[0][0], 1000 * moved))
                while progress and now - progress[0][0] > float(TR.stall_s):
                    progress.popleft()
            near = float(np.clip((to_mouth - STOP_M) / max(1e-6, float(TR.slow_from_m) - STOP_M), 0.0, 1.0))
            vmax = float(TR.near_mps) + (float(TR.far_mps) - float(TR.near_mps)) * near
            want = np.zeros(3) if hold else e / max(1e-9, dist) * min(vmax, float(TR.gain_per_s) * dist)
            dv = want - v
            lim = float(TR.accel_mps2) * dt
            v = v + (dv * lim / np.linalg.norm(dv) if np.linalg.norm(dv) > lim else dv)
            ah = np.array([ax[0], ax[1], 0.0]) / max(1e-9, np.hypot(ax[0], ax[1]))
            ax0 = ah if ax0 is None else ax0
            h0 = float(np.degrees(np.arctan2(ax0[1], ax0[0])))
            turned = float(np.degrees(np.arctan2(ax0[0] * ah[1] - ax0[1] * ah[0], ax0 @ ah)))
            want_turn = fold(float(np.degrees(np.arctan2(dh[1], dh[0]))) - h0)
            want_turn = float(np.clip(want_turn, -float(TR.max_turn_deg), float(TR.max_turn_deg)))
            yaw = want_turn - turned                            # toward the mouth, within max_turn_deg of the start
            cycles += 1
            held += float(np.linalg.norm(v)) < 0.005
            wz = float(np.clip(float(TR.yaw_gain_per_s) * yaw, -float(TR.max_yaw_dps), float(TR.max_yaw_dps)))
            dps, stop = _step(conn, q, T, R, v, wz)
            if stop:
                raise MoveError("tracking stopped: %s -- the edge of reach from here? come closer" % stop)
            _joint_speeds(conn, dps)
            time.sleep(max(0.0, dt - (time.time() - now)))
    finally:
        conn.base.Stop()
        if dog:
            dog.close()
        runlog.save_arrays("track", **{k: np.array(x, float) for k, x in tel.items()})
        p = tool_pose(conn)
        Tn, Rn = np.array(p[:3]), R_from_kortex(*p[3:])
        # the retract backs off from here straight along the utensil, then plans its way home
        STATE.write_text(json.dumps(dict(end_tool=list(Tn - float(TR.back_off_m) * axis(Rn)))))
        if mouth is not None:
            _keep_mouth(mouth)
    tipn = Tn + Rn @ tip_in_tool()
    log.info("    presented: utensil tip (%.3f %.3f %.3f), %.0f mm from the mouth (%.3f %.3f %.3f) (tracked, %.1f s, "
             "%.0f mm travelled; %d of %d cycles under 5 mm/s; joints moved at most %.0f deg, J%d)", *tipn,
             1000 * np.linalg.norm(mouth - tipn), *mouth, time.time() - t0, 1000 * np.linalg.norm(Tn - p0), held, cycles,
             max_step(joints(conn), q_start), 1 + int(np.argmax([abs(fold(a - b)) for a, b in zip(joints(conn), q_start)])))
    runlog.event("presented", tip=tipn, mouth_base=mouth, tip_to_mouth_m=float(np.linalg.norm(mouth - tipn)), tracked=True)
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
    """One continuous move away from the user: straight back along the utensil (back_off_m of the way to where the
    swing ended), on in the same trajectory toward the base's axis, level, to tuck_radii_m -- unless that nears
    the mouth -- never stopped on torque (moving away is the safe direction, and stopping could leave the food in
    the mouth); then one cuRobo path to food_detect_start keeping the tip no further out from the base than where
    it starts or ends (+ reach_margin_m): never sweeping wide, past people beside the arm."""
    go = not validate_only
    p, q = tool_pose(conn), joints(conn)
    T0, R0 = np.array(p[:3]), R_from_kortex(*p[3:])
    path, Tn = [q], T0                                  # the horizontal retract: one joint path, run once
    if abs(axis(R0)[2]) <= LEVEL_TOL:                  # at the user: first straight back along the utensil
        if not STATE.exists():
            raise MoveError("no swing recorded (%s) -- not guessing the way back from the user" % STATE)
        Te = np.array(json.loads(STATE.read_text())["end_tool"])
        L = float(np.linalg.norm(Te - T0))
        if L > float(DL.retract.max_from_swing_end_m):
            raise MoveError("%.0f mm from the swing's end -- not backing off from here" % (1000 * L))
        if BACK_OFF_M is not None and L > BACK_OFF_M:   # only this far back; cuRobo takes it from there
            Te, L = T0 + (Te - T0) * (BACK_OFF_M / L), BACK_OFF_M
        if L > ARRIVED_M:
            back, err = homotopy(conn, q, lambda s: (T0 + (Te - T0) * s, R0), 40)
            if back is None:
                raise MoveError("back off " + err + " -- nothing moved")
            path, Tn = back, Te
    # ... then on in toward the base's axis, level at its height, in the same motion (not past the mouth)
    tip = Tn + R0 @ tip_in_tool()
    r = float(np.hypot(tip[0], tip[1]))
    mouth = np.array(mouth_now())
    for tuck in (float(x) for x in DL.retract.tuck_radii_m):     # the closest to the base the arm can pull in to
        if r <= tuck + ARRIVED_M:
            break
        Tt = Tn - np.array([tip[0], tip[1], 0.0]) * (r - tuck) / r
        line = [tip + (Tt - Tn) * k / 20 for k in range(21)]
        if min(np.linalg.norm(x - mouth) for x in line) < np.linalg.norm(tip - mouth) - ARRIVED_M:
            log.info("    pulling in would pass nearer the mouth -- straight to the planned path")
            break
        more, err = homotopy(conn, path[-1], lambda s, Tt=Tt: (Tn + (Tt - Tn) * s, R0), 40)
        if more is None:
            log.info("    pulling in to %.0f mm from the base: %s", 1000 * tuck, err)
            continue
        path, r = path + more[1:], tuck
        Tn = Tt
        break
    start = None
    if len(path) > 1:
        L = float(np.linalg.norm(Tn - T0))
        run_path(conn, path, dps_for(path, L, RETRACT_MPS), "back off and in toward the arm, %.0f mm"
                 % (1000 * L), go, torque_stop=False)   # moving away from the user: never stopped on torque
        start = None if go else path[-1]
    goal = scene.overlook_joints(conn)
    r_goal = float(np.hypot(*fk_tip(conn, goal)[:2]))
    curobo_client.travel(conn, goal, "back to food_detect_start", go=go, dps=RETRACT_HOME_DPS,
                         torque_limit=RETRACT_TORQUE_NM, start=start,
                         max_reach_m=max(r, r_goal) + float(DL.retract.reach_margin_m))
    if go:
        STATE.unlink(missing_ok=True)
        runlog.event("retracted")


# ------------------------------------------------------------------ command line

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", metavar="step", choices=["swing", "track", "approach", "sidestep", "bite", "retract"])
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
        track=lambda c: track(c, a.validate_only),
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
