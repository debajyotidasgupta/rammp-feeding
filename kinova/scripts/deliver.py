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
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from kortex_api.autogen.messages import Base_pb2

from kinova.connection import kinova_connection
from kinova.scripts.acquire import (CAM_IN_TOOL, LIMITS, R_from_kortex, _ik, _timing, _waypoints, cal,
                                    kortex_from_R, tip_in_tool)
from kinova.scripts.safe_move import MoveError, TorqueGuard, _events, fold, goto, goto_cartesian, joints, tool_pose

STATE = Path("/tmp/deliver_state.json")
# The scoop always ends on the calibrated scoop_end pose, so the swing -- and the retract's reverse swing and
# descent -- are the same joint paths every time: planned once, then reused (planning cost ~10 s of stillness)
SWING_CACHE = Path("/tmp/deliver_swing_plan.json")
RETRACT_CACHE = Path("/tmp/deliver_retract_plan.json")


def _cached(cache, q_start, tol=1.0):
    try:
        c = json.loads(cache.read_text())
    except (OSError, ValueError):
        return None
    if max(abs(fold(x - y)) for x, y in zip(c["q_start"], q_start)) > tol:
        return None
    return c
LIFT_M = 0.175                 # above the scoop end, before the swing
RETRACT_TRIES = (0.120, 0.100, 0.080, 0.060, 0.040, 0.0)   # 150 never fit
SWING_DPS = 20.0               # was 10 (user: faster); the food sits level on the utensil throughout
DETECT_SIDESTEP_M = 0.050      # after the swing, to the user's right (-Y): the food otherwise sits between
                               # the camera and the mouth (user, 2026-09-24)
STOP_M = 0.080                 # utensil tip short of the mouth
AIM_DZ = -0.010                # below the detected mouth
APPROACH_MPS = 0.035           # was 20 mm/s; still eases in and out, and stops 80 mm short
MAX_APPROACH_M = 0.300
MAX_SHIFT_S = 120.0            # detection must be this fresh
NOSE_TO_MOUTH = 0.84           # nose heights, upright image (3 clean detections: 0.81-0.87)
MOUTH_BEHIND_NOSE = 0.005      # m (measured 0-13 mm)
BITE_N = 1.0
BITE_HOLD_S = 0.10
# retract carries no food and moves away from the user: fast (user, 2026-09-24: "it's otherwise too slow")
RETRACT_MPS = 0.060            # back off along the utensil
RETRACT_SWING_DPS = 25.0
RETRACT_DOWN_DPS = 20.0
RETRACT_HOME_DPS = 20.0
MIN_TIP_Z = 0.300              # the return's joint move must keep the utensil tip above this
UP_W, UP_H = 720, 1280         # upright (rotated) image


def margin(q):
    return min(LIMITS[i] - abs(q[i]) for i in LIMITS)


def Rz(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def axis(R):
    return R @ np.array([0, 0, 1.0])


def fk_tip(conn, q):
    ja = Base_pb2.JointAngles()
    for i, v in enumerate(q):
        j = ja.joint_angles.add(); j.joint_identifier = i; j.value = v % 360.0
    for attempt in range(5):             # has failed once (METHOD_FAILED) right after a trajectory
        try:
            p = conn.base.ComputeForwardKinematics(ja)
            break
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise MoveError("forward kinematics keeps failing -- stopped")
            time.sleep(0.3)
    return np.array([p.x, p.y, p.z]) + R_from_kortex(p.theta_x, p.theta_y, p.theta_z) @ tip_in_tool()


def homotopy(conn, q0, at, n, toward=None):
    """Continuous IK path along at(s) -> (tool xyz, R). The 7th DoF is steered away from J2/J4/J6 limits
    (without it the swing only fits 60 mm of retraction), or blended `toward` a known arm shape."""
    prev, path = q0, [q0]
    for k in range(1, n + 1):
        s = k / n
        T, R = at(s)
        if toward is not None:
            g = [fold(a + fold(b - a) * s) for a, b in zip(q0, toward)]
        else:
            g = list(prev)
            for j in (1, 3, 5):
                if abs(prev[j]) > LIMITS[j] - 25:
                    g[j] = prev[j] - math.copysign(min(3.0, abs(prev[j])), prev[j])
        q = _ik(conn, T, kortex_from_R(R), g)
        if q is None:
            return None, "IK fails at %d%%" % (100 * s)
        step = max(abs(fold(a - b)) for a, b in zip(q, prev))
        if step > 8:
            return None, "joint jump %.0f deg at %d%%" % (step, 100 * s)
        prev = q; path.append(q)
    return path, None


def run_path(conn, path, dps, label, go, torque_stop=True, ramp=0.25, from_here=True):
    """Validate a joint path with the controller; with go, execute it and watch it. A path that starts
    somewhere the arm is not yet (from_here=False, validate-only) can only be IK-checked: the controller
    validates relative to the arm's current joints."""
    if not from_here:
        print("    %-26s IK-continuous, joint margin >= %.1f deg  (controller validates it just before it runs)"
              % (label, min(margin(q) for q in path)))
        return
    wl, sp = _waypoints(conn, path, dps, ramp, label)
    total = sum(w.angular_waypoint.duration for w in wl.waypoints)
    print("    %-26s controller accepted at %.1f deg/s: %.1f s%s" % (label, sp, total, "" if go else "  (validate only)"))
    if not go:
        return
    ev = _events(conn); ev.clear(); guard = TorqueGuard()
    conn.base.ExecuteWaypointTrajectory(wl)
    t0 = time.time()
    while time.time() - t0 < total + 15:
        time.sleep(0.02)
        hit = guard.update([x.torque for x in conn.base_cyclic.RefreshFeedback().actuators])
        if any(e[0] == "ACTION_ABORT" for e in ev):
            conn.base.Stop(); raise MoveError("%s: controller aborted" % label)
        if hit and torque_stop:
            conn.base.Stop(); raise MoveError("%s: stopped, joint %d torque step" % (label, hit[0] + 1))
        if any(e[0] == "ACTION_END" for e in ev):
            break
    print("    %-26s done in %.1f s" % (label, time.time() - t0))


def dps_for(path, metres, mps):
    """Cruise joint speed that moves the tool `metres` at `mps`."""
    return sum(max(abs(fold(x - y)) for x, y in zip(q, p)) for p, q in zip(path, path[1:])) / (metres / mps)


def swing_tip(tipS, axS, rmax, u):
    """The swing's utensil-tip path: +90 deg about world z (== J1 -90), retraction front-loaded."""
    return Rz(90.0 * u) @ (tipS - rmax * (1 - (1 - u) ** 2) * axS)


# ------------------------------------------------------------------ steps

def swing(conn, a):
    """Lift, swing (retracting), and step to the user's right -- ONE continuous validated trajectory, no
    stops between them (user, 2026-09-24: 'painfully slow ... these need not be waited for')."""
    p = tool_pose(conn); q0 = joints(conn); R0 = R_from_kortex(*p[3:])
    if axis(R0)[1] < 0.99:
        raise MoveError("utensil is not pointing at the user (+Y) -- run from the end of the scoop")
    # The swing turns about the BASE axis: it was built and validated for scoops ending on the base's
    # -Y side (y -0.04 .. -0.10). From a bowl in front of the user (+Y) the same turn would sweep the
    # utensil through where the user's face is. Refuse anything outside the validated region.
    if not -0.15 < p[1] < 0.0:
        raise MoveError("scoop ended at y %+.3f, outside the region the swing was validated for (-0.15..0.0): "
                        "from here the swing could sweep into the user -- not moving" % p[1])
    z0 = p[2]; up = (p[0], p[1], z0 + LIFT_M); T0 = np.array(p[:3])
    c = _cached(SWING_CACHE, q0)
    if c:
        path, rmax, q80 = [q0] + c["path"][1:], c["retract_m"], c["q80"]
        print("    lift + swing + step right: planned before, reused (joint margin >= %.1f deg)" % min(margin(q) for q in path))
        return _swing_go(conn, a, path, rmax, q80, up, p, z0)
    # 1. lift: a steered joint path -- from a scoop that ends nearer the base, the controller's own straight
    #    lift runs a joint out at +70 mm; steering the redundancy reaches 175
    n_lift = 35
    lift, err = homotopy(conn, q0, lambda s: (T0 + np.array([0, 0, LIFT_M * s]), R0), n_lift)
    if lift is None:
        raise MoveError("the %.0f mm lift is not reachable (%s) -- nothing moved" % (1000 * LIFT_M, err))
    q80 = lift[round(n_lift * 0.080 / LIFT_M)]                  # the return's arm shape, for retract
    # 2. swing toward the user while retracting along the utensil, as far as the joints allow
    tip0 = np.array(up) + R0 @ tip_in_tool(); ax0 = axis(R0)
    best = None
    for rmax in RETRACT_TRIES:
        at = lambda s, r=rmax: (swing_tip(tip0, ax0, r, s) - Rz(90.0 * s) @ R0 @ tip_in_tool(), Rz(90.0 * s) @ R0)
        path, err = homotopy(conn, lift[-1], at, 60)
        if path and min(margin(q) for q in path) < min(3.0, margin(lift[-1]) - 0.5):
            err = "joint-limit margin only %.1f deg" % min(margin(q) for q in path); path = None
        if path:
            best = (rmax, path); break
    if best is None:
        raise MoveError("no swing toward the user from here -- nothing moved")
    rmax, swing_path = best
    # 3. to the user's right, so the food is not between the camera and the mouth
    Ts, Rs = (swing_tip(tip0, ax0, rmax, 1.0) - Rz(90.0) @ R0 @ tip_in_tool(), Rz(90.0) @ R0)
    side, err = homotopy(conn, swing_path[-1], lambda s: (Ts - np.array([0, DETECT_SIDESTEP_M * s, 0]), Rs), 8)
    if side is None:
        print("    (the %.0f mm step to the user's right is not reachable -- detecting from the swing's end)" % (1000 * DETECT_SIDESTEP_M))
        side = [swing_path[-1]]
    path = lift + swing_path[1:] + side[1:]
    print("    lift %.0f mm + swing retracting %.0f mm + %.0f mm to the user's right: one trajectory, joint margin >= %.1f deg"
          % (1000 * LIFT_M, 1000 * rmax, 1000 * DETECT_SIDESTEP_M if len(side) > 1 else 0, min(margin(q) for q in path)))
    SWING_CACHE.write_text(json.dumps(dict(q_start=q0, path=path, retract_m=rmax, q80=q80)))
    return _swing_go(conn, a, path, rmax, q80, up, p, z0)


def _swing_go(conn, a, path, rmax, q80, up, p, z0):
    run_path(conn, path, SWING_DPS, "lift + swing + step right", not a.validate_only)
    if a.validate_only:
        return
    pe = tool_pose(conn)          # the retract's back-off returns here
    STATE.write_text(json.dumps(dict(start_tool=list(up), start_rpy=list(p[3:]), retract_m=rmax,
                                     end_tool=list(pe[:3]), q80=q80, scoop_end_z=z0)))


def _mouth(f, allow_nose=False):
    """Mouth in camera coordinates: the plausible mouth box, else (allowed) the nose estimate."""
    c = cal()[1]
    n = f.get("nose")
    if not n:
        raise MoveError("no nose detected")
    # the nose's box is what the checks below need; its depth can be missing (the food hides it from the depth
    # sensor, which sits 27 mm beside the colour camera) -- then only the nose-based fallback is unavailable
    x1, y1, x2, y2 = n["box_upright"]; ncx, ncy, nw, nh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    if nw * nh > 0.03 * UP_W * UP_H:
        raise MoveError("nose box is %.0f%% of the frame -- junk" % (100 * nw * nh / (UP_W * UP_H)))
    m = f.get("mouth"); why = None
    if not m or not m.get("depth_m"):
        why = "no mouth box"
    else:
        a1, b1, a2, b2 = m["box_upright"]; mcx, mcy = (a1 + a2) / 2, (b1 + b2) / 2
        if (a2 - a1) * (b2 - b1) > 0.03 * UP_W * UP_H:
            why = "mouth box covers %.0f%% of the frame" % (100 * (a2 - a1) * (b2 - b1) / (UP_W * UP_H))
        elif not 0.4 * nh < mcy - ncy < 1.5 * nh:
            why = "mouth box %.2f nose heights below the nose" % ((mcy - ncy) / nh)
        elif abs(mcx - ncx) > nw:
            why = "mouth box %.2f nose widths sideways" % (abs(mcx - ncx) / nw)
        elif n.get("depth_m") and abs(m["depth_m"] - n["depth_m"]) > 0.030:
            why = "mouth depth %.0f mm off the nose's" % (1000 * abs(m["depth_m"] - n["depth_m"]))
    if why is None:
        return m["cam_xyz"], "mouth box (%s)" % m.get("prompt", "")
    if not allow_nose:
        return None, why
    if not n.get("depth_m"):
        raise MoveError("mouth not usable (%s) and the nose has no depth -- cannot estimate it" % why)
    u, v = ncy + NOSE_TO_MOUTH * nh, UP_W - 1 - ncx          # upright -> original colour pixel
    z = n["depth_m"] + MOUTH_BEHIND_NOSE
    return [(u - c["cx"]) * z / c["fx"], (v - c["cy"]) * z / c["fy"], z], "nose estimate (%s)" % why


def approach(conn, a):
    if os.path.getmtime(a.face) < os.path.getmtime(a.colour):
        raise MoveError("the detection is older than its capture -- not moving")
    if time.time() - os.path.getmtime(a.face) > MAX_SHIFT_S:
        raise MoveError("the detection is more than %.0f s old -- not moving" % MAX_SHIFT_S)
    f = json.loads(Path(a.face).read_text())
    cam, source = _mouth(f, allow_nose=a.allow_nose)
    if cam is None:
        print("    mouth not usable (%s) -- sidestep and look again" % source)
        return 3
    p = tool_pose(conn); q0 = joints(conn); T0 = np.array(p[:3]); R = R_from_kortex(*p[3:]); ax = axis(R)
    if abs(ax[2]) > 0.1:
        raise MoveError("utensil is not level -- run the swing first")
    mouth = T0 + R @ (CAM_IN_TOOL + np.array([-cam[0], -cam[1], cam[2]]))
    goal_tip = mouth - STOP_M * ax + np.array([0, 0, AIM_DZ])
    Tg = goal_tip - R @ tip_in_tool(); L = np.linalg.norm(Tg - T0)
    print("    mouth (%.3f %.3f %.3f) from the %s" % (*mouth, source))
    print("    stop: utensil tip (%.3f %.3f %.3f), %.0f mm short and %.0f mm below; %.0f mm to go"
          % (*goal_tip, 1000 * STOP_M, -1000 * AIM_DZ, 1000 * L))
    if L > MAX_APPROACH_M:
        raise MoveError("an approach of %.0f mm is too long -- not moving" % (1000 * L))
    if L < 0.003:
        print("    already there"); return 0
    path, err = homotopy(conn, q0, lambda s: (T0 + (Tg - T0) * s, R), 40)
    if path is None:
        raise MoveError("approach " + err + " -- not moving")
    run_path(conn, path, dps_for(path, L, APPROACH_MPS), "approach at %.0f mm/s" % (1000 * APPROACH_MPS), not a.validate_only)
    if not a.validate_only:
        tip = np.array(tool_pose(conn)[:3]) + R @ tip_in_tool()
        print("    utensil tip %.0f mm from the detected mouth" % (1000 * np.linalg.norm(mouth - tip)))
    return 0


def sidestep(conn, a):
    p = tool_pose(conn); q0 = joints(conn)
    if not all(_ik(conn, (p[0], p[1] - a.step * k / 5, p[2]), tuple(p[3:]), q0) for k in range(1, 6)):
        raise MoveError("the sidestep is not reachable -- not moving")
    goto_cartesian(conn, (p[0], p[1] - a.step, p[2]), tuple(p[3:]), speed=0.02,
                   label="sidestep -Y %.0f mm" % (1000 * a.step), max_tilt=95.0)


def bite(conn, a):
    def F():
        b = conn.base_cyclic.RefreshFeedback().base
        return np.array([b.tool_external_wrench_force_x, b.tool_external_wrench_force_y, b.tool_external_wrench_force_z])
    base = np.median([F() for _ in range(40)], 0)                    # ~1 s at rest
    print("    waiting up to %.0f s for the bite (> %.1f N for %.2f s)" % (a.timeout, BITE_N, BITE_HOLD_S))
    t0 = time.time(); above = None; peak = 0.0
    while time.time() - t0 < a.timeout:
        f = np.linalg.norm(F() - base); t = time.time() - t0; peak = max(peak, f)
        if f <= BITE_N:
            above = None
        elif above is None:
            above = t
        elif t - above >= BITE_HOLD_S:
            break
    else:
        print("    no bite in %.0f s" % a.timeout)
        return 4
    print("    BITE at %.1f s (%.1f N)" % (above, peak))
    tail = []; ts = time.time()
    while time.time() - ts < 10:
        tail.append(F() - base)
        if len(tail) > 40 and max(np.linalg.norm(x) for x in tail[-40:]) < BITE_N:
            break
    print("    settled; resting shift %.2f N (the food's weight gone: ~0.2-0.4 N)" % np.linalg.norm(np.median(tail[-40:], 0)))
    return 0


def retract(conn, a):
    if not STATE.exists():
        raise MoveError("no swing recorded (%s) -- cannot retrace" % STATE)
    st = json.loads(STATE.read_text())
    go = not a.validate_only
    p = tool_pose(conn); q0 = joints(conn); T0 = np.array(p[:3]); R0 = R_from_kortex(*p[3:])
    if abs(axis(R0)[2]) > 0.1:
        raise MoveError("utensil is not level -- not at the user")
    RS = R_from_kortex(*st["start_rpy"]); axS = axis(RS); TS = np.array(st["start_tool"])
    # resuming after legs 1-2 (utensil already turned back, at the swing's start): straight to leg 3.
    # Leg 1 from here would "back off" toward the swing's END, i.e. toward the user.
    resumed = axis(R0) @ axS > 0.99 and np.linalg.norm(T0 - TS) < 0.020
    if not resumed and axis(R0) @ axS > 0.5:
        raise MoveError("utensil is neither at the user nor back at the swing's start -- not guessing a path")
    # 1. back off along the utensil, to where the swing ended
    Te = np.array(st["end_tool"]); L = 0.0 if resumed else np.linalg.norm(Te - T0)
    if L > 0.400:
        raise MoveError("%.0f mm from the swing's end -- not retracing from here" % (1000 * L))
    if resumed:
        print("    legs 1-2 already done (at the swing's start): going home")
        q1 = q0
    elif L > 0.003:
        path, err = homotopy(conn, q0, lambda s: (T0 + (Te - T0) * s, R0), 40)
        if path is None:
            raise MoveError("back off " + err + " -- nothing moved")
        run_path(conn, path, dps_for(path, L, RETRACT_MPS), "back off %.0f mm" % (1000 * L), go, torque_stop=False)
        q1 = joints(conn) if go else path[-1]
    else:
        q1 = q0
    # 2. reverse the swing
    tipv_t = tip_in_tool(); tipS = TS + RS @ tipv_t
    R1 = R0 if not go else R_from_kortex(*tool_pose(conn)[3:]); ax1 = axis(R1)
    th = math.degrees(math.atan2(np.cross(axS, ax1)[2], axS @ ax1)); u0 = th / 90.0
    tip1 = (Te if L > 0.003 else T0) + R1 @ tipv_t
    e = tip1 - swing_tip(tipS, axS, st["retract_m"], u0)

    def at2(s):
        u = u0 * (1 - s); R = Rz(90.0 * u) @ RS
        return swing_tip(tipS, axS, st["retract_m"], u) + (1 - s) * e - R @ tipv_t, R
    rc = _cached(RETRACT_CACHE, q1, tol=1.5)
    if resumed:
        q2 = q0
    elif rc:
        path2 = [q1] + rc["path2"][1:]
        print("    reverse swing + descent: planned before, reused")
        run_path(conn, path2, RETRACT_SWING_DPS, "reverse swing", go, from_here=go or L <= 0.003)
        q2 = joints(conn) if go else path2[-1]
    else:
        path2, err = homotopy(conn, q1, at2, 60)
        if path2 is None:
            raise MoveError("reverse swing " + err)
        run_path(conn, path2, RETRACT_SWING_DPS, "reverse swing %.0f deg" % th, go, from_here=go or L <= 0.003)
        q2 = joints(conn) if go else path2[-1]
    # 3. down to the scoop end + 80 mm in the scoop's own arm shape, then the proven joint move
    Ta = TS; Td = np.array([TS[0], TS[1], st["scoop_end_z"] + 0.080])
    qd = cal()[0]["joints_deg"]
    if rc and not resumed:
        path3, lo = [q2] + rc["path3"][1:], rc["lo"]
    else:
        path3, err = homotopy(conn, q2, lambda s: (Ta + (Td - Ta) * s, RS), 40, toward=st["q80"])
        if path3 is None:
            raise MoveError("descent " + err)
        lo = min(fk_tip(conn, [fold(x + fold(y - x) * k / 20) for x, y in zip(path3[-1], qd)])[2] for k in range(21))
        if not resumed and go:
            RETRACT_CACHE.write_text(json.dumps(dict(q_start=q1, path2=path2, path3=path3, lo=lo)))
    print("    return: utensil tip never below %.0f mm" % (1000 * lo))
    if lo < MIN_TIP_Z:
        raise MoveError("the return would dip the utensil tip to %.0f mm -- stopped" % (1000 * lo))
    run_path(conn, path3, RETRACT_DOWN_DPS, "down to scoop end + 80 mm", go, from_here=go or resumed)
    if go:
        try:
            goto(conn, qd, speed=RETRACT_HOME_DPS, label="-> food_detect_start", max_tilt=95.0)
        except MoveError as e:
            # braking at the end of this faster move reads as a torque step (J2 -7.8 -> -4.8 N.m on
            # 2026-09-24, nothing touched): if the arm is already there, it arrived
            off = max(abs(fold(x - y)) for x, y in zip(joints(conn), qd))
            if "torque" not in str(e) or off > 0.2:
                raise
            print("    -> food_detect_start: arrived (%.2f deg off); the torque step was the braking" % off)
        STATE.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["swing", "approach", "sidestep", "bite", "retract"])
    ap.add_argument("--validate-only", action="store_true", help="plan and let the controller validate; do not move")
    ap.add_argument("--face", help="approach: detect_mouth.py output, made at the current pose")
    ap.add_argument("--colour", help="approach: the capture that detection was made from")
    ap.add_argument("--allow-nose", action="store_true", help="approach: estimate the mouth from the nose")
    ap.add_argument("--step", type=float, default=0.05, help="sidestep: metres")
    ap.add_argument("--timeout", type=float, default=60.0, help="bite: seconds")
    a = ap.parse_args()
    with kinova_connection() as conn:
        try:
            rc = dict(swing=swing, approach=approach, sidestep=sidestep, bite=bite, retract=retract)[a.step](conn, a)
        except MoveError as e:
            print("\n  STOPPED: %s" % e, file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001 -- a controller error mid-sequence: stop, say where
            print("\n  STOPPED on a controller error: %s" % " ".join(str(e).split())[:300], file=sys.stderr)
            return 1
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
