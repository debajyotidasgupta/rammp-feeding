"""Motion building blocks shared by every stage: frames, IK, continuous IK paths, time scaling, validated
trajectories, forward kinematics. Numbers come from config/ (motion.*, guards.*, the camera and utensil profiles).

Used by acquire.py (over / plunge / scoop / pick) and deliver.py (swing / approach / retract); a new stage should
be built from these rather than copying them.
"""

from __future__ import annotations

import math
import time

import numpy as np
from kortex_api.autogen.messages import Base_pb2

from settings import cfg
from feeding.camera import CAM_IN_TOOL, FXC, VP
from kinova.scripts.safe_move import MoveError, TorqueGuard, _events, fold

# J2 / J4 / J6 limits, keyed by 0-based joint index
LIMITS = {int(j) - 1: float(v) for j, v in cfg.motion.joint_limits_deg.items()}
A_MAX = float(cfg.motion.joint_accel_cap_dps2)        # deg/s^2 per joint (the controller refuses above 80)


def tip_in_tool() -> np.ndarray:
    """Utensil tip in the tool frame, from calibration only.

    tip = camera + camera_to_tip along the axis + the tip's sideways offset,
    read off the image as (tip pixel - image of true vertical) scaled to that
    distance. Image +u is tool -x and image +v is tool -y at the overlook
    orientation (from motion probes). For the fork this gives (0, 8.8, 142) mm.
    """
    zt = float(cfg.utensil.camera_to_tip_m)
    d = (np.array(cfg.utensil.tip_px, float) - VP) * zt / FXC
    return CAM_IN_TOOL + np.array([-d[0], -d[1], zt])


def rot_x(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def R_from_kortex(tx, ty, tz):
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    Rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    Rz = np.array([[math.cos(tz), -math.sin(tz), 0], [math.sin(tz), math.cos(tz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def kortex_from_R(R):
    ty = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    return (math.degrees(math.atan2(R[2, 1], R[2, 2])), ty,
            math.degrees(math.atan2(R[1, 0], R[0, 0])))


def _ik(conn, T, rpy, g):
    d = Base_pb2.IKData(); p = d.cartesian_pose
    p.x, p.y, p.z = map(float, T); p.theta_x, p.theta_y, p.theta_z = map(float, rpy)
    for v in g:
        d.guess.joint_angles.add().value = v % 360.0
    try:
        return [fold(j.value) for j in conn.base.ComputeInverseKinematics(d).joint_angles]
    except Exception:  # noqa: BLE001 -- unreachable is an answer
        return None


def _time_scale(path, vmax, amax=A_MAX):
    """Segment durations so that EVERY joint's speed stays under vmax and changes by at most amax -- from
    rest and back to rest. The earlier leading-joint profile let a joint that barely moved in one segment
    lead the next: J1 went from ~0 (the lift) to full speed (the swing) in one segment, 628 deg/s^2, and
    the controller refused the lift+swing trajectory at every speed down to 7 deg/s."""
    D = np.array([[fold(x - y) for x, y in zip(q, p)] for p, q in zip(path, path[1:])])
    n = len(D)
    vcap = np.full(n, vmax) if np.isscalar(vmax) else np.asarray(vmax, float)
    dt = np.maximum(np.abs(D).max(1) / vcap, 0.02)
    for _ in range(4000):
        V = D / dt[:, None]
        Vp = np.vstack([np.zeros(7), V, np.zeros(7)])          # at rest before and after
        mid = np.concatenate([[dt[0] / 2], (dt[1:] + dt[:-1]) / 2, [dt[-1] / 2]])
        acc = np.abs(np.diff(Vp, axis=0)).max(1) / mid          # n + 1 junctions
        bad = acc > amax
        if not bad.any():
            break
        grow = np.zeros(n, bool)
        idx = np.nonzero(bad)[0]
        grow[np.clip(idx - 1, 0, n - 1)] = True; grow[np.clip(idx, 0, n - 1)] = True
        dt[grow] *= 1.03
    return list(dt)


def _waypoints(conn, path, top_speed, ramp, what):
    """The joint path as one WaypointList the controller has validated; slower if refused.
    (`ramp` is kept for the callers; the time scaling now bounds acceleration itself.)"""
    raw = [x.position for x in conn.base_cyclic.RefreshFeedback().actuators]
    for speed in (top_speed, top_speed * 0.75, top_speed * 0.5, top_speed * 0.35):
        durs = _time_scale(path, speed, A_MAX * (speed / top_speed))
        wl = Base_pb2.WaypointList(); wl.duration = 0.0; wl.use_optimal_blending = False
        cur, prev = list(raw), path[0]
        for k, (q, dt) in enumerate(zip(path[1:], durs), 1):
            cur = [c + fold(x - y) for c, x, y in zip(cur, q, prev)]
            w = wl.waypoints.add(); w.name = "w%02d" % k
            w.angular_waypoint.angles.extend(cur)
            w.angular_waypoint.duration = float(dt)
            prev = q
        errs = list(conn.base.ValidateWaypointList(wl).trajectory_error_report.trajectory_error_elements)
        if not errs:
            return wl, speed
        print("    %.0f deg/s refused by the controller: %s" % (speed, " ".join(str(errs[0]).split())))
    raise MoveError("controller refused the %s even at %.0f deg/s -- nothing moved" % (what, top_speed * 0.35))


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


def run_path(conn, path, dps, label, go, torque_stop=True, ramp=0.25, from_here=True,
             torque_limit=float(cfg.guards.torque_step_nm.default)):
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
    ev = _events(conn); ev.clear(); guard = TorqueGuard(limit=torque_limit)
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
