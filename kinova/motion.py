"""Motion building blocks shared by every stage: frames, IK, continuous IK paths, time scaling, validated
trajectories, forward kinematics. Numbers come from config/ (motion.*, guards.*, the camera and utensil profiles).

Used by acquire.py (over / plunge / scoop / pick) and deliver.py (swing / approach / retract); a new stage should
be built from these rather than copying them.

Conventions: joint vectors are 7 angles in degrees folded to (-180, 180]; poses are Kortex's (x y z m, theta
x y z deg, fixed-frame XYZ Euler); R is a 3x3 rotation, base <- tool; a tool "axis" is where the utensil
points (tool +z).
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from kortex_api.autogen.messages import Base_pb2

import runlog
from settings import cfg
from feeding.camera import CAM_IN_TOOL, FXC, VP
from kinova.scripts.safe_move import Joints, MoveError, TorqueGuard, _events, fold

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

log = runlog.get("motion")

# J2 / J4 / J6 limits, keyed by 0-based joint index
LIMITS = {int(j) - 1: float(v) for j, v in cfg.motion.joint_limits_deg.items()}
A_MAX = float(cfg.motion.joint_accel_cap_dps2)        # deg/s^2 per joint (the controller refuses above 80)
MAX_STEP_DEG = float(cfg.motion.max_joint_step_deg)   # a bigger IK step on any joint is a jump, not a path
STEER_WITHIN = float(cfg.motion.steer_within_deg)     # steer J2/J4/J6 off a limit once within this of it
STEER_STEP = float(cfg.motion.steer_step_deg)
SPEED_FALLBACKS = [float(f) for f in cfg.motion.speed_fallbacks]   # refused: retry at these fractions
WATCH_EXTRA_S = float(cfg.motion.watch_extra_s)       # stop watching this long after the planned end
ACCEL_FACTOR = float(cfg.motion.controller_accel_factor)   # the controller sees this much more acceleration
ACCEL_RETRIES = int(cfg.motion.accel_retries)         # refused on acceleration: rescale it by the controller's ratio

PoseAt = Callable[[float], "tuple[np.ndarray, np.ndarray]"]   # s in [0, 1] -> (tool xyz, R)


# ------------------------------------------------------------------ frames

def tip_in_tool() -> np.ndarray:
    """Utensil tip in the tool frame, from calibration only.

    tip = camera + camera_to_tip along the axis + the tip's sideways offset,
    read off the image as (tip pixel - image of true vertical) scaled to that
    distance. Image +u is tool -x and image +v is tool -y at the overlook
    orientation (from motion probes). Plus the utensil's tine offset (tool frame: a curved fork head meets the
    food off where the image puts its tip).
    """
    zt = float(cfg.utensil.camera_to_tip_m)
    d = (np.array(cfg.utensil.tip_px, float) - VP) * zt / FXC
    tine = np.array(cfg.get("utensil.tine_offset_tool_m", [0.0, 0.0, 0.0]), float)
    return CAM_IN_TOOL + np.array([-d[0], -d[1], zt]) + tine


def rot_x(deg: float) -> np.ndarray:
    """Rotation about x by `deg`."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(deg: float) -> np.ndarray:
    """Rotation about z by `deg` (about the base's vertical: J1's direction)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def R_from_kortex(tx: float, ty: float, tz: float) -> np.ndarray:
    """Kortex theta x y z (deg) -> R = Rz @ Ry @ Rx."""
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    Rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    Rz_ = np.array([[math.cos(tz), -math.sin(tz), 0], [math.sin(tz), math.cos(tz), 0], [0, 0, 1]])
    return Rz_ @ Ry @ Rx


def kortex_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """R -> Kortex theta x y z (deg)."""
    ty = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    return (math.degrees(math.atan2(R[2, 1], R[2, 2])), ty,
            math.degrees(math.atan2(R[1, 0], R[0, 0])))


def axis(R: np.ndarray) -> np.ndarray:
    """Where the utensil points (tool +z) in base coordinates."""
    return R @ np.array([0, 0, 1.0])


def margin(q: Sequence[float]) -> float:
    """Smallest distance (deg) from any limited joint to its limit."""
    return min(LIMITS[i] - abs(q[i]) for i in LIMITS)


def max_step(a: Sequence[float], b: Sequence[float]) -> float:
    """Largest single-joint difference between two joint vectors, deg (the short way round)."""
    return max(abs(fold(x - y)) for x, y in zip(a, b))


# ------------------------------------------------------------------ kinematics (the controller's)

def _ik(conn: KinovaConnection, T: Sequence[float], rpy: Sequence[float], g: Sequence[float]) -> Joints | None:
    """The controller's IK for tool pose (T, rpy), seeded with joints g; None when unreachable."""
    d = Base_pb2.IKData()
    p = d.cartesian_pose
    p.x, p.y, p.z = map(float, T)
    p.theta_x, p.theta_y, p.theta_z = map(float, rpy)
    for v in g:
        d.guess.joint_angles.add().value = v % 360.0
    try:
        return [fold(j.value) for j in conn.base.ComputeInverseKinematics(d).joint_angles]
    except Exception:  # noqa: BLE001 -- unreachable is an answer
        return None


def fk_tip(conn: KinovaConnection, q: Sequence[float]) -> np.ndarray:
    """Utensil tip position (base frame) at joints q, from the controller's FK."""
    ja = Base_pb2.JointAngles()
    for i, v in enumerate(q):
        j = ja.joint_angles.add()
        j.joint_identifier = i
        j.value = v % 360.0
    for attempt in range(5):             # has failed once (METHOD_FAILED) right after a trajectory
        try:
            p = conn.base.ComputeForwardKinematics(ja)
            break
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise MoveError("forward kinematics keeps failing -- stopped")
            time.sleep(0.3)
    return np.array([p.x, p.y, p.z]) + R_from_kortex(p.theta_x, p.theta_y, p.theta_z) @ tip_in_tool()


def homotopy(conn: KinovaConnection, q0: Joints, at: PoseAt, n: int,
             toward: Sequence[float] | None = None) -> tuple[list[Joints] | None, str | None]:
    """Continuous IK path along at(s) -> (tool xyz, R), s = 1/n .. 1: (path including q0, None), or (None, why).

    The 7th DoF is steered away from J2/J4/J6 limits (without it the swing only fits 60 mm of retraction), or
    the IK guess is blended `toward` a known arm shape. A step over MAX_STEP_DEG on any joint is a jump.
    """
    prev, path = q0, [q0]
    for k in range(1, n + 1):
        s = k / n
        T, R = at(s)
        if toward is not None:
            g = [fold(a + fold(b - a) * s) for a, b in zip(q0, toward)]
        else:
            g = list(prev)
            for j in (1, 3, 5):
                if abs(prev[j]) > LIMITS[j] - STEER_WITHIN:
                    g[j] = prev[j] - math.copysign(min(STEER_STEP, abs(prev[j])), prev[j])
        q = _ik(conn, T, kortex_from_R(R), g)
        if q is None:
            return None, "IK fails at %d%%" % (100 * s)
        step = max_step(q, prev)
        if step > MAX_STEP_DEG:
            return None, "joint jump %.0f deg at %d%%" % (step, 100 * s)
        prev = q
        path.append(q)
    return path, None


# ------------------------------------------------------------------ trajectories

def _time_scale(path: Sequence[Sequence[float]], vmax: float | Sequence[float], amax: float = A_MAX) -> list[float]:
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
        grow[np.clip(idx - 1, 0, n - 1)] = True
        grow[np.clip(idx, 0, n - 1)] = True
        dt[grow] *= 1.03
    return list(dt)


def waypoint_list(raw: Sequence[float], path: Sequence[Sequence[float]], durations: Sequence[float],
                  prefix: str = "w") -> Base_pb2.WaypointList:
    """An angular WaypointList for path[1:] with the given segment durations.

    Kortex takes absolute (unfolded) angles and interpolates literally, so each waypoint is the arm's current
    raw reading `raw` plus the folded (short-way) joint deltas accumulated along the path.
    """
    wl = Base_pb2.WaypointList()
    wl.duration = 0.0
    wl.use_optimal_blending = False
    cur, prev = list(raw), path[0]
    for k, (q, dt) in enumerate(zip(path[1:], durations), 1):
        cur = [c + fold(x - y) for c, x, y in zip(cur, q, prev)]
        w = wl.waypoints.add()
        w.name = "%s%02d" % (prefix, k)
        w.angular_waypoint.angles.extend(cur)
        w.angular_waypoint.duration = float(dt)
        prev = q
    return wl


def arm_caps(conn: KinovaConnection) -> tuple[float, float]:
    """The arm's own ANGULAR_TRAJECTORY soft limits (slowest joint): (joint speed deg/s, acceleration deg/s^2),
    read once per connection; (inf, inf) if they cannot be read (the fake controller in the tests)."""
    if not hasattr(conn, "_arm_caps"):
        caps = (math.inf, math.inf)
        try:
            from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
            from kortex_api.autogen.messages import ControlConfig_pb2 as CC
            for m in ControlConfigClient(conn._tcp_router).GetAllKinematicSoftLimits().kinematic_limits_list:
                if m.control_mode == CC.ANGULAR_TRAJECTORY and list(m.joint_speed_limits):
                    caps = (min(m.joint_speed_limits), min(m.joint_acceleration_limits or [math.inf]))
        except Exception:  # noqa: BLE001 -- no ControlConfig (the fake), or a read error: no cap
            pass
        conn._arm_caps = caps
    return conn._arm_caps


def capped(conn: KinovaConnection, speed: float, accel: float = A_MAX) -> tuple[float, float]:
    """speed and acceleration held just under the arm's soft limits: asking for more is refused, not clamped."""
    v, a = arm_caps(conn)
    return min(speed, 0.98 * v), min(accel, 0.98 * a)


def validate(conn: KinovaConnection, wl: Base_pb2.WaypointList) -> list:
    """The controller's objections to a waypoint list (empty = it would run it)."""
    return list(conn.base.ValidateWaypointList(wl).trajectory_error_report.trajectory_error_elements)


def record_plan(label: str, wl: Base_pb2.WaypointList, **info) -> None:
    """The validated trajectory, to the run record: waypoint angles and durations."""
    runlog.save_json("plan_" + "".join(c if c.isalnum() else "_" for c in label)[:40],
                     dict(label=label, angles=[list(w.angular_waypoint.angles) for w in wl.waypoints],
                          durations=[w.angular_waypoint.duration for w in wl.waypoints], **info))


def _waypoints(conn: KinovaConnection, path: Sequence[Sequence[float]], top_speed: float, ramp: float,
               what: str) -> tuple[Base_pb2.WaypointList, float]:
    """The joint path as one WaypointList the controller has validated, and the speed it took; slower if
    refused. (`ramp` is kept for the callers; the time scaling now bounds acceleration itself.)"""
    raw = [x.position for x in conn.base_cyclic.RefreshFeedback().actuators]
    asked = top_speed
    top_speed, amax = capped(conn, top_speed)
    amax /= ACCEL_FACTOR                  # as the controller will measure it: accepted at full speed, not halved
    if top_speed < asked:
        log.info("    %-26s %.0f deg/s asked, the arm's limit allows %.0f", what, asked, top_speed)
    for speed in [top_speed * f for f in SPEED_FALLBACKS]:
        a_try = amax * (speed / top_speed)
        for _ in range(ACCEL_RETRIES + 1):
            wl = waypoint_list(raw, path, _time_scale(path, speed, a_try))
            errs = validate(conn, wl)
            if not errs:
                record_plan(what, wl, speed_dps=speed, top_speed_dps=top_speed, accel_dps2=a_try)
                return wl, speed
            e = errs[0]
            # Refused on acceleration: the controller says what it measured and its limit -- rescale only the
            # acceleration by that and ask again at the SAME speed (a fixed factor was not enough for every path:
            # the retract's measured 93-97 vs 80 with 1.6, 2026-09-25). Anything else: the next, slower speed.
            if getattr(e, "error_type", None) != Base_pb2.TRAJECTORY_ERROR_TYPE_INVALID_JOINT_ACCELERATION \
                    or not getattr(e, "error_value", 0) > 0:
                break
            a_try *= 0.95 * e.max_value / e.error_value
        log.info("    %.0f deg/s refused by the controller: %s", speed, " ".join(str(e).split()))
    raise MoveError("controller refused the %s even at %.0f deg/s -- nothing moved" % (what, top_speed * SPEED_FALLBACKS[-1]))


def run_path(conn: KinovaConnection, path: Sequence[Sequence[float]], dps: float, label: str, go: bool,
             torque_stop: bool = True, ramp: float = 0.25, from_here: bool = True,
             torque_limit: float = float(cfg.guards.torque_step_nm.default)) -> None:
    """Validate a joint path with the controller; with go, execute it and watch it.

    A path that starts somewhere the arm is not yet (from_here=False, validate-only) can only be IK-checked:
    the controller validates relative to the arm's current joints. While moving, a torque step beyond
    torque_limit stops the arm (unless torque_stop=False: moving away from the user is the safe direction).
    """
    if not from_here:
        log.info("    %-26s IK-continuous, joint margin >= %.1f deg  (controller validates it just before it runs)",
                 label, min(margin(q) for q in path))
        return
    wl, sp = _waypoints(conn, path, dps, ramp, label)
    total = sum(w.angular_waypoint.duration for w in wl.waypoints)
    log.info("    %-26s controller accepted at %.1f deg/s: %.1f s%s", label, sp, total, "" if go else "  (validate only)")
    if not go:
        return
    ev = _events(conn)
    ev.clear()
    guard = TorqueGuard(limit=torque_limit)
    tel = runlog.Telemetry(label)
    outcome = "error"
    conn.base.ExecuteWaypointTrajectory(wl)
    t0 = time.time()
    try:
        while time.time() - t0 < total + WATCH_EXTRA_S:
            time.sleep(0.02)
            fb = conn.base_cyclic.RefreshFeedback()
            tel.add(fb)
            hit = guard.update([x.torque for x in fb.actuators])
            if any(e[0] == "ACTION_ABORT" for e in ev):
                conn.base.Stop()
                raise MoveError("%s: controller aborted" % label)
            if hit and torque_stop:
                conn.base.Stop()
                raise MoveError("%s: stopped, joint %d torque step" % (label, hit[0] + 1))
            if any(e[0] == "ACTION_END" for e in ev):
                break
        outcome = "done"
    except MoveError as e:
        outcome = str(e)
        raise
    finally:
        tel.save(outcome=outcome, planned_s=total, speed_dps=sp)
    log.info("    %-26s done in %.1f s", label, time.time() - t0)


def dps_for(path: Sequence[Sequence[float]], metres: float, mps: float) -> float:
    """Cruise joint speed that moves the tool `metres` at `mps`."""
    return sum(max_step(q, p) for p, q in zip(path, path[1:])) / (metres / mps)
