#!/usr/bin/env python3
"""Every arm motion goes through here. Importable helper, not a command.

goto() for joint moves, goto_cartesian() for straight-line tool moves. Both
refuse before moving if the command cannot be safe, and both watch the whole
move and stop the arm the moment it stops being safe. Each check below exists
because its absence cost something on this arm:

  UTENSIL DOWN. Tool pitch is set by the SUM J2 + J4 + J6; ~-180 points the
  utensil at the floor. On 2026-09-23 a diagnostic moved J6 alone while J2/J4
  stayed put (-4.10 - 57.08 - 40.08 = -101.3) and the fork swung 79 deg off
  vertical. A simultaneous interpolation between two utensil-down poses keeps
  the sum between their values; moving one of the three does not. goto() only
  accepts complete 7-joint vectors, and checks the destination's pitch sum.

  CONTACT. Orientation was not enough: in the same session the fork tines
  swung into the arm's OWN joint 6 housing and broke, with the tool angle legal
  throughout. There is no self-collision model yet (sim/safety.py was deleted
  in the cleanup and has not been restored with the new utensil), so a
  joint-torque step against a trailing median is the only thing that notices
  contact. It runs on every move.

  JOINT LIMITS. Kortex SILENTLY REJECTS a trajectory whose target is past a
  joint limit -- no error, no clamp, the arm just does not move. Four retries
  were lost to J6 = -129.6 against a +-120.3 limit. Checked before sending.

  NO PROGRESS. A trajectory the controller ignored looks exactly like a slow
  one. If nothing has moved after NO_PROGRESS_S the move fails at once, and it
  reports the controller's OWN reason: every move subscribes to the action
  notifications and surfaces the ACTION_ABORT sub-error code.

  TIME-SYNCHRONISED JOINT MOVES. Multi-joint moves were refused with a bare
  METHOD_FAILED whenever one joint had only a sliver of travel. The reason is
  only visible in base.GetTrajectoryErrorReport(), which is read automatically
  on every abort -- see goto(). Several wrong theories (angle wrap encoding,
  the deprecated API, the gamepad) were tested and disproved before that report
  was read. Read the report first.

  ExecuteAction is used rather than PlayJointTrajectory/PlayCartesianTrajectory,
  which are deprecated in this SDK.

  Do NOT infer the cause from the arm state. On this firmware the state reads
  SERVOING_MANUALLY_CONTROLLED for a while after ANY trajectory completes --
  including our own -- and then drops back to SERVOING_READY. An earlier
  version of this module refused to move in that state and blamed the gamepad;
  the gamepad was off and blameless.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from kortex_api.autogen.messages import Base_pb2

import runlog
from settings import cfg

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

Joints = list[float]          # 7 joint angles, deg, folded to (-180, 180]
Pose = list[float]            # tool x y z (m), theta x y z (deg): Kortex's fixed-frame XYZ Euler angles

log = runlog.get("safe_move")

MAX_TILT_DEG = float(cfg.motion.max_tilt_deg)                # how far off straight-down the utensil may ever get
TORQUE_STEP_NM = float(cfg.guards.torque_step_nm.default)    # a step this big against recent history = contact
NO_PROGRESS_S = float(cfg.motion.no_progress_s)              # nothing moved by then = the controller ignored it
# Gen3 soft limits for the three limited joints (J2, J4, J6; 0-indexed), with a
# little margin: a target AT the limit is as likely to be refused as one past it.
LIMITS = {int(j) - 1: float(v) for j, v in cfg.motion.joint_limits_deg.items()}
LIMIT_MARGIN = float(cfg.motion.joint_limit_margin_deg)


def fold(d: float) -> float:
    """Kortex reports [0, 360); joints 1/3/5/7 are continuous and sit near the
    wrap, so always compare and interpolate on (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def approach_axis(tx: float, ty: float, tz: float) -> tuple[float, float, float]:
    """Tool +z (where the utensil points) in base coords, from Kortex's
    fixed-frame XYZ Euler angles: R = Rz @ Ry @ Rx, third column."""
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    return (cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx)


def tilt_of(base: Any) -> float:
    """Angle between the utensil and straight down, from a feedback `base` message."""
    ax = approach_axis(base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z)
    return math.degrees(math.acos(max(-1.0, min(1.0, -ax[2]))))


def tilt_deg(conn: KinovaConnection) -> float:
    """Angle between the utensil and straight down, right now."""
    return tilt_of(conn.base_cyclic.RefreshFeedback().base)


def joints(conn: KinovaConnection) -> Joints:
    """The arm's joint angles now, folded."""
    return [fold(a.position) for a in conn.base_cyclic.RefreshFeedback().actuators]


def tool_pose(conn: KinovaConnection) -> Pose:
    """The tool pose now (the configured tool frame, not the flange)."""
    b = conn.base_cyclic.RefreshFeedback().base
    return [b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
            b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z]


def pitch_sum(q: Sequence[float]) -> float:
    """J2 + J4 + J6, folded: ~-180 points the utensil straight down."""
    return fold(q[1] + q[3] + q[5])


class TorqueGuard:
    """Fires when any joint's torque steps away from its own recent history.

    `gap` samples separate the baseline window from the present sample, so a
    fast-rising contact torque cannot contaminate the baseline it is compared
    against.
    """

    def __init__(self, n: int = 7, span: int = int(cfg.guards.torque_window.span),
                 gap: int = int(cfg.guards.torque_window.gap), limit: float = TORQUE_STEP_NM):
        self.hist = [deque(maxlen=span + gap) for _ in range(n)]
        self.span, self.gap, self.limit = span, gap, limit

    def update(self, tau: Sequence[float]) -> tuple[int, float, float] | None:
        """Add one sample per joint; (joint index, torque, its baseline) on a step, else None."""
        for i, t in enumerate(tau):
            self.hist[i].append(t)
        for i, h in enumerate(self.hist):
            if len(h) < self.span + self.gap:
                continue
            base = statistics.median(list(h)[: self.span])
            if abs(h[-1] - base) > self.limit:
                return i, h[-1], base
        return None


def tool_mismatch(conn: Any) -> list[str]:
    """How the arm's configured tool differs from config deployment.arm.tool ([] = it is ours). Read from the arm;
    the fake controller (no ControlConfig) counts as ours."""
    try:
        router = conn._tcp_router
    except AttributeError:
        return []
    from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
    t = ControlConfigClient(router).GetToolConfiguration()
    want, tol = cfg.deployment.arm.tool, cfg.deployment.arm.tool.tolerance
    tr = t.tool_transform
    got = dict(transform_m=[tr.x, tr.y, tr.z], rotation_deg=[tr.theta_x, tr.theta_y, tr.theta_z], mass_kg=t.tool_mass)
    bad = []
    for key in ("transform_m", "rotation_deg"):
        if max(abs(g - float(w)) for g, w in zip(got[key], want[key])) > float(tol[key]):
            bad.append("%s %s (ours %s)" % (key, [round(v, 4) for v in got[key]], list(want[key])))
    if abs(got["mass_kg"] - float(want.mass_kg)) > float(tol.mass_kg):
        bad.append("mass %.3f kg (ours %.3f)" % (got["mass_kg"], float(want.mass_kg)))
    return bad


def check_tool(conn: Any) -> None:
    """Refuse to move unless the arm's tool is ours: a different tool frame puts every cartesian target (and the tip
    model) off by the difference -- 80 mm low on 2026-09-30, when another user's 200 mm frame was left on the arm.
    Once per connection."""
    if getattr(conn, "_tool_ok", False):
        return
    bad = tool_mismatch(conn)
    if bad:
        raise MoveError("REFUSED: the arm's tool configuration is not ours: %s. `make tool` shows it; "
                        "`make tool APPLY=1` sets ours (saving these values)" % "; ".join(bad))
    conn._tool_ok = True


class MoveError(RuntimeError):
    """A move was refused, ignored, or stopped. The arm is stationary."""


_BAD_STATES = {Base_pb2.ARMSTATE_IN_FAULT, Base_pb2.ARMSTATE_MAINTENANCE,
               Base_pb2.ARMSTATE_BASE_INITIALIZATION}


def _events(conn: KinovaConnection) -> list[tuple[str, str]]:
    """Action notifications for this connection, subscribed once: (event name, abort detail) in order."""
    if not hasattr(conn, "_move_events"):
        conn._move_events = []

        def cb(n):
            detail = ""
            if n.action_event == Base_pb2.ACTION_ABORT:
                try:
                    detail = Base_pb2.SubErrorCodes.Name(n.abort_details)
                except ValueError:
                    detail = str(n.abort_details)
            conn._move_events.append((Base_pb2.ActionEvent.Name(n.action_event), detail))

        conn.base.OnNotificationActionTopic(cb, Base_pb2.NotificationOptions())
    return conn._move_events


def _preflight(conn: KinovaConnection, label: str) -> list[tuple[str, str]]:
    """Refuse in a fault / maintenance / initialising state; clear and return the event list."""
    s = conn.base.GetArmState().active_state
    if s in _BAD_STATES:
        raise MoveError("%s REFUSED: arm state is %s" % (label, Base_pb2.ArmState.Name(s)))
    ev = _events(conn)
    ev.clear()
    return ev


def _report(conn: KinovaConnection) -> str:
    """The controller's itemised reason for a rejected trajectory."""
    try:
        r = conn.base.GetTrajectoryErrorReport()
    except Exception:  # noqa: BLE001 - diagnostics must not mask the real error
        return ""
    out = []
    for el in r.trajectory_error_elements:
        try:
            kind = Base_pb2.TrajectoryErrorType.Name(el.error_type)
        except ValueError:
            kind = str(el.error_type)
        out.append("%s on index %d (value %.3f, allowed %.3f..%.3f)"
                   % (kind.replace("TRAJECTORY_ERROR_TYPE_", ""), el.index,
                      el.error_value, el.min_value, el.max_value))
    return (" -- " + "; ".join(out)) if out else ""


def _abort_reason(ev: list[tuple[str, str]]) -> str | None:
    aborts = [d for e, d in ev if e == "ACTION_ABORT"]
    return aborts[-1] if aborts else None


class ContactGuard:
    """Stops on CONTACT during a descent, from the arm's own force estimate (fz).

    Not used by the current stages (the plunge became one joint trajectory watched by TorqueGuard); kept, with
    its measurements, for any Cartesian descent that needs a force stop: goto_cartesian(..., contact=...).

    Triggers on STIFFNESS -- dF/dz, newtons per millimetre of descent -- not on
    force level and not on force rate. Measured on this arm:
      * a LEVEL threshold is unusable: fz drifts +1.0 .. -2.4 N over a 170 mm
        descent as the arm changes shape; 2 N tripped after 5 mm of free air;
      * a RATE threshold (N/s) is speed-dependent: the drift comes from changing
        SHAPE, so it scales with distance, not time. 0.8 N/s fired falsely at
        20 mm/s on drift of 1.2 N/s;
      * per millimetre the drift is ~0.014 N/mm typical, 0.06 N/mm worst over a
        short window, while strawberry flesh is 0.15-0.27 N/mm and a rigid
        surface far more. 0.10 N/mm sits between.
    Every move also starts with a ~4 N/s kick lasting ~0.4 s (acceleration), so
    the first `settle_s` is ignored and the baseline is taken after it.
    Upward push on the utensil tip reads as +fz (finger-press test).
    """

    def __init__(self, k_npmm: float = 0.10, window_mm: float = 3.0, settle_s: float = 0.8, cap_n: float = 3.0,
                 confirm: int = 3):
        self.k, self.win, self.settle, self.cap, self.confirm = k_npmm, window_mm, settle_s, cap_n, confirm
        self.d: list[float] = []
        self.f: list[float] = []
        self.base: float | None = None
        self.z0: float | None = None
        self.hits = 0

    def update(self, t: float, fz: float, z: float) -> str | None:
        """t s since the move began, fz N, z tool height m. Reason string on contact."""
        if t < self.settle:
            return None
        if self.base is None:
            self.base, self.z0 = fz, z
        d = 1000.0 * (self.z0 - z)                   # mm descended since settling
        if fz - self.base > self.cap:
            return "fz %.2f N above its settled value (cap %.1f)" % (fz - self.base, self.cap)
        self.d.append(d)
        self.f.append(fz)
        while self.d and self.d[-1] - self.d[0] > self.win:
            self.d.pop(0)
            self.f.pop(0)
        if len(self.d) >= 4 and self.d[-1] - self.d[0] > 0.5 * self.win:
            k = float(np.polyfit(self.d, self.f, 1)[0])
            self.hits = self.hits + 1 if k > self.k else 0
            if self.hits >= self.confirm:
                return "stiffness %.3f N/mm (limit %.2f) for %d samples" % (k, self.k, self.hits)
        return None


def _watch(conn: KinovaConnection, label: str, arrived: Callable[[Any, Joints], bool], start_q: Joints,
           max_tilt: float, torque_limit: float, timeout: float, ev: list[tuple[str, str]],
           contact: ContactGuard | None = None) -> tuple[float, float]:
    """Shared monitor for both move types: (worst tilt deg, seconds) on arrival; raises MoveError on every
    failure, after stopping the arm. Every sample goes to the run record (runlog.Telemetry)."""
    tel = runlog.Telemetry(label)
    outcome = "error"
    try:
        worst, dt = _watch_loop(conn, label, arrived, start_q, max_tilt, torque_limit, timeout, ev, contact, tel)
        outcome = "done"
        return worst, dt
    except MoveError as e:
        outcome = str(e)
        raise
    finally:
        tel.save(outcome=outcome)


def _watch_loop(conn, label, arrived, start_q, max_tilt, torque_limit, timeout, ev, contact, tel):
    guard = TorqueGuard(limit=torque_limit)
    t0 = time.time()
    worst = 0.0
    moved = False
    while time.time() - t0 < timeout:
        time.sleep(0.05)
        fb = conn.base_cyclic.RefreshFeedback()
        tel.add(fb)
        t = tilt_of(fb.base)
        worst = max(worst, t)
        fz = fb.base.tool_external_wrench_force_z
        now = time.time() - t0
        if contact is not None:
            why = contact.update(now, fz, fb.base.tool_pose_z)
            if why:
                conn.base.Stop()
                raise MoveError("%s STOPPED ON CONTACT: %s, at z %.4f"
                                % (label, why, fb.base.tool_pose_z))
        if t > max_tilt:
            conn.base.Stop()
            raise MoveError("%s ABORTED: utensil %.1f deg off vertical (limit %.0f)"
                            % (label, t, max_tilt))
        hit = guard.update([a.torque for a in fb.actuators])
        if hit:
            j, now, base = hit
            conn.base.Stop()
            raise MoveError("%s ABORTED: joint %d torque stepped %.2f -> %.2f N.m -- "
                            "something is in contact" % (label, j + 1, base, now))
        why = _abort_reason(ev)
        if why:
            raise MoveError("%s ABORTED by the controller: %s%s" % (label, why, _report(conn)))
        q = [fold(a.position) for a in fb.actuators]
        if not moved and max(abs(fold(a - b)) for a, b in zip(q, start_q)) > float(cfg.motion.moved_deg):
            moved = True
        if arrived(fb, q):
            # Let it actually stop. A new trajectory sent while the last is
            # still decelerating can be dropped -- that is the other way a
            # command gets silently ignored.
            t_settle = time.time()
            while time.time() - t_settle < float(cfg.motion.settle_max_s):
                vel = conn.base_cyclic.RefreshFeedback().actuators
                if max(abs(a.velocity) for a in vel) < float(cfg.motion.settle_speed_dps):
                    break
                time.sleep(0.05)
            return worst, time.time() - t0
        if not moved and time.time() - t0 > NO_PROGRESS_S:
            conn.base.Stop()
            raise MoveError("%s IGNORED: nothing moved in %.0f s. Controller events: %s"
                            % (label, NO_PROGRESS_S, ev or "none received"))
    conn.base.Stop()
    raise MoveError("%s TIMED OUT after %.0f s" % (label, timeout))


def _braked_on_arrival(e: MoveError, arrived: bool, label: str) -> bool:
    """A torque-step stop with the arm already at its target: the move was complete, the step was the arm's own
    braking (seen at 65 deg/s and 0.22 m/s, nothing touched). Anything else is a real stop."""
    if arrived and "torque stepped" in str(e):
        log.info("    %-24s arrived; the torque step was the braking (%s)", label, str(e).split(": ", 1)[-1][:60])
        runlog.event("braked_on_arrival", label=label, stop=str(e))
        return True
    return False


def goto(conn: KinovaConnection, target: Sequence[float], speed: float = 10.0, tol: float = 0.08,
         label: str = "", max_tilt: float = MAX_TILT_DEG, torque_limit: float = TORQUE_STEP_NM,
         verbose: bool = True) -> None:
    """Joint move to a COMPLETE 7-joint target, keeping the utensil down.

    speed is the fastest joint's average deg/s (the move is time-synchronised: every joint starts and ends
    together); tol the arrival tolerance per joint, deg. Refuses (MoveError, nothing moved) a target near a
    joint limit or one that does not point the utensil down within max_tilt.
    """
    target = [fold(v) for v in target]
    if len(target) != 7:
        raise MoveError("%s REFUSED: need all 7 joints, never a single-joint move" % label)
    for i, lim in LIMITS.items():
        if abs(target[i]) > lim - LIMIT_MARGIN:
            raise MoveError("%s REFUSED: J%d target %.2f is at/past its +-%.1f limit; "
                            "Kortex would silently ignore it" % (label, i + 1, target[i], lim))
    s = pitch_sum(target)
    if abs(fold(s + 180.0)) > max_tilt:
        raise MoveError("%s REFUSED: target J2+J4+J6 = %.1f, %.1f deg off the -180 that "
                        "points the utensil down" % (label, s, abs(fold(s + 180.0))))
    ev = _preflight(conn, label)
    q0 = joints(conn)
    travel = max(abs(fold(t - c)) for t, c in zip(target, q0))
    if travel < tol:
        if verbose:
            log.info("    %-24s already there", label)
        return
    a = Base_pb2.Action()
    a.name = label or "goto"
    rj = a.reach_joint_angles
    for i, v in enumerate(target):
        j = rj.joint_angles.joint_angles.add()
        j.joint_identifier = i
        j.value = v % 360.0            # Kortex's own [0, 360); it takes the short way
    # DURATION, not SPEED. A speed constraint asks EVERY joint to reach that
    # speed, so a joint with a sliver of travel must accelerate and stop within
    # it -- the controller rejects that as INVALID_JOINT_ACCELERATION (read via
    # GetTrajectoryErrorReport: J3, 0.7 deg of travel, 8.0 requested, 7.56 max).
    # A duration makes all joints start and finish together, so each joint's
    # acceleration scales with its own travel. It is also what makes the
    # utensil-down argument hold: only a time-synchronised move interpolates
    # J2+J4+J6 linearly between its endpoints. Under a per-joint speed limit a
    # short-travel joint finishes early and the sum wanders.
    rj.constraint.type = Base_pb2.JOINT_CONSTRAINT_DURATION
    from kinova.motion import capped              # (motion imports this module)
    v, acc = capped(conn, speed, math.inf)        # under the arm's own limits: faster is refused, not clamped
    ramp = v / acc if math.isfinite(acc) else 0.0  # the time the arm's acceleration limit needs to reach v
    rj.constraint.value = max(float(cfg.motion.goto_min_s), travel / v + max(float(cfg.motion.goto_pad_s), ramp))
    tries = int(cfg.motion.goto_duration_retries) + 1
    check_tool(conn)
    for attempt in range(1, tries + 1):
        conn.base.ExecuteAction(a)
        try:
            worst, dt = _watch(conn, label,
                               lambda fb, q: all(abs(fold(a - b)) < tol for a, b in zip(target, q)),
                               q0, max_tilt, torque_limit, max(20.0, 6.0 * travel / speed), ev)
            break
        except MoveError as e:
            if _braked_on_arrival(e, all(abs(fold(a - b)) < tol for a, b in zip(target, joints(conn))), label):
                worst, dt = float("nan"), float("nan")
                break
            # Faster than the arm's configured joint limits allow: the controller refuses the duration BEFORE any
            # motion and says its minimum -- which is not stable (a 75 deg swing half: 1.69, then 1.93 s for the
            # re-sent one). Follow it, a few times, each at the minimum just reported + 5%
            m = re.search(r"INVALID_DURATION on index \d+ \(value [\d.]+, allowed ([\d.]+)\.\.", str(e))
            if attempt == tries or not m or max(abs(fold(t - c)) for t, c in zip(q0, joints(conn))) > 0.05:
                raise
            rj.constraint.value = 1.05 * float(m.group(1))
            if verbose:
                log.info("    %-24s (the arm's minimum for this move: %.2f s)", label, float(m.group(1)))
            ev.clear()
    runlog.event("goto", label=label, target=target, travel_deg=travel, seconds=dt, worst_tilt_deg=worst)
    if verbose and dt == dt:                      # (nan: arrived on a braking step, already said)
        log.info("    %-24s ok  %6.1f deg in %4.1f s, worst tilt %.1f deg", label, travel, dt, worst)


def goto_cartesian(conn: KinovaConnection, xyz: Sequence[float], rpy: Sequence[float], speed: float = 0.05,
                   tol_m: float = 0.002, label: str = "", max_tilt: float = MAX_TILT_DEG,
                   torque_limit: float = TORQUE_STEP_NM, verbose: bool = True,
                   contact: ContactGuard | None = None) -> None:
    """One smooth straight-line tool move to (xyz m, rpy deg) at `speed` m/s -- not a staircase."""
    ev = _preflight(conn, label)
    p0 = tool_pose(conn)
    dist = math.dist(p0[:3], xyz)
    if dist < tol_m:
        if verbose:
            log.info("    %-24s already there", label)
        return
    a = Base_pb2.Action()
    a.name = label or "goto_cartesian"
    cp = a.reach_pose
    t = cp.target_pose
    t.x, t.y, t.z = xyz
    t.theta_x, t.theta_y, t.theta_z = rpy
    cp.constraint.speed.translation = speed
    cp.constraint.speed.orientation = float(cfg.motion.cartesian_orientation_dps)
    check_tool(conn)
    conn.base.ExecuteAction(a)

    def arrived(fb: Any, q: Joints) -> bool:
        b = fb.base
        return math.dist((b.tool_pose_x, b.tool_pose_y, b.tool_pose_z), xyz) < tol_m

    try:
        worst, dt = _watch(conn, label, arrived, joints(conn), max_tilt, torque_limit,
                           max(20.0, 4.0 * dist / speed), ev, contact=contact)
    except MoveError as e:
        b = conn.base_cyclic.RefreshFeedback().base
        if not _braked_on_arrival(e, math.dist((b.tool_pose_x, b.tool_pose_y, b.tool_pose_z), xyz) < tol_m, label):
            raise
        worst, dt = float("nan"), float("nan")
    runlog.event("goto_cartesian", label=label, target=list(xyz) + list(rpy), distance_m=dist, seconds=dt,
                 worst_tilt_deg=worst)
    if verbose and dt == dt:
        log.info("    %-24s ok  %5.0f mm in %4.1f s, worst tilt %.1f deg", label, 1000 * dist, dt, worst)
