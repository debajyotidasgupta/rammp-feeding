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
import statistics
import time
from collections import deque

from kortex_api.autogen.messages import Base_pb2

MAX_TILT_DEG = 25.0          # how far off straight-down the utensil may ever get
TORQUE_STEP_NM = 2.5         # a step this big against recent history = contact
NO_PROGRESS_S = 3.0          # nothing moved by then = the controller ignored it
# Gen3 soft limits for the three limited joints (J2, J4, J6; 0-indexed), with a
# little margin: a target AT the limit is as likely to be refused as one past it.
LIMITS = {1: 128.9, 3: 147.8, 5: 120.3}
LIMIT_MARGIN = 0.3


def fold(d: float) -> float:
    """Kortex reports [0, 360); joints 1/3/5/7 are continuous and sit near the
    wrap, so always compare and interpolate on (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def approach_axis(tx: float, ty: float, tz: float):
    """Tool +z (where the utensil points) in base coords, from Kortex's
    fixed-frame XYZ Euler angles: R = Rz @ Ry @ Rx, third column."""
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    return (cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx)


def tilt_of(base) -> float:
    ax = approach_axis(base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z)
    return math.degrees(math.acos(max(-1.0, min(1.0, -ax[2]))))


def tilt_deg(conn) -> float:
    """Angle between the utensil and straight down, right now."""
    return tilt_of(conn.base_cyclic.RefreshFeedback().base)


def joints(conn) -> list[float]:
    return [fold(a.position) for a in conn.base_cyclic.RefreshFeedback().actuators]


def tool_pose(conn) -> list[float]:
    b = conn.base_cyclic.RefreshFeedback().base
    return [b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
            b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z]


def pitch_sum(q) -> float:
    return fold(q[1] + q[3] + q[5])


class TorqueGuard:
    """Fires when any joint's torque steps away from its own recent history.

    `gap` samples separate the baseline window from the present sample, so a
    fast-rising contact torque cannot contaminate the baseline it is compared
    against.
    """

    def __init__(self, n=7, span=12, gap=5, limit=TORQUE_STEP_NM):
        self.hist = [deque(maxlen=span + gap) for _ in range(n)]
        self.span, self.gap, self.limit = span, gap, limit

    def update(self, tau):
        for i, t in enumerate(tau):
            self.hist[i].append(t)
        for i, h in enumerate(self.hist):
            if len(h) < self.span + self.gap:
                continue
            base = statistics.median(list(h)[: self.span])
            if abs(h[-1] - base) > self.limit:
                return i, h[-1], base
        return None


class MoveError(RuntimeError):
    """A move was refused, ignored, or stopped. The arm is stationary."""


_BAD_STATES = {Base_pb2.ARMSTATE_IN_FAULT, Base_pb2.ARMSTATE_MAINTENANCE,
               Base_pb2.ARMSTATE_BASE_INITIALIZATION}


def _events(conn):
    """Action notifications for this connection, subscribed once."""
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


def _preflight(conn, label):
    s = conn.base.GetArmState().active_state
    if s in _BAD_STATES:
        raise MoveError("%s REFUSED: arm state is %s" % (label, Base_pb2.ArmState.Name(s)))
    ev = _events(conn)
    ev.clear()
    return ev


def _report(conn) -> str:
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


def _abort_reason(ev):
    aborts = [d for e, d in ev if e == "ACTION_ABORT"]
    return aborts[-1] if aborts else None


def _watch(conn, label, arrived, start_q, max_tilt, torque_limit, timeout, ev,
           force_limit=None, log=None):
    """Shared monitor for both move types. Raises on every failure.

    force_limit: stop if the arm's own external-force estimate along base z
    rises this much above its value at the start of the move. An upward push on
    the utensil tip reads as +fz (finger-press test: rest noise +-0.3 N, light
    press +0.7 N, firm +3 N). log: a list to append per-sample records to.
    """
    guard = TorqueGuard(limit=torque_limit)
    fz0 = conn.base_cyclic.RefreshFeedback().base.tool_external_wrench_force_z
    t0 = time.time()
    worst = 0.0
    moved = False
    while time.time() - t0 < timeout:
        time.sleep(0.05)
        fb = conn.base_cyclic.RefreshFeedback()
        t = tilt_of(fb.base)
        worst = max(worst, t)
        fz = fb.base.tool_external_wrench_force_z
        if log is not None:
            log.append(dict(t=time.time() - t0, z=fb.base.tool_pose_z, fz=fz,
                            fx=fb.base.tool_external_wrench_force_x,
                            fy=fb.base.tool_external_wrench_force_y,
                            tau=[a.torque for a in fb.actuators], label=label))
        if force_limit is not None and fz - fz0 > force_limit:
            conn.base.Stop()
            raise MoveError("%s STOPPED ON FORCE: fz rose %.2f N (limit %.1f) at z %.4f -- "
                            "something is pushing up on the utensil"
                            % (label, fz - fz0, force_limit, fb.base.tool_pose_z))
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
        if not moved and max(abs(fold(a - b)) for a, b in zip(q, start_q)) > 0.05:
            moved = True
        if arrived(fb, q):
            # Let it actually stop. A new trajectory sent while the last is
            # still decelerating can be dropped -- that is the other way a
            # command gets silently ignored.
            t_settle = time.time()
            while time.time() - t_settle < 2.0:
                vel = conn.base_cyclic.RefreshFeedback().actuators
                if max(abs(a.velocity) for a in vel) < 0.2:
                    break
                time.sleep(0.05)
            return worst, time.time() - t0
        if not moved and time.time() - t0 > NO_PROGRESS_S:
            conn.base.Stop()
            raise MoveError("%s IGNORED: nothing moved in %.0f s. Controller events: %s"
                            % (label, NO_PROGRESS_S, ev or "none received"))
    conn.base.Stop()
    raise MoveError("%s TIMED OUT after %.0f s" % (label, timeout))


def goto(conn, target, speed: float = 10.0, tol: float = 0.08, label: str = "",
         max_tilt: float = MAX_TILT_DEG, torque_limit: float = TORQUE_STEP_NM,
         verbose: bool = True):
    """Joint move to a COMPLETE 7-joint target, keeping the utensil down."""
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
            print("    %-24s already there" % label)
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
    rj.constraint.value = max(1.5, travel / speed + 1.0)
    conn.base.ExecuteAction(a)
    worst, dt = _watch(conn, label,
                       lambda fb, q: all(abs(fold(a - b)) < tol for a, b in zip(target, q)),
                       q0, max_tilt, torque_limit, max(20.0, 6.0 * travel / speed), ev)
    if verbose:
        print("    %-24s ok  %6.1f deg in %4.1f s, worst tilt %.1f deg" % (label, travel, dt, worst))


def goto_cartesian(conn, xyz, rpy, speed: float = 0.05, tol_m: float = 0.002,
                   label: str = "", max_tilt: float = MAX_TILT_DEG,
                   torque_limit: float = TORQUE_STEP_NM, verbose: bool = True,
                   force_limit=None, log=None):
    """One smooth straight-line tool move to (xyz, rpy) -- not a staircase."""
    ev = _preflight(conn, label)
    p0 = tool_pose(conn)
    dist = math.dist(p0[:3], xyz)
    if dist < tol_m:
        if verbose:
            print("    %-24s already there" % label)
        return
    a = Base_pb2.Action()
    a.name = label or "goto_cartesian"
    cp = a.reach_pose
    t = cp.target_pose
    t.x, t.y, t.z = xyz
    t.theta_x, t.theta_y, t.theta_z = rpy
    cp.constraint.speed.translation = speed
    cp.constraint.speed.orientation = 10.0
    conn.base.ExecuteAction(a)

    def arrived(fb, q):
        b = fb.base
        return math.dist((b.tool_pose_x, b.tool_pose_y, b.tool_pose_z), xyz) < tol_m

    worst, dt = _watch(conn, label, arrived, joints(conn), max_tilt, torque_limit,
                       max(20.0, 4.0 * dist / speed), ev, force_limit=force_limit, log=log)
    if verbose:
        print("    %-24s ok  %5.0f mm in %4.1f s, worst tilt %.1f deg"
              % (label, 1000 * dist, dt, worst))
