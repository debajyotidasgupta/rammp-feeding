#!/usr/bin/env python3
"""Joint moves that keep the utensil pointing at the floor, or refuse to run.

Importable helper, not a command. Every motion in the stage-1 sequence goes
through goto(), which watches the tool's approach axis for the whole move and
stops the arm the moment it leaves the downward cone.

WHY THIS EXISTS. On 2026-09-23 a diagnostic moved J6 alone, from -119 to -40,
while J2 and J4 were still in the overlook configuration. Tool pitch on this arm
is set by the SUM:

    J2 + J4 + J6 ~= -180  =>  utensil points down

  feeding_start    8.07 - 139.20 -  40.08 = -171.2   ok
  food_detect     -4.10 -  57.08 - 119.24 = -180.4   ok
  after that move -4.10 -  57.08 -  40.08 = -101.3   fork 79 deg off, horizontal

A simultaneous interpolation between two utensil-down poses keeps the sum between
their two values, so it stays safe. Moving ONE joint of the three does not --
it walks the sum straight out of the safe band. The fork ended up pointing
sideways at a person. Never move J2, J4 or J6 in isolation.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque

from kortex_api.autogen.messages import Base_pb2

MAX_TILT_DEG = 25.0          # how far off straight-down the utensil may ever get
PITCH_JOINTS = (1, 3, 5)     # J2, J4, J6 -- the three that set tool pitch
TORQUE_STEP_NM = 2.5         # a step this big against recent history = contact

# Orientation is NOT enough. On 2026-09-23 the fork tines swung into the arm's
# OWN joint 6 housing and were dragged across it. The tool angle was legal
# throughout -- a self-collision does not need an illegal angle, and a utensil
# is clamped with a lateral offset from the tool axis, so wrist rotations sweep
# the tines through arcs that can re-enter the wrist body.
#
# The geometric check for that (sim/safety.py, whole-arm mj_geomDistance with
# the tool modelled) was deleted in the repo cleanup and is not back yet. Until
# it is, TORQUE is the only thing that notices contact, so no wrist motion
# should run without it.


def fold(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def approach_axis(tx: float, ty: float, tz: float):
    """Tool +z in base coords, from Kortex fixed-frame XYZ Euler angles."""
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    return (cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx)


def tilt_deg(conn) -> float:
    """Angle between the utensil and straight down, right now."""
    b = conn.base_cyclic.RefreshFeedback().base
    ax = approach_axis(b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z)
    return math.degrees(math.acos(max(-1.0, min(1.0, -ax[2]))))


def joints(conn) -> list[float]:
    return [fold(a.position) for a in conn.base_cyclic.RefreshFeedback().actuators]


def pitch_sum(q) -> float:
    return fold(q[1] + q[3] + q[5])


class TorqueGuard:
    """Fires when any joint's torque steps away from its own recent history.

    Geometry-free, so it still works when the collision model is missing --
    which is exactly the situation it was written for. `gap` samples separate
    the baseline window from the present sample so a fast-rising contact
    torque cannot contaminate the baseline it is compared against.
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


def torques(conn):
    return [a.torque for a in conn.base_cyclic.RefreshFeedback().actuators]


def goto(conn, target, speed: float = 10.0, tol: float = 0.08,
         max_tilt: float = MAX_TILT_DEG, label: str = "", verbose: bool = True,
         torque_limit: float = TORQUE_STEP_NM):
    """Command a full 7-joint target and hold the utensil down for the whole move.

    `target` MUST be a complete joint vector whose own pitch sum is safe.
    Refuses before moving if the destination would point the utensil out, and
    stops the arm mid-move if the measured tilt ever exceeds `max_tilt`.
    """
    q0 = joints(conn)
    end_sum = pitch_sum(target)
    if abs(fold(end_sum + 180.0)) > max_tilt:
        raise RuntimeError(
            "%s REFUSED: target pitch sum J2+J4+J6 = %.1f deg, %.1f deg off the "
            "-180 that points the utensil down" % (label, end_sum, abs(fold(end_sum + 180.0))))

    m = Base_pb2.ConstrainedJointAngles()
    for i, v in enumerate(target):
        j = m.joint_angles.joint_angles.add()
        j.joint_identifier = i
        j.value = v
    m.constraint.type = Base_pb2.JOINT_CONSTRAINT_SPEED
    m.constraint.value = speed
    conn.base.PlayJointTrajectory(m)

    worst_tilt = 0.0
    guard = TorqueGuard(limit=torque_limit)
    travel = max(abs(fold(t - c)) for t, c in zip(target, q0))
    deadline = time.time() + max(20.0, 6.0 * travel / speed)
    while time.time() < deadline:
        time.sleep(0.05)
        fb = conn.base_cyclic.RefreshFeedback()
        b = fb.base
        ax = approach_axis(b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z)
        t = math.degrees(math.acos(max(-1.0, min(1.0, -ax[2]))))
        worst_tilt = max(worst_tilt, t)
        if t > max_tilt:
            conn.base.Stop()
            raise RuntimeError("%s ABORTED mid-move: utensil reached %.1f deg off "
                               "vertical (limit %.0f)" % (label, t, max_tilt))
        hit = guard.update([a.torque for a in fb.actuators])
        if hit:
            j, now, base = hit
            conn.base.Stop()
            raise RuntimeError("%s ABORTED mid-move: joint %d torque stepped "
                               "%.2f -> %.2f N.m (%.2f over baseline) -- something "
                               "is in contact" % (label, j + 1, base, now, abs(now - base)))
        q = [fold(a.position) for a in fb.actuators]
        if all(abs(fold(a - b)) < tol for a, b in zip(target, q)):
            if verbose:
                print("    %-22s ok   travel %6.1f deg, worst tilt %4.1f deg"
                      % (label, travel, worst_tilt))
            return True
    q = joints(conn)
    err = max(abs(fold(a - b)) for a, b in zip(target, q))
    if verbose:
        print("    %-22s STUCK  %.2f deg short, worst tilt %.1f deg"
              % (label, err, worst_tilt))
    return False
