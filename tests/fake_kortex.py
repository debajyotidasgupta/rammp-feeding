"""A stand-in for the Kortex controller, so the arm stages run without the arm.

Good enough to exercise every code path of sequence / acquire / deliver deterministically, and to record
exactly what they asked the controller for -- not a physics or kinematics model:

  * IK is a smooth, deterministic function of the requested pose: the pose encoded in the joints (position
    * 100, orientation quaternion * 60, its sign kept continuous with the guess) plus a correction that
    blends (inverse-distance) into the REAL joints at every measured pose (calibration/*.json, the bowl's
    scoop end). So a small tool move is a small joint move, paths starting at a measured pose are continuous,
    and goto() to a measured pose, the joint-limit and utensil-down checks all see real values.
  * FK returns the pose each IK answer was computed for (remembered), the measured poses for their joints,
    and for anything else (blends) the pose of the nearest joint vector it knows.
  * Moves take their planned time on a simulated clock (`clock`): every RefreshFeedback is 1 ms, every
    time.sleep() is its argument. Joint and tool pose interpolate along the move; ACTION_END fires at its
    end. Nothing depends on the wall clock.

`FakeArm.calls` is the trace: every controller call with its request (runs of RefreshFeedback are counted).
"""

from __future__ import annotations

import math
import time as _time
from types import SimpleNamespace

import numpy as np
from google.protobuf.json_format import MessageToDict
from kortex_api.autogen.messages import Base_pb2


# ------------------------------------------------------------------ clock

class Clock:
    def __init__(self, start: float):
        self.t = start

    def time(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(0.0, float(s))


clock = Clock(_time.time())


def install_clock() -> None:
    """Route time.time / time.sleep through the simulated clock (for every module)."""
    _time.time = clock.time
    _time.sleep = clock.sleep


# ------------------------------------------------------------------ rotations (independent of the code under test)

def R_rpy(tx, ty, tz):
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    Rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    Rz = np.array([[math.cos(tz), -math.sin(tz), 0], [math.sin(tz), math.cos(tz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rpy_R(R):
    ty = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])), ty, math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def quat_R(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def R_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fold(d):
    return (d + 180.0) % 360.0 - 180.0


def encode(pose, guess=None):
    """Pose (x y z m, rpy deg) -> 7 'joints'."""
    q = quat_R(R_rpy(*pose[3:]))
    if guess is not None and np.dot(q, np.array([fold(g) for g in guess[3:7]]) / 60.0) < 0:
        q = -q
    return [100 * pose[0], 100 * pose[1], 100 * pose[2]] + list(60 * q)


def decode(joints):
    j = [fold(v) for v in joints]
    return [j[0] / 100, j[1] / 100, j[2] / 100] + rpy_R(R_quat(np.array(j[3:7]) / 60.0))


def rot_angle(a, b):
    """Angle (rad) between two rpy orientations."""
    R = R_rpy(*a[3:]).T @ R_rpy(*b[3:])
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))


def lerp_pose(a, b, s):
    qa, qb = quat_R(R_rpy(*a[3:])), quat_R(R_rpy(*b[3:]))
    if np.dot(qa, qb) < 0:
        qb = -qb
    return [a[i] + (b[i] - a[i]) * s for i in range(3)] + rpy_R(R_quat(qa + (qb - qa) * s))


def rounded(x, sig=9):
    """Floats to `sig` significant digits and 1e-10 absolute, recursively (traces compare as text; platform float
    noise in near-zero differences stays out)."""
    if isinstance(x, float):
        return round(float("%.*g" % (sig, x)), 10)
    if isinstance(x, dict):
        return {k: rounded(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, sig) for v in x]
    return x


# ------------------------------------------------------------------ the arm

class FakeArm:
    def __init__(self):
        self.q = [0.0] * 7
        self.pose = [0.3, 0.0, 0.4, 180.0, 0.0, 90.0]
        self.known = []                  # (joints, pose, offset, encoding): measured poses with real joints
        self.memo = {}                   # IK answers -> the pose they were computed for
        self.motion = None               # (t0, [(t_end, q_a, q_b, pose_a, pose_b)], total)
        self.cb = None
        self.calls = []
        self.torque = lambda t: [0.0] * 7          # scenario hooks: f(clock time) -> values
        self.force = lambda t: [0.0, 0.0, 0.0]
        self.refuse_above_dps = None     # ValidateWaypointList refuses any segment faster than this
        self.ik_unreachable = lambda pose: False

    # --- scenario setup
    def register(self, joints, pose):
        q = [fold(v) for v in joints]
        e = encode(pose, q)
        # the sign whose encoding is nearer the real joints, so the offset is small where it can be
        e2 = [e[0], e[1], e[2]] + [-v for v in e[3:]]
        if sum(abs(fold(a - b)) for a, b in zip(e2, q)) < sum(abs(fold(a - b)) for a, b in zip(e, q)):
            e = e2
        self.known.append((q, list(pose), [fold(a - b) for a, b in zip(q, e)], e))

    def place(self, joints=None, pose=None):
        if pose is None:
            pose = self.fk_pose(joints)
        if joints is None:
            joints = self.joints_for(pose)
        self.q, self.pose, self.motion = [fold(v) for v in joints], list(pose), None

    def _offset(self, pose):
        num, den = np.zeros(7), 0.0
        for q, p, off, _ in self.known:
            d = math.dist(p[:3], pose[:3]) + 0.1 * rot_angle(p, pose)
            if d < 1e-9:
                return np.array(off)
            w = 1.0 / d ** 2
            num += w * np.array(off)
            den += w
        return num / den if den else num

    def solve(self, pose, guess):
        """The fake IK: pose -> joints, the quaternion sign nearest the guess. Remembered for FK."""
        off = self._offset(pose)
        best = None
        for sign in (1, -1):
            e = encode(pose)
            e = e[:3] + [sign * v for v in e[3:]]
            q = [fold(a + b) for a, b in zip(e, off)]
            d = sum(abs(fold(a - b)) for a, b in zip(q, guess))
            if best is None or d < best[0]:
                best = (d, q)
        q = best[1]
        self.memo[self._key(q)] = list(pose)
        return q

    @staticmethod
    def _key(q):
        return tuple(round(fold(v), 4) for v in q)

    def fk_pose(self, joints):
        j = [fold(v) for v in joints]
        for kj, kp, _, _ in self.known:
            if max(abs(fold(a - b)) for a, b in zip(kj, j)) < 1e-4:
                return list(kp)
        hit = self.memo.get(self._key(j))
        if hit is not None:
            return list(hit)
        # a blend of two measured poses, or of a measured pose and an IK answer (e.g. a joint midpoint): blend
        # their poses
        ends = [(kj, kp) for kj, kp, _, _ in self.known] + [(list(k), v) for k, v in self.memo.items()]
        for i, (qa, pa) in enumerate(ends[:len(self.known)]):
            for qb, pb in ends[i + 1:]:
                dq = np.array([fold(b - a) for a, b in zip(qa, qb)])
                dj = np.array([fold(x - a) for a, x in zip(qa, j)])
                if dq @ dq < 1e-12:
                    continue
                s = float(dj @ dq / (dq @ dq))
                if 0 <= s <= 1 and np.abs(dj - s * dq).max() < 1e-3:
                    return lerp_pose(pa, pb, s)
        cands = [(kj, kp) for kj, kp, _, _ in self.known] + [(list(k), v) for k, v in self.memo.items()]
        return list(min(cands, key=lambda c: max(abs(fold(a - b)) for a, b in zip(c[0], j)))[1])

    def joints_for(self, pose, guess=None):
        for kj, kp, _, _ in self.known:
            if max(abs(a - b) for a, b in zip(kp[:3], pose[:3])) < 2e-4 and rot_angle(kp, pose) < 3e-4:
                return list(kj)
        return self.solve(pose, guess if guess is not None else self.q)

    # --- trace
    def log(self, name, req=None):
        if name == "RefreshFeedback" and self.calls and self.calls[-1][0] == "RefreshFeedback":
            self.calls[-1][1] += 1
            return
        if name == "RefreshFeedback":
            self.calls.append(["RefreshFeedback", 1])
            return
        body = None
        if req is not None:
            body = rounded(MessageToDict(req, preserving_proto_field_name=True)) if hasattr(req, "DESCRIPTOR") else rounded(req)
        self.calls.append([name, body])

    # --- motion
    def _advance(self):
        if self.motion is None:
            return
        t0, segs, total = self.motion
        t = clock.time() - t0
        for t_end, qa, qb, pa, pb, t_start in segs:
            if t <= t_end:
                s = 0.0 if t_end <= t_start else (t - t_start) / (t_end - t_start)
                s = min(1.0, max(0.0, s))
                self.q = [fold(a + fold(b - a) * s) for a, b in zip(qa, qb)]
                self.pose = lerp_pose(pa, pb, s)
                return
        _, qa, qb, pa, pb, _ = segs[-1]
        self.q, self.pose, self.motion = list(qb), list(pb), None
        self._notify(Base_pb2.ACTION_END)

    def _start(self, points, durations):
        """points: [(joints, pose)] from the current state; durations per segment."""
        segs, t = [], 0.0
        prev_q, prev_p = list(self.q), list(self.pose)
        for (q, p), dt in zip(points, durations):
            segs.append((t + dt, prev_q, q, prev_p, p, t))
            t += dt
            prev_q, prev_p = q, p
        self.motion = (clock.time(), segs, t)
        self._notify(Base_pb2.ACTION_START)

    def _notify(self, event, detail=0):
        if self.cb:
            self.cb(SimpleNamespace(action_event=event, abort_details=detail))

    def feedback(self):
        clock.t += 0.001
        self._advance()
        t = clock.time()
        tau, f = self.torque(t), self.force(t)
        p = self.pose
        base = SimpleNamespace(tool_pose_x=p[0], tool_pose_y=p[1], tool_pose_z=p[2],
                               tool_pose_theta_x=p[3], tool_pose_theta_y=p[4], tool_pose_theta_z=p[5],
                               tool_external_wrench_force_x=f[0], tool_external_wrench_force_y=f[1],
                               tool_external_wrench_force_z=f[2])
        acts = [SimpleNamespace(position=v % 360.0, torque=tau[i], velocity=0.0) for i, v in enumerate(self.q)]
        return SimpleNamespace(base=base, actuators=acts)


class _Base:
    def __init__(self, arm: FakeArm):
        self.arm = arm

    def ComputeInverseKinematics(self, d):
        a = self.arm
        a.log("ComputeInverseKinematics", d)
        p = d.cartesian_pose
        pose = [p.x, p.y, p.z, p.theta_x, p.theta_y, p.theta_z]
        if a.ik_unreachable(pose):
            raise RuntimeError("IK: unreachable (fake)")
        guess = [j.value for j in d.guess.joint_angles]
        return SimpleNamespace(joint_angles=[SimpleNamespace(value=v % 360.0) for v in a.solve(pose, guess)])

    def ComputeForwardKinematics(self, ja):
        a = self.arm
        a.log("ComputeForwardKinematics", ja)
        p = a.fk_pose([j.value for j in ja.joint_angles])
        # blends between an encoded and a measured joint vector decode to nonsense heights: floor them, so the
        # retract's "tip never below" check sees a plausible return rather than a fake artefact
        return SimpleNamespace(x=p[0], y=p[1], z=max(p[2], 0.35), theta_x=p[3], theta_y=p[4], theta_z=p[5])

    def ValidateWaypointList(self, wl):
        a = self.arm
        a.log("ValidateWaypointList", wl)
        errs = []
        if a.refuse_above_dps is not None:
            prev = list(a.q)
            for k, w in enumerate(wl.waypoints):
                ang = list(w.angular_waypoint.angles)
                v = max(abs(fold(x - y)) for x, y in zip(ang, prev)) / max(1e-6, w.angular_waypoint.duration)
                if v > a.refuse_above_dps:
                    errs.append("INVALID_JOINT_SPEED on index %d (value %.3f)" % (k, v))
                    break
                prev = ang
        return SimpleNamespace(trajectory_error_report=SimpleNamespace(trajectory_error_elements=errs))

    def ExecuteWaypointTrajectory(self, wl):
        a = self.arm
        a.log("ExecuteWaypointTrajectory", wl)
        pts = [([fold(v) for v in w.angular_waypoint.angles], None) for w in wl.waypoints]
        pts = [(q, a.fk_pose(q)) for q, _ in pts]
        a._start(pts, [w.angular_waypoint.duration for w in wl.waypoints])

    def ExecuteAction(self, act):
        a = self.arm
        a.log("ExecuteAction", act)
        if act.HasField("reach_joint_angles"):
            rj = act.reach_joint_angles
            q = [fold(j.value) for j in rj.joint_angles.joint_angles]
            a._start([(q, a.fk_pose(q))], [float(rj.constraint.value)])
        else:
            t = act.reach_pose.target_pose
            pose = [t.x, t.y, t.z, t.theta_x, t.theta_y, t.theta_z]
            dist = math.dist(a.pose[:3], pose[:3])
            a._start([(a.joints_for(pose), pose)], [max(0.2, dist / act.reach_pose.constraint.speed.translation)])

    def Stop(self):
        a = self.arm
        a.log("Stop")
        a.motion = None

    def OnNotificationActionTopic(self, cb, opts):
        self.arm.log("OnNotificationActionTopic")
        self.arm.cb = cb

    def GetArmState(self):
        self.arm.log("GetArmState")
        return SimpleNamespace(active_state=Base_pb2.ARMSTATE_SERVOING_READY)

    def GetTrajectoryErrorReport(self):
        self.arm.log("GetTrajectoryErrorReport")
        return SimpleNamespace(trajectory_error_elements=[])


class _Cyclic:
    def __init__(self, arm: FakeArm):
        self.arm = arm

    def RefreshFeedback(self):
        self.arm.log("RefreshFeedback")
        return self.arm.feedback()


class FakeConnection:
    """What kinova_connection() yields: .base and .base_cyclic."""

    def __init__(self, arm: FakeArm):
        self.base, self.base_cyclic = _Base(arm), _Cyclic(arm)
