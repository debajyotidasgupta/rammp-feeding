"""Guarded arm motion on top of the arm driver: every move is a joint path we compute (our IK, a cuRobo plan), timed
here, executed by the driver (/execute_joint_trajectory, position or impedance), and watched while it runs:

  CONTACT   a joint-torque step against recent history (TorqueGuard; limits per move, guards.torque_step_nm)
  TILT      the utensil further off straight down than motion.max_tilt_deg (a move's own cap where it has one), or
            than 5 deg past the move's own start or end where either is further (a move that turns it on purpose);
            anything else the gripper holds (Held), its own axis against its own reference and limit
  CHECKS    a move's own (the plunge's depth zone, the scoop's food path)
  DRIVER    its path tolerance (the arm diverging from the path) and its own refusals
Any of them cancels the goal (the driver halts the arm) and raises MoveError. Free-space moves are cuRobo's paths
(travel), checked against our tip floor and, for the utensil, our boxes before they run.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import deque
from typing import Callable, Sequence

import numpy as np

from rammp_feeding import record
from rammp_feeding.config import cfg, unmeasured
from rammp_feeding.geometry import densify, fold, max_step, sphere_box_clearance, tilt_deg
from rammp_feeding.kinematics import Chain

log = logging.getLogger("rammp")
M = cfg.motion
Joints = list[float]
PoseAt = Callable[[float], "tuple[np.ndarray, np.ndarray]"]   # s in [0, 1] -> (tool position, R)


class MoveError(RuntimeError):
    """A move was refused, ignored, or stopped. The arm is stationary."""


def chain(robot) -> Chain:
    """The arm's kinematics from the model its description node serves (ros.robot_description), rooted at
    arm.base_frame."""
    urdf = robot.robot_description()
    if urdf is None:
        raise RuntimeError("no arm model on %s -- is the arm's description node (arm_state_publisher) up?"
                           % cfg.ros.robot_description)
    kin = Chain(urdf, cfg.arm.tool_from_flange_m)
    if kin.base != str(cfg.arm.base_frame):
        raise RuntimeError("the arm's model is rooted at %s, not arm.base_frame (%s)" % (kin.base, cfg.arm.base_frame))
    return kin


class TorqueGuard:
    """Fires when any joint's torque steps away from its own recent history: `gap` samples separate the baseline
    window from the present sample, so a fast-rising contact cannot contaminate the baseline it is compared with."""

    def __init__(self, limit: float, span: int = int(cfg.guards.torque_window.span), gap: int = int(cfg.guards.torque_window.gap)):
        self.hist = [deque(maxlen=span + gap) for _ in range(7)]
        self.span, self.limit = span, limit

    def update(self, tau: Sequence[float]) -> tuple[int, float, float] | None:
        for h, t in zip(self.hist, tau):
            h.append(float(t))
        for i, h in enumerate(self.hist):
            if len(h) == h.maxlen:
                base = statistics.median(list(h)[: self.span])
                if abs(h[-1] - base) > self.limit:
                    return i, h[-1], base
        return None


def time_scale(path: Sequence[Sequence[float]], vmax, amax: float) -> list[float]:
    """Segment durations keeping every joint under vmax (deg/s, per joint or one) and its speed changing by at most
    amax (deg/s^2), from rest and back to rest."""
    D = np.array([fold(np.asarray(q) - np.asarray(p)) for p, q in zip(path, path[1:])])
    vcap = np.broadcast_to(np.asarray(vmax, float), (7,))
    dt = np.maximum((np.abs(D) / vcap).max(1), 0.02)
    for _ in range(4000):
        V = D / dt[:, None]
        Vp = np.vstack([np.zeros(7), V, np.zeros(7)])
        mid = np.concatenate([[dt[0] / 2], (dt[1:] + dt[:-1]) / 2, [dt[-1] / 2]])
        bad = np.abs(np.diff(Vp, axis=0)).max(1) / mid > amax
        if not bad.any():
            break
        idx = np.nonzero(bad)[0]
        grow = np.zeros(len(D), bool)
        grow[np.clip(idx - 1, 0, len(D) - 1)] = grow[np.clip(idx, 0, len(D) - 1)] = True
        dt[grow] *= 1.03
    return list(dt)


def smooth_timing(path: Sequence[Sequence[float]], vmax: float, amax: float,
                  sample_s: float) -> tuple[list[np.ndarray], list[float]]:
    """A path that is one smooth curve (a planned path, a joint-space line) timed by one speed profile along it, the
    furthest joint's travel the measure -- from rest at amax (deg/s^2) up to vmax (deg/s), cruising, back to rest --
    and sampled every sample_s: (points, durations). Each joint follows a monotone cubic through the waypoints (no
    overshoot), so its speed has no jump at a waypoint: the arm follows timed points directly, and time_scale's speed
    held from one waypoint to the next jumps at each. Where the curve bends more than amax allows at that speed, the
    whole profile is slowed."""
    from scipy.interpolate import PchipInterpolator
    P = [np.asarray(path[0], float)]
    for q in path[1:]:                                     # unwrapped, repeated waypoints dropped
        d = fold(np.asarray(q, float) - P[-1])
        if np.abs(d).max() > 1e-6:
            P.append(P[-1] + d)
    P = np.array(P)
    if len(P) < 2:
        return [P[0], P[0]], [sample_s]
    s = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(P, axis=0)).max(1))])
    curve, S = PchipInterpolator(s, P, axis=0), float(s[-1])
    for _ in range(6):
        vp = min(vmax, math.sqrt(S * amax))
        ta, tc = vp / amax, max(0.0, S / vp - vp / amax)
        T = 2 * ta + tc
        n = max(2, int(math.ceil(T / sample_s)))
        t = np.linspace(0.0, T, n + 1)
        st = np.where(t < ta, 0.5 * amax * t ** 2,
                      np.where(t < ta + tc, 0.5 * vp * ta + vp * (t - ta), S - 0.5 * amax * (T - t) ** 2))
        Q, dt = curve(np.clip(st, 0.0, S)), T / n
        V = np.vstack([np.zeros(P.shape[1]), np.diff(Q, axis=0) / dt, np.zeros(P.shape[1])])
        k = max(float(np.abs(V).max()) / vmax, float(np.abs(np.diff(V, axis=0)).max()) / dt / amax)
        if k <= 1.1:
            break
        vmax, amax = vmax / k, amax / k            # (speed / k and acceleration / k at most: both back under)
    Q[0], Q[-1] = P[0], P[-1]
    return list(Q), [dt] * n


class Held:
    """What the gripper holds, as the arm's checks take it. An arm without one holds this module's fork: its tip from the
    wrist camera's calibration, and the scene's floor, boxes and world. A module built on this one subclasses it for
    anything else (Arm(robot, held=...), Arm.hold): its tilt is the angle between R @ tilt_axis_in_tool and tilt_ref."""
    name = "the utensil"                                   # in the checks' messages
    camera = True                                          # the arm takes the wrist camera's model
    tip_in_tool = np.zeros(3)                              # the point that leads, tool frame (the tines, a drink's rim)
    tilt_axis_in_tool = np.array([0.0, 0.0, 1.0])          # its tilt: this tool axis ...
    tilt_ref = np.array([0.0, 0.0, -1.0])                  # ... against this base-frame direction (the fork: straight down)
    max_tilt_deg = 25.0                                    # the most it may tilt on a move (a move's own cap aside)

    def spheres(self, arm, carrying: bool) -> list[list[float]]:
        """Its shape as spheres [x, y, z, r] in the tool frame, kept clear of boxes() along a planned path."""
        return []

    def boxes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """What it is kept clear of: (centres, rotations base <- box, half extents, names), as scene.utensil_boxes."""
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)), []

    def floor(self, x: float, y: float) -> float:
        """The lowest its tip may pass at (x, y) on a planned move, as scene.tip_floor."""
        return -math.inf

    def push_world(self, robot) -> None:
        """Its collision world into the shared planner, before each planned move, as scene.push_world."""


def joint_cap(dps: float) -> float:
    """A cruise speed held under the cap every joint runs under (motion.joint_speed_cap_dps)."""
    return min(float(dps), float(M.joint_speed_cap_dps))


class Arm:
    def __init__(self, robot, kin: Chain | None = None, measured: bool = True, held: Held | None = None):
        """measured=False: without the utensil's tip model, and so without the wrist camera (rammp state, capture,
        home --direct, calibrate-wrist). held: what the gripper holds when not the fork (Held), with its own tip and
        checks: none of the fork's measurements is needed."""
        if held is None and measured and unmeasured():
            raise RuntimeError("not measured on this robot yet: %s (docs/porting.md)" % ", ".join(unmeasured()))
        from rammp_feeding.scene import tip_in_tool       # (the tip model: the wrist camera's intrinsics and mount)
        self.robot, self.kin, self.held = robot, kin or chain(robot), held
        if held is None:
            self.cam = robot.camera_model("camera") if measured else None
            self.tip_tool = tip_in_tool(self.cam) if measured else None
        else:
            self.cam = robot.camera_model("camera") if held.camera else None
            self.tip_tool = np.asarray(held.tip_in_tool, float)
        self.limits = {i: min(-lo, hi) for i, (lo, hi) in self.kin.limits.items()}   # J2 / J4 / J6, the model's

    def hold(self, held: Held) -> None:
        """What the gripper holds from here on (a grasp, a release): its tip and checks."""
        self.held, self.tip_tool = held, np.asarray(held.tip_in_tool, float)

    @property
    def held_name(self) -> str:
        return "the utensil" if self.held is None else self.held.name

    # ------------------------------------------------------------------ state
    def q(self) -> Joints:
        return self.robot.joints()

    def tool(self, q=None) -> tuple[np.ndarray, np.ndarray]:
        return self.kin.fk(self.q() if q is None else q)

    def tip(self, q=None) -> np.ndarray:
        T, R = self.tool(q)
        return T + R @ self.tip_tool

    def margin(self, q) -> float:
        """Smallest distance (deg) from a limited joint to its limit."""
        return min(lim - abs(q[i]) for i, lim in self.limits.items())

    def ik(self, T, R, seed) -> Joints | None:
        q = self.kin.ik(T, R, seed)
        return None if q is None or self.margin(q) < 0 else q

    def tilt(self, R) -> float:
        """How far the held object is tilted at tool orientation R (deg): the fork's angle off straight down, or a Held's
        axis against its reference."""
        if self.held is None:
            return tilt_deg(R)
        a, b = R @ np.asarray(self.held.tilt_axis_in_tool, float), np.asarray(self.held.tilt_ref, float)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))))

    def tilt_limit(self, path, cap: float | None = None) -> float:
        """How far a move may tilt what it holds: cap (motion.max_tilt_deg, or the Held's max_tilt_deg), or 5 deg past
        its own start or end where either is further."""
        if cap is None:
            cap = M.max_tilt_deg if self.held is None else self.held.max_tilt_deg
        return max(float(cap), self.tilt(self.tool(path[0])[1]) + 5.0, self.tilt(self.tool(path[-1])[1]) + 5.0)

    def steered(self, prev: Sequence[float]) -> list[float]:
        """An IK seed from prev with J2 / J4 / J6 nudged off a limit they are within steer_within_deg of."""
        g = list(prev)
        for j in (1, 3, 5):
            if abs(prev[j]) > self.limits[j] - float(M.steer_within_deg):
                g[j] = prev[j] - math.copysign(min(float(M.steer_step_deg), abs(prev[j])), prev[j])
        return g

    def homotopy(self, q0: Joints, at: PoseAt, n: int, toward: Sequence[float] | None = None,
                 max_jump: float = float(M.max_joint_step_deg)) -> tuple[list[Joints] | None, str | None]:
        """A continuous IK path along at(s), s = 1/n .. 1 (including q0), or (None, why). The IK seed is the last
        point steered off the limits, or blended toward a known arm shape `toward`."""
        prev, path = list(q0), [list(q0)]
        for k in range(1, n + 1):
            s = k / n
            T, R = at(s)
            g = [a + float(fold(b - a)) * s for a, b in zip(q0, toward)] if toward is not None else self.steered(prev)
            q = self.ik(T, R, g)
            if q is None:
                return None, "IK fails at %d%%" % (100 * s)
            if max_step(q, prev) > max_jump:
                return None, "a joint jump of %.0f deg at %d%%" % (max_step(q, prev), 100 * s)
            prev = q
            path.append(q)
        return path, None

    # ------------------------------------------------------------------ running a path
    def run(self, path: Sequence[Sequence[float]], label: str, dps: float | None = None,
            durations: Sequence[float] | None = None, torque_limit: float | None = None, torque_stop: bool = True,
            tilt_cap: float | None = None, check: Callable | None = None, impedance: str | None = None,
            path_tolerance_deg: float = float(M.path_tolerance_deg)) -> float:
        """Execute a joint path from where the arm is (path[0] must be within near_pose_deg of it), at a cruise of
        `dps` deg/s on the furthest joint (or at the given segment durations), watched as above. check(q, tau, t) ->
        a reason to stop, or None. impedance: the motion.impedance entry to run it compliant. Returns its seconds."""
        if len(path) < 2:
            return 0.0
        max_tilt = self.tilt_limit(path, tilt_cap)
        t_now, raw, _ = self.robot.joint_sample()
        q_now = fold(np.degrees(raw))
        if max_step(q_now, path[0]) > float(M.near_pose_deg):
            raise MoveError("%s: the arm is %.1f deg from the path's start -- not moving" % (label, max_step(q_now, path[0])))
        if durations is None:
            durations = time_scale(path, joint_cap(dps or float(M.free_space.dps)), float(M.joint_accel_dps2))
        pos, cur = [raw.copy()], raw.copy()            # the arm's own (unwrapped) radians + the short-way deltas
        for p, q in zip(path, path[1:]):
            cur = cur + np.radians(fold(np.asarray(q, float) - np.asarray(p, float)))
            pos.append(cur.copy())
        if impedance:                                  # the driver's compliant mode takes a continuous joint the long
            wrapped = fold(np.degrees(np.array(pos)))  # way round once its target crosses +-180 (it spun joint 1)
            for i in (i for i in range(7) if i not in self.kin.limits):
                if np.any(np.abs(np.diff(wrapped[:, i])) > 180.0):
                    raise MoveError("%s REFUSED: joint %d would cross +-180 deg in compliant mode, which the arm driver "
                                    "runs the long way round -- not moving" % (label, i + 1))
        times = np.concatenate([[0.0], np.cumsum(durations)])
        total = float(times[-1])
        gains = dict(cfg.motion.impedance[impedance]) if impedance else None
        guard = TorqueGuard(float(torque_limit if torque_limit is not None else cfg.guards.torque_step_nm.default))
        tel = dict(t=[], q=[], tau=[])
        traj = self.robot.execute(np.array(pos), list(times), impedance=gains,
                                  path_tolerance_rad=math.radians(path_tolerance_deg),
                                  goal_tolerance_rad=math.radians(float(M.goal_tolerance_deg)))
        t0, why, last = time.time(), None, 0.0
        try:
            while not traj.done():
                if time.time() - t0 > total + float(M.watch_extra_s):
                    why = "not finished %.0f s after its planned %.1f s" % (float(M.watch_extra_s), total)
                    break
                ts, raw_q, tau = self.robot.joint_sample()
                if ts == last:
                    time.sleep(0.005)
                    continue
                last = ts
                q = [float(v) for v in fold(np.degrees(raw_q))]
                tel["t"].append(ts - t0); tel["q"].append(q); tel["tau"].append(list(tau))
                hit = guard.update(tau)
                tilt = self.tilt(self.kin.fk(q)[1])
                if hit and torque_stop:
                    why = "joint %d torque stepped %.2f -> %.2f N.m -- something is in contact" % (hit[0] + 1, hit[2], hit[1])
                elif tilt > max_tilt:
                    why = "%s %.0f deg off vertical (limit %.0f)" % (self.held_name, tilt, max_tilt)
                elif self.robot.estopped:
                    why = "the e-stop is engaged"
                elif check is not None:
                    why = check(q, tau, ts - t0)
                if why:
                    break
            if why:
                raise MoveError("%s STOPPED: %s" % (label, why))
            code, msg = traj.result()
            if code != 0:
                raise MoveError("%s: the arm driver ended it (%d %s)%s" % (label, code, msg, {
                    -11: " -- its own contact stop tripped, and stays latched until /clear_faults",
                    -9: " -- the arm is not available (faulted or reconnecting)"}.get(code, "")))
        except BaseException:                          # a guard, Ctrl-C, sheppy stopping us: the arm halts first
            if not traj.done():
                traj.cancel()
            raise
        finally:
            record.save_arrays("arm_" + "".join(c if c.isalnum() else "_" for c in label)[:40],
                               **{k: np.array(v, float) for k, v in tel.items()}, planned_s=total)
            record.event("move", label=label, planned_s=round(total, 2), seconds=round(time.time() - t0, 2),
                         outcome=why or "done")
        log.info("    %-28s done in %.1f s", label, time.time() - t0)
        return time.time() - t0

    def goto(self, goal: Sequence[float], label: str, dps: float | None = None, **kw) -> None:
        """A joint-space line to a complete 7-joint goal (every joint the short way), refused if the goal is past a
        joint limit."""
        q0 = self.q()
        goal = [float(v) for v in fold(goal)]
        travel = max_step(q0, goal)
        if travel < 0.05:
            return
        if self.margin(goal) < 0:
            raise MoveError("%s REFUSED: the goal is %.1f deg past a joint limit" % (label, -self.margin(goal)))
        n = max(2, int(math.ceil(travel / 2.0)))
        path = [list(np.asarray(q0) + fold(np.asarray(goal) - np.asarray(q0)) * k / n) for k in range(n + 1)]
        path, durations = smooth_timing(path, joint_cap(dps or float(M.settle_dps)), float(M.joint_accel_dps2),
                                        float(M.trajectory_sample_s))
        self.run(path, label, durations=durations, **kw)

    def straight(self, Tg, Rg, mps: float, label: str, step_m: float = 0.005, **kw) -> list[Joints]:
        """A straight tool line to (Tg, Rg) at mps, the orientation turned evenly on the way (a continuous IK path);
        nothing to do within motion.arrived_m of it."""
        q0 = self.q()
        T0, R0 = self.tool(q0)
        from rammp_feeding.geometry import rodrigues, rot_log
        w = rot_log(np.asarray(Rg) @ R0.T)
        L = float(np.linalg.norm(np.asarray(Tg) - T0))
        if L < float(M.arrived_m) and np.linalg.norm(w) < 1e-3:
            log.info("    %-28s already there", label)
            return [q0]
        n = max(4, int(math.ceil(L / step_m)))
        path, err = self.homotopy(q0, lambda s: (T0 + (np.asarray(Tg) - T0) * s, rodrigues(w * s) @ R0), n)
        if path is None:
            raise MoveError("%s: %s -- not moving" % (label, err))
        self.run(path, label, dps=dps_for(path, max(L, 1e-4), mps), **kw)
        return path

    # ------------------------------------------------------------------ free space: cuRobo
    def check_path(self, path, floor: Callable[[float, float], float], max_reach_m: float | None = None,
                   spheres=None, boxes=None, tilt_cap: float | None = None) -> tuple[float, float]:
        """(lowest, highest) utensil tip z along the path as it runs (sampled every check_step_deg); MoveError if a joint
        passes our limits, the tip dips under floor(x, y) or reaches past max_reach_m, the utensil tilts past
        tilt_limit, or a sphere ([x, y, z, r], tool frame) comes nearer a box (scene.utensil_boxes) than
        utensil_clearance_m -- or than it already is at the start or will be at the goal, where that is nearer
        (feeding_start parks the utensil beside the table): the planner knows the arm, not what it holds."""
        FS = M.free_space
        qs = densify(path, float(FS.check_step_deg))
        near = min(qs, key=self.margin)
        if self.margin(near) < 0:
            raise MoveError("the path takes a joint %.1f deg past its limit (the arm's model): %s"
                            % (-self.margin(near), np.round(near, 1).tolist()))
        poses = [self.tool(q) for q in qs]
        tips = [T + R @ self.tip_tool for T, R in poses]
        max_tilt, worst = self.tilt_limit(path, tilt_cap), max(self.tilt(R) for _, R in poses)
        if worst > max_tilt + 1e-6:
            raise MoveError("the path tilts %s %.1f deg off vertical (limit %.0f)" % (self.held_name, worst, max_tilt))
        for t in tips:
            if t[2] < floor(t[0], t[1]) - 1e-6:
                raise MoveError("the path takes %s tip to z %.3f at (%.3f, %.3f), under the floor there (%.3f)"
                                % (self.held_name, t[2], t[0], t[1], floor(t[0], t[1])))
            if max_reach_m is not None and math.hypot(t[0], t[1]) > max_reach_m + 1e-6:
                raise MoveError("the path swings %s tip %.0f mm out from the arm's base (at most %.0f)"
                                % (self.held_name, 1000 * math.hypot(t[0], t[1]), 1000 * max_reach_m))
        if spheres is not None and boxes is not None and len(boxes[0]):
            sp = np.asarray(spheres, float)
            centres = np.array([T + sp[:, :3] @ R.T for T, R in poses])                 # (samples, spheres, 3)
            gap = sphere_box_clearance(centres, sp[:, 3], *boxes[:3])               # (samples, spheres, boxes)
            need = np.minimum(float(FS.utensil_clearance_m), np.minimum(gap[0], gap[-1]))
            short = gap - need + 1e-6
            k, s, b = np.unravel_index(int(np.argmin(short)), short.shape)
            if short[k, s, b] < 0:
                raise MoveError("the path takes %s within %.0f mm of %s (at least %.0f here): its sphere %d at "
                                "(%.3f, %.3f, %.3f)" % (self.held_name, 1000 * gap[k, s, b], boxes[3][b], 1000 * need[s, b],
                                                        s, *centres[k, s]))
        return min(t[2] for t in tips), max(t[2] for t in tips)

    def travel(self, goal: Sequence[float], label: str, dps: float | None = None, go: bool = True,
               max_reach_m: float | None = None, carrying: bool = False, level: bool = False, **kw) -> list[Joints]:
        """A collision-free path from the shared planner, from here to goal (go=False: plan only), with our world pushed
        to it first; thinned, checked as it will run against our tip floor, the utensil's tilt (tilt_limit) and, for
        the utensil (and, carrying, the piece), against our boxes -- re-planned when it fails any -- run, and settled
        exactly onto goal. level: planned to goal's tool pose with its tilt held all the way (the planner's HOLD_LEVEL:
        something that must stay upright, a cup), and not settled onto goal's joints (the plan may reach the pose in
        others)."""
        from rammp_feeding.scene import push_world, tip_floor, utensil_boxes, utensil_spheres
        FS = M.free_space
        q = self.q()
        if max_step(q, goal) <= float(M.near_pose_deg):
            if go:
                self.goto(goal, "settle " + label)
            return [q, list(goal)]
        H = self.held                                 # (none: the fork, and the scene's floor, boxes and world)
        (push_world if H is None else H.push_world)(self.robot)   # another client may have changed the planner's world
        spheres, boxes = (utensil_spheres(self, carrying), utensil_boxes()) if H is None else (H.spheres(self, carrying),
                                                                                                H.boxes())
        floor = tip_floor if H is None else H.floor
        for k in range(int(FS.replans) + 1):
            try:                                      # level: to goal's tool pose, its tilt held on the way (a cup)
                pos, _ = (self.robot.plan_pose(q, *self.tool(goal), float(FS.timeout_s)) if level else
                          self.robot.plan_joints(q, goal, float(FS.timeout_s)))
            except RuntimeError as e:
                raise MoveError("%s: %s" % (label, e))
            pos[0] = q
            keep = [0]
            for i in range(1, len(pos) - 1):
                if max_step(pos[i], pos[keep[-1]]) >= float(FS.path_step_deg):
                    keep.append(i)
            path = [list(map(float, pos[i])) for i in keep + [len(pos) - 1]]
            try:
                lo, hi = self.check_path(path, floor, max_reach_m, spheres, boxes, kw.get("tilt_cap"))
                break
            except MoveError as e:
                if k == int(FS.replans):
                    raise MoveError("%s: %s -- not moving" % (label, e))
                log.info("    %-28s re-planning (%d/%d): %s", label, k + 1, int(FS.replans), e)
        log.info("    %-28s planned: %d waypoints, %s tip z %.3f .. %.3f%s", label, len(path),
                 self.held_name.replace("the ", ""), lo, hi, "" if go else "  (plan only)")
        if go:
            timed, durations = smooth_timing(path, joint_cap(dps or float(FS.dps)), float(M.joint_accel_dps2),
                                             float(M.trajectory_sample_s))
            self.run(timed, label, durations=durations,
                     torque_limit=kw.pop("torque_limit", float(cfg.guards.torque_step_nm.free_space)), **kw)
            if not level:                             # (a level plan ends at goal's pose, maybe in other joints: kept)
                self.goto(goal, "settle " + label)
        return path

    # ------------------------------------------------------------------ the tool's force, from the joint torques
    def tool_force(self, tau, q=None) -> np.ndarray:
        """The force at the tool frame that explains joint torques tau (N.m, against a baseline): J^T f = tau. At the
        tool frame, where Kortex's own estimate was, which delivery.bite_n is tuned against."""
        J = self.kin.jacobian(self.q() if q is None else q)[:3]
        return np.linalg.lstsq(J.T, np.asarray(tau, float), rcond=None)[0]


def dps_for(path, metres: float, mps: float) -> float:
    """The cruise joint speed (furthest joint) that moves the tool `metres` at `mps`."""
    return sum(max_step(q, p) for p, q in zip(path, path[1:])) / (metres / mps)
