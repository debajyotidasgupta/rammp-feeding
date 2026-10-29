"""Guarded arm motion on top of the arm driver: every move is a joint path we compute (our IK, a cuRobo plan), timed
here, executed by the driver (/execute_joint_trajectory, position or impedance), and watched while it runs:

  CONTACT   a joint-torque step against recent history (TorqueGuard; limits per move, guards.torque_step_nm)
  TILT      the utensil further than max_tilt off straight down (but where a move turns it on purpose)
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
from rammp_feeding.config import ROOT, cfg, unmeasured
from rammp_feeding.geometry import densify, fold, max_step, sphere_box_clearance, tilt_deg
from rammp_feeding.kinematics import Chain

log = logging.getLogger("rammp")
M = cfg.motion
Joints = list[float]
PoseAt = Callable[[float], "tuple[np.ndarray, np.ndarray]"]   # s in [0, 1] -> (tool position, R)


class MoveError(RuntimeError):
    """A move was refused, ignored, or stopped. The arm is stationary."""


def chain(robot=None) -> Chain:
    """The arm's kinematics from its node's /robot_description; the bundled copy of the driver's model without one."""
    urdf = robot.robot_description() if robot is not None else None
    if urdf is None:
        if robot is not None:
            log.info("  (no /robot_description from the arm's node: the bundled %s)", cfg.arm.urdf)
        urdf = ROOT / str(cfg.arm.urdf)
    return Chain(urdf, cfg.arm.tool_from_flange_m)


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


def joint_cap(dps: float) -> float:
    """A cruise speed held under the cap every joint runs under (motion.joint_speed_cap_dps)."""
    return min(float(dps), float(M.joint_speed_cap_dps))


class Arm:
    def __init__(self, robot, kin: Chain | None = None, measured: bool = True):
        """measured=False: without the utensil's tip model, and so without the wrist camera (rammp state, capture,
        home --direct, calibrate-wrist)."""
        if measured and unmeasured():
            raise RuntimeError("not measured on this robot yet: %s (docs/porting.md)" % ", ".join(unmeasured()))
        from rammp_feeding.scene import tip_in_tool       # (the tip model: the wrist camera's intrinsics and mount)
        self.robot, self.kin = robot, kin or chain(robot)
        self.cam = robot.camera_model("camera") if measured else None
        self.tip_tool = tip_in_tool(self.cam) if measured else None
        self.limits = {int(j) - 1: float(v) for j, v in M.joint_limits_deg.items()}

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
        if q is None or self.margin(q) < float(M.joint_limit_margin_deg):
            return None
        return q

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
            max_tilt: float = float(M.max_tilt_deg), check: Callable | None = None, impedance: str | None = None,
            path_tolerance_deg: float = float(M.path_tolerance_deg)) -> float:
        """Execute a joint path from where the arm is (path[0] must be within near_pose_deg of it), at a cruise of
        `dps` deg/s on the furthest joint (or the given segment durations), watched as above. check(q, tau, t) ->
        a reason to stop, or None. impedance: the motion.impedance entry to run it compliant. Returns its seconds."""
        if len(path) < 2:
            return 0.0
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
                tilt = tilt_deg(self.kin.fk(q)[1])
                if hit and torque_stop:
                    why = "joint %d torque stepped %.2f -> %.2f N.m -- something is in contact" % (hit[0] + 1, hit[2], hit[1])
                elif tilt > max_tilt:
                    why = "the utensil %.0f deg off vertical (limit %.0f)" % (tilt, max_tilt)
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
                raise MoveError("%s: the arm driver ended it (%d %s)" % (label, code, msg))
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

    def goto(self, goal: Sequence[float], label: str, dps: float | None = None, max_tilt: float = float(M.max_tilt_deg),
             **kw) -> None:
        """A joint-space line to a complete 7-joint goal (every joint the short way), refused if the goal does not
        hold the utensil within max_tilt of straight down or comes near a joint limit."""
        q0 = self.q()
        goal = [float(v) for v in fold(goal)]
        travel = max_step(q0, goal)
        if travel < 0.05:
            return
        if self.margin(goal) < float(M.joint_limit_margin_deg):
            raise MoveError("%s REFUSED: the goal comes within %.1f deg of a joint limit" % (label, self.margin(goal)))
        if tilt_deg(self.kin.fk(goal)[1]) > max_tilt:
            raise MoveError("%s REFUSED: the goal holds the utensil %.0f deg off vertical" % (label, tilt_deg(self.kin.fk(goal)[1])))
        n = max(2, int(math.ceil(travel / 2.0)))
        path = [list(np.asarray(q0) + fold(np.asarray(goal) - np.asarray(q0)) * k / n) for k in range(n + 1)]
        self.run(path, label, dps=dps or float(M.settle_dps), max_tilt=max(max_tilt, tilt_deg(self.tool(q0)[1]) + 1), **kw)

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
                   spheres=None, boxes=None, max_tilt: float | None = None) -> tuple[float, float]:
        """(lowest, highest) utensil tip z along the path as it runs (sampled every check_step_deg); MoveError if a joint
        comes within joint_limit_margin_deg of our limits, the tip dips under floor(x, y) or reaches past max_reach_m,
        the utensil tilts past max_tilt off straight down, or a
        sphere ([x, y, z, r], tool frame) comes nearer a box (scene.utensil_boxes) than utensil_clearance_m -- or than
        it already is at the start or will be at the goal, where that is nearer (feeding_start parks the utensil beside
        the table): the planner knows the arm, not what it holds."""
        FS = M.free_space
        qs = densify(path, float(FS.check_step_deg))
        near = min(qs, key=self.margin)
        if self.margin(near) < float(M.joint_limit_margin_deg):
            raise MoveError("the path takes a joint within %.1f deg of its limit (motion.joint_limits_deg): %s"
                            % (self.margin(near), np.round(near, 1).tolist()))
        poses = [self.tool(q) for q in qs]
        tips = [T + R @ self.tip_tool for T, R in poses]
        if max_tilt is not None:
            worst = max(tilt_deg(R) for _, R in poses)
            if worst > max_tilt + 1e-6:
                raise MoveError("the path tilts the utensil %.1f deg off vertical (limit %.0f)" % (worst, max_tilt))
        for t in tips:
            if t[2] < floor(t[0], t[1]) - 1e-6:
                raise MoveError("the path takes the utensil tip to z %.3f at (%.3f, %.3f), under the floor there (%.3f)"
                                % (t[2], t[0], t[1], floor(t[0], t[1])))
            if max_reach_m is not None and math.hypot(t[0], t[1]) > max_reach_m + 1e-6:
                raise MoveError("the path swings the utensil tip %.0f mm out from the arm's base (at most %.0f)"
                                % (1000 * math.hypot(t[0], t[1]), 1000 * max_reach_m))
        if spheres is not None and boxes is not None and len(boxes[0]):
            sp = np.asarray(spheres, float)
            centres = np.array([T + sp[:, :3] @ R.T for T, R in poses])                 # (samples, spheres, 3)
            gap = sphere_box_clearance(centres, sp[:, 3], *boxes[:3])               # (samples, spheres, boxes)
            need = np.minimum(float(FS.utensil_clearance_m), np.minimum(gap[0], gap[-1]))
            short = gap - need + 1e-6
            k, s, b = np.unravel_index(int(np.argmin(short)), short.shape)
            if short[k, s, b] < 0:
                raise MoveError("the path takes the utensil within %.0f mm of %s (at least %.0f here): its sphere %d at "
                                "(%.3f, %.3f, %.3f)" % (1000 * gap[k, s, b], boxes[3][b], 1000 * need[s, b], s, *centres[k, s]))
        return min(t[2] for t in tips), max(t[2] for t in tips)

    def travel(self, goal: Sequence[float], label: str, dps: float | None = None, go: bool = True,
               max_reach_m: float | None = None, carrying: bool = False, **kw) -> list[Joints]:
        """A collision-free path from the shared planner, from here to goal (go=False: plan only), with our world pushed
        to it first; thinned, checked as it will run against our tip floor, the utensil's tilt (no further than its
        start or its goal already is, unless max_tilt says) and, for the utensil (and, carrying, the piece), against
        our boxes -- re-planned when it fails any -- run, and settled exactly onto goal."""
        from rammp_feeding.scene import push_world, tip_floor, utensil_boxes, utensil_spheres
        FS = M.free_space
        q = self.q()
        kw.setdefault("max_tilt", max(float(M.max_tilt_deg), tilt_deg(self.tool(q)[1]) + 5, tilt_deg(self.tool(goal)[1]) + 5))
        if max_step(q, goal) <= float(M.near_pose_deg):
            if go:
                self.goto(goal, "settle " + label, max_tilt=kw["max_tilt"])
            return [q, list(goal)]
        push_world(self.robot)                        # another client may have changed the planner's world since
        spheres, boxes = utensil_spheres(self, carrying), utensil_boxes()
        for k in range(int(FS.replans) + 1):
            try:
                pos, _ = self.robot.plan_joints(q, goal, float(FS.timeout_s))
            except RuntimeError as e:
                raise MoveError("%s: %s" % (label, e))
            pos[0] = q
            keep = [0]
            for i in range(1, len(pos) - 1):
                if max_step(pos[i], pos[keep[-1]]) >= float(FS.path_step_deg):
                    keep.append(i)
            path = [list(map(float, pos[i])) for i in keep + [len(pos) - 1]]
            try:
                lo, hi = self.check_path(path, tip_floor, max_reach_m, spheres, boxes, kw["max_tilt"])
                break
            except MoveError as e:
                if k == int(FS.replans):
                    raise MoveError("%s: %s -- not moving" % (label, e))
                log.info("    %-28s re-planning (%d/%d): %s", label, k + 1, int(FS.replans), e)
        log.info("    %-28s planned: %d waypoints, utensil tip z %.3f .. %.3f%s", label, len(path), lo, hi,
                 "" if go else "  (plan only)")
        if go:
            self.run(path, label, dps=dps or float(FS.dps),
                     torque_limit=kw.pop("torque_limit", float(cfg.guards.torque_step_nm.free_space)), **kw)
            self.goto(goal, "settle " + label, max_tilt=kw["max_tilt"])
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
