"""Whole-arm clearance to the user.

Written because a tip-only check was wrong in a way that mattered. The first
version of the meal demo measured the distance from the FORK TIP to the user's
no-go volumes, reported a comfortable 84-90 mm at PRESENT, and was driving the
wrist and forearm into the person's chest at several hundred newtons the whole
time. At the instant of first contact the fork tip was 381 mm from anything.

A feeding arm reaches AROUND a seated person. Most of its length is closer to
them than the tool is. Any monitor that watches only the tool is watching the
one part of the arm that is aimed at a place the user has agreed to put their
mouth.

Distances come from `mj_geomDistance`, which is exact for the convex meshes the
menagerie ships, so this is ground truth rather than a primitive approximation.
On hardware the same job needs the URDF and a collision library (cuRobo, which
is what the GPU is for); the interface is the same -- a signed clearance and the
pair that produced it.
"""
from __future__ import annotations

import mujoco
import numpy as np

USER_GEOMS = ("head_nogo", "torso_nogo", "lap_nogo", "pelvis_nogo")
DISTMAX = 0.6          # metres; beyond this we do not care and can stop early


class ClearanceMonitor:
    def __init__(self, world, arm_prefix: str = "arm_"):
        self.w = world
        m = world.model
        self.user = [world.gid(g) for g in USER_GEOMS]
        self.arm = []
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            if body and body.startswith(arm_prefix) and m.geom_contype[g]:
                self.arm.append(g)
        self.names = {g: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                          or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]))
                      for g in self.arm + self.user}

    def closest(self) -> tuple[float, str, str]:
        """(clearance_m, arm_part, user_part). Negative means interpenetrating."""
        best, pa, pu = DISTMAX, "", ""
        for ga in self.arm:
            for gu in self.user:
                d = mujoco.mj_geomDistance(self.w.model, self.w.data, ga, gu, DISTMAX, None)
                if d < best:
                    best, pa, pu = float(d), self.names[ga], self.names[gu]
        return best, pa, pu

    def tip_only(self) -> float:
        """What a tool-frame-only monitor would report. Kept to show the gap."""
        tip = self.w.site_pos("arm_tcp")
        best = DISTMAX
        for gu in self.user:
            c = self.w.data.geom_xpos[gu]
            best = min(best, float(np.linalg.norm(tip - c) - self.w.model.geom_size[gu, 0]))
        return best


def sweep(world, monitor, qa, qb, samples: int = 60):
    """Worst clearance along the joint-space straight line from qa to qb."""
    worst = (DISTMAX, "", "", 0.0)
    keep = world.data.qpos[world.arm_qadr].copy()
    for s in np.linspace(0.0, 1.0, samples):
        world.data.qpos[world.arm_qadr] = qa + (qb - qa) * s
        mujoco.mj_forward(world.model, world.data)
        d, pa, pu = monitor.closest()
        if d < worst[0]:
            worst = (d, pa, pu, float(s))
    world.data.qpos[world.arm_qadr] = keep
    mujoco.mj_forward(world.model, world.data)
    return worst


def plan_leg(world, monitor, ik, qa, pos_b, axis_b, margin: float = 0.02,
             detours=(0.10, 0.18, 0.26), seed_tries: int = 6):
    """Joint waypoints from qa to a Cartesian goal, keeping the WHOLE arm clear.

    Two knobs, in the order that matters:

    1. WHICH IK SOLUTION. The Gen3 is 7-DoF, so a pose goal leaves a free
       dimension, and that dimension is where the elbow sits. Two solves of the
       same goal from different seeds can differ by "elbow beside the chair" and
       "elbow inside the user's chest" while both report sub-millimetre
       residuals. Re-seeding is therefore the first thing to try, not the last.
    2. A DETOUR. If no solution has a clear straight line from qa, route via a
       waypoint pushed onto the arm's own side of the chair.

    Crude by design; a real planner is cuRobo's job and needs the GPU that is
    still blocked. What it must not do is return a path it has not checked.

    Returns (waypoints, worst_clearance, how).
    """
    import numpy as _np
    R = axis_b
    best = None
    seeds = [qa, ik.HOME, _np.array([0, -0.35, 3.14, -2.55, 0, -0.87, 1.57])]
    for k in range(max(1, seed_tries)):
        seed = seeds[k] if k < len(seeds) else _np.clip(
            qa + world.rng.normal(0, 0.7, 7), world.arm_lo, world.arm_hi)
        qb, pe, re = ik.solve(_np.asarray(pos_b, float), R, q0=seed)
        if pe > 3e-3 or re > 1.0:
            continue
        w = sweep(world, monitor, qa, qb)
        if best is None or w[0] > best[1]:
            best = ([qb], w[0], "direct" if k == 0 else "reseeded IK (#%d)" % k)
        if w[0] >= margin:
            return best

    for dy in detours:
        via = _np.asarray(pos_b, float) + _np.array([0.0, -dy, 0.0])
        qv, pe, _ = ik.solve(via, R, q0=qa)
        if pe > 5e-3:
            continue
        qb = best[0][-1] if best else ik.solve(_np.asarray(pos_b, float), R, q0=qv)[0]
        w1 = sweep(world, monitor, qa, qv)
        w2 = sweep(world, monitor, qv, qb)
        if min(w1[0], w2[0]) >= margin:
            return [qv, qb], min(w1[0], w2[0]), "via -%.0f mm in Y" % (dy * 1000)

    if best is None:
        return [], -1.0, "NO IK SOLUTION"
    return best[0], best[1], "NO SAFE PATH FOUND"
