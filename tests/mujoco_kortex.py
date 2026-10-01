"""A KINEMATIC stand-in for the Kortex controller: fake_kortex's clock, trajectories and trace, with real FK / IK.

fake_kortex.FakeArm answers IK by blending into the measured poses -- right near them, nonsense far from them --
which is what the golden trace needs (deterministic, about the recorded deployment). This answers IK and FK with
the Gen3's actual kinematics (MuJoCo Menagerie's model, sim/assets/kinova_gen3: within ~6 mm / 1.2 deg of this
arm's controller over a recorded session, tests/twin_check.py), so the planning code can be checked for a NEW
workspace -- another table height, another bowl position -- without the arm:

    arm = KinematicArm(); conn = KinematicConnection(arm)     # what kinova_connection() would yield

IK: damped least squares on the tool frame (the flange + deployment.arm.tool.transform_m), seeded with the
request's guess, joint limits held; no solution within tolerance = unreachable (the controller raises; so does
this). Like the controller's, it is local: a far target from a bad guess may not converge.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

import fake_kortex as fk

ROOT = Path(__file__).resolve().parent.parent
LIMITS_RAD = {1: 2.2497, 3: 2.5796, 5: 2.0996}          # J2, J4, J6 (0-based 1, 3, 5): the Gen3's hard limits


class KinematicArm(fk.FakeArm):
    def __init__(self, tool_z: float = 0.12):
        super().__init__()
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "assets" / "kinova_gen3" / "gen3.xml"))
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.tool_z = tool_z
        self.ik_calls = 0

    # --- kinematics
    def _tool(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:7] = q_rad
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3).copy()
        return self.d.site_xpos[self.site] + R[:, 2] * self.tool_z, R

    def fk_pose(self, joints):
        p, R = self._tool(np.radians([fold(v) for v in joints]))
        return [float(p[0]), float(p[1]), float(p[2])] + fk.rpy_R(R)

    def solve(self, pose, guess, iters: int = 300):
        """Tool pose (Kortex x y z, theta xyz deg) -> joints (deg), from the guess; None if it does not converge."""
        self.ik_calls += 1
        p_goal = np.array(pose[:3], float)
        R_goal = fk.R_rpy(*pose[3:])
        q = np.radians([fold(v) for v in guess])
        lam = 1e-3
        for _ in range(iters):
            p, R = self._tool(q)
            ep = p_goal - p
            er = 0.5 * (np.cross(R[:, 0], R_goal[:, 0]) + np.cross(R[:, 1], R_goal[:, 1]) + np.cross(R[:, 2], R_goal[:, 2]))
            if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-5:
                return [fold(math.degrees(v)) for v in q]
            mujoco.mj_comPos(self.m, self.d)
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jp, jr, p, self.m.site_bodyid[self.site])
            J = np.vstack([jp[:, :7], jr[:, :7]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            step = float(np.abs(dq).max())
            if step > 0.2:
                dq *= 0.2 / step
            q = q + dq
            for j, lim in LIMITS_RAD.items():
                q[j] = min(lim, max(-lim, q[j]))
        return None

    def joints_for(self, pose, guess=None):
        q = self.solve(pose, guess if guess is not None else self.q)
        if q is None:
            raise RuntimeError("no IK for %s" % [round(v, 3) for v in pose])
        return q


def fold(v: float) -> float:
    return (v + 180.0) % 360.0 - 180.0


class _KBase(fk._Base):
    def ComputeInverseKinematics(self, d):
        a = self.arm
        a.log("ComputeInverseKinematics", d)
        p = d.cartesian_pose
        pose = [p.x, p.y, p.z, p.theta_x, p.theta_y, p.theta_z]
        q = a.solve(pose, [j.value for j in d.guess.joint_angles])
        if q is None or a.ik_unreachable(pose):
            raise RuntimeError("IK: unreachable (kinematic fake)")
        return SimpleNamespace(joint_angles=[SimpleNamespace(value=v % 360.0) for v in q])

    def ComputeForwardKinematics(self, ja):
        a = self.arm
        a.log("ComputeForwardKinematics", ja)
        p = a.fk_pose([j.value for j in ja.joint_angles])
        return SimpleNamespace(x=p[0], y=p[1], z=p[2], theta_x=p[3], theta_y=p[4], theta_z=p[5])


class KinematicConnection(fk.FakeConnection):
    def __init__(self, arm: KinematicArm):
        super().__init__(arm)
        self.base = _KBase(arm)
