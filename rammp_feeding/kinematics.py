"""The Gen3's kinematics from the arm's own model -- the URDF its node publishes on /robot_description (the bundled
copy of the driver's, urdf/, only without one): forward kinematics of every link, the Jacobian, and a damped
least-squares IK seeded from a nearby configuration (no IK service is offered: straight-line and scoop paths are ours).

The tool frame is the flange (the URDF's end_effector_link) plus arm.tool_from_flange_m: the 2F-85's TCP, the frame the
utensil's tip model and the camera mount are measured in (and RAMMP-CuRobo's tool_frame).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rammp_feeding.geometry import fold, rot_log, rpy_matrix


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    o = el.find("origin")
    if o is not None:
        T[:3, :3] = rpy_matrix(*(float(v) for v in o.get("rpy", "0 0 0").split()))
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray            # parent <- joint frame at q = 0
    axis: np.ndarray
    moving: bool


class Chain:
    """base_link -> the flange, through the URDF's seven arm joints; every link on it (and the fixed ones hanging off
    it, the gripper's) placed by fk()."""

    def __init__(self, urdf: str | Path, tool_from_flange_m):
        root = ET.fromstring(urdf) if isinstance(urdf, str) and urdf.lstrip().startswith("<") else ET.parse(str(urdf)).getroot()
        self.joints = []
        for j in root.findall("joint"):
            ax = j.find("axis")
            self.joints.append(Joint(j.get("name"), j.find("parent").get("link"), j.find("child").get("link"),
                                     _origin(j), np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()]),
                                     j.get("type") in ("revolute", "continuous")))
        by_child = {j.child: j for j in self.joints}
        flange = next(lk.get("name") for lk in root.findall("link") if lk.get("name").endswith("end_effector_link"))
        chain, link = [], flange
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].parent
        self.base = link
        self.chain = chain[::-1]                        # base -> flange
        self.arm = [j for j in self.chain if j.moving]
        assert len(self.arm) == 7, "expected 7 arm joints, got %d" % len(self.arm)
        self.tool = np.eye(4)
        self.tool[:3, 3] = np.asarray(tool_from_flange_m, float)
        self.limits = {}                                # joint index -> (lo, hi) deg, the revolute ones
        for i, j in enumerate(self.arm):
            el = root.find("joint[@name='%s']/limit" % j.name)
            if el is not None and el.get("lower") is not None:
                self.limits[i] = (math.degrees(float(el.get("lower"))), math.degrees(float(el.get("upper"))))

    def frames(self, q_deg) -> dict[str, np.ndarray]:
        """Every link's pose (base <- link, 4x4) at joints q (deg): the chain and the fixed links off it, by name and
        by name without a gen3_ prefix (the two models of the same arm)."""
        q = np.radians(np.asarray(q_deg, float))
        out = {self.base: np.eye(4)}
        k = 0
        for j in self.chain:
            T = out[j.parent] @ j.origin
            if j.moving:
                T = T @ _axis_rot(j.axis, q[k])
                k += 1
            out[j.child] = T
        for j in self.joints:                           # fixed links hanging off the chain (gripper, camera, ...)
            if j.child not in out and j.parent in out and not j.moving:
                out[j.child] = out[j.parent] @ j.origin
        out.update({k[len("gen3_"):]: v for k, v in list(out.items()) if k.startswith("gen3_")})
        return out

    def flange(self, q_deg) -> np.ndarray:
        q = np.radians(np.asarray(q_deg, float))
        T, k = np.eye(4), 0
        for j in self.chain:
            T = T @ j.origin
            if j.moving:
                T = T @ _axis_rot(j.axis, q[k])
                k += 1
        return T

    def fk(self, q_deg) -> tuple[np.ndarray, np.ndarray]:
        """The tool frame at joints q: (position m, R)."""
        T = self.flange(q_deg) @ self.tool
        return T[:3, 3].copy(), T[:3, :3].copy()

    def jacobian(self, q_deg, point_in_tool=(0.0, 0.0, 0.0)) -> np.ndarray:
        """6 x 7 (linear; angular) Jacobian of a point fixed in the tool frame, per radian."""
        q = np.radians(np.asarray(q_deg, float))
        T, k, axes, origins = np.eye(4), 0, [], []
        for j in self.chain:
            T = T @ j.origin
            if j.moving:
                axes.append(T[:3, :3] @ j.axis)
                origins.append(T[:3, 3].copy())
                T = T @ _axis_rot(j.axis, q[k])
                k += 1
        Tt = T @ self.tool
        p = Tt[:3, 3] + Tt[:3, :3] @ np.asarray(point_in_tool, float)
        J = np.zeros((6, 7))
        for i, (a, o) in enumerate(zip(axes, origins)):
            J[:3, i] = np.cross(a, p - o)
            J[3:, i] = a
        return J

    def ik(self, T, R, seed_deg, iters: int = 60, tol_m: float = 1e-4, tol_rad: float = 1e-3,
           point_in_tool=(0.0, 0.0, 0.0)) -> list[float] | None:
        """Joints (deg, folded) placing the tool (or a point fixed in it) at T with orientation R, from seed: damped
        least squares, the null space drawn toward the seed (the arm keeps its shape). None if it does not converge
        or passes a joint limit."""
        q = np.asarray(seed_deg, float).copy()
        seed = q.copy()
        pt = np.asarray(point_in_tool, float)
        for _ in range(iters):
            p, Rn = self.fk(q)
            e = np.concatenate([np.asarray(T, float) - (p + Rn @ pt), rot_log(np.asarray(R) @ Rn.T)])
            if np.linalg.norm(e[:3]) < tol_m and np.linalg.norm(e[3:]) < tol_rad:
                break
            J = self.jacobian(q, pt)
            lam = 0.01
            Jp = J.T @ np.linalg.inv(J @ J.T + lam * lam * np.eye(6))
            dq = Jp @ e + 0.1 * (np.eye(7) - Jp @ J) @ np.radians(fold(seed - q))
            step = float(np.abs(dq).max())
            if step > 0.2:
                dq *= 0.2 / step
            q = q + np.degrees(dq)
        else:
            return None
        q = fold(q)
        for i, (lo, hi) in self.limits.items():
            if not lo <= q[i] <= hi:
                return None
        return [float(v) for v in q]


def _axis_rot(axis: np.ndarray, a: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return T
