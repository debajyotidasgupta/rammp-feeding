"""Angles and rotations. Joint vectors are 7 angles in degrees folded to (-180, 180]; R is a 3x3 rotation, base <- tool;
a tool "axis" is where the utensil points (tool +z). Poses kept as (xyz m, R): no Euler angles in the logic."""

from __future__ import annotations

import math

import numpy as np


def fold(d):
    """Joints 1/3/5/7 are continuous: compare and interpolate on (-180, 180]."""
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0 if np.ndim(d) else (d + 180.0) % 360.0 - 180.0


def max_step(a, b) -> float:
    """The largest single-joint difference between two joint vectors, deg, the short way round."""
    return float(np.max(np.abs(fold(np.asarray(a, float) - np.asarray(b, float)))))


def rot(axis: str, deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis].astype(float)


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF / ROS fixed-axis roll-pitch-yaw (rad): R = Rz(y) Ry(p) Rx(r)."""
    return rot("z", math.degrees(y)) @ rot("y", math.degrees(p)) @ rot("x", math.degrees(r))


def axis(R: np.ndarray) -> np.ndarray:
    """Where the utensil points (tool +z), base coordinates."""
    return np.asarray(R)[:, 2]


def tilt_deg(R: np.ndarray) -> float:
    """The utensil's angle off straight down."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -axis(R)[2]))))


def quat_matrix(q) -> np.ndarray:
    """(x, y, z, w) -> R."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quat(R: np.ndarray) -> list[float]:
    """R -> (x, y, z, w)."""
    R = np.asarray(R, float)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [x, y, z, w]


def rot_log(R: np.ndarray) -> np.ndarray:
    """The rotation vector of R (axis * angle, rad)."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if a < 1e-9:
        return v / 2
    if a > math.pi - 1e-6:                    # a half turn: the axis from the symmetric part
        k = np.sqrt(np.maximum(0.0, (np.diag(R) + 1) / 2))
        i = int(np.argmax(k))
        k[[j for j in range(3) if j != i]] *= np.sign(R[i, [j for j in range(3) if j != i]] + 1e-12)
        return a * k / np.linalg.norm(k)
    return a * v / (2 * math.sin(a))


def rodrigues(w) -> np.ndarray:
    """The rotation of the rotation vector w."""
    w = np.asarray(w, float)
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def densify(path, step_deg: float) -> list[np.ndarray]:
    """A joint path as it runs (straight in joint space between its waypoints, the short way round), sampled so that no
    joint moves more than step_deg between two samples."""
    out = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        a = np.asarray(a, float)
        d = fold(np.asarray(b, float) - a)
        n = max(1, int(math.ceil(float(np.max(np.abs(d))) / step_deg)))
        out += [a + d * k / n for k in range(1, n + 1)]
    return out


def sphere_box_clearance(centres, radii, box_centres, box_rots, half_dims) -> np.ndarray:
    """Surface distances (m) from spheres to oriented boxes: centres (..., S, 3), radii (S,); boxes B as centres (B, 3),
    rotations base <- box (B, 3, 3) and half extents (B, 3). Returns (..., S, B), negative where a sphere overlaps."""
    d = np.asarray(centres, float)[..., :, None, :] - np.asarray(box_centres, float)        # (..., S, B, 3)
    local = np.einsum("...bj,bjk->...bk", d, np.asarray(box_rots, float))                  # in each box's axes
    q = np.abs(local) - np.asarray(half_dims, float)
    out = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
    return out - np.asarray(radii, float)[:, None]

