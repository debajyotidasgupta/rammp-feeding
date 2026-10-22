"""A camera as its driver describes it: the colour and depth intrinsics (their camera_info) and the depth -> colour
transform (its /tf_static). Pure numpy: rammp_feeding.ros fills it from the messages, the tests from numbers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, m) -> "Intrinsics":
        return cls(float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]), int(m.width), int(m.height))


@dataclass
class CameraModel:
    colour: Intrinsics
    depth: Intrinsics
    d2c: np.ndarray                     # 4 x 4: colour optical frame <- depth optical frame

    def colour_to_depth(self, u, v, z):
        """A colour pixel at colour-frame depth z -> its depth pixel."""
        c, d = self.colour, self.depth
        p = np.array([(u - c.cx) * z / c.fx, (v - c.cy) * z / c.fy, z])
        q = self.d2c[:3, :3].T @ (p - self.d2c[:3, 3])
        return d.fx * q[0] / q[2] + d.cx, d.fy * q[1] / q[2] + d.cy

    def depth_to_colour(self, ud, vd, z):
        """A depth pixel at depth-frame z -> (u, v, z) in the colour frame."""
        c, d = self.colour, self.depth
        p = self.d2c[:3, :3] @ np.array([(ud - d.cx) * z / d.fx, (vd - d.cy) * z / d.fy, z]) + self.d2c[:3, 3]
        return c.fx * p[0] / p[2] + c.cx, c.fy * p[1] / p[2] + c.cy, float(p[2])

    def pixel(self, p) -> tuple[float, float]:
        """A point in the colour optical frame -> its colour pixel."""
        c = self.colour
        return c.fx * p[0] / p[2] + c.cx, c.fy * p[1] / p[2] + c.cy

    def ray(self, u, v, z) -> np.ndarray:
        """A colour pixel at depth z -> the point in the colour optical frame."""
        c = self.colour
        return np.array([(u - c.cx) * z / c.fx, (v - c.cy) * z / c.fy, z])

    def depth_points(self, depth_mm: np.ndarray) -> np.ndarray:
        """Every valid depth pixel as a point in the colour optical frame."""
        d = self.depth
        v, u = np.nonzero(depth_mm > 0)
        z = depth_mm[v, u] / 1000.0
        p = np.stack([(u - d.cx) * z / d.fx, (v - d.cy) * z / d.fy, z], axis=1)
        return p @ self.d2c[:3, :3].T + self.d2c[:3, 3]

    def depth_to_colour_shift_m(self) -> float:
        """How far the depth sensor sits from the colour camera along its axis (0 for a D405: one sensor)."""
        return float(self.d2c[2, 3])
