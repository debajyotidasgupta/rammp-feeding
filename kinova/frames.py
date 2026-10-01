"""Angles and Kortex's pose convention. Plain numpy (no Kortex): the detector's venv and the planner's container
import it as well as the arm side.

Joint vectors are 7 angles in degrees folded to (-180, 180]; poses are Kortex's (x y z m, theta x y z deg,
fixed-frame XYZ Euler: R = Rz @ Ry @ Rx); R is base <- tool; a tool "axis" is where the utensil points (tool +z).
"""

from __future__ import annotations

import math

import numpy as np


def fold(d):
    """Kortex reports [0, 360); joints 1/3/5/7 are continuous: compare and interpolate on (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def R_from_kortex(tx: float, ty: float, tz: float) -> np.ndarray:
    """Kortex theta x y z (deg) -> R = Rz @ Ry @ Rx."""
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    Rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    Rz_ = np.array([[math.cos(tz), -math.sin(tz), 0], [math.sin(tz), math.cos(tz), 0], [0, 0, 1]])
    return Rz_ @ Ry @ Rx


def kortex_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """R -> Kortex theta x y z (deg)."""
    ty = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    return (math.degrees(math.atan2(R[2, 1], R[2, 2])), ty, math.degrees(math.atan2(R[1, 0], R[0, 0])))


def Rz(deg: float) -> np.ndarray:
    """Rotation about z by `deg` (about the base's vertical: J1's direction)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def axis(R: np.ndarray) -> np.ndarray:
    """Where the utensil points (tool +z) in base coordinates."""
    return R @ np.array([0, 0, 1.0])
