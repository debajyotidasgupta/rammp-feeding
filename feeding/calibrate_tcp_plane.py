"""Tool-offset calibration by touching a PLANE, not a point.

Classic pivot calibration needs the tip on ONE physical point across all poses,
which is awkward to drive and awkward to hit. Touching a flat surface is far
easier to automate, and carries enough information:

    for every touch i:   n . (R_i @ p_tool + t_i) = d

with |n| = 1. Unknowns are the plane (n, d -> 3 dof) and the tool offset
(p_tool -> 3), so six or more touches at varied orientations determine both. The
touches may land anywhere on the surface; only the ORIENTATIONS need to vary.

Solved by alternating least squares: fix the plane, p_tool is linear; fix
p_tool, the plane is a total-least-squares fit to the implied tip points. Both
steps are closed-form, and it converges in a handful of iterations from a
sensible start.
"""
from __future__ import annotations

import numpy as np


def solve_plane_touch(Rs, ts, p0=None, iters=60):
    """Rs: (N,3,3) tool rotations. ts: (N,3) tool positions. -> p_tool, n, d, resid."""
    Rs = np.asarray(Rs, float)
    ts = np.asarray(ts, float)
    n = np.array([0.0, 0.0, 1.0])
    d = float(np.mean(ts[:, 2]))
    p = np.zeros(3) if p0 is None else np.asarray(p0, float).copy()

    for _ in range(iters):
        # 1. plane fixed -> p_tool is linear:  (n^T R_i) p = d - n.t_i
        A = np.array([n @ R for R in Rs])
        b = np.array([d - n @ t for t in ts])
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
        # 2. p_tool fixed -> plane is a TLS fit through the implied tips
        tips = np.array([R @ p + t for R, t in zip(Rs, ts)])
        c = tips.mean(axis=0)
        _, _, vt = np.linalg.svd(tips - c)
        n = vt[2] / np.linalg.norm(vt[2])
        d = float(n @ c)

    tips = np.array([R @ p + t for R, t in zip(Rs, ts)])
    resid = np.abs(tips @ n - d)
    return p, n, d, resid


def orientation_spread(Rs):
    return max(float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))
               for a in Rs for b in Rs)
