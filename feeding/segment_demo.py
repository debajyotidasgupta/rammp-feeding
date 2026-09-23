#!/usr/bin/env python3
"""Break a teleop demo into skill segments, with measured parameters.

Segmentation runs on the STATE log, not the video: the state log is uniform
40 Hz with monotonic timestamps, whereas the wrist video drops frames and its
container clock does not preserve wall time (docs/demo-analysis.md).

The cut points are physical rather than learned -- motion start/stop, direction
reversal in z, and region changes between the bowl and the person. A demo is
short and the phases are gross; a change-point detector would add opacity
without adding accuracy.

Output is the thing the state machine actually needs: for each phase, how far,
how fast, how long, and what force.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# Workspace regions, from the demo: the bowl sits at +y, the person at -y.
BOWL_Y = 0.15
PERSON_Y = -0.10


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    t = np.array([r["t"] for r in rows]); t -= t[0]
    P = np.array([[r["tool"]["x"], r["tool"]["y"], r["tool"]["z"]] for r in rows])
    A = np.array([[r["tool"]["tx"], r["tool"]["ty"], r["tool"]["tz"]] for r in rows])
    F = np.array([[r["wrench"]["fx"], r["wrench"]["fy"], r["wrench"]["fz"]] for r in rows])
    TAU = np.array([[j["torque"] for j in r["joints"]] for r in rows])
    return t, P, A, F, TAU


def segments(t, P, still_mps=0.004, min_s=0.6):
    hz = len(t) / t[-1]
    v = np.linalg.norm(np.gradient(P, axis=0), axis=1) * hz
    moving = v > still_mps
    # close short gaps so a momentary dip does not split one motion in two
    k = max(1, int(0.3 * hz))
    for _ in range(2):
        for i in range(k, len(moving) - k):
            if moving[i - k] and moving[i + k]:
                moving[i] = True
    out, i = [], 0
    while i < len(t):
        j = i
        while j < len(t) and moving[j] == moving[i]:
            j += 1
        if t[j - 1] - t[i] >= min_s:
            if moving[i]:
                out.extend(_split_by_direction(t, P, i, j, hz, min_s))
            else:
                out.append((i, j, False))
        i = j
    return out, hz


def _split_by_direction(t, P, i, j, hz, min_s):
    """Cut a motion where it changes character: vertical <-> horizontal.

    The descent and the scoop are one continuous movement -- the operator never
    stops between them -- so a start/stop segmenter merges them and the scoop
    disappears into a 'traverse'. What separates them is DIRECTION: the descent
    is z-dominant, the scoop is xy-dominant at constant z.
    """
    w = max(3, int(0.25 * hz))
    d = np.gradient(P[i:j], axis=0)
    vert = np.abs(d[:, 2]) > np.hypot(d[:, 0], d[:, 1])
    # smooth the classification so noise does not shred the segment
    sm = np.convolve(vert.astype(float), np.ones(w) / w, mode="same") > 0.5
    out, a = [], 0
    n = j - i
    while a < n:
        b = a
        while b < n and sm[b] == sm[a]:
            b += 1
        if t[i + b - 1] - t[i + a] >= min_s * 0.6:
            out.append((i + a, i + b, True))
        elif out:                      # too short to stand alone: merge back
            out[-1] = (out[-1][0], i + b, True)
        a = b
    return out or [(i, j, True)]


def label(P, A, i, j, moving, z_floor=None):
    """Name a segment from what the tool actually did.

    `z_floor` is the lowest z reached during this visit to the bowl. A lateral
    move only counts as a SCOOP if it happens AT that floor -- the operator also
    traverses sideways at intermediate heights while lining up on the piece, and
    those look identical in isolation. What distinguishes the scoop is that it
    happens at the bottom, after the tines are in the food.
    """
    d = P[j - 1] - P[i]
    dz, dxy = d[2], np.hypot(d[0], d[1])
    y0, y1 = P[i, 1], P[j - 1, 1]
    region = lambda y: "bowl" if y > BOWL_Y else ("person" if y < PERSON_Y else "mid")
    r0, r1 = region(y0), region(y1)
    if not moving:
        return {"bowl": "HOLD@BOWL", "person": "PRESENT", "mid": "HOLD"}[r1]
    if r0 != r1 and dxy > 0.15:
        return "TRANSPORT->PERSON" if r1 == "person" else "RETURN->BOWL"
    if dz < -0.04 and abs(dz) > dxy:
        return "DESCEND"
    if dz > 0.04 and abs(dz) > dxy:
        return "LIFT"
    at_floor = z_floor is not None and abs(P[i, 2] - z_floor) < 0.012
    if abs(dz) < 0.020 and 0.02 < dxy < 0.25 and r0 == "bowl":
        return "SCOOP" if at_floor else "ALIGN"
    if dxy > 0.05:
        return "TRAVERSE"
    return "ADJUST"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--from-t", type=float, default=0.0)
    ap.add_argument("--to-t", type=float, default=1e9)
    a = ap.parse_args()

    t, P, A, F, TAU = load(a.state)
    Fm = np.linalg.norm(F, axis=1)
    segs, hz = segments(t, P)

    print("%d samples, %.0f s, %.0f Hz\n" % (len(t), t[-1], hz))
    print("  %-7s %-18s %-6s %-22s %-8s %-9s %s"
          % ("t (s)", "phase", "dur", "travel dx,dy,dz (mm)", "speed", "z end", "|F| min/mean/max"))
    # lowest z reached while over the bowl, per contiguous visit
    over_bowl = P[:, 1] > BOWL_Y
    z_floor = np.full(len(t), np.nan)
    k = 0
    while k < len(t):
        if not over_bowl[k]:
            k += 1; continue
        m = k
        while m < len(t) and over_bowl[m]:
            m += 1
        z_floor[k:m] = P[k:m, 2].min()
        k = m

    for i, j, mv in segs:
        if t[j - 1] < a.from_t or t[i] > a.to_t:
            continue
        d = (P[j - 1] - P[i]) * 1000
        dur = t[j - 1] - t[i]
        dist = np.linalg.norm(d) / 1000
        sl = slice(i, j)
        print("  %-7.1f %-18s %-6.1f %-22s %-8s %-9.0f %.1f/%.1f/%.1f"
              % (t[i], label(P, A, i, j, mv, z_floor[i] if not np.isnan(z_floor[i]) else None), dur,
                 "%.0f, %.0f, %.0f" % tuple(d),
                 ("%.0f mm/s" % (dist / dur * 1000)) if mv else "-",
                 P[j - 1, 2] * 1000,
                 Fm[sl].min(), Fm[sl].mean(), Fm[sl].max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
