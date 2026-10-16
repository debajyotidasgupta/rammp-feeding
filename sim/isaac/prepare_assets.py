#!/usr/bin/env python3
"""The replay's scene meshes, decimated for thousands of copies: the wheelchair (ramms-sim's Unreal model), the seated
SMPL-X person, the round bowl and the utensil -> one npz per asset (vertices, faces, colour per part), in the arm's
base frame (the person with its mouth at the origin: the replay moves it onto each scenario's mouth).

    .venv/bin/python sim/isaac/prepare_assets.py out/sim/isaac/assets
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent.parent
CHAIR = ROOT / "sim" / "assets" / "wheelchair"
HUMAN = ROOT / "sim" / "assets" / "human"


def decimate(m: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    if len(m.faces) <= faces:
        return m
    import fast_simplification
    v, f = fast_simplification.simplify(m.vertices, m.faces, target_reduction=1 - faces / len(m.faces))
    return trimesh.Trimesh(v, f, process=False)


def lathe(profile, n=64):
    """A surface of revolution about z from (r, z) points."""
    v, f = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        v += [(r * math.cos(a), r * math.sin(a), z) for r, z in profile]
    m = len(profile)
    for k in range(n):
        for i in range(m - 1):
            a, b = k * m + i, ((k + 1) % n) * m + i
            f += [(a, b, a + 1), (b, b + 1, a + 1)]
    return trimesh.Trimesh(np.array(v), np.array(f), process=False)


def save(path: Path, parts: dict) -> None:
    out = {}
    for name, (mesh, rgb) in parts.items():
        out[name + "_v"] = np.asarray(mesh.vertices, np.float32)
        out[name + "_f"] = np.asarray(mesh.faces, np.int32)
        out[name + "_c"] = np.asarray(rgb, np.float32)
    np.savez_compressed(path, names=np.array(list(parts)), **out)
    print("%s: %s" % (path.name, ", ".join("%s %d faces" % (n, len(m.faces)) for n, (m, _) in parts.items())))


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "out" / "sim" / "isaac" / "assets")
    out.mkdir(parents=True, exist_ok=True)
    # the wheelchair: the MEBot base and the RAMMP seat, already in base_link (identity), decimated
    chair = {"seat": (decimate(trimesh.load(CHAIR / "seat.obj", force="mesh"), 12000), (0.16, 0.17, 0.19)),
             "base": (decimate(trimesh.load(CHAIR / "mebot.obj", force="mesh"), 20000), (0.32, 0.33, 0.36)),
             "rail_l": (trimesh.load(CHAIR / "rail_l.obj", force="mesh"), (0.55, 0.56, 0.58)),
             "rail_r": (trimesh.load(CHAIR / "rail_r.obj", force="mesh"), (0.55, 0.56, 0.58)),
             "mount": (trimesh.load(CHAIR / "mount_arm.obj", force="mesh"), (0.20, 0.21, 0.23))}
    save(out / "wheelchair.npz", chair)
    # the person (SMPL-X, seated, mouth closed), moved so the mouth is at the origin
    lm = json.loads((HUMAN / "landmarks.json").read_text())
    mouth = np.array(lm["closed"]["mouth"], float)
    colours = {"skin": (0.86, 0.68, 0.56), "shirt": (0.10, 0.10, 0.12), "trousers": (0.16, 0.20, 0.30),
               "shoes": (0.92, 0.92, 0.93), "hair": (0.09, 0.06, 0.04), "eyes": (0.15, 0.09, 0.05), "sclera": (0.93, 0.92, 0.9)}
    budget = {"skin": 16000, "shirt": 6000, "trousers": 5000, "shoes": 2000, "hair": 5000, "eyes": 600, "sclera": 1500}
    person = {}
    for part, rgb in colours.items():
        m = trimesh.load(HUMAN / ("%s_closed.obj" % part), force="mesh")
        m = decimate(m, budget[part])
        m.vertices = m.vertices - mouth
        person[part] = (m, rgb)
    save(out / "person.npz", person)
    # the round bowl (ceramic, rim radius 54 mm, 70 mm tall), a wall of thickness 4 mm and a floor
    prof = [(0.000, 0.004), (0.038, 0.004), (0.046, 0.012), (0.052, 0.040), (0.054, 0.070), (0.058, 0.070),
            (0.056, 0.040), (0.050, 0.010), (0.040, 0.000), (0.000, 0.000)]
    save(out / "bowl.npz", {"bowl": (lathe(prof, 72), (0.93, 0.93, 0.95))})
    # the utensil: a spoon (an elliptic head and a handle) along +x, the head's centre at the origin, food side +z
    a, b = 0.022, 0.0135
    head = []
    for k in range(48):
        th = 2 * math.pi * k / 48
        head.append((a * math.cos(th), b * math.sin(th)))
    v = [(0.0, 0.0, -0.004)] + [(x, y, 0.0) for x, y in head]
    f = [(0, k + 1, (k + 1) % 48 + 1) for k in range(48)]
    spoon = trimesh.Trimesh(np.array(v), np.array(f), process=False)
    handle = trimesh.creation.box(extents=(0.12, 0.009, 0.003))
    handle.apply_translation((-0.082, 0.0, 0.002))
    save(out / "spoon.npz", {"head": (spoon, (0.82, 0.84, 0.88)), "handle": (handle, (0.82, 0.84, 0.88))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
