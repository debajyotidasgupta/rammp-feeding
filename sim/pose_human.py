#!/usr/bin/env python3
"""Pose SMPL-X into the seated user and export the meshes the twin scene loads. A one-time step:

    .venv/bin/python sim/pose_human.py        # needs torch + smplx and sim/assets/smplx/SMPLX_NEUTRAL.npz

SMPL-X is licensed, so neither the body model nor these meshes are in the repo (.gitignore). Writes
sim/assets/human/<part>_<closed|open>.obj -- one mesh per material (skin, hair, eyes, sclera, shirt, trousers, shoes),
with the jaw closed and open -- and landmarks.json. Trousers and shoes stay on the seat; the rest (skin, hair,
eyes, sclera, shirt) is the upper body, which sim/scene.py hinges at the lumbar spine so the user can lean in to
bite. Frame: Z-up, the person faces +X, their left is +Y, the floor is z = 0. sim/scene.py moves the whole person
so the mouth lands where the recording measured it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import smplx
import torch

HERE = Path(__file__).parent
OUT = HERE / "assets" / "human"

BODY = {"left_hip": 0, "right_hip": 1, "spine1": 2, "left_knee": 3, "right_knee": 4, "spine2": 5,
        "left_ankle": 6, "right_ankle": 7, "spine3": 8, "neck": 11, "left_collar": 12, "right_collar": 13,
        "head": 14, "left_shoulder": 15, "right_shoulder": 16, "left_elbow": 17, "right_elbow": 18,
        "left_wrist": 19, "right_wrist": 20}              # index into body_pose (21 x 3, axis-angle)
JOINT = {"pelvis": 0, "spine1": 3, "left_knee": 4, "neck": 12, "head": 15, "jaw": 22}   # index into the joints output

# the material each vertex takes, by the joint that drives it most (SMPL-X joint ids)
TROUSERS = {0, 1, 2, 4, 5}
SHOES = {7, 8, 10, 11}
SUBDIVIDE = 2                                         # 16x the triangles: clean material edges
JAW_OPEN_RAD = 0.30                                   # ~17 deg: a mouth open for a bite


def seated_pose() -> torch.Tensor:
    """Hips and knees at ~90 deg, the forearms resting forward on the thighs, the head level."""
    bp = torch.zeros(1, 21, 3)
    for side, s in (("left", 1.0), ("right", -1.0)):
        bp[0, BODY[side + "_hip"], 0] = -np.pi / 2
        bp[0, BODY[side + "_knee"], 0] = np.pi / 2
        bp[0, BODY[side + "_shoulder"], 2] = -1.20 * s        # the arm down out of the T-pose
        bp[0, BODY[side + "_shoulder"], 1] = -0.25 * s        # ... and a little forward
        bp[0, BODY[side + "_elbow"], 1] = -1.25 * s           # the forearm forward, over the thigh
        bp[0, BODY[side + "_collar"], 2] = -0.10 * s
        bp[0, BODY[side + "_wrist"], 2] = 0.25 * s
    bp[0, BODY["spine2"], 0] = 0.06                          # a slight lean toward the table
    bp[0, BODY["head"], 0] = -0.06                           # ... with the eyes level
    return bp


def to_mj(a: np.ndarray) -> np.ndarray:
    """(x, y, z)_smplx -> (z, x, y): a cyclic permutation (det +1) from Y-up/+Z-forward to Z-up/+X-forward."""
    a = np.asarray(a)
    return np.stack([a[..., 2], a[..., 0], a[..., 1]], axis=-1)


def write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    keep = np.unique(f)
    remap = np.full(len(v), -1, np.int64)
    remap[keep] = np.arange(len(keep))
    with path.open("w") as fh:
        fh.writelines("v %.5f %.5f %.5f\n" % tuple(p) for p in v[keep])
        fh.writelines("f %d %d %d\n" % tuple(t + 1) for t in remap[f])


def subdivide(v: np.ndarray, f: np.ndarray, attr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split every triangle in four at its edge midpoints (shared, so the surface stays closed); `attr` (per
    vertex) is interpolated. Material edges are drawn per triangle, so this makes them 2x finer."""
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    mid = len(v) + np.arange(len(uniq))
    v2 = np.vstack([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2])
    a2 = np.vstack([attr, (attr[uniq[:, 0]] + attr[uniq[:, 1]]) / 2])
    n = len(f)
    m01, m12, m20 = mid[inv[:n]], mid[inv[n:2 * n]], mid[inv[2 * n:]]
    f2 = np.vstack([np.column_stack([f[:, 0], m01, m20]), np.column_stack([m01, f[:, 1], m12]),
                    np.column_stack([m20, m12, f[:, 2]]), np.column_stack([m01, m12, m20])])
    return v2, f2, a2


def labels(v: np.ndarray, lbs: np.ndarray, j: np.ndarray) -> np.ndarray:
    """The material of each vertex. Body regions come from summed skinning weights, but every VISIBLE edge (the
    neckline, the sleeves, the hairline) is a smooth geometric cut -- a weight threshold makes a saw-tooth."""
    def w(ids):
        return lbs[:, sorted(ids)].sum(axis=1)
    lab = np.full(len(v), "shirt", dtype=object)
    head, neck = j[JOINT["head"]], j[JOINT["neck"]]
    # skin above a crew neckline: high at the back, dipping at the front, round the neck only
    r_neck = np.hypot(v[:, 0] - neck[0], v[:, 1] - neck[1])
    neckline = neck[2] - 0.012 - 0.030 * np.clip((v[:, 0] - neck[0] + 0.02) / 0.08, 0, 1)   # higher at the back
    lab[((v[:, 2] > neckline) & (r_neck < 0.095)) | (v[:, 2] > neck[2] + 0.03)] = "skin"
    for k in (18, 19):                                                   # T-shirt sleeves: to 110 mm above the elbow
        elbow, shoulder = j[k], j[k - 2]
        axis = (shoulder - elbow) / np.linalg.norm(shoulder - elbow)
        rel = v - elbow
        along = rel @ axis
        radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)
        lab[(along < 0.11) & (along > -0.02) & (radial < 0.07)] = "skin"
        lab[w({k, k + 2} | set(range(25, 55))) > 0.5] = "skin"           # the forearm, the wrist, the hand
    lab[w(TROUSERS) + w(SHOES) > 0.5] = "trousers"
    lab[w(SHOES) > 0.5] = "shoes"
    # hair: above a hairline that runs from the forehead (+65 mm over the head joint) down to the nape (-50 mm)
    rel = v - head
    on_head = (w({12, 15, 22, 23, 24}) > 0.5) | (rel[:, 2] > 0.0)
    front = np.clip((rel[:, 0] + 0.06) / 0.14, 0, 1)                     # 0 at the back of the head, 1 at the brow
    hairline = -0.050 + 0.115 * front ** 1.4
    lab[on_head & (rel[:, 2] > hairline) & (rel[:, 0] < 0.085) & (np.linalg.norm(rel, axis=1) < 0.16)] = "hair"
    for k in (23, 24):                                                   # the eyeballs: white, iris to the front
        e = lbs[:, k] > 0.5
        if e.any():
            c = v[e].mean(axis=0)
            r = np.linalg.norm(v[e] - c, axis=1).max()
            lab[e] = "sclera"
            lab[e & ((v[:, 0] - c[0]) > 0.72 * r)] = "eyes"
    return lab


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = smplx.create(str(HERE / "assets"), model_type="smplx", gender="neutral", use_pca=False,
                         batch_size=1)
    lbs = model.lbs_weights.detach().numpy()
    faces = np.asarray(model.faces, dtype=np.int64)
    landmarks: dict = {"frame": "Z-up, faces +X, left +Y, floor z=0 (metres)"}
    lab = None
    z_floor = None
    for name, jaw in (("closed", 0.0), ("open", JAW_OPEN_RAD)):
        out = model(body_pose=seated_pose(), jaw_pose=torch.tensor([[jaw, 0.0, 0.0]]), return_verts=True)
        v = to_mj(out.vertices.detach().numpy()[0])
        j = to_mj(out.joints.detach().numpy()[0])
        if z_floor is None:
            z_floor = float(v[:, 2].min())
        v[:, 2] -= z_floor
        j[:, 2] -= z_floor
        vs, fs, ws = v, faces, lbs
        for _ in range(SUBDIVIDE):
            vs, fs, ws = subdivide(vs, fs, ws)
        if lab is None:                                     # the same materials for both (the jaw only moves)
            lab = labels(vs, ws, j)
        votes = lab[fs]
        face_lab = np.where(votes[:, 1] == votes[:, 2], votes[:, 1], votes[:, 0])   # the majority of three
        for part in ("skin", "hair", "eyes", "sclera", "shirt", "trousers", "shoes"):
            write_obj(OUT / ("%s_%s.obj" % (part, name)), vs, fs[face_lab == part])
        # the mouth: on the midline, between the lips -- the most forward point of the band under the nose
        head = j[JOINT["head"]]
        mid = np.abs(v[:, 1] - head[1]) < 0.006
        front = mid & (v[:, 2] > head[2] - 0.12) & (v[:, 2] < head[2] + 0.04)
        nose = v[np.where(front)[0][np.argmax(v[front, 0])]]
        band = mid & (v[:, 2] > nose[2] - 0.05) & (v[:, 2] < nose[2] - 0.02)
        lips = v[band]
        mouth = np.array([lips[:, 0].max() - 0.004, head[1], float(np.median(lips[:, 2]))])
        landmarks[name] = {"mouth": mouth.round(4).tolist(), "nose": nose.round(4).tolist(),
                           "head": head.round(4).tolist(), "pelvis": j[0].round(4).tolist(),
                           "knee": j[JOINT["left_knee"]].round(4).tolist(),
                           "lean_hinge": j[JOINT["spine1"]].round(4).tolist()}
        print("%-6s mouth %s  nose %s  pelvis z %.3f" % (name, mouth.round(3), nose.round(3), j[0][2]))
    landmarks["bounds_min"] = v.min(axis=0).round(4).tolist()
    landmarks["bounds_max"] = v.max(axis=0).round(4).tolist()
    (OUT / "landmarks.json").write_text(json.dumps(landmarks, indent=1))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
