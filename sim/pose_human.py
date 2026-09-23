#!/usr/bin/env python3
"""Pose SMPL-X into a seated posture and export assets for the MuJoCo scene.

One-time offline step: the simulation itself needs neither torch nor smplx,
only the OBJ and the landmark JSON this writes.

    python sim/pose_human.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
import smplx

HERE = pathlib.Path(__file__).parent
MODEL_DIR = HERE / "assets"          # expects assets/smplx/SMPLX_NEUTRAL.npz
OUT = HERE / "assets" / "human"

# SMPL-X body_pose is 21 joints x 3 (axis-angle), global_orient separate.
BODY = {  # index into body_pose
    "left_hip": 0, "right_hip": 1, "spine1": 2, "left_knee": 3, "right_knee": 4,
    "spine2": 5, "left_ankle": 6, "right_ankle": 7, "spine3": 8,
    "left_foot": 9, "right_foot": 10, "neck": 11, "left_collar": 12,
    "right_collar": 13, "head": 14, "left_shoulder": 15, "right_shoulder": 16,
    "left_elbow": 17, "right_elbow": 18, "left_wrist": 19, "right_wrist": 20,
}
# Full joint indices in the returned `joints` tensor
JOINT = {"pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3, "left_knee": 4,
         "right_knee": 5, "left_ankle": 7, "right_ankle": 8, "neck": 12, "head": 15,
         "left_shoulder": 16, "right_shoulder": 17, "jaw": 22}

# The user must be able to LEAN. "User leans into the fork during approach" is
# the scenario the whole safety architecture in the architecture doc Sec. 0
# exists for, and a single rigid mesh cannot express it. So the body is split at
# the lumbar spine into a part that stays on the seat and a part that pitches
# forward about a hinge there.
#
# Split by LBS weight: each vertex belongs to whichever joint drives it most.
# These joints keep the seat; everything else (spine, neck, head, arms, hands)
# rides the hinge.
LOWER_JOINTS = {0, 1, 2, 4, 5, 7, 8, 10, 11}   # pelvis, hips, knees, ankles, feet


def seated_pose() -> torch.Tensor:
    """Hips and knees flexed ~90 deg. Sign of the X rotation is verified below."""
    bp = torch.zeros(1, 21, 3)
    hip, knee = -np.pi / 2, np.pi / 2          # flex about X
    for side in ("left", "right"):
        bp[0, BODY[f"{side}_hip"], 0] = hip
        bp[0, BODY[f"{side}_knee"], 0] = knee
    # Bring the arms down out of T-pose and rest the hands in the lap.
    # In SMPL-X the shoulder lowers by a large rotation about Z; the sign is
    # opposite per side. Verified below by checking the wrist ends up BELOW the
    # shoulder rather than beside it.
    bp[0, BODY["left_shoulder"], 2] = -1.15
    bp[0, BODY["right_shoulder"], 2] = 1.15
    bp[0, BODY["left_elbow"], 1] = -1.05
    bp[0, BODY["right_elbow"], 1] = 1.05
    # small inward collar tuck so the upper arms sit against the torso
    bp[0, BODY["left_collar"], 2] = -0.12
    bp[0, BODY["right_collar"], 2] = 0.12
    return bp


def write_obj(path: pathlib.Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w") as fh:
        for p in verts:
            fh.write("v %.6f %.6f %.6f\n" % tuple(p))
        for tri in faces:
            fh.write("f %d %d %d\n" % (tri[0] + 1, tri[1] + 1, tri[2] + 1))


def split_body(v: np.ndarray, f: np.ndarray, lbs: np.ndarray):
    """Split the mesh into (lower, upper) at the lumbar spine.

    Faces are assigned by MAJORITY VOTE of their vertices and keep all three
    vertices, so boundary vertices are duplicated into both parts. Dropping
    straddling faces instead would tear a ring of holes around the waist that
    opens up the moment the body leans.
    """
    dom = np.asarray(lbs).argmax(axis=1)               # dominant joint per vertex
    v_lower = np.isin(dom, list(LOWER_JOINTS))
    f_lower = v_lower[f].sum(axis=1) >= 2              # majority vote

    parts = []
    for mask in (f_lower, ~f_lower):
        tris = f[mask]
        keep = np.unique(tris)
        remap = np.full(len(v), -1, np.int64)
        remap[keep] = np.arange(len(keep))
        parts.append((v[keep], remap[tris]))
    return parts[0], parts[1], v_lower


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = smplx.create(str(MODEL_DIR), model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=1)
    out = model(body_pose=seated_pose(), return_verts=True)
    v = out.vertices.detach().numpy()[0]
    j = out.joints.detach().numpy()[0]
    f = model.faces

    print(f"  vertices {v.shape}  faces {f.shape}  joints {j.shape}")
    print("  --- joint positions (model frame, metres) ---")
    for name, idx in JOINT.items():
        print("    %-14s x=%+.3f y=%+.3f z=%+.3f" % (name, *j[idx]))

    # Verify the pose really is seated rather than trusting the sign convention:
    # knees should sit forward of the pelvis and roughly level with it, ankles below.
    pelvis, knee, ankle, head = j[JOINT["pelvis"]], j[JOINT["left_knee"]], j[JOINT["left_ankle"]], j[JOINT["head"]]
    up = int(np.argmax(np.abs(head - pelvis)))     # the axis the torso extends along
    fwd = [a for a in (0, 1, 2) if a != up and a != int(np.argmax(np.abs(j[JOINT["left_hip"]] - j[JOINT["right_hip"]])))][0]
    print("  --- geometry check ---")
    print("    inferred up axis=%d  forward axis=%d" % (up, fwd))
    print("    knee forward of pelvis by %+.3f m (want positive)" % (knee[fwd] - pelvis[fwd]))
    print("    knee vs pelvis along up %+.3f m (want near 0 for seated)" % (knee[up] - pelvis[up]))
    print("    ankle below knee by %+.3f m" % (ankle[up] - knee[up]))
    lsh, lwr = j[16], j[20]      # left shoulder, left wrist (SMPL-X joint ids)
    print("    wrist below shoulder by %+.3f m (want clearly negative)" % (lwr[up] - lsh[up]))
    print("    wrist lateral offset from shoulder %+.3f m (want small)" % (lwr[0] - lsh[0]))

    # --- face landmarks from the mesh surface -------------------------------
    # The SMPL-X `jaw` joint sits at the head centre, not on the face, so it is
    # useless as a feeding target. Derive nose tip and mouth from the surface:
    # on the midline, at head height, the most-forward vertex is the nose tip.
    head_y = j[JOINT["head"]][up]
    midline = np.abs(v[:, 0] - j[JOINT["head"]][0]) < 0.02
    band = (v[:, up] > head_y - 0.10) & (v[:, up] < head_y + 0.06)
    face = np.where(midline & band)[0]
    nose_i = face[np.argmax(v[face, fwd])]
    nose = v[nose_i]
    # mouth: midline, ~25 mm below the nose tip, most-forward vertex in that band
    mband = (v[:, up] > nose[up] - 0.045) & (v[:, up] < nose[up] - 0.015)
    mcand = np.where(midline & mband)[0]
    mouth = v[mcand[np.argmax(v[mcand, fwd])]] if len(mcand) else nose
    print("  --- face landmarks (from mesh surface) ---")
    print("    nose tip   x=%+.3f y=%+.3f z=%+.3f" % tuple(nose))
    print("    mouth      x=%+.3f y=%+.3f z=%+.3f" % tuple(mouth))
    print("    nose is %+.3f m forward of head joint" % (nose[fwd] - j[JOINT["head"]][fwd]))
    print("    mouth is %+.3f m below nose" % (mouth[up] - nose[up]))
    head_r = float(np.linalg.norm(v[face] - j[JOINT["head"]], axis=1).max())
    print("    head radius (no-go sphere) %.3f m" % head_r)

    # --- convert SMPL-X (Y-up, +Z forward) to MuJoCo (Z-up, +X forward) ------
    # (x,y,z)_smplx -> (z,x,y)_mujoco. Cyclic permutation, det=+1, so handedness
    # is preserved: person faces +X, up is +Z, their LEFT is +Y (SMPL +x is left),
    # which puts a right-side arm mount at -Y.
    def to_mj(a):
        a = np.asarray(a)
        return np.stack([a[..., 2], a[..., 0], a[..., 1]], axis=-1)

    v, j, nose, mouth = to_mj(v), to_mj(j), to_mj(nose), to_mj(mouth)
    # drop the model to the floor so z=0 is the ground plane
    z_min = float(v[:, 2].min())
    v[:, 2] -= z_min; j[:, 2] -= z_min; nose[2] -= z_min; mouth[2] -= z_min
    print("  --- converted to MuJoCo frame (Z-up, +X forward), floor at z=0 ---")
    print("    mouth      x=%+.3f y=%+.3f z=%+.3f" % tuple(mouth))
    print("    nose tip   x=%+.3f y=%+.3f z=%+.3f" % tuple(nose))
    print("    head       x=%+.3f y=%+.3f z=%+.3f" % tuple(j[JOINT["head"]]))
    print("    seat height (pelvis z) %.3f m" % j[JOINT["pelvis"]][2])

    # --- visual meshes ------------------------------------------------------
    # Whole body, kept for reference and for anything that does not need to lean.
    write_obj(OUT / "human_seated.obj", v, f)

    # Split at the lumbar spine so the upper body can pitch forward on a hinge.
    (vl, fl), (vu, fu), v_lower = split_body(v, f, model.lbs_weights.detach().numpy())
    write_obj(OUT / "human_lower.obj", vl, fl)
    write_obj(OUT / "human_upper.obj", vu, fu)
    hinge = j[JOINT["spine1"]]
    print("  --- lumbar split ---")
    print("    lower %5d verts %5d faces   upper %5d verts %5d faces"
          % (len(vl), len(fl), len(vu), len(fu)))
    print("    faces kept %d of %d (no face dropped: %s)"
          % (len(fl) + len(fu), len(f), len(fl) + len(fu) == len(f)))
    print("    lean hinge (spine1) x=%+.3f y=%+.3f z=%+.3f" % tuple(hinge))
    # A lean must actually move the mouth forward, and the head must be entirely
    # in the upper part or the no-go sphere would stay behind when the user leans.
    assert vu[:, 2].max() > vl[:, 2].max(), "head landed in the lower part"
    lean = np.radians(20.0)
    c, sn = np.cos(lean), np.sin(lean)
    R = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])       # pitch about +Y
    mouth_leaned = R @ (mouth - hinge) + hinge
    print("    at 20 deg lean the mouth moves %+.3f m forward, %+.3f m down"
          % (mouth_leaned[0] - mouth[0], mouth_leaned[2] - mouth[2]))

    # landmarks the feeding stack needs, in the same frame as the mesh
    landmarks = {
        "frame": "mujoco",
        "note": "metres, Z-up, +X forward, floor at z=0; same frame as human_seated.obj",
        "joints": {k: j[i].tolist() for k, i in JOINT.items()},
        "nose_tip": nose.tolist(),
        "mouth": mouth.tolist(),
        "head_radius": head_r,
        "lean_hinge": hinge.tolist(),
        "bounds_min": v.min(axis=0).tolist(),
        "bounds_max": v.max(axis=0).tolist(),
    }
    (OUT / "landmarks.json").write_text(json.dumps(landmarks, indent=2))
    print("  wrote landmarks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
