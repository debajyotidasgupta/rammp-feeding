#!/usr/bin/env python3
"""The scene camera's pose, measured from the arm itself: cuRobo's collision spheres of the arm, gripper and
utensil (at the arm's joints) fitted to the scene camera's depth -- no marker, no other stack's calibration file.

    python -m feeding.calibrate_scene --joints J1,...,J7 [--tool x,y,z,tx,ty,tz]   # add a capture, solve
    python -m feeding.calibrate_scene --solve               # solve again over the kept captures
    python -m feeding.calibrate_scene --reset ...           # forget the captures first (the camera was moved)
    (make calibrate-scene [RESET=1]: the arm's joints read by the driver; nothing moves)

Each capture: the joints, the spheres (the planner server's POST /spheres) and a depth median, kept in
<runs_dir>/scene_calib/; with the wrist camera looking straight down (food_detect_start, --tool given), also the
table's height from the wrist camera (its own depth, its pose from the arm). The fit: every depth point near the
arm's surface pulls the camera pose so it lies on the nearest sphere, and the table's points so they lie level at
that height (each group weighted alike) -- coarse to fine (perception.scene.calibration.gates_m), from the current
extrinsic and turned starts round it (a knocked camera). One pose's chain of spheres leaves a rotation weak; the
table pins it (and its height), else a second pose in view is needed.
Accepted only with enough points on the arm and a small residual; then written to <runs_dir>/scene_extrinsic.json,
which locate_scene.extrinsic() reads first.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from settings import cfg
import feeding.locate_scene as ls

CAL = cfg.perception.scene.calibration
RUNS = Path(str(cfg.deployment.runs_dir)).expanduser()
CAPS = RUNS / "scene_calib"
OUT = RUNS / "scene_extrinsic.json"


def rot(w: np.ndarray) -> np.ndarray:
    """The rotation of the small rotation vector w (Rodrigues)."""
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def spheres(joints: list[float]) -> np.ndarray:
    req = urllib.request.Request("http://127.0.0.1:%d/spheres" % int(cfg.deployment.servers.planner_port),
                                 data=json.dumps(dict(joints=joints)).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return np.array(json.loads(r.read())["spheres"], float)


def wrist_table_z(tool: list[float]) -> float | None:
    """The table's height from the wrist camera looking straight down: its pose from the arm, its depth's most
    common distance (the table fills most of the view at food_detect_start). None when not looking down."""
    from feeding.camera import CAM_IN_TOOL, corrected
    from kinova.frames import R_from_kortex
    R = R_from_kortex(*tool[3:])
    look = R @ np.array([0.0, 0.0, 1.0])
    if look[2] > -float(CAL.wrist_down_cos):
        return None
    with urllib.request.urlopen("http://127.0.0.1:%d/median?frames=15&wait=5&min=5" % int(cfg.deployment.servers.camera_port),
                                timeout=15) as r:
        d = corrected(np.load(__import__("io").BytesIO(r.read()))).astype(float) / 1000.0
    v = d[d > 0]
    h, e = np.histogram(v, bins=np.arange(0.1, 1.0, 0.005))
    mode = e[int(np.argmax(h))] + 0.0025
    v = v[np.abs(v - mode) < 0.01]
    if len(v) < 0.3 * d.size:
        return None                                        # no one dominant surface: not the table
    cam = np.array(tool[:3]) + R @ CAM_IN_TOOL
    return float(cam[2] + look[2] * np.median(v))


def capture(joints: list[float], tool: list[float] | None) -> Path:
    _, depth = ls.fetch(int(cfg.deployment.servers.scene_camera_port))
    pts, _ = ls.depth_points(depth)
    pts = pts[::int(CAL.stride)]
    tz = wrist_table_z(tool) if tool else None
    CAPS.mkdir(parents=True, exist_ok=True)
    f = CAPS / time.strftime("cap_%Y%m%d_%H%M%S.npz")
    np.savez_compressed(f, joints=np.array(joints), spheres=spheres(joints), points=pts.astype(np.float32), t=time.time(),
                        table_z=np.nan if tz is None else tz)
    if tz is not None:
        print("the table at z %.4f (the wrist camera)" % tz)
    return f


def residuals(caps, R, t, gate):
    """Per capture's points near the arm: the distance to the nearest sphere's surface, and its outward normal."""
    out = []
    for P, S in caps:
        X = P @ R.T + t
        D = np.linalg.norm(X[:, None, :] - S[None, :, :3], axis=2)
        k = np.argmin(np.abs(D - S[None, :, 3]), axis=1)
        d = D[np.arange(len(X)), k] - S[k, 3]
        m = np.abs(d) < gate
        n = (X[m] - S[k[m], :3]) / np.maximum(D[np.arange(len(X)), k][m], 1e-9)[:, None]
        out.append((X[m], n, d[m]))
    X = np.vstack([o[0] for o in out]) if out else np.zeros((0, 3))
    return X, np.vstack([o[1] for o in out]), np.concatenate([o[2] for o in out])


def table_rows(caps, R, t, gate, table_z):
    """The table's points (within gate of the level plane at table_z, away from the arm): their heights' residuals."""
    if table_z is None:
        return np.zeros((0, 3)), np.zeros(0)
    X = np.vstack([P @ R.T + t for P, S in caps])
    near_arm = np.zeros(len(X), bool)
    k = 0
    for P, S in caps:
        Xi = X[k:k + len(P)]
        near_arm[k:k + len(P)] = (np.linalg.norm(Xi[:, None, :] - S[None, :, :3], axis=2) - S[None, :, 3]).min(1) < 0.05
        k += len(P)
    m = (np.abs(X[:, 2] - table_z) < gate) & ~near_arm
    return X[m], X[m, 2] - table_z


def fit(caps, R, t, table_z=None):
    """Gauss-Newton on the arm's points to its spheres (and the table's to its level plane), coarse to fine;
    (R, t, rms on the arm, points on the arm) at the finest gate."""
    ez = np.array([0.0, 0.0, 1.0])
    for gate in (float(g) for g in CAL.gates_m):
        for _ in range(int(CAL.iterations)):
            X, n, d = residuals(caps, R, t, gate)
            if len(d) < 30:
                break
            J = np.hstack([np.cross(X, n), n])                 # d(w, v) ~ d + (X x n).w + n.v
            w = np.minimum(1.0, 0.01 / np.maximum(np.abs(d), 1e-9))   # Huber, 1 cm
            Xt, dt = table_rows(caps, R, t, gate, table_z)
            if len(dt):                                       # z(w, v) ~ z + (X x ez).w + v_z: as much weight as the arm
                Jt = np.hstack([np.cross(Xt, ez), np.tile(ez, (len(Xt), 1))])
                wt = np.minimum(1.0, 0.01 / np.maximum(np.abs(dt), 1e-9)) * len(d) / len(dt)
                J, d, w = np.vstack([J, Jt]), np.concatenate([d, dt]), np.concatenate([w, wt])
            x = np.linalg.solve((J * w[:, None]).T @ J + 1e-9 * np.eye(6), -(J * w[:, None]).T @ d)
            dR = rot(x[:3])
            R, t = dR @ R, dR @ t + x[3:]
    _, _, d = residuals(caps, R, t, float(CAL.gates_m[-1]))
    return R, t, float(np.sqrt(np.mean(d ** 2))) if len(d) else float("inf"), int(len(d))


def crop(cap, R0, t0, table_z):
    """Only what the fit can use: points within crop_m of the arm's spheres under the guess, and (with a table
    height) a sample of the rest near the table's level -- a few thousand points, not the whole frame."""
    P, S = cap
    X = P @ R0.T + t0
    near = (np.linalg.norm(X[:, None, :] - S[None, :, :3], axis=2) - S[None, :, 3]).min(1) < float(CAL.crop_m)
    keep = near
    if table_z is not None:
        flat = np.flatnonzero(~near & (np.abs(X[:, 2] - table_z) < float(CAL.gates_m[0])))
        keep = near.copy()
        keep[flat[:: max(1, len(flat) // int(CAL.table_points))]] = True
    return P[keep], S


def best_fit(caps, R0, t0, table_z=None):
    """The best fit from the guess and starts turned round the camera's own centre (one pose's chain of spheres
    leaves a rotation weak: the starts, and more poses, keep the fit out of a wrong minimum)."""
    best = None
    turns = [s * np.radians(float(a)) * e for a in CAL.start_turns_deg for e in np.eye(3) for s in (-1, 1)]
    for w in [np.zeros(3)] + turns:
        R, t, rms, n = fit(caps, R0 @ rot(w), t0, table_z)   # turned about the camera's axes, round its centre
        if best is None or rms < best[2]:
            best = (R, t, rms, n)
    return best


def solve(R0, t0):
    """best_fit over the kept captures; (R, t, rms, points, captures that see the arm: min_points of their own)."""
    caps, tzs = [], []
    for f in sorted(CAPS.glob("cap_*.npz")):
        z = np.load(f)
        if time.time() - float(z["t"]) < float(CAL.max_age_s):
            caps.append((z["points"].astype(float), z["spheres"]))
            if "table_z" in z and np.isfinite(z["table_z"]):
                tzs.append(float(z["table_z"]))
    if not caps:
        raise SystemExit("no captures -- make calibrate-scene with the arm in the scene camera's view")
    table_z = float(np.median(tzs)) if tzs else None
    caps = [crop(c, R0, t0, table_z) for c in caps]
    R, t, rms, n = best_fit(caps, R0, t0, table_z)
    seeing = sum(len(residuals([c], R, t, float(CAL.gates_m[-1]))[2]) >= int(CAL.min_points) for c in caps)
    return R, t, rms, n, seeing, table_z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", help="the arm's joints (deg, comma-separated): add a capture at this pose")
    ap.add_argument("--tool", help="the tool pose (x, y, z m, theta x, y, z deg): the wrist camera's table height")
    ap.add_argument("--solve", action="store_true", help="solve over the kept captures only")
    ap.add_argument("--reset", action="store_true", help="forget the kept captures first")
    a = ap.parse_args()
    if a.reset:
        for f in CAPS.glob("cap_*.npz"):
            f.unlink()
    if a.joints:
        f = capture([float(v) for v in a.joints.split(",")], [float(v) for v in a.tool.split(",")] if a.tool else None)
        print("capture %s" % f.name)
    R0, t0, source = ls.extrinsic()
    R, t, rms, n, ncap, table_z = solve(R0, t0)
    moved = float(np.linalg.norm(t - t0)), float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1))))
    print("fit: %d points on the arm, rms %.1f mm, %d capture(s) seeing it, %s; the camera %.0f mm / %.1f deg from %s"
          % (n, 1000 * rms, ncap, "the table level at z %.4f (wrist camera)" % table_z if table_z is not None
             else "no table height", 1000 * moved[0], moved[1], source))
    need = 1 if table_z is not None else int(CAL.min_captures)      # the table pins what one pose leaves weak
    if rms > float(CAL.max_rms_m) or n < int(CAL.min_points) or ncap < need:
        print("NOT accepted (needs rms <= %.0f mm, >= %d points, >= %d capture(s) seeing the arm -- one at "
              "food_detect_start gives the table too)" % (1000 * float(CAL.max_rms_m), int(CAL.min_points), need))
        return 1
    from scipy.spatial.transform import Rotation
    OUT.write_text(json.dumps(dict(parent_frame="base_link", child_frame="scene_camera_link",
                                   xyz=[round(float(v), 5) for v in t],
                                   quat_xyzw=[round(float(v), 6) for v in Rotation.from_matrix(R).as_quat()],
                                   rms_m=round(rms, 4), points=n, captures=ncap, t=time.time(), table_z=table_z,
                                   method="arm spheres ICP (feeding/calibrate_scene.py)"), indent=1))
    print("accepted -> %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
