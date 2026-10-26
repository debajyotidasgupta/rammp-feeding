"""The fixed scene camera: the COARSE bowl and table (to its error_m: it only places food_detect_start; the wrist
camera aims), and the camera's own pose, measured from the arm.

The camera's model -- its intrinsics, depth -> colour, and where its optical frames sit in its link frame -- is its
driver's (camera_info, /tf_static); where its link frame sits on the robot is ours: state_dir/scene_extrinsic.json
(rammp calibrate-scene), published as the static transform base_link -> <its link frame> for everyone on the graph.

locate     the bowl: the always-on scene segmenter's latest "food container" (/perception/object_detections, with
           masks; SAM3 with perception.scene.bowl_prompts when it has none); its points in the base frame; the table
           round it (a plane fit to the dominant flat surface), its centre (a circle through the rim) and its top
           -> state_dir/scene_state.json, the reachable bowl nearest workspace.bowl_centre_m (where it is laid)
calibrate  the arm's collision spheres (config/arm_spheres.yaml, at its joints by our FK) fitted to the camera's depth:
           every depth point near the arm's surface pulls the camera's pose so it lies on the nearest sphere, and, with
           the wrist camera looking straight down, the table's points so they lie level at the height it measures
           (point-to-sphere Gauss-Newton, coarse to fine, from the current pose and turned starts round it); accepted
           only with enough points on the arm and a small residual
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from rammp_feeding import record
from rammp_feeding.config import ROOT, cfg, state_dir
from rammp_feeding.geometry import matrix_quat, quat_matrix, rodrigues
from rammp_feeding.perception import _in, mask_of

log = logging.getLogger("rammp")
SC, PS = cfg.scene_camera, cfg.perception.scene
CAL = PS.calibration


def extrinsic(guess: bool = False) -> tuple[np.ndarray, str]:
    """base_link <- the camera's link frame (4 x 4): our calibration (state_dir/scene_extrinsic.json); guess=True (the
    calibration's start): the configured guess where there is none yet."""
    f = state_dir() / "scene_extrinsic.json"
    T = np.eye(4)
    if f.exists():
        d = json.loads(f.read_text())
        T[:3, :3], T[:3, 3] = quat_matrix(d["quat_xyzw"]), d["xyz"]
        return T, str(f)
    e = SC.extrinsic
    if not guess or e.get("xyz") is None or e.get("quat_xyzw") is None:
        raise RuntimeError("the scene camera is not calibrated (%s): rammp calibrate-scene, from a rough guess in "
                           "local.yaml (scene_camera.extrinsic xyz, quat_xyzw)" % f)
    T[:3, :3], T[:3, 3] = quat_matrix(e.quat_xyzw), e.xyz
    return T, "the configured guess"


def publish(robot) -> None:
    """Our measured pose of the camera on /tf_static, base_link -> its link frame, if there is one."""
    if (state_dir() / "scene_extrinsic.json").exists():
        robot.publish_static("base_link", str(SC.link_frame), extrinsic()[0])


def _points(robot, D: np.ndarray, base_from_link: np.ndarray):
    """The depth frame's points in the base frame, and their colour pixels."""
    cam = robot.camera_model("scene_camera")
    p = cam.depth_points(D)                                          # the colour optical frame
    T = base_from_link @ robot.transform(str(SC.link_frame), robot.cameras["scene_camera"].colour_frame)
    return p @ T[:3, :3].T + T[:3, 3], np.round(np.stack(cam.pixel(p.T), 1)).astype(int)


def table_height(p: np.ndarray, at_xy) -> float:
    """The dominant flat surface (the most common height, 1 cm bins), fitted as a plane and read at at_xy."""
    z = p[:, 2]
    lo = np.floor(z.min() * 100) / 100
    peak = lo + (int(np.argmax(np.bincount(((z - lo) * 100).astype(int)))) + 0.5) / 100
    q = p[np.abs(z - peak) < 0.03]
    for _ in range(4):
        A = np.c_[q[:, 0], q[:, 1], np.ones(len(q))]
        c, *_ = np.linalg.lstsq(A, q[:, 2], rcond=None)
        r = q[:, 2] - A @ c
        q = q[np.abs(r) < 2.5 * r.std() + 1e-4]
    return float(c[0] * at_xy[0] + c[1] * at_xy[1] + c[2])


def bowl_at(mask: np.ndarray, box, uv: np.ndarray, p: np.ndarray) -> dict:
    """One candidate -> the bowl it is (centre, top, the table round it), or why not."""
    inside = _in(mask, uv)
    w, h = box[2] - box[0], box[3] - box[1]
    ring = ((uv[:, 0] >= box[0] - 0.6 * w) & (uv[:, 0] <= box[2] + 0.6 * w) & (uv[:, 1] >= box[1] - 0.3 * h)
            & (uv[:, 1] <= box[3] + 0.6 * h) & ~inside)
    res = dict(box=[round(float(v), 1) for v in box])
    if inside.sum() < 50 or ring.sum() < 50:
        return dict(res, rejected="too little depth at or round it")
    b = p[inside]
    tz = table_height(p[ring], np.median(b[:, :2], axis=0))
    above = b[(b[:, 2] > tz + 0.010) & (b[:, 2] < tz + float(PS.max_bowl_height_m))]
    if len(above) < 30:
        return dict(res, rejected="nothing bowl-sized on the table")
    top = float(np.percentile(above[:, 2], 95))
    rim = above[above[:, 2] > top - 0.015][:, :2]
    centre, radius = above[:, :2].mean(0), None
    if len(rim) >= 20:                                  # a circle through the rim: the near and the far edge alike
        A = np.column_stack([2 * rim[:, 0], 2 * rim[:, 1], np.ones(len(rim))])
        cx, cy, c = np.linalg.lstsq(A, (rim ** 2).sum(1), rcond=None)[0]
        rr = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
        if 0.03 < rr < float(PS.max_bowl_radius_m):
            centre, radius = np.array([cx, cy]), rr
    return dict(res, bowl_xy=[float(v) for v in centre], table_z=tz, bowl_top_z=top,
                rim_radius_m=None if radius is None else round(radius, 3))


def locate(robot, out: Path) -> dict:
    T, source = extrinsic()
    cam = robot.cameras["scene_camera"]
    since = time.time()
    _, rgb = cam.frame()
    D = cam.depth_median(15, since, wait_s=5.0)
    Image.fromarray(rgb).save(out / "scene_colour.jpg", quality=90)
    np.save(out / "scene_depth.npy", D)
    p, uv = _points(robot, D, T)
    dets = [d for d in (robot.segmented_objects(float(PS.segmenter_max_age_s)) or []) if d.label in PS.segmenter_classes]
    how = "the scene segmenter"
    if not dets:
        dets, how = robot.detect("scene_camera", list(PS.bowl_prompts), float(PS.score_threshold)), "SAM3"
    ref = list(cfg.workspace.bowl_centre_m)                # where the operator lays the bowl
    cands = []
    for d in sorted(dets, key=lambda d: -d.score):
        m = bowl_at(mask_of(d, rgb.shape[:2]), d.box, uv, p)
        if "bowl_xy" in m and np.hypot(*m["bowl_xy"]) > float(PS.max_reach_m):
            m["rejected"] = "out of reach"
        cands.append(m)
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for m in cands:
        dr.rectangle(m["box"], outline=(239, 68, 68) if m.get("rejected") else (34, 197, 94), width=3)
        dr.text((m["box"][0], m["box"][1] - 14), m.get("rejected") or "(%.3f, %.3f)" % tuple(m["bowl_xy"]), fill=(255, 255, 255))
    im.save(out / "scene_detection.jpg", quality=90)
    ok = [m for m in cands if not m.get("rejected")]
    if not ok:
        raise RuntimeError("no reachable bowl in the scene camera's view (%s: %d candidates; %s)" % (how, len(cands), out))
    best = min(ok, key=lambda m: np.hypot(m["bowl_xy"][0] - ref[0], m["bowl_xy"][1] - ref[1]))
    st = dict(best, t=time.time(), extrinsic=source, found_by=how, record=str(out))
    (state_dir() / "scene_state.json").write_text(json.dumps(st, indent=1))
    log.info("  the bowl at (%.3f, %.3f) (+-%.2f m; %s), table z %.3f, top z %.3f", *best["bowl_xy"], float(SC.error_m),
             how, best["table_z"], best["bowl_top_z"])
    record.event("located", **st)
    return st


# ------------------------------------------------------------------ calibration
def spheres(arm, q) -> np.ndarray:
    """The arm's collision spheres at joints q (base frame), and the gripper and the utensil along the tool axis."""
    frames = arm.kin.frames(q)
    out = []
    for link, ss in yaml.safe_load((ROOT / "config" / "arm_spheres.yaml").read_text()).items():
        F = frames[link]
        out += [[*(F[:3, :3] @ np.array(s[:3]) + F[:3, 3]), s[3]] for s in ss]
    T, R = arm.tool(q)
    flange = T - R @ np.asarray(cfg.arm.tool_from_flange_m, float)
    for z, r in ((0.04, 0.04), (0.08, 0.035)):          # the gripper's body
        out.append([*(flange + R @ np.array([0, 0, z])), r])
    tip = T + R @ arm.tip_tool
    for s in np.linspace(0.3, 1.0, 4):                  # the utensil, from the fingers to the tip
        out.append([*(T + s * (tip - T)), 0.012])
    return np.array(out)


def wrist_table_z(robot, arm) -> float | None:
    """The table's height from the wrist camera looking straight down: its pose from the arm, its depth's most common
    distance (the table fills most of the view at food_detect_start). None when not looking down."""
    from rammp_feeding.scene import camera_pose
    o, Rc = camera_pose(arm)
    look = Rc[:, 2]
    if look[2] > -float(CAL.wrist_down_cos):
        return None
    d = robot.cameras["camera"].depth_median(15, time.time(), wait_s=5.0).astype(float) / 1000.0
    v = d[d > 0]
    h, e = np.histogram(v, bins=np.arange(0.1, 1.0, 0.005))
    mode = e[int(np.argmax(h))] + 0.0025
    v = v[np.abs(v - mode) < 0.01]
    if len(v) < 0.3 * d.size:
        return None
    return float(o[2] + look[2] * np.median(v))


def _residuals(caps, R, t, gate):
    X_, n_, d_ = [], [], []
    for P, S in caps:
        X = P @ R.T + t
        D = np.linalg.norm(X[:, None, :] - S[None, :, :3], axis=2)
        k = np.argmin(np.abs(D - S[None, :, 3]), axis=1)
        dist = D[np.arange(len(X)), k]
        d = dist - S[k, 3]
        m = np.abs(d) < gate
        X_.append(X[m]); n_.append((X[m] - S[k[m], :3]) / np.maximum(dist[m], 1e-9)[:, None]); d_.append(d[m])
    return np.vstack(X_), np.vstack(n_), np.concatenate(d_)


def _table_rows(caps, R, t, gate, table_z):
    if table_z is None:
        return np.zeros((0, 3)), np.zeros(0)
    X = np.vstack([P @ R.T + t for P, _ in caps])
    S = np.vstack([S for _, S in caps])
    near = (np.linalg.norm(X[:, None, :] - S[None, :, :3], axis=2) - S[None, :, 3]).min(1) < 0.05
    m = (np.abs(X[:, 2] - table_z) < gate) & ~near
    return X[m], X[m, 2] - table_z


def _fit(caps, R, t, table_z):
    ez = np.array([0.0, 0.0, 1.0])
    for gate in (float(g) for g in CAL.gates_m):
        for _ in range(int(CAL.iterations)):
            X, n, d = _residuals(caps, R, t, gate)
            if len(d) < 30:
                break
            J, w = np.hstack([np.cross(X, n), n]), np.minimum(1.0, 0.01 / np.maximum(np.abs(d), 1e-9))
            Xt, dt = _table_rows(caps, R, t, gate, table_z)
            if len(dt):
                J = np.vstack([J, np.hstack([np.cross(Xt, ez), np.tile(ez, (len(Xt), 1))])])
                w = np.concatenate([w, np.minimum(1.0, 0.01 / np.maximum(np.abs(dt), 1e-9)) * len(d) / len(dt)])
                d = np.concatenate([d, dt])
            x = np.linalg.solve((J * w[:, None]).T @ J + 1e-9 * np.eye(6), -(J * w[:, None]).T @ d)
            dR = rodrigues(x[:3])
            R, t = dR @ R, dR @ t + x[3:]
    _, _, d = _residuals(caps, R, t, float(CAL.gates_m[-1]))
    return R, t, float(np.sqrt(np.mean(d ** 2))) if len(d) else float("inf"), int(len(d))


def calibrate(robot, arm, reset: bool = False) -> dict | None:
    """One capture at the arm's pose now, then the fit over every capture kept (--reset forgets them: the camera was
    moved); written and published if accepted."""
    caps_dir = state_dir() / "scene_calib"
    caps_dir.mkdir(parents=True, exist_ok=True)
    if reset:
        for f in caps_dir.glob("cap_*.npz"):
            f.unlink()
    q = arm.q()
    D = robot.cameras["scene_camera"].depth_median(15, time.time(), wait_s=5.0)
    p, _ = _points(robot, D, np.eye(4))                 # in the camera's link frame
    tz = wrist_table_z(robot, arm)
    np.savez_compressed(caps_dir / time.strftime("cap_%Y%m%d_%H%M%S.npz"), points=p[::int(CAL.stride)].astype(np.float32),
                        spheres=spheres(arm, q), t=time.time(), table_z=np.nan if tz is None else tz)
    caps, tzs = [], []
    for f in sorted(caps_dir.glob("cap_*.npz")):
        z = np.load(f)
        caps.append((z["points"].astype(float), z["spheres"]))
        if np.isfinite(z["table_z"]):
            tzs.append(float(z["table_z"]))
    table_z = float(np.median(tzs)) if tzs else None
    T0, source = extrinsic(guess=True)
    R0, t0 = T0[:3, :3], T0[:3, 3]
    cropped = []
    for P, S in caps:                                   # only what the fit can use: near the arm, and the table's level
        X = P @ R0.T + t0
        near = (np.linalg.norm(X[:, None, :] - S[None, :, :3], axis=2) - S[None, :, 3]).min(1) < float(CAL.crop_m)
        keep = near.copy()
        if table_z is not None:
            flat = np.flatnonzero(~near & (np.abs(X[:, 2] - table_z) < float(CAL.gates_m[0])))
            keep[flat[:: max(1, len(flat) // int(CAL.table_points))]] = True
        cropped.append((P[keep], S))
    best = None
    for w in [np.zeros(3)] + [s * np.radians(float(a)) * e for a in CAL.start_turns_deg for e in np.eye(3) for s in (-1, 1)]:
        fit = _fit(cropped, R0 @ rodrigues(w), t0, table_z)
        if best is None or fit[2] < best[2]:
            best = fit
    R, t, rms, n = best
    seeing = sum(len(_residuals([c], R, t, float(CAL.gates_m[-1]))[2]) >= int(CAL.min_points) for c in cropped)
    log.info("  fit: %d points on the arm, rms %.1f mm, %d capture(s) seeing it, %s; the camera %.0f mm from %s",
             n, 1000 * rms, seeing, "the table at z %.4f" % table_z if table_z is not None else "no table height",
             1000 * float(np.linalg.norm(t - t0)), source)
    need = 1 if table_z is not None else int(CAL.min_captures)
    if rms > float(CAL.max_rms_m) or n < int(CAL.min_points) or seeing < need:
        log.info("  NOT accepted (rms <= %.0f mm, >= %d points, >= %d capture(s) seeing the arm)",
                 1000 * float(CAL.max_rms_m), int(CAL.min_points), need)
        return None
    res = dict(parent_frame="base_link", child_frame=str(SC.link_frame), xyz=[round(float(v), 5) for v in t],
               quat_xyzw=[round(v, 6) for v in matrix_quat(R)], rms_m=round(rms, 4), points=n, captures=seeing,
               table_z=table_z, t=time.time())
    (state_dir() / "scene_extrinsic.json").write_text(json.dumps(res, indent=1))
    publish(robot)
    log.info("  accepted -> %s (and on /tf_static: base_link -> %s)", state_dir() / "scene_extrinsic.json", SC.link_frame)
    return res
