"""The fixed scene camera: the COARSE bowl and table (to its error_m: it only places food_detect_start; the wrist
camera aims), and the camera's own pose, measured through a ChArUco board both cameras see.

The camera's model -- its intrinsics, depth -> colour, and where its optical frames sit in its link frame -- is its
driver's (camera_info, /tf_static); where its link frame sits on the robot is ours: state_dir/scene_extrinsic.json
(rammp calibrate-scene), published as the static transform base_link -> <its link frame> for everyone on the graph.

locate     the bowl: SAM3 with perception.scene.bowl_prompts (boxes and masks); its points in the base frame; the
           table round it (a plane fit to the dominant flat surface), its centre and radius (a circle through its rim)
           and its top -> state_dir/scene_state.json, the reachable bowl nearest workspace.bowl_centre_m (where the
           operator lays it)
refine     the same fit on the wrist camera's view from food_detect_start (rammp detect): its bowl centre and radius
           replace this one's
calibrate  through the ChArUco board on the table (perception.wrist_calibration's, as rammp calibrate-wrist uses it):
           this camera sees it from where it is; the wrist camera, through its measured mount, sees it from the route
           over it and so places it in the base frame; this camera's pose follows. Neither the arm nor a guess needs to
           be in its view (the print's scale is the wrist camera's to check, face-on, on the route). Coarse is
           enough: the wrist camera refines
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rammp_feeding import record
from rammp_feeding.config import cfg, state_dir
from rammp_feeding.geometry import matrix_quat, quat_matrix
from rammp_feeding.perception import _in

log = logging.getLogger("rammp")
SC, PS = cfg.scene_camera, cfg.perception.scene


def extrinsic() -> tuple[np.ndarray, str]:
    """base_link <- the camera's link frame (4 x 4): our calibration (state_dir/scene_extrinsic.json)."""
    f = state_dir() / "scene_extrinsic.json"
    if not f.exists():
        raise RuntimeError("the scene camera is not calibrated (%s): rammp calibrate-scene, the ChArUco board where it "
                           "sees it" % f)
    d = json.loads(f.read_text())
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_matrix(d["quat_xyzw"]), d["xyz"]
    return T, str(f)


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
    if len(rim) < 20:
        return dict(res, rejected="too little of its rim in depth")
    A = np.column_stack([2 * rim[:, 0], 2 * rim[:, 1], np.ones(len(rim))])   # a circle through the rim: the near and
    cx, cy, c = np.linalg.lstsq(A, (rim ** 2).sum(1), rcond=None)[0]         # the far edge alike
    radius = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    if not 0.03 < radius < float(PS.max_bowl_radius_m):
        return dict(res, rejected="its rim is no bowl's (radius %.0f mm)" % (1000 * radius))
    return dict(res, bowl_xy=[float(cx), float(cy)], table_z=tz, bowl_top_z=top, rim_radius_m=round(radius, 3))


def locate(robot, out: Path) -> dict:
    T, source = extrinsic()
    cam = robot.cameras["scene_camera"]
    since = time.time()
    _, rgb = cam.frame()
    D = cam.depth_median(15, since, wait_s=5.0)
    Image.fromarray(rgb).save(out / "scene_colour.jpg", quality=90)
    np.save(out / "scene_depth.npy", D)
    p, uv = _points(robot, D, T)
    dets = robot.detect("scene_camera", list(PS.bowl_prompts), float(PS.score_threshold))
    ref = list(cfg.workspace.bowl_centre_m)                # where the operator lays the bowl
    cands = []
    for d in sorted(dets, key=lambda d: -d.score):
        m = bowl_at(d.mask, d.box, uv, p)
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
        raise RuntimeError("no reachable bowl in the scene camera's view (%d candidates: %s; %s)"
                           % (len(cands), ", ".join(m["rejected"] for m in cands) or "none", out / "scene_detection.jpg"))
    best = min(ok, key=lambda m: np.hypot(m["bowl_xy"][0] - ref[0], m["bowl_xy"][1] - ref[1]))
    st = dict(best, t=time.time(), extrinsic=source, record=str(out))
    (state_dir() / "scene_state.json").write_text(json.dumps(st, indent=1))
    log.info("  the bowl at (%.3f, %.3f) (+-%.2f m), rim radius %.3f m, table z %.3f, top z %.3f", *best["bowl_xy"],
             float(SC.error_m), best["rim_radius_m"], best["table_z"], best["bowl_top_z"])
    record.event("located", **st)
    return st


def refine(robot, arm) -> dict:
    """The bowl as the wrist camera sees it from food_detect_start (SAM3 with perception.bowl_prompts, its depth through
    the measured mount, locate's own fit): scene_state.json's bowl centre and radius take its reading. The heights stay
    the scene camera's: from above, the rim is too thin to read its top."""
    from rammp_feeding.scene import camera_pose
    cam, model = robot.cameras["camera"], robot.camera_model("camera")
    since = time.time()
    _, rgb = cam.frame(after=since)
    D = cam.depth_median(int(cfg.perception.depth_frames), since, wait_s=float(cfg.perception.depth_wait_s))
    o, Rc = camera_pose(arm)
    pc = model.depth_points(D)
    uv = np.round(np.stack(model.pixel(pc.T), 1)).astype(int)
    dets = robot.detect("camera", list(cfg.perception.bowl_prompts), float(cfg.perception.bowl_score_threshold))
    fits = [bowl_at(d.mask, d.box, uv, pc @ Rc.T + o) for d in sorted(dets, key=lambda d: -d.score)]
    fit = next((f for f in fits if "bowl_xy" in f), None)
    if fit is None:
        raise RuntimeError("the wrist camera finds no bowl from food_detect_start (%d candidates: %s)"
                           % (len(fits), ", ".join(f["rejected"] for f in fits) or "none"))
    f = state_dir() / "scene_state.json"
    st = json.loads(f.read_text())
    log.info("  the wrist camera: the bowl at (%.3f, %.3f), %.0f mm from the scene camera's, rim radius %.3f m",
             *fit["bowl_xy"], 1000 * float(np.hypot(*np.subtract(fit["bowl_xy"], st["bowl_xy"]))), fit["rim_radius_m"])
    st.update({k: fit[k] for k in ("bowl_xy", "rim_radius_m")}, refined_by="the wrist camera")
    f.write_text(json.dumps(st, indent=1))
    record.event("refined", **st)
    return st


# ------------------------------------------------------------------ calibration
def camera_in_base(B: np.ndarray, C: np.ndarray, link_from_colour: np.ndarray) -> np.ndarray:
    """base <- the camera's link frame, from the board in the base frame (B), the board in its colour optical frame (C),
    and its link <- colour optical frame (its driver's /tf_static)."""
    return B @ np.linalg.inv(C) @ np.linalg.inv(link_from_colour)


def calibrate(robot, arm) -> dict | None:
    """This camera's pose through the board: what it sees of it here, then the wrist camera's route over it (from
    feeding_start and back; the board kept still); written and published."""
    from rammp_feeding import wrist_calibration as wc
    from rammp_feeding.scene import CAM_IN_TOOL, CAM_ROT
    model, cam = robot.camera_model("scene_camera"), robot.cameras["scene_camera"]
    _, rgb = cam.frame(after=time.time())
    Image.fromarray(rgb).save(record.here() / "scene_board.jpg", quality=90)
    C, n = wc.board_pose(model, rgb)
    if C is None:
        log.info("  NOT measured: the scene camera sees %d of the board's corners, %d needed: lay it where it sees it",
                 n, int(wc.WC.min_corners))
        return None
    log.info("  the scene camera: the board, %d corners, %.2f m away", n, float(np.linalg.norm(C[:3, 3])))
    caps = wc.board_views(robot, arm)                   # the wrist camera's route over it: MOVES, back to feeding_start
    if caps is None:
        return None
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = CAM_ROT, CAM_IN_TOOL
    B, spread_m, _ = wc.board_in_base(caps, X)
    T = camera_in_base(B, C, robot.transform(str(SC.link_frame), cam.colour_frame))
    res = dict(parent_frame="base_link", child_frame=str(SC.link_frame), xyz=[round(float(v), 5) for v in T[:3, 3]],
               quat_xyzw=[round(v, 6) for v in matrix_quat(T[:3, :3])], board_spread_m=round(spread_m, 4), corners=n,
               method="the ChArUco board, through the wrist camera", t=time.time())
    (state_dir() / "scene_extrinsic.json").write_text(json.dumps(res, indent=1))
    publish(robot)
    record.event("scene_extrinsic", **res)
    log.info("  -> %s (and on /tf_static: base_link -> %s)", state_dir() / "scene_extrinsic.json", SC.link_frame)
    return res
