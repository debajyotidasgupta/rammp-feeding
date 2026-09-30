#!/usr/bin/env python3
"""The COARSE bowl and table position, from the fixed scene camera (profiles.scene_camera). No arm motion.

    ~/venvs/la/bin/python -m feeding.locate_scene [--out DIR]      # make locate-scene

Detects the bowl in the scene camera's colour frame, turns its depth into the arm's base frame (the camera's
extrinsic: scene_camera.extrinsic, e.g. another stack's calibration file read in place; the device's own
depth -> colour transform from its profile), and writes

    <runs_dir>/scene_state.json   {"bowl_xy": [x, y], "table_z": z, "bowl_top_z": z, ...}   (base frame, m)

which kinova/scene.py uses to place food_detect_start over the bowl at a set height. Coarse by design
(scene_camera.extrinsic.error_m, ~5 cm): the wrist camera, from the overlook, finds the food itself.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from settings import cfg

SC = cfg.scene_camera


def quat_mat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# optical frame (x right, y down, z forward) -> the device's link-style frame (x forward, y left, z up)
OPT_TO_LINK = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def extrinsic() -> tuple[np.ndarray, np.ndarray, str]:
    """(R, t): base_link <- the camera's link frame, and where it came from."""
    e = SC.extrinsic
    path = str(e.get("file") or "")
    if path:
        doc = yaml.safe_load(open(os.path.expanduser(path)))
        if doc.get("parent_frame") not in ("base_link", None):
            raise ValueError("%s: parent frame %r, expected base_link" % (path, doc.get("parent_frame")))
        return quat_mat(doc["quat_xyzw"]), np.array(doc["xyz"], float), path
    if e.get("xyz") is None:
        raise ValueError("no scene camera extrinsic (scene_camera.extrinsic.file or .xyz / .quat_xyzw)")
    return quat_mat(e.quat_xyzw), np.array(e.xyz, float), "config"


def fetch(port: int) -> tuple[Image.Image, np.ndarray]:
    with urllib.request.urlopen("http://127.0.0.1:%d/frame" % port, timeout=5) as r:
        colour = Image.open(io.BytesIO(r.read())).convert("RGB")
    with urllib.request.urlopen("http://127.0.0.1:%d/median?frames=15&wait=5&min=5" % port, timeout=15) as r:
        depth = np.load(io.BytesIO(r.read())).astype(np.float64) / 1000.0            # m, 0 = no return
    return colour, depth


def depth_points(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every valid depth pixel as a 3D point in the depth LINK frame, and its (row, col)."""
    d = SC.depth
    v, u = np.nonzero(depth > 0)
    z = depth[v, u] * float(d.scale) + float(d.offset_m)
    opt = np.stack([(u - float(d.cx)) * z / float(d.fx), (v - float(d.cy)) * z / float(d.fy), z], axis=1)
    link = opt @ OPT_TO_LINK.T
    Rl, tl = quat_mat(SC.link_to_depth_frame.quat_xyzw), np.array(SC.link_to_depth_frame.xyz, float)
    return link @ Rl.T + tl, np.stack([v, u], axis=1)           # depth frame expressed in the camera link frame


def project_colour(p_link: np.ndarray) -> np.ndarray:
    """Points in the camera link frame -> colour pixels (u, v); NaN behind the camera."""
    Rd, td = quat_mat(SC.link_to_depth_frame.quat_xyzw), np.array(SC.link_to_depth_frame.xyz, float)
    Rc, tc = quat_mat(SC.depth_frame_to_colour_frame.quat_xyzw), np.array(SC.depth_frame_to_colour_frame.xyz, float)
    p_d = (p_link - td) @ Rd                                    # link -> depth frame
    p_c = (p_d - tc) @ Rc                                       # depth frame -> colour frame
    opt = p_c @ OPT_TO_LINK                                     # colour frame -> colour optical
    c = SC.colour
    with np.errstate(divide="ignore", invalid="ignore"):
        u = float(c.fx) * opt[:, 0] / opt[:, 2] + float(c.cx)
        v = float(c.fy) * opt[:, 1] / opt[:, 2] + float(c.cy)
    bad = opt[:, 2] <= 0
    u[bad] = v[bad] = np.nan
    return np.stack([u, v], axis=1)


def circle_fit(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle (Kasa) through 2D points: (centre, radius)."""
    A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([cx, cy]), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="the directory for the images and the result (default: a new one in runs_dir)")
    ap.add_argument("--prompt", default=str(cfg.get("perception.scene.bowl_prompt", "bowl")))
    a = ap.parse_args()
    if not SC:
        print("no scene camera configured (profiles.scene_camera)")
        return 2
    runs = Path(str(cfg.deployment.runs_dir)).expanduser()
    out = Path(a.out) if a.out else runs / (time.strftime("%Y%m%d_%H%M%S") + "_scene")
    out.mkdir(parents=True, exist_ok=True)
    R, t, source = extrinsic()
    t0 = time.time()
    colour, depth = fetch(int(cfg.deployment.servers.scene_camera_port))
    colour.save(out / "scene_colour.jpg")
    np.save(out / "scene_depth.npy", (depth * 1000).astype(np.uint16))
    from feeding.detector_server import connect
    det = connect()
    colour.filename = str(out / "scene_colour.jpg")
    box = det.locate_one(colour, a.prompt)
    if box is None:
        print("no %r in the scene camera's view -- nothing located (see %s)" % (a.prompt, out))
        return 3
    bx = [float(getattr(box, k)) for k in ("x1", "y1", "x2", "y2")]
    p_link, _ = depth_points(depth)
    uv = project_colour(p_link)
    p_base = p_link @ R.T + t
    inside = (uv[:, 0] >= bx[0]) & (uv[:, 0] <= bx[2]) & (uv[:, 1] >= bx[1]) & (uv[:, 1] <= bx[3])
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    ring = ((uv[:, 0] >= bx[0] - 0.6 * w) & (uv[:, 0] <= bx[2] + 0.6 * w) & (uv[:, 1] >= bx[1] - 0.3 * h)
            & (uv[:, 1] <= bx[3] + 0.6 * h) & ~inside)
    if inside.sum() < 50 or ring.sum() < 50:
        print("too little depth at the bowl (%d points) or around it (%d)" % (inside.sum(), ring.sum()))
        return 3
    table_z = float(np.median(p_base[ring, 2]))
    b = p_base[inside]
    above = b[b[:, 2] > table_z + 0.010]                        # the bowl, not the table seen through the box
    if len(above) < 30:
        print("the box holds nothing standing above the table (table z %.3f)" % table_z)
        return 3
    lo, hi = np.percentile(above[:, :2], 5, axis=0), np.percentile(above[:, :2], 95, axis=0)
    top_z = float(np.percentile(above[:, 2], 95))
    # the centre: a circle through the RIM (the top 15 mm), which a side view sees on both the near and the far
    # edge -- the middle of the points' extent leans toward the camera (only the near wall is visible)
    rim = above[above[:, 2] > top_z - 0.015][:, :2]
    bowl_xy, radius = circle_fit(rim) if len(rim) >= 20 else ((lo + hi) / 2, None)
    if radius is None or not 0.03 < radius < 0.15:
        bowl_xy, radius = (lo + hi) / 2, None                  # no believable rim: the extent's middle
    res = dict(t=time.time(), bowl_xy=[round(float(v), 4) for v in bowl_xy], table_z=round(table_z, 4),
               bowl_top_z=round(top_z, 4), bowl_extent_m=[round(float(v), 3) for v in hi - lo],
               rim_radius_m=None if radius is None else round(radius, 3),
               box_px=[round(v, 1) for v in bx], points=int(len(above)), extrinsic=source,
               error_m=float(SC.extrinsic.error_m), seconds=round(time.time() - t0, 2), record=str(out))
    (out / "scene_state.json").write_text(json.dumps(res, indent=1))
    (runs / "scene_state.json").write_text(json.dumps(res, indent=1))
    im = colour.copy()
    dr = ImageDraw.Draw(im)
    dr.rectangle(bx, outline=(34, 197, 94), width=3)
    dr.text((bx[0], bx[1] - 16), "bowl: (%.3f, %.3f) m, table z %.3f m" % (bowl_xy[0], bowl_xy[1], table_z),
            fill=(34, 197, 94))
    im.save(out / "scene_detection.jpg")
    print("bowl at (%.3f, %.3f) m in the base frame (+-%.2f m), table z %.3f m, bowl top z %.3f m, rim radius %s m"
          % (bowl_xy[0], bowl_xy[1], res["error_m"], table_z, top_z, res["rim_radius_m"]))
    print("  %d points, %.1f s; %s" % (len(above), res["seconds"], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
