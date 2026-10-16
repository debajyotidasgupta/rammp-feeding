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
    """(R, t): base_link <- the camera's link frame, and where it came from: our own calibration
    (make calibrate-scene: <runs_dir>/scene_extrinsic.json) when there is one, else the configured file / values."""
    e = SC.extrinsic
    ours = Path(os.path.expanduser(str(cfg.deployment.runs_dir))) / "scene_extrinsic.json"
    path = str(ours) if ours.exists() else str(e.get("file") or "")
    if path:
        doc = yaml.safe_load(open(os.path.expanduser(path)))
        if doc.get("parent_frame") not in ("base_link", None):
            raise ValueError("%s: parent frame %r, expected base_link" % (path, doc.get("parent_frame")))
        return quat_mat(doc["quat_xyzw"]), np.array(doc["xyz"], float), path
    if e.get("xyz") is None:
        raise ValueError("no scene camera extrinsic (scene_camera.extrinsic.file or .xyz / .quat_xyzw)")
    return quat_mat(e.quat_xyzw), np.array(e.xyz, float), "config"


def fetch(port: int) -> tuple[Image.Image, np.ndarray]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/frame" % port, timeout=5) as r:
            colour = Image.open(io.BytesIO(r.read())).convert("RGB")
        with urllib.request.urlopen("http://127.0.0.1:%d/median?frames=15&wait=5&min=5" % port, timeout=15) as r:
            depth = np.load(io.BytesIO(r.read())).astype(np.float64) / 1000.0        # m, 0 = no return
    except OSError as e:
        raise SystemExit("the scene camera server (:%d) has no frames (%s) -- is the camera's driver running? "
                         "(then restart ours: make camera-server)" % (port, e))
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


def table_height(p: np.ndarray, at_xy: np.ndarray) -> float:
    """The table round the bowl, at the bowl: the dominant flat surface (the most common height, 1 cm bins, not
    whatever else the ring takes in -- an arm, a person, the floor past the table's edge), fitted as a plane and read
    at at_xy. A plane, not the mode: through a slightly rotated extrinsic the table reads sloped, and the ring's side
    nearest the camera has the most pixels."""
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


def measure(bx: list[float], uv: np.ndarray, p_base: np.ndarray) -> dict:
    """One candidate box -> the bowl it holds: centre (a circle through the rim), top, table around it; or why not."""
    res = dict(box_px=[round(v, 1) for v in bx])
    inside = (uv[:, 0] >= bx[0]) & (uv[:, 0] <= bx[2]) & (uv[:, 1] >= bx[1]) & (uv[:, 1] <= bx[3])
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    ring = ((uv[:, 0] >= bx[0] - 0.6 * w) & (uv[:, 0] <= bx[2] + 0.6 * w) & (uv[:, 1] >= bx[1] - 0.3 * h)
            & (uv[:, 1] <= bx[3] + 0.6 * h) & ~inside)
    if inside.sum() < 50 or ring.sum() < 50:
        return dict(res, rejected="too little depth at it (%d points) or around it (%d)" % (inside.sum(), ring.sum()))
    b = p_base[inside]
    table_z = table_height(p_base[ring], np.median(b[:, :2], axis=0))
    max_h, gate = float(cfg.perception.scene.max_bowl_height_m), float(cfg.perception.scene.max_bowl_radius_m)
    blocked = float(np.mean(b[:, 2] > table_z + max_h))
    if blocked > float(cfg.perception.scene.max_blocked_frac):
        return dict(res, rejected="something stands in front of the bowl (%.0f%% of its box is taller than a bowl)"
                    % (100 * blocked))
    above = b[(b[:, 2] > table_z + 0.010) & (b[:, 2] < table_z + max_h)]   # the bowl: not the table through the
    if len(above) >= 30:                                        # box, not an arm or a head above or in front of it
        mid = np.median(above[:, :2], axis=0)
        above = above[np.hypot(*(above[:, :2] - mid).T) < gate]
    if len(above) < 30:
        return dict(res, rejected="nothing bowl-sized standing on the table (z %.3f)" % table_z)
    lo, hi = np.percentile(above[:, :2], 5, axis=0), np.percentile(above[:, :2], 95, axis=0)
    top_z = float(np.percentile(above[:, 2], 95))
    # the centre: a circle through the RIM (the top 15 mm), which a side view sees on both the near and the far
    # edge -- the middle of the points' extent leans toward the camera (only the near wall is visible)
    rim = above[above[:, 2] > top_z - 0.015][:, :2]
    centre, radius = circle_fit(rim) if len(rim) >= 20 else ((lo + hi) / 2, None)
    if radius is None or not 0.03 < radius < gate:
        centre, radius = (lo + hi) / 2, None                   # no believable rim: the extent's middle
    return dict(res, bowl_xy=[float(v) for v in centre], table_z=table_z, bowl_top_z=top_z,
                bowl_extent_m=[round(float(v), 3) for v in hi - lo],
                rim_radius_m=None if radius is None else round(radius, 3), points=int(len(above)))


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def reference(runs: Path) -> list[float]:
    """Where the bowl is expected: the last reading (any age), else rammp.yaml's workspace."""
    try:
        return [float(v) for v in json.loads((runs / "scene_state.json").read_text())["bowl_xy"]]
    except (OSError, ValueError, KeyError):
        return [float(v) for v in cfg.workspace.bowl_centre_m]


def draw(colour: Image.Image, cands: list[dict], best: dict | None, out: Path) -> None:
    im = colour.copy()
    dr = ImageDraw.Draw(im)
    for m in cands:
        chosen = m is best
        c = (34, 197, 94) if chosen else (239, 68, 68) if m.get("rejected") else (148, 163, 184)
        dr.rectangle(m["box_px"], outline=c, width=3 if chosen else 2)
        text = (m.get("rejected") or "(%.3f, %.3f) m" % tuple(m["bowl_xy"])) if not chosen else \
            "BOWL (%.3f, %.3f) m, table z %.3f" % (m["bowl_xy"][0], m["bowl_xy"][1], m["table_z"])
        dr.text((m["box_px"][0], m["box_px"][1] - 14), text, fill=c)
    im.save(out / "scene_detection.jpg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="the directory for the images and the result (default: a new one in runs_dir)")
    ap.add_argument("--prompt", nargs="+", default=list(cfg.get("perception.scene.bowl_prompts", ["bowl"])),
                    help="what to ask for; every prompt's boxes are candidates (overlaps merged)")
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
    # detect on a copy no wider than perception.scene.detect_width (detector time grows with pixels); boxes scaled back
    k = min(1.0, float(cfg.get("perception.scene.detect_width", 848)) / colour.width)
    small = colour.resize((round(colour.width * k), round(colour.height * k)), Image.BILINEAR) if k < 1 else colour
    small.save(out / "scene_colour_detect.jpg")
    small = Image.open(out / "scene_colour_detect.jpg"); small.filename = str(out / "scene_colour_detect.jpg")
    p_link, _ = depth_points(depth)
    uv = project_colour(p_link)
    p_base = p_link @ R.T + t
    # the prompts in order, stopping at the first that yields a reachable bowl: every candidate placed in the base
    # frame; only one the arm can reach, on the table, and of those the nearest to the last reading (else the
    # configured workspace) -- "bowl" also names the food containers around it
    ref = reference(runs)
    boxes, cands = [], []
    for p in a.prompt:
        for b in det.locate(small, p) or []:
            bb = [float(getattr(b, key)) / k for key in ("x1", "y1", "x2", "y2")]
            if any(iou(bb, o) >= 0.5 for o in boxes):
                continue
            boxes.append(bb)
            m = dict(measure(bb, uv, p_base), prompt=p)
            if m.get("bowl_xy") is not None:
                r = float(np.hypot(*m["bowl_xy"]))
                if r > float(cfg.get("perception.scene.max_reach_m", 0.75)):
                    m["rejected"] = "%.2f m from the arm's base: out of reach" % r
                m["from_ref_m"] = float(np.hypot(m["bowl_xy"][0] - ref[0], m["bowl_xy"][1] - ref[1]))
            cands.append(m)
        if any(not m.get("rejected") for m in cands):
            break
    if not cands:
        print("no %s in the scene camera's view -- nothing located (see %s)" % (" / ".join(a.prompt), out))
        return 3
    for m in cands:
        print("  candidate %s (%s): %s" % (m["box_px"], m["prompt"], m.get("rejected") or
                                           "bowl (%.3f, %.3f), %.2f m from the reference"
                                           % (m["bowl_xy"][0], m["bowl_xy"][1], m["from_ref_m"])))
    ok = [m for m in cands if not m.get("rejected")]
    if not ok:
        draw(colour, cands, None, out)
        print("no reachable %s on the table -- nothing located (see %s)" % (" / ".join(a.prompt), out))
        return 3
    best = min(ok, key=lambda m: m["from_ref_m"])
    bowl_xy, table_z, top_z, radius, bx = best["bowl_xy"], best["table_z"], best["bowl_top_z"], best["rim_radius_m"], best["box_px"]
    res = dict(t=time.time(), bowl_xy=[round(float(v), 4) for v in bowl_xy], table_z=round(table_z, 4),
               bowl_top_z=round(top_z, 4), bowl_extent_m=best["bowl_extent_m"], rim_radius_m=radius,
               box_px=bx, points=best["points"], candidates=len(cands), reference_xy=[round(v, 3) for v in ref],
               extrinsic=source, error_m=float(SC.extrinsic.error_m), seconds=round(time.time() - t0, 2),
               record=str(out))
    (out / "scene_state.json").write_text(json.dumps(res, indent=1))
    (runs / "scene_state.json").write_text(json.dumps(res, indent=1))
    draw(colour, cands, best, out)
    print("bowl at (%.3f, %.3f) m in the base frame (+-%.2f m), table z %.3f m, bowl top z %.3f m, rim radius %s m"
          % (bowl_xy[0], bowl_xy[1], res["error_m"], table_z, top_z, res["rim_radius_m"]))
    print("  %d points, %d candidate(s), %.1f s; %s" % (res["points"], len(cands), res["seconds"], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
