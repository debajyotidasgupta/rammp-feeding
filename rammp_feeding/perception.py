"""The wrist camera: at food_detect_start, which piece to pick and where the tool goes; from any hover, what is straight
under the utensil tip. Everything is placed in the base frame: each depth pixel through the camera's model (its
driver's camera_info and /tf_static) and the camera's pose from the arm.

perceive: one colour frame and a depth median from frames after the arm stopped; SAM3, one call, for the bowl and the
food (boxes and masks); the food filtered (slivers, the utensil's own pixels, outside or too big for the bowl,
overlaps); the piece nearest the bowl's centre (the most rim clearance), aimed at the centroid of its points; the
tool's goal the level slide that puts the utensil tip right above it. Refused without the bowl in view. Exit 3: the
bowl is empty; 5: food is left but no piece is pickable.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rammp_feeding import record, scene
from rammp_feeding.config import cfg
from rammp_feeding.ros import Detection

log = logging.getLogger("rammp")
P = cfg.perception


class NoTarget(Exception):
    def __init__(self, why: str, rc: int):
        super().__init__(why)
        self.rc = rc


def world_points(cam, arm, D: np.ndarray, q=None):
    """Every valid depth pixel in the base frame: (points N x 3, their colour pixels N x 2 (int, may fall outside), their
    depth from the camera N)."""
    o, Rc = scene.camera_pose(arm, q)
    p = cam.depth_points(D)
    return o + p @ Rc.T, np.round(np.stack(cam.pixel(p.T), 1)).astype(int), p[:, 2]


def _in(mask: np.ndarray, uv: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    out = np.zeros(len(uv), bool)
    out[ok] = mask[uv[ok, 1], uv[ok, 0]]
    return out


def food(rgb: np.ndarray, dets, bowl: np.ndarray) -> list[Detection]:
    """The food detections worth aiming at, best first."""
    ute = np.zeros(rgb.shape[:2], bool)
    x1, y1, x2, y2 = cfg.utensil.image_region_px
    ute[y1:y2, x1:x2] = True
    out, masks = [], []
    for d in sorted(dets, key=lambda d: -d.score):
        w, h = d.box[2] - d.box[0], d.box[3] - d.box[1]
        if min(w, h) < float(P.min_box_side_px) or max(w, h) / max(1.0, min(w, h)) > float(P.max_box_aspect):
            continue
        m = d.mask
        n = max(1, int(m.sum()))
        if (m & ute).sum() / n >= float(P.inside_utensil_max):
            continue
        if (m & bowl).sum() / n <= float(P.food_inside_bowl_min) or n >= float(P.food_max_frac_of_bowl) * bowl.sum():
            continue
        if any((m & o).sum() / max(1, (m | o).sum()) > float(P.overlap_iou) for o in masks):
            continue
        out.append(d)
        masks.append(m)
    return out


def piece_geometry(W: np.ndarray, uv: np.ndarray, mask: np.ndarray) -> dict | None:
    """A piece from its mask: where to aim (the centroid of its points) and its top (their highest), or None with too
    little depth on it."""
    inside = _in(mask, uv)
    if inside.sum() < 20:
        return None
    pts = W[inside]
    return dict(xy=[float(v) for v in pts[:, :2].mean(0)], top_z=float(np.percentile(pts[:, 2], 95)))


def food_left(W: np.ndarray, uv: np.ndarray, bowl: np.ndarray) -> float:
    """The fraction of the bowl's area standing more than food_left_above_floor_mm above its own floor (its lowest
    surface)."""
    z = W[_in(bowl, uv), 2]
    if len(z) < 50:
        raise ValueError("too little depth inside the bowl to tell whether food is left")
    return float((z > np.percentile(z, 10) + float(P.food_left_above_floor_mm) / 1000.0).mean())


def perceive(robot, arm, out: Path) -> dict:
    """At food_detect_start: the target -> out/target.json (tool_goal, the piece in the base frame); NoTarget else."""
    from rammp_feeding.arm import MoveError
    from rammp_feeding.geometry import max_step
    q = arm.q()
    off = max_step(q, scene.overlook_joints(arm))
    if off > float(cfg.motion.near_pose_deg):
        raise MoveError("not at food_detect_start (%.1f deg away) -- rammp detect" % off)
    since = time.time()
    cam = robot.cameras["camera"]
    _, rgb = cam.frame(after=since)
    D = cam.depth_median(int(P.depth_frames), since, wait_s=float(P.depth_wait_s))
    Image.fromarray(rgb).save(out / "colour.jpg", quality=95)
    np.save(out / "depth.npy", D)
    t0 = time.time()
    dets = robot.detect("camera", list(P.bowl_prompts) + [str(cfg.food.prompt)], float(P.bowl_score_threshold))
    bowls = [d for d in dets if d.label in P.bowl_prompts]       # the bowl at its own threshold, the food at its
    if not bowls:
        _draw(rgb, None, [], out / "detection.jpg")
        raise MoveError("SAM3 finds no bowl in the wrist camera's view (%s)" % (out / "detection.jpg"))
    from scipy.ndimage import binary_fill_holes                  # its mask stops at the food on it: the food is inside
    bowl = binary_fill_holes(max(bowls, key=lambda d: d.score).mask)
    pieces = food(rgb, [d for d in dets if d.label == str(cfg.food.prompt) and d.score >= float(P.score_threshold)], bowl)
    log.info("  SAM3 %.2f s: the bowl, %d piece(s) of %d detections", time.time() - t0, len(pieces), len(dets))
    W, uv, _ = world_points(arm.cam, arm, D, q)
    geo = [(d, piece_geometry(W, uv, d.mask)) for d in pieces]
    geo = [(d, g) for d, g in geo if g is not None]
    _draw(rgb, bowl, pieces, out / "detection.jpg")
    if not geo:
        left = food_left(W, uv, bowl)
        if left > float(P.food_left_min_frac):
            (out / "food_left").write_text("food left, no pickable piece detected\n")
            raise NoTarget("food is left in the bowl (%.0f%% stands above its floor) but no piece is pickable" % (100 * left), 5)
        (out / "no_food").write_text("no food detected\n")
        raise NoTarget("no food in the bowl", 3)
    ws = scene.workspace()
    d, g = min(geo, key=lambda dg: math.hypot(dg[1]["xy"][0] - ws["bowl_xy"][0], dg[1]["xy"][1] - ws["bowl_xy"][1]))
    T, _ = arm.tool(q)
    tip = arm.tip(q)
    goal = [float(T[0] + g["xy"][0] - tip[0]), float(T[1] + g["xy"][1] - tip[1]), float(T[2])]
    tgt = dict(piece=dict(g, box=d.box, score=d.score), tool_now=[float(v) for v in T], tool_goal=goal, t=time.time())
    (out / "target.json").write_text(json.dumps(tgt, indent=1))
    log.info("  target: the piece at (%.3f, %.3f), its top z %.3f; the tool slides %+.1f, %+.1f mm", *g["xy"], g["top_z"],
             1000 * (goal[0] - T[0]), 1000 * (goal[1] - T[1]))
    record.event("perceived", target=tgt, pieces=len(geo))
    return tgt


def _draw(rgb, bowl: np.ndarray | None, pieces, path: Path) -> None:
    im = rgb.copy()
    if bowl is not None:
        im[bowl] = (0.6 * im[bowl] + 0.4 * np.array([234, 179, 8])).astype(np.uint8)
    pil = Image.fromarray(im)
    dr = ImageDraw.Draw(pil)
    for d in pieces:
        dr.rectangle(d.box, outline=(34, 197, 94), width=3)
    dr.rectangle(list(cfg.utensil.image_region_px), outline=(148, 163, 184), width=1)
    pil.save(path, quality=90)


# ------------------------------------------------------------------ under the tip
def under_tip(D: np.ndarray, arm) -> dict:
    """Straight under the utensil tip (the arm's pose now): the highest surface on its vertical line (the food's top)
    and the lowest round it (the floor) -- gap_top / gap_floor: how far straight down from the tip. The utensil itself
    (depth no farther than the tip's, and a border of pixels round it) is dropped first."""
    tip = arm.tip()
    o, Rc = scene.camera_pose(arm)
    zt = float(Rc[:, 2] @ (tip - o))                    # the tip's own depth from the camera
    D = D.astype(float)
    ute = (D > 0) & (D / 1000.0 <= zt + float(P.utensil_depth_margin_m))
    n = int(P.utensil_depth_border_px)
    grown = ute.copy()
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            grown |= np.roll(np.roll(ute, dy, 0), dx, 1)
    W, _, _ = world_points(arm.cam, arm, np.where(grown, 0, D).astype(np.uint16))
    W = W[W[:, 2] < tip[2]]
    d = np.hypot(W[:, 0] - tip[0], W[:, 1] - tip[1])
    line = W[d < float(P.tip_line_radius_m), 2]
    if len(line) < 10:
        raise ValueError("no depth return straight under the tip")
    top = float(np.percentile(line, 90))
    half = float(P.food_half_width_m)
    ring = W[(d > 1.3 * half) & (d < 2.6 * half), 2]
    out = dict(top_z=top, gap_top=float(tip[2] - top), tip=[float(v) for v in tip])
    if len(ring) > 10:
        f = float(np.percentile(ring, 15))
        out.update(floor_z=f, gap_floor=float(tip[2] - f), height=top - f)
    return out
