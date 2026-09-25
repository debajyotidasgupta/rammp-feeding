#!/usr/bin/env python3
"""What is directly under the utensil tip, from one depth frame.

    python -m feeding.measure_under --depth frame.npy --out under.json [--guess 0.30]

Reports, along the utensil tip's vertical line:
  top    nearest surface (the food's top, when over food), m from the camera
  floor  the bowl floor just beside it, m from the camera
  plus the tip's clearance to each, using the calibrated camera-to-tip.

Read at close range on purpose. Every approach today re-measured at a ~30 mm
hover and found the food 4.7-6.7 mm closer than the overlook reading predicted;
a reading taken from where the arm now is beats one taken 150 mm further away.

The vertical line is projected through the image of TRUE vertical
(camera profile: colour.vertical_vp) and the MEASURED tip pixel
(utensil profile: tip_px) -- not the utensil box's centre, which
was ~10 mm off for a fork.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from settings import cfg
from feeding.camera import CXC, CXD, CYC, CYD, FXC, FXD, FYC, FYD, T_D2C, VP, corrected

FOOD_HALF_M = float(cfg.perception.food_half_width_m)        # the floor ring beside the food


def measure(depth_mm: np.ndarray, guess: float = 0.30, filled: np.ndarray | None = None,
            target: tuple | None = None) -> dict:
    cx, cy, fx, fy = CXC, CYC, FXC, FYC
    vp = VP
    tip = np.array(cfg.utensil.tip_px, dtype=float)
    zt = float(cfg.utensil.camera_to_tip_m)
    if not zt:
        raise SystemExit("camera_to_tip_m is not set for this utensil -- measure it first")
    D = corrected(depth_mm)
    H, W = D.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # target = (u, v, z): read along the vertical line through that scene point instead of the tip's -- i.e.
    # from the overlook frame, what will be under the tip once OVER has put it there (a pure horizontal slide)
    z = target[2] if target else guess
    for _ in range(8):
        p = (vp + (np.array(target[:2]) - vp) * target[2] / z) if target else (vp + (tip - vp) * zt / z)
        X = (p[0] - cx) * z / fx - T_D2C[0]
        Y = (p[1] - cy) * z / fy - T_D2C[1]
        Zd = z - T_D2C[2]
        ud, vd = FXD * X / Zd + CXD, FYD * Y / Zd + CYD
        rr = np.hypot(xx - ud, yy - vd)
        inner = D[(rr <= 7) & (D > 0)]
        if inner.size < 10:
            raise SystemExit("no depth return under the tip (depth px %.0f,%.0f)" % (ud, vd))
        z = np.percentile(inner, 10) / 1000.0 + T_D2C[2]
    top = z
    r_b = FOOD_HALF_M * FXD / top
    ring = D[(rr > 1.3 * r_b) & (rr < 2.6 * r_b) & (D > 0)]
    floor = np.percentile(ring, 85) / 1000.0 + T_D2C[2] if ring.size > 10 else None
    out = dict(camera_to_tip=zt, top=float(top), gap_top=float(top - zt),
               depth_px=[float(ud), float(vd)])
    if filled is not None:                   # how much of each reading came from the DA3 fallback
        m_in = (rr <= 7) & (D > 0); m_ring = (rr > 1.3 * r_b) & (rr < 2.6 * r_b) & (D > 0)
        out["filled_top"] = float(filled[m_in].mean()) if m_in.any() else 0.0
        out["filled_floor"] = float(filled[m_ring].mean()) if m_ring.any() else 0.0
    if floor is not None:
        out.update(floor=float(floor), gap_floor=float(floor - zt), height=float(floor - top))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True, help="depth_probe .npy (mm, 480x270)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--guess", type=float, default=0.30, help="rough depth, m, to start from")
    ap.add_argument("--target", help="perceive's target.json: measure what OVER will put under the tip, from this frame")
    a = ap.parse_args()
    dp = Path(a.depth)
    mask_p = dp.with_name(dp.stem + "_filled_mask.npy"); fill_p = dp.with_name(dp.stem + "_fill.json")
    tgt = None
    if a.target:
        t = json.loads(Path(a.target).read_text())
        u, v = t["berry_bbox_centre"]; zb = t["berry_depth_m"]; dx, dy = cfg.food.aim_offset_base_m
        # the aim offset moves the tool by (dx, dy): the point under the tip is that far from the berry centre.
        # At the overlook orientation a scene point further +X appears higher (-v), further +Y further left (-u).
        tgt = (u - dy * FXC / zb, v - dx * FYC / zb, zb)
    m = measure(np.load(a.depth), a.guess, np.load(mask_p) if mask_p.exists() else None, target=tgt)
    if fill_p.exists():
        m["depth_source"] = json.loads(fill_p.read_text()).get("source", "sensor")
    Path(a.out).write_text(json.dumps(m, indent=2))
    from feeding.viz import depth_png                  # the debug image: where it read the top and the floor
    ud, vd = m["depth_px"]
    r_b = FOOD_HALF_M * FXD / m["top"]
    depth_png(np.load(a.depth), Path(a.out).with_suffix(".png"), filled=np.load(mask_p) if mask_p.exists() else None,
              circles=[(ud, vd, 7, "white"), (ud, vd, 1.3 * r_b, "yellow"), (ud, vd, 2.6 * r_b, "yellow")],
              text="top %.0f mm, floor %s" % (1000 * m["top"], "%.0f mm" % (1000 * m["floor"]) if "floor" in m else "none"))
    s = "  under the tip: top %.1f mm (tip %.1f mm above it)" % (1000 * m["top"], 1000 * m["gap_top"])
    if m.get("filled_floor") or m.get("filled_top"):
        s += " [%.0f%% of the top, %.0f%% of the floor from Depth Anything 3: %s]" % (
            100 * m.get("filled_top", 0), 100 * m.get("filled_floor", 0), m.get("depth_source", "?"))
    if "floor" in m:
        s += ", floor %.1f mm (tip %.1f mm above it), food %.1f mm tall" % (
            1000 * m["floor"], 1000 * m["gap_floor"], 1000 * m["height"])
    print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
