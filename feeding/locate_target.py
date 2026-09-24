#!/usr/bin/env python3
"""From one overlook frame: where is each piece of food, and where must the tool go.

    python -m feeding.locate_target --det scene_det.json --depth scene_depth.npy \
        --da3 da3.npy --tool 0.3439 0.0616 0.5641 --out target

Inputs are all from ONE pose (food_detect_start): colour detections, the depth
sensor frame, and Depth Anything 3 metric depth on the colour frame.

1. DEPTH. The sensor is the metric truth. Inside the bowl it resolves three
   surfaces cleanly -- table ~480 mm, rim ~440 mm (40 mm proud), floor ~472 mm
   -- and each berry as a 30 mm bump. Berries are found as blobs nearer than
   the floor INSIDE the rim; without that restriction the rim itself looks like
   food.

2. REGISTRATION. The three berries are visible in both frames, so they are the
   correspondences: each gives the colour principal point directly under the
   factory depth->colour extrinsic. Measured cx 657 +- 9, cy 226 +- 3.5 px
   against factory 621 / 238 -- close, and 1 px is ~0.34 mm at berry depth.

3. DA3 FUSION. DA3's metric output is NOT used as-is. It invents a ~45 mm tilt
   across a table the camera is looking straight down at, and exaggerates local
   relief ~4x. It is fitted to the sensor with scale, shift AND a plane,
       Z = a*DA3 + b + c*(u/W) + d*(v/H)
   which absorbs the tilt and keeps only its local shape. It is reported as a
   cross-check at colour resolution, not as the source of truth.

4. TARGET. With the camera vertical, a vertical line in the world projects
   through the principal point. So the utensil tip's vertical line, seen at
   berry depth Zb, lands at
       p* = pp + (p_tip - pp) * Z_tip / Zb
   The tool must translate until the berry sits at p*. From probes at this
   orientation, +X_base shifts the scene +v and +Y_base shifts it +u, both at
   f/Z, so
       dX = (p*_v - berry_v) * Zb / fy        dY = (p*_u - berry_u) * Zb / fx
   No hand-eye calibration is needed; what remains is Z_tip, the camera-to-tip
   distance along the utensil in the depth sensor's frame: 260 mm for the knife.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FXD = FYD = 334.810; CXD = 238.536; CYD = 139.694          # depth, device
FXC, FYC = 1297.673, 1298.631                              # colour, device
T_D2C = np.array([-0.02706, -0.00997, -0.00471])            # factory extrinsic
FACTORY_PP = (620.914, 238.280)                             # colour, device-reported
MIN_FOOD_MM, MAX_FOOD_MM = 12.0, 60.0                       # food height above the floor


def colour_to_depth(u, v, z_m, cx, cy):
    """Colour pixel at depth z_m (colour-frame z) -> depth-frame pixel."""
    X = (u - cx) * z_m / FXC - T_D2C[0]
    Y = (v - cy) * z_m / FYC - T_D2C[1]
    Z = z_m - T_D2C[2]
    return FXD * X / Z + CXD, FYD * Y / Z + CYD


def to_colour(u, v, z_m, cx, cy):
    X = (u - CXD) * z_m / FXD + T_D2C[0]
    Y = (v - CYD) * z_m / FYD + T_D2C[1]
    Z = z_m + T_D2C[2]
    return FXC * X / Z + cx, FYC * Y / Z + cy, Z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--depth", required=True, help="sensor depth .npy, mm, 480x270")
    ap.add_argument("--da3", default=None,
                    help="DA3 metric .npy (m, colour size). OPTIONAL cross-check only: "
                         "scored against physical checks it overstated berry relief 2-3x "
                         "raw and flattened it to 9-14 mm fused; the sensor got 31-32 mm")
    ap.add_argument("--image", default=None)
    ap.add_argument("--intrinsics", default=str(Path(__file__).resolve().parent.parent
                                                / "calibration" / "colour_intrinsics.json"))
    ap.add_argument("--tool", type=float, nargs=3, required=True, help="current tool xyz, m")
    ap.add_argument("--z-tip", type=float, default=0.265,
                    help="camera -> utensil tip along the axis, m. KNIFE 260 mm, ruler-closed "
                         "the fork; NOT yet measured for the spoon")
    ap.add_argument("--out", default="target")
    a = ap.parse_args()

    det = json.loads(open(a.det).read())
    D = np.load(a.depth).astype(np.float64)
    A = np.load(a.da3).astype(np.float64) if a.da3 else None

    # 1+2. DEPTH PER DETECTION, and a registration check that falls out of it.
    #
    # No blob segmentation. The bowl floor is CONCAVE, rising toward the rim,
    # so no single floor level separates berries from the bowl wall; every
    # threshold tried let a different slice of wall merge into a different
    # berry, and blob centroids wandered by tens of pixels. Instead each colour
    # box is mapped into the depth frame through the frozen principal point and
    # depth is read locally:
    #   top    = nearest surface within ~20 mm of the mapped centre
    #   floor  = far end of a ~45 mm window around it (the bowl just beside it)
    # The offset between where that nearest surface actually is and where the
    # mapping predicted it is the per-run registration residual.
    cal = json.loads(open(a.intrinsics).read())
    cx, cy = cal["cx"], cal["cy"]
    H, W = D.shape
    col = list(det["food"])
    areas = [(f["x2"] - f["x1"]) * (f["y2"] - f["y1"]) for f in col]
    med = float(np.median(areas)) if areas else 0.0
    print("DEPTH (sensor, read at each detection)")
    res = []
    for f, ar in zip(col, areas):
        ud, vd = colour_to_depth(f["cx"], f["cy"], 0.44, cx, cy)
        r_in = max(4, int(round(0.25 * (f["x2"] - f["x1"]) * FXD / FXC)))   # ~ berry half-width
        r_out = int(round(2.2 * r_in))
        u0, u1 = int(max(0, ud - r_out)), int(min(W, ud + r_out + 1))
        v0, v1 = int(max(0, vd - r_out)), int(min(H, vd + r_out + 1))
        win = D[v0:v1, u0:u1]
        yy, xx = np.mgrid[v0:v1, u0:u1]
        rr = np.hypot(xx - ud, yy - vd)
        inner = win[(rr <= r_in) & (win > 0)]
        outer = win[(rr > r_in) & (rr <= r_out) & (win > 0)]
        f["_occluded"] = ar < 0.6 * med
        if inner.size < 5 or outer.size < 5:
            print("  berry at (%4.0f,%4.0f): no depth return here -- skipped" % (f["cx"], f["cy"]))
            continue
        top = float(np.percentile(inner, 5))
        floor_here = float(np.percentile(outer, 85))
        near = (win > 0) & (win <= top + 3) & (rr <= r_out)
        pu, pv = float(xx[near].mean()), float(yy[near].mean())
        f["_depth"] = dict(top=top, floor=floor_here, u=pu, v=pv)
        tag = "  (partly hidden)" if f["_occluded"] else ""
        print("  berry at (%4.0f,%4.0f)  top %.0f mm  floor beside it %.0f mm  height %.0f mm%s"
              % (f["cx"], f["cy"], top, floor_here, floor_here - top, tag))
        if not f["_occluded"]:
            cu_, cv_, _ = to_colour(pu, pv, (top + 8.0) / 1000.0, cx, cy)
            res.append((f["cx"] - cu_, f["cy"] - cv_))
    col = [f for f in col if "_depth" in f]
    floor = float(np.median([f["_depth"]["floor"] for f in col])) if col else float("nan")
    print("\nREGISTRATION CHECK against the frozen principal point (%.1f, %.1f)" % (cx, cy))
    if res:
        r = np.array(res)
        print("  %d unoccluded landmark(s): mean offset (%+.1f, %+.1f) px, spread %.1f px"
              "   (1 px ~ 0.34 mm at berry depth)"
              % (len(r), r[:, 0].mean(), r[:, 1].mean(), float(np.hypot(*r.std(0)))))
        if np.hypot(*r.mean(0)) > 25:
            print("  WARNING: >25 px systematic offset -- camera, utensil or pose may have changed")

    # 3. DA3 fused against the sensor -- optional cross-check
    F = None
    if A is not None:
        vs, us = np.nonzero(D)
        Zs = D[vs, us] / 1000.0
        k = (Zs > 0.30) & (Zs < 0.65)
        uc, vc, Zc = to_colour(us[k], vs[k], Zs[k], cx, cy)
        ui, vi = np.round(uc).astype(int), np.round(vc).astype(int)
        ok = (ui >= 0) & (ui < A.shape[1]) & (vi >= 0) & (vi < A.shape[0])
        da, zt = A[vi[ok], ui[ok]], Zc[ok]
        M = np.stack([da, np.ones_like(da), ui[ok] / A.shape[1], vi[ok] / A.shape[0]], 1)
        p, *_ = np.linalg.lstsq(M, zt, rcond=None)
        r = np.abs(M @ p - zt)
        inl = r < np.percentile(r, 80)
        p, *_ = np.linalg.lstsq(M[inl], zt[inl], rcond=None)
        fit_err = float(np.median(np.abs(M @ p - zt)))
        H, W = A.shape
        yy, xx = np.mgrid[0:H, 0:W]
        F = p[0] * A + p[1] + p[2] * xx / W + p[3] * yy / H
        np.save(a.out + "_fused.npy", F.astype(np.float32))
        print("\nDA3 FUSED to the sensor   Z = %.3f*DA3 %+.3f %+.3f*u/W %+.3f*v/H   median |err| %.1f mm"
              % (p[0], p[1], p[2], p[3], 1000 * fit_err))

    # 4. per-berry 3D and the target
    ute = det["utensil"][0]
    p_tip = np.array([(ute["x1"] + ute["x2"]) / 2.0, ute["y1"]])   # stored region, top-centre
    bowl = det["bowl"][0]
    bc = np.array([bowl["cx"], bowl["cy"]])
    print("\nFOOD")
    rows = []
    for i, f in enumerate(col):
        zb_s = (f["_depth"]["top"] + 8.0) / 1000.0     # centre of the visible cap
        zb_f = (float(np.median(F[int(f["y1"]):int(f["y2"]), int(f["x1"]):int(f["x2"])]))
                if F is not None else float("nan"))
        # camera-frame point of the bbox centre at the sensor depth
        Xc = (f["cx"] - cx) * zb_s / FXC; Yc = (f["cy"] - cy) * zb_s / FYC
        d_rim = float(np.hypot(*(np.array([f["cx"], f["cy"]]) - bc)))
        rows.append((i, f, zb_s, zb_f, Xc, Yc, d_rim))
        print("  berry %d  bbox centre (%4.0f,%4.0f)  depth sensor %.0f mm | DA3-fused %.0f mm  "
              "cam (%+.0f,%+.0f) mm   %.0f px from bowl centre"
              % (i + 1, f["cx"], f["cy"], 1000 * zb_s, 1000 * zb_f, 1000 * Xc, 1000 * Yc, d_rim))

    visible = [r for r in rows if not r[1].get("_occluded")]
    pick = min(visible or rows, key=lambda r: r[6])   # most central visible: most rim clearance
    i, f, zb, _, _, _, _ = pick
    # The tip's vertical line projects through the image of TRUE VERTICAL, not
    # through the optical centre -- measured, see colour_intrinsics.json.
    pp = np.array(cal.get("vertical_vp", [cx, cy]))
    p_star = pp + (p_tip - pp) * (a.z_tip / zb)
    du, dv = p_star[0] - f["cx"], p_star[1] - f["cy"]
    dX, dY = dv * zb / FYC, du * zb / FXC
    tgt = (a.tool[0] + dX, a.tool[1] + dY, a.tool[2])
    # sensitivity to the one unmeasured quantity
    lo = pp + (p_tip - pp) * ((a.z_tip - 0.025) / zb)
    hi = pp + (p_tip - pp) * ((a.z_tip + 0.025) / zb)
    sens = 1000 * np.hypot(*((hi - lo) * zb / FXC)) / 2

    print("\nTARGET  berry %d (most central -> most rim clearance)" % (i + 1))
    print("  utensil tip seen at (%4.0f,%4.0f); its vertical line at berry depth -> p* (%.0f,%.0f)"
          % (p_tip[0], p_tip[1], p_star[0], p_star[1]))
    print("  move tool  dX %+.1f mm  dY %+.1f mm  (z unchanged)" % (1000 * dX, 1000 * dY))
    print("  tool now   (%.4f %.4f %.4f) m" % tuple(a.tool))
    print("  tool goal  (%.4f %.4f %.4f) m" % tgt)
    print("  uncertainty from Z_tip (%.0f mm assumed, +-25 mm): +-%.1f mm" % (1000 * a.z_tip, sens))

    json.dump(dict(principal_point=[cx, cy], bowl_floor_mm=floor,
                   berry=i + 1, berry_bbox_centre=[f["cx"], f["cy"]], berry_depth_m=zb,
                   p_star=list(map(float, p_star)), delta_xy_m=[dX, dY],
                   tool_now=list(a.tool), tool_goal=list(tgt), z_tip_assumed_m=a.z_tip),
              open(a.out + ".json", "w"), indent=2)

    if a.image:
        im = Image.open(a.image).convert("RGB"); d = ImageDraw.Draw(im)
        for r_ in rows:
            ff = r_[1]
            col_ = (34, 197, 94) if r_[0] == i else (239, 68, 68)
            d.rectangle([ff["x1"], ff["y1"], ff["x2"], ff["y2"]], outline=col_, width=4)
            d.text((ff["x1"] + 4, ff["y1"] + 4), "%d  %.0fmm" % (r_[0] + 1, 1000 * r_[2]), fill=col_)
        d.ellipse([f["cx"] - 7, f["cy"] - 7, f["cx"] + 7, f["cy"] + 7], outline=(34, 197, 94), width=3)
        d.ellipse([p_star[0] - 9, p_star[1] - 9, p_star[0] + 9, p_star[1] + 9], outline=(59, 130, 246), width=3)
        d.line([f["cx"], f["cy"], p_star[0], p_star[1]], fill=(59, 130, 246), width=2)
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(234, 179, 8))
        d.text((p_star[0] + 12, p_star[1] - 8), "utensil tip lands here", fill=(59, 130, 246))
        d.text((cx + 8, cy - 8), "principal point", fill=(234, 179, 8))
        im.save(a.out + ".jpg", quality=92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
