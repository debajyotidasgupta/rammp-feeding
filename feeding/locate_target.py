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
   No hand-eye calibration is needed; the unknown that remains is Z_tip, the
   camera-to-tip distance, which was ~105 mm for the fork and has NOT been
   measured for the spoon.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

FXD = FYD = 334.810; CXD = 238.536; CYD = 139.694          # depth, device
FXC, FYC = 1297.673, 1298.631                              # colour, device
T_D2C = np.array([-0.02706, -0.00997, -0.00471])            # factory extrinsic


def berries_in_depth(D, centre=(240, 170), rim_r=64):
    from scipy import ndimage as ndi
    H, W = D.shape
    yy, xx = np.mgrid[0:H, 0:W]
    interior = np.hypot(xx - centre[0], yy - centre[1]) < rim_r - 12
    floor = float(np.median(D[interior & (D > 465) & (D < 485)]))
    mask = ndi.binary_opening(interior & (D > 0) & (D < floor - 15), iterations=1)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    out = []
    for i in np.argsort(-sizes)[:3]:
        ys, xs = np.nonzero(lab == i + 1)
        z = D[ys, xs]
        w = floor - z
        out.append(dict(u=float(np.average(xs, weights=w)), v=float(np.average(ys, weights=w)),
                        top=float(np.percentile(z, 10)), mean=float(z.mean()), px=int(len(xs))))
    return floor, out


def to_colour(u, v, z_m, cx, cy):
    X = (u - CXD) * z_m / FXD + T_D2C[0]
    Y = (v - CYD) * z_m / FYD + T_D2C[1]
    Z = z_m + T_D2C[2]
    return FXC * X / Z + cx, FYC * Y / Z + cy, Z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--depth", required=True, help="sensor depth .npy, mm, 480x270")
    ap.add_argument("--da3", required=True, help="DA3 metric .npy, m, colour size")
    ap.add_argument("--image", default=None)
    ap.add_argument("--tool", type=float, nargs=3, required=True, help="current tool xyz, m")
    ap.add_argument("--z-tip", type=float, default=0.105,
                    help="camera -> utensil tip along the axis, m. MEASURED 105 mm for "
                         "the fork; NOT yet measured for the spoon")
    ap.add_argument("--out", default="target")
    a = ap.parse_args()

    det = json.loads(open(a.det).read())
    D = np.load(a.depth).astype(np.float64)
    A = np.load(a.da3).astype(np.float64)

    # 1. depth
    floor, bd = berries_in_depth(D)
    print("DEPTH (sensor)")
    print("  bowl floor %.0f mm" % floor)
    for b in bd:
        print("  berry  depth-px (%5.1f,%5.1f)  top %.0f mm  height %.0f mm" % (b["u"], b["v"], b["top"], floor - b["top"]))

    # 2. registration from berry correspondences (match by order along u)
    col = sorted(det["food"], key=lambda f: f["cx"])
    bd = sorted(bd, key=lambda b: b["u"])
    cxs, cys = [], []
    for f, b in zip(col, bd):
        z = (b["top"] + 8.0) / 1000.0
        u0, v0, _ = to_colour(b["u"], b["v"], z, 0.0, 0.0)
        cxs.append(f["cx"] - u0); cys.append(f["cy"] - v0)
        f["_depth"] = b
    cx, cy = float(np.mean(cxs)), float(np.mean(cys))
    print("\nREGISTRATION (3 berry correspondences)")
    print("  colour principal point  cx %.1f +- %.1f   cy %.1f +- %.1f px  (factory 620.9 / 238.3)"
          % (cx, np.std(cxs), cy, np.std(cys)))

    # 3. DA3 fused against the sensor: scale + shift + plane
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
    ute = det["utensil"][0] if det.get("utensil") else None
    p_tip = np.array([(ute["x1"] + ute["x2"]) / 2.0, ute["y1"]]) if ute else None
    bowl = det["bowl"][0]
    bc = np.array([bowl["cx"], bowl["cy"]])
    print("\nFOOD")
    rows = []
    for i, f in enumerate(col):
        zb_s = (f["_depth"]["top"] + 8.0) / 1000.0
        zb_f = float(np.median(F[int(f["y1"]):int(f["y2"]), int(f["x1"]):int(f["x2"])]))
        # camera-frame point of the bbox centre at the sensor depth
        Xc = (f["cx"] - cx) * zb_s / FXC; Yc = (f["cy"] - cy) * zb_s / FYC
        d_rim = float(np.hypot(*(np.array([f["cx"], f["cy"]]) - bc)))
        rows.append((i, f, zb_s, zb_f, Xc, Yc, d_rim))
        print("  berry %d  bbox centre (%4.0f,%4.0f)  depth sensor %.0f mm | DA3-fused %.0f mm  "
              "cam (%+.0f,%+.0f) mm   %.0f px from bowl centre"
              % (i + 1, f["cx"], f["cy"], 1000 * zb_s, 1000 * zb_f, 1000 * Xc, 1000 * Yc, d_rim))

    pick = min(rows, key=lambda r: r[6])          # most central: most clearance from the rim
    i, f, zb, _, _, _, _ = pick
    pp = np.array([cx, cy])
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

    json.dump(dict(principal_point=[cx, cy], da3_fit=list(map(float, p)), bowl_floor_mm=floor,
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
