#!/usr/bin/env python3
"""Fill the depth sensor's holes from Depth Anything 3 -- the fallback when depth is missing.

    python -m feeding.depth_fill RUN_DIR [--colour colour.jpg] [--depth depth.npy]

Reads the sensor frame (depth.npy, mm, 0 = no return; may be missing: an RGB-only camera) and the colour
frame, asks the models server for DA3 metric depth (resident, ~0.1 s), projects it into the SENSOR's frame,
and fits  sensor = s * da3 + b  on the pixels the sensor did return. Only the holes are filled; every sensor
pixel is kept as measured. Writes, beside the inputs:
    depth.npy               filled (the original is kept as depth_sensor.npy)
    depth_filled_mask.npy   True where a pixel came from DA3
    depth_fill.json         source: "sensor" | "filled" | "da3-raw", fit, fraction filled

Why fit, not trust: DA3's own metric scale made berries 2-3x too tall and invented a 39 mm table tilt
(2026-09-23). With no sensor pixels at all the fill is RAW DA3 -- good enough to find and rank food and a
face, NOT to stop a plunge at the bowl floor: acquire refuses a plunge measured on raw DA3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from feeding.measure_under import CXD, CYD, FXD, FYD, T_D2C

CAL = Path(__file__).resolve().parent.parent / "calibration"
SHAPE = (270, 480)               # the depth sensor's frame
MIN_FIT_PX = 500


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--colour", default="colour.jpg")
    ap.add_argument("--depth", default="depth.npy")
    a = ap.parse_args()
    D_dir = Path(a.run_dir); cpath = D_dir / a.colour; dpath = D_dir / a.depth
    stem = dpath.stem
    D = np.load(dpath).astype(np.float64) if dpath.exists() else np.zeros(SHAPE)
    D[~np.isfinite(D)] = 0
    holes = D <= 0
    info = dict(source="sensor", filled_frac=0.0)
    if not holes.any():
        (D_dir / (stem + "_fill.json")).write_text(json.dumps(info)); print("depth: sensor complete, nothing to fill")
        return 0
    from feeding.detector_server import da3_depth
    tmp = D_dir / (stem + "_da3.npy")
    if not cpath.exists() or not da3_depth(str(cpath), str(tmp)):
        info["source"] = "sensor (holes left: Depth Anything 3 not running)"
        (D_dir / (stem + "_fill.json")).write_text(json.dumps(info))
        print("depth: %.0f%% no return, NOT filled -- the models server is not up (make models)" % (100 * holes.mean()))
        return 0
    A = np.load(tmp).astype(np.float64)                     # colour-frame z, metres
    c = json.loads((CAL / "colour_intrinsics.json").read_text())
    cx, cy, fx, fy = c["cx"], c["cy"], c["fx"], c["fy"]
    Hc, Wc = A.shape
    vd, ud = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(np.float64)
    zd = np.full(SHAPE, float(np.median(A)) - T_D2C[2])     # depth-frame z, iterated to consistency
    ok = np.ones(SHAPE, bool)
    for _ in range(4):
        X = (ud - CXD) * zd / FXD + T_D2C[0]; Y = (vd - CYD) * zd / FYD + T_D2C[1]; Zc = zd + T_D2C[2]
        uc = np.round(fx * X / Zc + cx).astype(int); vc = np.round(fy * Y / Zc + cy).astype(int)
        ok = (uc >= 0) & (uc < Wc) & (vc >= 0) & (vc < Hc)
        zd = np.where(ok, A[np.clip(vc, 0, Hc - 1), np.clip(uc, 0, Wc - 1)] - T_D2C[2], zd)
    da3_mm = 1000.0 * zd
    fit = (~holes) & ok
    if fit.sum() >= MIN_FIT_PX:
        x, y = da3_mm[fit], D[fit]
        s, b = 1.0, float(np.median(y - x))
        for _ in range(3):                                  # robust: refit on the inliers
            r = y - (s * x + b); keep = np.abs(r - np.median(r)) < 3 * 1.4826 * np.median(np.abs(r - np.median(r))) + 1
            s, b = np.polyfit(x[keep], y[keep], 1)
        resid = float(np.median(np.abs(y - (s * x + b))))
        fill = s * da3_mm + b
        # one global fit cannot follow DA3's LOCAL distortions (a 60x60 px hole came out 14 mm off, 52 mm worst).
        # Correct each hole by the fit's own error on the sensor pixels around it: spread the residual field
        # into the holes (normalised Gaussian), so DA3 supplies the shape and the sensor the level.
        r = np.zeros(SHAPE); w = np.zeros(SHAPE)
        r[fit] = (D - fill)[fit]; w[fit] = 1.0
        from scipy.ndimage import gaussian_filter
        for sig in (6.0, 20.0, 60.0):                       # near first; wider only where nothing is near
            rs, ws = gaussian_filter(r, sig), gaussian_filter(w, sig)
            corr = np.where(ws > 0.02, rs / np.maximum(ws, 1e-9), np.nan)
            if sig == 6.0:
                local = corr
            else:
                local = np.where(np.isnan(local), corr, local)
        fill = fill + np.nan_to_num(local)
        info.update(source="filled", fit=dict(scale=float(s), offset_mm=float(b), median_resid_mm=resid,
                                              n=int(fit.sum()), local_correction=True))
    else:
        fill = da3_mm; info["source"] = "da3-raw"
    mask = holes & ok
    out = D.copy(); out[mask] = fill[mask]
    info["filled_frac"] = float(mask.mean())
    if dpath.exists():
        np.save(D_dir / (stem + "_sensor.npy"), np.load(dpath))
    np.save(dpath, out.astype(np.float32)); np.save(D_dir / (stem + "_filled_mask.npy"), mask)
    (D_dir / (stem + "_fill.json")).write_text(json.dumps(info))
    print("depth: %.0f%% filled from Depth Anything 3 (%s%s)" % (
        100 * info["filled_frac"], info["source"],
        "; fit x%.3f %+.0f mm, median residual %.1f mm" % (info["fit"]["scale"], info["fit"]["offset_mm"],
                                                          info["fit"]["median_resid_mm"]) if "fit" in info else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
