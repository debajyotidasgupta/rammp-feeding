#!/usr/bin/env python3
"""Run Depth Anything 3 (metric) on one image and save metric depth at colour size.

    python -m feeding.run_depth_anything3 --image scene.jpg --out /tmp/da3

Model: depth-anything/DA3METRIC-LARGE (0.35B, Apache 2.0), the monocular
METRIC model. Its raw output converts to metres as

    depth_m = focal_px * net_output / 300

where focal_px is the focal length AT THE PROCESSED RESOLUTION, not the
camera's native one. DA3 resizes the long side to `process_res` before the
network sees it, so fx must be scaled by the same factor or the whole map is
off by that ratio. Upsampled back to the colour size afterwards so it lines up
pixel-for-pixel with the detector's boxes.

process_res defaults to 504 upstream, which turns 1280x720 into ~504x284 and
throws away exactly the resolution this is for. 1008 (a multiple of the 14 px
ViT patch) keeps berry-scale detail.

The metric scale is a MODEL ESTIMATE. depth_fusion.py fits it to the depth
sensor rather than trusting it.

Install note (aarch64): upstream pins xformers, open3d, pycolmap, moviepy and
friends. None are needed for inference and xformers has no aarch64 wheel, so
install with --no-deps, add einops/omegaconf/safetensors/imageio/addict/evo/
scipy, and make utils/export/__init__.py's imports lazy.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

from settings import cfg
from feeding.camera import FXC as FX_COLOUR, FYC as FY_COLOUR

MODEL = cfg.deployment.models.depth_anything


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="/tmp/da3")
    ap.add_argument("--process-res", type=int, default=1008)
    a = ap.parse_args()

    import torch
    from depth_anything_3.api import DepthAnything3

    img = Image.open(a.image).convert("RGB")
    W, H = img.size
    model = DepthAnything3.from_pretrained(MODEL).to("cuda").eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model.inference([img], process_res=a.process_res,
                               process_res_method="upper_bound_resize")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    raw = np.asarray(pred.depth[0], dtype=np.float32)          # (h, w) processed
    h, w = raw.shape
    focal_proc = 0.5 * (FX_COLOUR + FY_COLOUR) * (w / W)       # scale to processed size
    metric = focal_proc * raw / 300.0                           # metres

    up = torch.nn.functional.interpolate(
        torch.from_numpy(metric)[None, None], size=(H, W),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    np.save(a.out + ".npy", up.astype(np.float32))
    conf = getattr(pred, "conf", None)
    if conf is not None:
        c = torch.nn.functional.interpolate(
            torch.from_numpy(np.asarray(conf[0], np.float32))[None, None],
            size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
        np.save(a.out + "_conf.npy", c)

    print("DA3 %s" % MODEL)
    print("  input      : %dx%d   processed %dx%d   (process_res %d)" % (W, H, w, h, a.process_res))
    print("  focal used : %.1f px at processed size" % focal_proc)
    print("  metric     : %.3f - %.3f m   median %.3f m" % (up.min(), up.max(), np.median(up)))
    print("  time       : %.2f s" % dt)
    print("  saved      : %s.npy (metres, %dx%d)" % (a.out, W, H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
