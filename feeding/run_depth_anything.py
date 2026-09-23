#!/usr/bin/env python3
"""Run Depth Anything V2 on one image and save its raw output.

    python -m feeding.run_depth_anything --image x.jpg --out /tmp/da

Depth Anything is MONOCULAR: it infers structure from a single RGB frame, so
its output is affine-invariant inverse depth (disparity-like), with NO metric
scale and NO absolute offset. On its own it cannot tell the arm how far
anything is. What it does have is resolution and sharp object boundaries, which
is exactly what the 480x270 sensor lacks.

So this is half of a pair: the sensor supplies metric truth at low resolution,
this supplies structure at colour resolution, and depth_fusion.py fits one to
the other.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from PIL import Image

MODEL = "depth-anything/Depth-Anything-V2-Base-hf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="/tmp/da")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    img = Image.open(a.image).convert("RGB")
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModelForDepthEstimation.from_pretrained(a.model).to("cuda").eval()

    t0 = time.perf_counter()
    inputs = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs).predicted_depth
    # Back to the ORIGINAL image size: the processor resizes to a multiple of
    # 14 for the ViT patches, and a fused result has to line up with the colour
    # pixels the detector gave boxes in.
    out = torch.nn.functional.interpolate(
        out.unsqueeze(1), size=(img.height, img.width),
        mode="bicubic", align_corners=False).squeeze()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    d = out.float().cpu().numpy()
    np.save(a.out + ".npy", d)
    print("depth anything %s" % a.model)
    print("  input      : %dx%d" % (img.width, img.height))
    print("  output     : %s  range %.3f - %.3f (relative inverse depth)"
          % (d.shape, d.min(), d.max()))
    print("  time       : %.2f s" % dt)
    print("  saved      : %s.npy" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
