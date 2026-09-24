#!/usr/bin/env python3
"""Face, nose and mouth from the wrist camera at the delivery pose, with depth at each.

    python -m feeding.detect_mouth RUN_DIR      # reads colour.jpg + depth.npy, writes face.json + face_upright.jpg

At the delivery pose the camera sits beside the level utensil, so the image comes out rolled 90 deg
(world-up is image-left). It is rotated upright before detection; boxes are reported in the upright frame,
pixels and camera coordinates in the original one.

- depth is the NEAREST surface at each point (20th percentile), not the median: a median once read the mouth
  67 mm behind the nose, off the face's edge.
- the "mouth" query has returned a full-frame box with the mouth in plain view; when the box is not
  anatomically plausible (small, just below the nose), it is retried as "lips". deliver.py makes the
  final call on what is usable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from feeding.locate_anything import LocateAnything
from feeding.measure_under import CXD, CYD, FXD, FYD, T_D2C

CAL = Path(__file__).resolve().parent.parent / "calibration"


def main() -> int:
    D = Path(sys.argv[1])
    c = json.loads((CAL / "colour_intrinsics.json").read_text())
    cx, cy, fx, fy = c["cx"], c["cy"], c["fx"], c["fy"]
    img = Image.open(D / "colour.jpg").convert("RGB"); H = img.size[1]
    up = img.transpose(Image.ROTATE_270)                    # world-up was image-left: 90 deg clockwise
    dep = np.load(D / "depth.npy").astype(float); yy, xx = np.mgrid[0:dep.shape[0], 0:dep.shape[1]]

    def to_orig(x, y):
        return y, H - 1 - x

    def depth_at(u, v, z=0.35):
        for _ in range(8):
            X = (u - cx) * z / fx - T_D2C[0]; Y = (v - cy) * z / fy - T_D2C[1]; Zd = z - T_D2C[2]
            ud, vd = FXD * X / Zd + CXD, FYD * Y / Zd + CYD
            m = (np.hypot(xx - ud, yy - vd) <= 5) & (dep > 0)
            if m.sum() < 5:
                return None
            z = np.percentile(dep[m], 20) / 1000.0 + T_D2C[2]
        return z

    la = LocateAnything()

    def run(prompt):
        b = la.locate_one(up, prompt)
        if b is None:
            return None
        u, v = to_orig((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2); z = depth_at(u, v)
        return dict(box_upright=[b.x1, b.y1, b.x2, b.y2], px=[u, v], depth_m=z, prompt=prompt,
                    cam_xyz=None if z is None else [(u - cx) * z / fx, (v - cy) * z / fy, z])

    def plausible(m, n):
        if not m or not n:
            return False
        x1, y1, x2, y2 = m["box_upright"]; a1, b1, a2, b2 = n["box_upright"]
        nh, nw = b2 - b1, a2 - a1; dy = (y1 + y2) / 2 - (b1 + b2) / 2
        return ((x2 - x1) * (y2 - y1) < 0.03 * up.width * up.height and 0.4 * nh < dy < 1.5 * nh
                and abs((x1 + x2) / 2 - (a1 + a2) / 2) < nw)

    out = dict(face=run("a human face"), nose=run("the person's nose"), mouth=run("the person's mouth"))
    if not plausible(out["mouth"], out["nose"]):
        out["mouth"] = run("the person's lips")
        print("mouth: 'mouth' box implausible, retried 'lips': %s"
              % ("usable" if plausible(out["mouth"], out["nose"]) else "still not usable"))
    d = ImageDraw.Draw(up)
    for name, col in (("face", "lime"), ("nose", "cyan"), ("mouth", "red")):
        r = out[name]
        if r is None:
            print("%-5s not found" % name); continue
        d.rectangle(r["box_upright"], outline=col, width=4)
        d.text((r["box_upright"][0] + 4, r["box_upright"][1] + 4), name, fill="white")
        print("%-5s depth %s  [%s]" % (name, "none" if r["depth_m"] is None else "%.0f mm" % (1000 * r["depth_m"]), r["prompt"]))
    up.save(D / "face_upright.jpg")
    (D / "face.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
