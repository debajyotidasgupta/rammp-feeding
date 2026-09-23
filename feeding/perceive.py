#!/usr/bin/env python3
"""Detect the food, the bowl and the utensil in one frame, and write them out.

    python -m feeding.perceive --image scene.jpg --out scene_det

Writes <out>.json (pixel boxes, per category) and <out>.jpg (overlay).

ONE QUERY PER CATEGORY. Asking for several categories at once
("food pieces</c>bowl</c>spoon") is faster in principle and unreliable in
practice: the model gets stuck repeating one category's box until it exhausts
max_new_tokens, and every category after it is silently dropped. Which category
repeats is not stable -- on one frame it was the bowl, on the next the spoon,
then the food. Reordering only moves the casualty around. Separate queries cost
a few seconds and remove the failure outright.

  food     multi-instance -- there are several pieces
  bowl     single-instance -- one container; this path was 2.4 s and within
           ~12 px of the rim
  utensil  single-instance -- it is in the gripper, its position is known from
           kinematics; detected only as a cross-check

Boxes that cannot be a real object are dropped: a 10x310 px sliver came back as
'food pieces' on the same frame as three correct berries.

Food boxes lying mostly INSIDE the utensil box are dropped too. The spoon's
head is white and rounded and the detector reads it as food -- it was returned
as a fourth 'food piece' at (660,466)-(756,620), squarely inside the spoon.
The utensil is in every frame from this pose, so this is not a corner case.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from feeding.detect_scene import draw
from feeding.locate_anything import LocateAnything

MIN_SIDE_PX = 25
MAX_ASPECT = 3.0


def plausible(b) -> bool:
    w, h = b.x2 - b.x1, b.y2 - b.y1
    if min(w, h) < MIN_SIDE_PX:
        return False
    return max(w, h) / max(1.0, min(w, h)) <= MAX_ASPECT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="scene_det")
    ap.add_argument("--food", default="food pieces")
    ap.add_argument("--utensil", default="spoon")
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    la = LocateAnything()

    result, groups = {"image": a.image, "width": img.width, "height": img.height}, []
    t0 = time.perf_counter()
    food_all = la.locate(img, a.food, max_new_tokens=512)
    food = [b for b in food_all if plausible(b)]
    t_food = time.perf_counter() - t0

    def one(phrase):
        t = time.perf_counter()
        b = la.locate_one(img, phrase)
        return ([b] if b and plausible(b) else []), time.perf_counter() - t

    bowl, t_bowl = one("bowl")
    ute, t_ute = one(a.utensil)

    def inside_frac(b, u):
        ix = max(0.0, min(b.x2, u.x2) - max(b.x1, u.x1))
        iy = max(0.0, min(b.y2, u.y2) - max(b.y1, u.y1))
        return ix * iy / max(1.0, b.area)

    if ute:
        n0 = len(food)
        food = [b for b in food if inside_frac(b, ute[0]) < 0.5]
        n_utensil = n0 - len(food)
    else:
        n_utensil = 0

    for name, boxes, dt, extra in (("food", food, t_food, len(food_all) - len(food)),
                                   ("bowl", bowl, t_bowl, 0),
                                   ("utensil", ute, t_ute, 0)):
        note = ""
        if name == "food" and extra:
            note = "   (%d dropped: %d inside the utensil, %d implausible shape)" % (
                extra, n_utensil, extra - n_utensil)
        print("  %-8s %d box(es) in %5.2f s%s" % (name, len(boxes), dt, note))
        for i, b in enumerate(boxes):
            cx, cy = b.centre
            print("     %d  (%4.0f,%4.0f)-(%4.0f,%4.0f)  centre (%4.0f,%4.0f)  %3.0fx%3.0f px"
                  % (i + 1, b.x1, b.y1, b.x2, b.y2, cx, cy, b.x2 - b.x1, b.y2 - b.y1))
        result[name] = [dict(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
                             cx=b.centre[0], cy=b.centre[1]) for b in boxes]
        groups.append((name, boxes))

    Path(a.out + ".json").write_text(json.dumps(result, indent=2))
    draw(img, groups, a.out + ".jpg")
    print("  wrote %s.json, %s.jpg" % (a.out, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
