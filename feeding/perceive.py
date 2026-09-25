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
  utensil  NOT detected for use. It is fixed to the arm, and so is the camera,
           so at food_detect_start it occupies the same pixels every time; that
           region is stored in config/utensil/<name>.yaml. A 'spoon'
           query once returned the whole bowl box, which then suppressed every
           berry. It is still queried, only to warn if the stored region looks
           stale (utensil changed or re-gripped).

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

import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from feeding.locate_anything import LocateAnything

PALETTE = [(239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8),
           (168, 85, 247), (20, 184, 166), (249, 115, 22)]


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(img, groups, out_path):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    f = font(17)
    for gi, (name, boxes) in enumerate(groups):
        col = PALETTE[gi % len(PALETTE)]
        for i, b in enumerate(boxes):
            d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=col, width=4)
            tag = "%s %d" % (name, i + 1)
            tb = d.textbbox((b.x1 + 3, max(0, b.y1 - 21)), tag, font=f)
            d.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill=col)
            d.text((b.x1 + 3, max(0, b.y1 - 21)), tag, fill=(255, 255, 255), font=f)
    im.save(out_path, quality=92)


from settings import cfg

MIN_SIDE_PX = cfg.perception.min_box_side_px
MAX_ASPECT = float(cfg.perception.max_box_aspect)
FD = cfg.features.food_detector


def excluded_colour(img, b) -> bool:
    """A box whose centre is not food we pick (food profile exclude_grapes): red / purple with little green -- a
    grape, which rolls off a fork -- or blue -- the mat or the bowl's wall. Melon (G ~170) and pineapple (G ~200)
    are not. 2026-09-25: a brightness test let a lit red grape through; the green channel does not."""
    if not cfg.food.get("exclude_grapes", False):
        return False
    w, h = b.x2 - b.x1, b.y2 - b.y1
    c = img.crop((int(b.x1 + 0.3 * w), int(b.y1 + 0.3 * h), int(b.x2 - 0.3 * w), int(b.y2 - 0.3 * h))).convert("RGB")
    r, g, bl = (float(np.mean(np.asarray(c)[..., k])) for k in range(3))
    grape = g < 110 and r > 1.5 * g
    background = bl > r and bl > g
    return grape or background


def plausible(b) -> bool:
    w, h = b.x2 - b.x1, b.y2 - b.y1
    if min(w, h) < MIN_SIDE_PX:
        return False
    return max(w, h) / max(1.0, min(w, h)) <= MAX_ASPECT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="scene_det")
    ap.add_argument("--food", default=cfg.food.prompt)
    ap.add_argument("--utensil", default="spoon", help="phrase for the cross-check only")
    ap.add_argument("--check-utensil", action="store_true", default=bool(cfg.features.utensil_cross_check),
                    help="also detect the utensil, to cross-check its stored region")
    ap.add_argument("--bowl-cache", help="reuse / save this session's bowl box here (make feed clears it)")

    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB"); img.filename = a.image     # the detector server reads the file
    from feeding.detector_server import connect
    la = connect()                                  # the resident model if `make detector` is up, else load here

    result, groups = {"image": a.image, "width": img.width, "height": img.height}, []

    def one(phrase):
        t = time.perf_counter()
        b = la.locate_one(img, phrase)
        return ([b] if b and plausible(b) else []), time.perf_counter() - t

    # The bowl does not move between bites: detect it once per session (make feed clears the cache) and reuse
    # its box -- 1.8 s a bite saved. Its uses: food outside it is not food; the most central berry has the
    # most rim clearance; and the berry query can be run on it alone.
    from feeding.locate_anything import Box
    cache = Path(a.bowl_cache) if a.bowl_cache else None
    if cache and cache.exists():
        c = json.loads(cache.read_text()); bowl, t_bowl = [Box(*c["box"], "bowl")], 0.0
        print("  bowl     (this session's box, not re-detected)")
    else:
        bowl, t_bowl = one("bowl")
        if cache and bowl:
            cache.write_text(json.dumps(dict(box=[bowl[0].x1, bowl[0].y1, bowl[0].x2, bowl[0].y2], image=a.image)))

    # Berries: on the BOWL only (half the image-reading time), in the model's "fast" mode, ending at the first
    # repeated or sliver box. Measured on 8 saved frames: the same berries as the full frame in hybrid mode at
    # 128 tokens, 5.6 s -> 1.0 s. (Past the real pieces the model repeats itself; at 512 tokens, 80 boxes.)
    t0 = time.perf_counter()
    crop_at = (0, 0)
    if bowl and FD.crop_to_bowl:
        m = int(cfg.perception.bowl_crop_margin_px); bb = bowl[0]
        x0, y0 = max(0, int(bb.x1) - m), max(0, int(bb.y1) - m)
        x1, y1 = min(img.width, int(bb.x2) + m), min(img.height, int(bb.y2) + m)
        crop = img.crop((x0, y0, x1, y1)); crop_path = str(Path(a.out).with_name("food_crop.jpg"))
        crop.save(crop_path, quality=95); crop.filename = crop_path; crop_at = (x0, y0)
    else:
        crop = img
    def ask(phrase):
        boxes = la.locate(crop, phrase, max_new_tokens=int(FD.max_new_tokens), generation_mode=FD.generation_mode,
                          stop_on_repeat=bool(FD.stop_on_repeat))
        for b in boxes:
            b.x1 += crop_at[0]; b.x2 += crop_at[0]; b.y1 += crop_at[1]; b.y2 += crop_at[1]
        return boxes
    food_all = ask(a.food)
    food = [b for b in food_all if plausible(b) and not excluded_colour(img, b)]
    t_food = time.perf_counter() - t0
    used_prompt = a.food

    # the utensil is not detected (its region is stored); the cross-check query cost ~2 s a bite and is
    # now only run on request
    seen, t_ute = one(a.utensil) if a.check_utensil else ([], 0.0)

    ux1, uy1, ux2, uy2 = cfg.utensil.image_region_px
    ute = [Box(ux1, uy1, ux2, uy2, "utensil")]
    stale = ""
    if seen:
        s_ = seen[0]
        ix = max(0.0, min(s_.x2, ux2) - max(s_.x1, ux1))
        iy = max(0.0, min(s_.y2, uy2) - max(s_.y1, uy1))
        iou = ix * iy / max(1.0, s_.area + ute[0].area - ix * iy)
        if iou < 0.5:
            stale = ("   (cross-check: detector's '%s' box has IoU %.2f with the stored "
                     "region -- ignore unless the utensil was changed)" % (a.utensil, iou))

    def inside_frac(b, u):
        ix = max(0.0, min(b.x2, u.x2) - max(b.x1, u.x1))
        iy = max(0.0, min(b.y2, u.y2) - max(b.y1, u.y1))
        return ix * iy / max(1.0, b.area)

    if ute:
        n0 = len(food)
        food = [b for b in food if inside_frac(b, ute[0]) < cfg.perception.inside_utensil_max]
        n_utensil = n0 - len(food)
    else:
        n_utensil = 0

    # A bowl is round, seen from straight above. When its box comes back much
    # taller than wide, the extra is the utensil merging in at the bottom edge
    # (measured 549x644 against the true 549x544), which drags the bowl centre
    # down. Keep the width and the top edge, which are reliable.
    bowl_fixed = False
    if bowl:
        bb = bowl[0]
        w, h = bb.x2 - bb.x1, bb.y2 - bb.y1
        if h > 1.08 * w:
            bb.y2 = bb.y1 + w
            bowl_fixed = True
        # Food outside the bowl is not food we will pick: a 47x108 px object
        # at the image edge came back as 'food pieces' with three real berries.
        n0 = len(food)
        # ...and no single piece of food is a large fraction of its bowl: a
        # 599x484 px 'food pieces' box once covered most of a 585x432 bowl.
        in_bowl = lambda b: (inside_frac(b, bb) > cfg.perception.food_inside_bowl_min      # noqa: E731
                             and b.area < cfg.perception.food_max_frac_of_bowl * bb.area)
        food = [b for b in food if in_bowl(b)]
        n_outside = n0 - len(food)
        # Nothing usable (a pile drawn as one bowl-sized box, 2026-09-25): ask again in the food profile's other
        # words, keeping the same checks -- plausible shape, not the utensil, inside the bowl, not a grape.
        for phrase in ([] if food else list(cfg.food.get("fallback_prompts", []))):
            more = [b for b in ask(phrase) if plausible(b) and not excluded_colour(img, b) and in_bowl(b)
                    and (not ute or inside_frac(b, ute[0]) < cfg.perception.inside_utensil_max)]
            print("  (no usable box for %r: asked %r -> %d)" % (a.food, phrase, len(more)))
            if more:
                food, used_prompt = more, phrase
                break
        t_food = time.perf_counter() - t0
    else:
        n_outside = 0

    for name, boxes, dt, extra in (("food", food, t_food, len(food_all) - len(food)),
                                   ("bowl", bowl, t_bowl, 0),
                                   ("utensil", ute, t_ute, 0)):
        note = ""
        if name == "food" and extra:
            note = "   (%d dropped: %d inside the utensil, %d outside or too big for the bowl, %d implausible shape)" % (
                extra, n_utensil, n_outside, extra - n_utensil - n_outside)
        if name == "bowl" and bowl_fixed:
            note = "   (box squared: it had merged with the utensil below)"
        if name == "utensil":
            note = "   (stored region, not detected)" + stale
        print("  %-8s %d box(es) in %5.2f s%s" % (name, len(boxes), dt, note))
        for i, b in enumerate(boxes):
            cx, cy = b.centre
            print("     %d  (%4.0f,%4.0f)-(%4.0f,%4.0f)  centre (%4.0f,%4.0f)  %3.0fx%3.0f px"
                  % (i + 1, b.x1, b.y1, b.x2, b.y2, cx, cy, b.x2 - b.x1, b.y2 - b.y1))
        result[name] = [dict(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
                             cx=b.centre[0], cy=b.centre[1]) for b in boxes]
        groups.append((name, boxes))

    result["utensil_tip"] = list(cfg.utensil.tip_px)      # measured; the box centre is not the tip
    result["food_prompt"] = used_prompt                     # which words found the food (a fallback, or the prompt)
    Path(a.out + ".json").write_text(json.dumps(result, indent=2))
    draw(img, groups, a.out + ".jpg")
    print("  wrote %s.json, %s.jpg" % (a.out, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
