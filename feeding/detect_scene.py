#!/usr/bin/env python3
"""Run LocateAnything-3B on one image and draw what it found.

    python -m feeding.detect_scene --image overlook.jpg --out detected.jpg

Stage 1 of finding the goal: from the overlook pose the wrist camera sees the
whole workspace, and this says where the bowl and the food are in that frame.

Prompt choice is not cosmetic, and the SEPARATOR is the part that matters.
Categories must be joined with the model's own "</c>", not with commas.
Measured on the overlook frame, three strawberries in a bowl:

    'food pieces</c>bowl</c>fork'   4 boxes,  9.9 s   bowl + 3 berries, all correct
    'food pieces, bowl, fork'       9 boxes, 15.1 s   the same 3 berries, plus SIX
                                                      junk boxes (whole frame, the
                                                      table, the arm base, ...)

Commas get read as one long phrase, so the model has no category structure to
decode against and falls back to emitting regions. Same content, same order --
only the separator differs, and it is the difference between a usable detection
and one that needs filtering.

Asking for the container alongside its contents also helps: the bowl gives the
model a spatial anchor for what counts as food. With an EMPTY bowl, though,
"food pieces" has no valid target and latches onto the white fork instead -- so
a detection with no food in frame is not a null result, it is a wrong one.

Output boxes are PIXELS in this image. Turning them into something the arm can
reach still needs the depth frame; this stage only answers "what, and where in
the picture".
"""

from __future__ import annotations

import argparse
import time
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="detected.jpg")
    ap.add_argument("--prompts", default="food pieces</c>bowl",
                    help="'</c>'-joined categories, or ';'-separated separate queries")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--single", action="store_true",
                    help="locate_one per prompt: one instance, and MUCH faster. "
                         "Measured on the overlook frame: 'bowl' single-instance "
                         "is 1.8-2.4 s and lands within ~12 px of the rim, while "
                         "the multi-instance path takes 10-15 s and returns loose "
                         "or full-frame boxes. Use this for the bowl; the food "
                         "needs multi-instance because there are several pieces.")
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    print("image %s  %dx%d" % (a.image, img.width, img.height))

    t0 = time.perf_counter()
    la = LocateAnything()
    print("model loaded in %.1f s (%s)" % (time.perf_counter() - t0, la.dtype))

    groups = []
    for query in a.prompts.split(";"):
        query = query.strip()
        cats = [c.strip() for c in query.split("</c>")]
        t0 = time.perf_counter()
        if a.single:
            b = la.locate_one(img, query, max_new_tokens=a.max_new_tokens)
            boxes = [b] if b else []
        else:
            boxes = la.detect(img, cats, max_new_tokens=a.max_new_tokens)
        dt = time.perf_counter() - t0
        print("\n  %-28s %2d box(es) in %5.2f s" % (repr(query), len(boxes), dt))
        for i, b in enumerate(boxes):
            cx, cy = b.centre
            print("     %d  (%4.0f,%4.0f)-(%4.0f,%4.0f)  centre (%4.0f,%4.0f)  "
                  "%3.0fx%3.0f px" % (i + 1, b.x1, b.y1, b.x2, b.y2, cx, cy,
                                      b.x2 - b.x1, b.y2 - b.y1))
        groups.append((cats[0] if len(cats) == 1 else query.replace("</c>", "+"), boxes))

    draw(img, groups, a.out)
    print("\nwrote %s" % Path(a.out).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
