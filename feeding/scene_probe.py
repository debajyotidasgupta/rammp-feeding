#!/usr/bin/env python3
"""What can LocateAnything actually see in the feeding cell?

Loads the model ONCE and answers two questions together, because a model load
on Orin costs minutes and doing them separately doubles it:

  1. Which of the things the feeding loop depends on are detectable, and where.
  2. What each query costs, BF16 vs INT8-compiled, including how far the boxes
     move under quantisation.

The prompts are the actual objects in the bite cycle, not a generic demo list.
"""
from __future__ import annotations

import argparse
import time

import torch
from PIL import Image, ImageDraw

from feeding.locate_anything import LocateAnything, Box

# What the loop actually needs to find, and why.
TARGETS = [
    ("table",      "support plane; the hard floor for the skewer descent"),
    ("bowl",       "the food container"),
    ("fruit",      "what we are actually skewering"),
    ("strawberry", "the specific piece; a narrower prompt usually localises better"),
    ("fork",       "the tool, to cross-check against the commanded TCP"),
]


def draw(img: Image.Image, found: dict[str, list[Box]], path: str) -> None:
    im = img.copy()
    d = ImageDraw.Draw(im)
    colours = ["#ff3b30", "#34c759", "#0a84ff", "#ffd60a", "#bf5af2",
               "#ff9f0a", "#64d2ff", "#ff375f"]
    for i, (name, boxes) in enumerate(found.items()):
        c = colours[i % len(colours)]
        for b in boxes:
            d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=3)
            d.text((b.x1 + 3, max(0, b.y1 - 12)), name, fill=c)
    im.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/tmp/live_latest.jpg")
    ap.add_argument("--out", default="/tmp/scene_probe.png")
    ap.add_argument("--int8", action="store_true", help="also measure INT8+compile")
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    print("image %dx%d\n" % (img.width, img.height))

    t0 = time.perf_counter()
    la = LocateAnything()
    print("model loaded in %.0f s, %.1f GB resident\n"
          % (time.perf_counter() - t0, torch.cuda.memory_allocated() / 1e9))

    def sweep(tag: str) -> dict[str, list[Box]]:
        found: dict[str, list[Box]] = {}
        print("  %-14s %7s %8s  %s" % ("[%s] target" % tag, "n", "sec", "boxes (centre px)"))
        for phrase, _why in TARGETS:
            t = time.perf_counter()
            try:
                boxes = la.locate(img, phrase)
            except Exception as e:                            # noqa: BLE001
                print("  %-14s %7s %8s  ERROR %s" % (phrase, "-", "-", str(e)[:50]))
                continue
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            found[phrase] = boxes
            desc = ", ".join("(%.0f,%.0f)" % b.centre for b in boxes[:4]) or "NONE"
            print("  %-14s %7d %8.2f  %s" % (phrase, len(boxes), dt, desc))
        return found

    found = sweep("bf16")

    # Multi-category in ONE pass, the way architecture Sec. 4.2 specifies the
    # food detector: an open-vocabulary prompt rather than a per-object query.
    # One inference instead of N is the point -- detection is episodic and the
    # cycle cannot afford a query per object class.
    print("\n  [multi] one query, categories joined with the model's </c>")
    for cats in (["food pieces", "bowl"], ["food pieces"], ["fruit", "bowl", "table"]):
        t = time.perf_counter()
        try:
            boxes = la.detect(img, cats)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            desc = ", ".join("(%.0f,%.0f)" % b.centre for b in boxes[:6]) or "NONE"
            print("    %-34s %2d boxes %6.2fs  %s"
                  % ("</c>".join(cats), len(boxes), dt, desc))
            found["+".join(cats)] = boxes
        except Exception as e:                                  # noqa: BLE001
            print("    %-34s ERROR %s" % ("</c>".join(cats), str(e)[:50]))

    draw(img, found, a.out)
    print("\n  annotated -> %s" % a.out)

    if a.int8:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        print("\n  quantising to INT8 and compiling (first query pays the compile)...")
        quantize_(la.model, Int8WeightOnlyConfig())
        la.model = torch.compile(la.model)
        found8 = sweep("int8")
        draw(img, found8, a.out.replace(".png", "_int8.png"))
        print("\n  box displacement, INT8 vs BF16:")
        for phrase, _ in TARGETS:
            b0 = (found.get(phrase) or [None])[0]
            b1 = (found8.get(phrase) or [None])[0]
            if b0 and b1:
                dx = ((b0.centre[0] - b1.centre[0]) ** 2
                      + (b0.centre[1] - b1.centre[1]) ** 2) ** 0.5
                print("    %-14s %6.1f px" % (phrase, dx))
            else:
                print("    %-14s %6s" % (phrase, "n/a"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
