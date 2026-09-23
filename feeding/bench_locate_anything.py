#!/usr/bin/env python3
"""Measure LocateAnything on this machine: latency AND where the box lands.

Latency alone is the wrong metric for a detector that steers a fork. A variant
that is twice as fast and moves the mouth centre by 15 px has not helped, so
every quantised variant is compared against the BF16 reference on box centre
displacement as well as time.

    python -m feeding.bench_locate_anything --image /tmp/live_latest.jpg --phrase face
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from PIL import Image

from feeding.locate_anything import LocateAnything, Box


def centre_shift(ref: Box | None, got: Box | None) -> str:
    if ref is None or got is None:
        return "n/a"
    rx, ry = ref.centre
    gx, gy = got.centre
    return "%.1f px" % ((rx - gx) ** 2 + (ry - gy) ** 2) ** 0.5


def iou(a: Box | None, b: Box | None) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def timed(fn, n: int):
    """Median of n runs; the first is discarded as warm-up/compile."""
    ts = []
    out = None
    for _ in range(n + 1):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    steady = sorted(ts[1:])
    return out, ts[0], steady[len(steady) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/tmp/live_latest.jpg")
    ap.add_argument("--phrase", default="face")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--variants", default="bf16,int8")
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    print("image %dx%d, phrase %r\n" % (img.width, img.height, a.phrase))

    la = LocateAnything()
    print("loaded, %.1f GB on device\n" % (torch.cuda.memory_allocated() / 1e9))

    ref = None
    print("  %-22s %10s %10s %10s %8s %s"
          % ("variant", "first", "median", "Hz", "IoU", "centre shift"))
    for v in a.variants.split(","):
        if v == "int8":
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            quantize_(la.model, Int8WeightOnlyConfig())
            # Eager quantisation is a 4.5x PESSIMISATION on this stack -- the
            # dequant never fuses. compile() is what makes it a win at all.
            la.model = torch.compile(la.model)
        elif v == "int4":
            from torchao.quantization import quantize_, Int4WeightOnlyConfig
            quantize_(la.model, Int4WeightOnlyConfig(
                group_size=128, int4_packing_format="tile_packed_to_4d"))
            la.model = torch.compile(la.model)
        try:
            box, first, med = timed(lambda: la.locate_one(img, a.phrase), a.runs)
        except Exception as e:                                # noqa: BLE001
            print("  %-22s FAILED: %s" % (v, str(e).split("\n")[0][:80]))
            continue
        if v == a.variants.split(",")[0]:
            ref = box
        print("  %-22s %9.2fs %9.2fs %10.2f %8.3f %s"
              % (v, first, med, 1.0 / med if med else 0, iou(ref, box),
                 centre_shift(ref, box)))
        if box:
            print("      box (%.0f,%.0f)-(%.0f,%.0f) centre (%.0f,%.0f)"
                  % (box.x1, box.y1, box.x2, box.y2, *box.centre))
        else:
            print("      NO BOX RETURNED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
