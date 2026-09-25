"""Debug images for the run record: depth frames as colour PNGs, with marks. Never raises -- a failed debug image
must not fail a bite.

    from feeding.viz import depth_png
    depth_png(depth_mm, "run/depth.png")                                   # near = red, far = blue, none = black
    depth_png(depth_mm, "run/under.png", circles=[(u, v, 7, "white")], filled=mask)   # marks, filled = tinted
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def colourise(d_mm: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Near = red, far = blue, no return = black; lo / hi default to the 2nd / 98th percentile."""
    d = np.asarray(d_mm, dtype=np.float32)
    valid = d > 0
    if lo is None or hi is None:
        v = d[valid]
        lo, hi = (float(np.percentile(v, 2)), float(np.percentile(v, 98))) if v.size else (0.0, 1.0)
    t = np.zeros(d.shape, dtype=np.float32)
    t[valid] = np.clip((d[valid] - lo) / max(1.0, hi - lo), 0.0, 1.0)
    r = np.clip(1.5 - 2.0 * t, 0, 1)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0) * 2.0, 0, 1)
    b = np.clip(2.0 * t - 0.5, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def depth_png(d_mm: np.ndarray, out: str | Path, filled: np.ndarray | None = None,
              circles: Sequence[tuple] = (), text: str | None = None, scale: int = 2) -> Path | None:
    """Save a depth frame (mm) as a PNG, `scale` x larger. filled: bool mask drawn tinted (the DA3 fallback);
    circles: (u, v, radius, colour) in depth pixels; text: a caption, top left."""
    try:
        from PIL import Image, ImageDraw
        rgb = colourise(d_mm)
        if filled is not None and filled.shape == rgb.shape[:2] and filled.any():
            rgb[filled] = (0.5 * rgb[filled] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        im = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        for u, v, r, colour in circles:
            dr.ellipse([scale * (u - r), scale * (v - r), scale * (u + r), scale * (v + r)], outline=colour, width=2)
        if text:
            dr.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
            dr.text((4, 3), text, fill=(255, 255, 255))
        im.save(out)
        return Path(out)
    except Exception:  # noqa: BLE001 -- a debug image
        return None
