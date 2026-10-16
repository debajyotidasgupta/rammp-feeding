#!/usr/bin/env python3
"""One generation's film from feed_replay.py's frames: the grid, the close-up inset (the generation's first strategy,
the best so far), and what is happening (picked / on the utensil / dropped, the bite clock, the best scores).

    python3 sim/isaac/compose.py FRAMES_DIR OUT.mp4 --title "generation 3" [--history DIR/history.json] [--fps 30]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames"); ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--history", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=4 / 120, help="bite seconds per frame")
    a = ap.parse_args()
    import imageio.v2 as imageio
    fr = Path(a.frames)
    grid = sorted(fr.glob("grid_*.jpg"))
    best = ""
    if a.history and Path(a.history).exists():
        h = json.loads(Path(a.history).read_text())
        if h:
            b = h[-1]["strategies"][0]
            best = "best so far: %.0f%% delivered, %.1f s bite, jerk %.1f m/s3" % (100 * b["delivered"], b["time_s"], b["jerk_rms"])
    big, mid, small = font(34), font(26), font(22)
    w = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    for k, g in enumerate(grid):
        im = Image.open(g).convert("RGB")
        W, H = im.size
        hero = fr / g.name.replace("grid_", "hero_")
        if hero.exists():
            hi = Image.open(hero).convert("RGB")
            hw = int(W * 0.36); hh = int(hi.height * hw / hi.width)
            hi = hi.resize((hw, hh), Image.LANCZOS)
            x0, y0 = W - hw - 24, H - hh - 24
            d = ImageDraw.Draw(im)
            d.rectangle([x0 - 4, y0 - 4, x0 + hw + 4, y0 + hh + 4], fill=(240, 240, 245))
            im.paste(hi, (x0, y0))
        st = fr / ("%05d_state.npy" % int(g.stem.split("_")[1]))
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, W, 112], fill=(8, 10, 16, 190))
        d.text((28, 16), "RAMMP feeding, cuRobo plans  |  %s" % a.title, font=big, fill=(240, 242, 248))
        line = "t %5.1f s" % (k * a.dt)
        if st.exists():
            s = np.load(st)
            n = s.shape[1]
            line += "   |   %d arms   picked %d   on the utensil %d   dropped %d" % (n, s[0].sum(), s[1].sum(), s[2].sum())
        d.text((28, 64), line, font=mid, fill=(170, 220, 255))
        if best:
            d.text((28, H - 46), best, font=small, fill=(250, 220, 120))
        w.append_data(np.asarray(im))
    w.close()
    print("%s: %d frames" % (a.out, len(grid)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
