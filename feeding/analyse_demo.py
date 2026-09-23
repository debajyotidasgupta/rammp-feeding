#!/usr/bin/env python3
"""Find the fruit-insertion event in a teleop demo, and measure its force.

The point of recording the demo was never the video. It was this: the threshold
that decides when the tines are far enough into the flesh is currently a guess,
and a demo of a human doing it correctly contains the real number.

What it looks for, in order:
  1. DESCENT   -- tool z falling steadily, the approach.
  2. CONTACT   -- external force stepping away from its recent trend while
                  descending. Trend extrapolation, not a median: the arm's own
                  gravity torque ramps as the pose changes, and a median detector
                  reads that ramp as contact (it did exactly that on the table
                  scan).
  3. YIELD     -- the partial force DROP as skin breaks. This is the signature
                  that distinguishes fruit from a table: a rigid surface gives a
                  monotonic rise, flesh gives rise-then-give.
  4. STEADY    -- force while held inside the flesh.

Renders the traces with PIL; matplotlib is not installed and is not worth
adding for this.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw


def load(path):
    # Tolerate a truncated final record: the log is flushed periodically, so a
    # copy taken while recording is still running ends mid-line.
    rows = []
    bad = 0
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print("(skipped %d malformed line(s) -- log was still being written)" % bad)
    t = np.array([r["t"] for r in rows]); t -= t[0]
    z = np.array([r["tool"]["z"] for r in rows])
    tau = np.array([[j["torque"] for j in r["joints"]] for r in rows])
    F = np.array([[r["wrench"]["fx"], r["wrench"]["fy"], r["wrench"]["fz"]] for r in rows])
    return t, z, tau, F


def trend_dev(x, win=40):
    """Deviation of each sample from a line fit to the preceding window."""
    d = np.zeros(len(x))
    for i in range(win, len(x)):
        seg = x[i - win:i]
        a, b = np.polyfit(np.arange(win), seg, 1)
        d[i] = x[i] - (a * win + b)
    return d


def render(path, t, z, Fmag, tau, marks):
    w, h = 1400, 820
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    panes = [("tool z (m)", z, (20, 90, 200)),
             ("|external force| (N)", Fmag, (200, 40, 40)),
             ("joint 2 torque (Nm)", tau[:, 1], (30, 140, 60)),
             ("joint 4 torque (Nm)", tau[:, 3], (140, 60, 180))]
    ph = (h - 40) // len(panes)
    for pi, (label, y, col) in enumerate(panes):
        y0 = 20 + pi * ph
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-9:
            hi = lo + 1
        d.rectangle([60, y0, w - 20, y0 + ph - 30], outline=(220, 220, 220))
        d.text((64, y0 + 2), "%s   [%.3f .. %.3f]" % (label, lo, hi), fill=(0, 0, 0))
        pts = []
        for i in range(len(t)):
            px = 60 + (w - 80) * t[i] / max(t[-1], 1e-9)
            py = y0 + ph - 30 - (ph - 45) * (y[i] - lo) / (hi - lo)
            pts.append((px, py))
        d.line(pts, fill=col, width=2)
        for mt, mlabel, mcol in marks:
            px = 60 + (w - 80) * mt / max(t[-1], 1e-9)
            d.line([(px, y0), (px, y0 + ph - 30)], fill=mcol, width=1)
            if pi == 0:
                d.text((px + 3, y0 + 14), mlabel, fill=mcol)
    im.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state.jsonl")
    ap.add_argument("--out", default="demo_trace.png")
    ap.add_argument("--force-step", type=float, default=1.0)
    a = ap.parse_args()

    t, z, tau, F = load(a.state)
    Fmag = np.linalg.norm(F, axis=1)
    hz = len(t) / max(t[-1], 1e-9)
    print("%d samples, %.1f s, %.1f Hz" % (len(t), t[-1], hz))
    print("tool z: %.3f .. %.3f m" % (z.min(), z.max()))
    print("|F|   : %.2f .. %.2f N   (median %.2f)" % (Fmag.min(), Fmag.max(), np.median(Fmag)))

    win = max(10, int(hz))
    dz = np.gradient(z) * hz                      # m/s
    dev = trend_dev(Fmag, win)

    # descending AND force stepping up = something was touched on the way down
    cand = np.where((dz < -0.003) & (dev > a.force_step))[0]
    marks = []
    if len(cand):
        i0 = int(cand[0])
        print("\nfirst contact while descending:")
        print("  t=%.2f s  z=%.4f m  |F|=%.2f N  (step +%.2f N over trend)"
              % (t[i0], z[i0], Fmag[i0], dev[i0]))
        marks.append((t[i0], "contact", (200, 0, 0)))
        seg = slice(i0, min(i0 + int(4 * hz), len(t)))
        pk = i0 + int(np.argmax(Fmag[seg]))
        print("  peak |F| %.2f N at t=%.2f s (z=%.4f)" % (Fmag[pk], t[pk], z[pk]))
        marks.append((t[pk], "peak", (230, 120, 0)))
        after = slice(pk, min(pk + int(4 * hz), len(t)))
        if after.stop > after.start + 5:
            trough = pk + int(np.argmin(Fmag[after]))
            drop = Fmag[pk] - Fmag[trough]
            print("  then |F| falls to %.2f N by t=%.2f s  (drop %.2f N, %.0f%%)"
                  % (Fmag[trough], t[trough], drop, 100 * drop / max(Fmag[pk], 1e-9)))
            if drop > 0.25 * Fmag[pk]:
                print("  -> rise-then-give: consistent with SKIN BREAKING, not a rigid surface")
            else:
                print("  -> monotonic rise: looks like a rigid surface, not flesh")
            marks.append((t[trough], "yield", (0, 150, 0)))
    else:
        print("\nno descending contact found above %.1f N; "
              "lower --force-step or check the trace" % a.force_step)

    print("\nper-joint torque range over the demo (Nm):")
    for j in range(7):
        print("  joint_%d  %+7.2f .. %+7.2f   span %5.2f"
              % (j + 1, tau[:, j].min(), tau[:, j].max(), np.ptp(tau[:, j])))

    render(a.out, t, z, Fmag, tau, marks)
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
