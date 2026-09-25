#!/usr/bin/env python3
"""A depth frame (median of fresh frames) from the resident depth server; exit 1 if it is not up.

    python -m feeding.depth_grab OUT.npy [--frames 15]

The caller falls back to depth_probe.py in the vision container (the old ~5 s path).
"""
import argparse
import io
import sys
import time
import urllib.request

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--frames", type=int, default=15)
    a = ap.parse_args(); t0 = time.time()
    try:
        with urllib.request.urlopen("http://127.0.0.1:8767/median?frames=%d" % a.frames, timeout=15) as r:
            d = np.load(io.BytesIO(r.read()))
    except Exception as e:  # noqa: BLE001
        print("depth server: %s" % e, file=sys.stderr)
        return 1
    np.save(a.out, d)
    v = d > 0
    print("depth %dx%d, %.1f%% valid, median %.0f mm, %.2f s" % (d.shape[1], d.shape[0], 100 * v.mean(),
                                                               np.median(d[v]) if v.any() else 0, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
