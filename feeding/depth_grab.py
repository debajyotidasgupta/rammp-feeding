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

from settings import cfg


def fetch(frames: int, since: float | None = None) -> np.ndarray | None:
    """A median of fresh frames from the resident depth server: frames that arrived after `since` (unix time:
    when the arm stopped -- then frames buffered before a stream dropout still count; default: now), short
    waits, a few tries, None if it never answers."""
    wait, tries = float(cfg.perception.depth_fresh_wait_s), int(cfg.perception.depth_tries)
    since = time.time() if since is None else since
    url = "http://127.0.0.1:%d/median?frames=%d&wait=%.1f&since=%.3f&min=%d" % (
        int(cfg.deployment.servers.depth_port), frames, wait, since, int(cfg.perception.depth_min_frames))
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=wait + 5) as r:
                return np.load(io.BytesIO(r.read()))
        except Exception as e:  # noqa: BLE001
            print("depth server (try %d/%d): %s" % (k + 1, tries, e), file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=int(cfg.perception.depth_frames))
    ap.add_argument("--since", type=float, help="unix time the arm stopped: frames after it count (default: now)")
    a = ap.parse_args(); t0 = time.time()
    d = fetch(a.frames, a.since)
    if d is None:
        return 1
    np.save(a.out, d)
    from feeding.viz import depth_png
    depth_png(d, str(a.out)[:-4] + ".png" if str(a.out).endswith(".npy") else str(a.out) + ".png")
    v = d > 0
    print("depth %dx%d, %.1f%% valid, median %.0f mm, %.2f s" % (d.shape[1], d.shape[0], 100 * v.mean(),
                                                               np.median(d[v]) if v.any() else 0, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
