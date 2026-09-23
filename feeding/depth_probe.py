#!/usr/bin/env python3
"""Capture a depth frame and say how far away things actually are.

Runs INSIDE the kinova_vision container (it needs rclpy). Saves the raw 16-bit
depth as .npy plus a viewable colourised PNG.

    ros2 run ... python3 -m feeding.depth_probe --out /tmp/depth --frames 15

Takes the per-pixel MEDIAN over several frames rather than one frame. The
sensor drops out periodically and individual frames carry speckle; a median
over ~15 frames costs half a second and removes both.

Zero means NO RETURN, not "zero distance". It must be masked before any
statistic is taken -- averaging it in silently pulls every depth toward the
camera. Dark, shiny and steeply-angled surfaces produce it, which is exactly
what a dark glossy bowl interior is.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def colourise(d_mm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Near = red, far = blue, no-return = black. Plain numpy, no matplotlib."""
    valid = d_mm > 0
    t = np.zeros(d_mm.shape, dtype=np.float32)
    t[valid] = np.clip((d_mm[valid] - lo) / max(1.0, hi - lo), 0.0, 1.0)
    r = np.clip(1.5 - 2.0 * t, 0, 1)
    g = np.clip(1.5 - np.abs(2.0 * t - 1.0) * 2.0, 0, 1)
    b = np.clip(2.0 * t - 0.5, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/depth")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--timeout", type=float, default=25.0)
    a = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    frames: list[np.ndarray] = []

    class Probe(Node):
        def __init__(self):
            super().__init__("depth_probe")
            # The driver publishes BEST_EFFORT; a RELIABLE subscription simply
            # never matches and waits forever.
            self.create_subscription(Image, "/camera/depth/image_raw",
                                     self.cb, qos_profile_sensor_data)

        def cb(self, msg):
            if len(frames) >= a.frames:
                return
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            frames.append(arr.copy())
            self.w, self.h = msg.width, msg.height

    rclpy.init()
    node = Probe()
    t0 = time.time()
    while len(frames) < a.frames and time.time() - t0 < a.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()

    if not frames:
        print("no depth frames received in %.0f s" % a.timeout, file=sys.stderr)
        return 1

    stack = np.stack(frames)
    # Median over frames, ignoring no-returns so a pixel that returned in most
    # frames is not dragged to zero by the few where it did not.
    m = np.where(stack > 0, stack, np.nan).astype(np.float32)
    with np.errstate(all="ignore"):
        depth = np.nanmedian(m, axis=0)
    depth = np.nan_to_num(depth, nan=0.0).astype(np.uint16)

    valid = depth > 0
    pct = 100.0 * valid.mean()
    v = depth[valid]
    print("depth %dx%d from %d frames" % (depth.shape[1], depth.shape[0], len(frames)))
    print("  valid pixels : %.1f%%  (%d of %d are NO RETURN)"
          % (pct, (~valid).sum(), valid.size))
    if v.size:
        print("  range        : %d - %d mm" % (v.min(), v.max()))
        for q in (5, 25, 50, 75, 95):
            print("    p%-3d        : %d mm" % (q, int(np.percentile(v, q))))

    np.save(a.out + ".npy", depth)
    try:
        from PIL import Image as PImage
        lo = float(np.percentile(v, 2)) if v.size else 0.0
        hi = float(np.percentile(v, 98)) if v.size else 1.0
        PImage.fromarray(colourise(depth.astype(np.float32), lo, hi)).save(a.out + ".png")
        print("  colourised   : %s.png  (red %.0f mm -> blue %.0f mm, black = no return)"
              % (a.out, lo, hi))
    except ImportError:
        print("  (no PIL in this container; .npy written only)")
    print("  raw          : %s.npy" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
