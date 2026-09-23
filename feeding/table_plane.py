#!/usr/bin/env python3
"""Fit the table plane from the wrist depth stream. Runs in the vision container.

This is the measurement architecture Sec. 4.1 calls for, and the reason not to
reach for a monocular depth model: the Gen3's depth camera reports METRIC range
directly, so the support plane comes out in metres with no scale to guess.

Reports the plane in the CAMERA frame (range along the optical axis, and tilt),
plus an estimated table height in the base frame from the current end-effector
z. That second number is approximate -- it assumes the camera looks roughly
down and ignores the camera-to-flange offset, which is exactly the extrinsic
calibration still outstanding. It is a sanity check, not a calibrated value.
"""
from __future__ import annotations

import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo


class PlaneFit(Node):
    def __init__(self, args):
        super().__init__("table_plane")
        self.a = args
        self.depth = None
        self.K = None
        self.create_subscription(Image, "/camera/depth/image_raw",
                                 self._img, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/depth/camera_info",
                                 self._info, qos_profile_sensor_data)

    def _img(self, m: Image):
        a = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.width)
        self.depth = a.astype(np.float64) / 1000.0        # 16UC1 mm -> m
        self.encoding = m.encoding

    def _info(self, m: CameraInfo):
        self.K = dict(fx=m.k[0], fy=m.k[4], cx=m.k[2], cy=m.k[5],
                      w=m.width, h=m.height)

    def run(self) -> int:
        import time
        t0 = time.time()
        while (self.depth is None or self.K is None) and time.time() - t0 < 15:
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.depth is None:
            self.get_logger().error("no depth image received")
            return 1
        if self.K is None or not self.K["w"]:
            h, w = self.depth.shape
            self.K = dict(fx=self.a.fx, fy=self.a.fy, cx=self.a.cx, cy=self.a.cy,
                          w=w, h=h)
            print("camera_info was EMPTY -- using this arm's measured intrinsics")
        K = self.K
        print("camera_info the driver is publishing:")
        print("  %dx%d  fx=%.3f fy=%.3f cx=%.3f cy=%.3f"
              % (K["w"], K["h"], K["fx"], K["fy"], K["cx"], K["cy"]))
        print("  (this arm reports fx=fy=334.810 cx=238.536 cy=139.694;")
        print("   the package default would be 360.013 -- 7.5%% wrong)")

        z = self.depth
        valid = (z > 0.05) & (z < 3.0)
        print("\ndepth frame: %.1f%% valid, range %.3f-%.3f m"
              % (100 * valid.mean(), z[valid].min(), z[valid].max()))

        u, v = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
        x = (u - K["cx"]) * z / K["fx"]
        y = (v - K["cy"]) * z / K["fy"]
        pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        if len(pts) < 500:
            print("too few valid points (%d)" % len(pts))
            return 1

        rng = np.random.default_rng(0)

        def fit_one(P):
            best = (0, None)
            for _ in range(self.a.iters):
                idx = rng.choice(len(P), 3, replace=False)
                p0, p1, p2 = P[idx]
                n = np.cross(p1 - p0, p2 - p0)
                ln = np.linalg.norm(n)
                if ln < 1e-9:
                    continue
                n /= ln
                d = -n @ p0
                inl = np.abs(P @ n + d) < self.a.tol
                if inl.sum() > best[0]:
                    best = (int(inl.sum()), (n, d, inl))
            if best[1] is None:
                return None
            n, d, inl = best[1]
            Q = P[inl]
            c = Q.mean(axis=0)
            _, _, vt = np.linalg.svd(Q - c)
            n = vt[2] / np.linalg.norm(vt[2])
            if n[2] < 0:
                n = -n
            return n, float(-n @ c), inl, Q

        # A single RANSAC returns the DOMINANT plane, which in a wrist view of a
        # table is often the FLOOR -- it simply has more pixels. Peel off planes
        # in order so the table can be identified by range and tilt rather than
        # assumed to be the biggest thing in frame.
        remaining = pts
        print("\nplanes found, largest first (camera frame, +z = optical axis):")
        print("  %-3s %-9s %-26s %-9s %-8s %s"
              % ("#", "inliers", "normal", "range m", "tilt deg", "resid mm"))
        planes = []
        for i in range(self.a.planes):
            if len(remaining) < 800:
                break
            r = fit_one(remaining)
            if r is None:
                break
            n, d, inl, Q = r
            rng_m = -d / n[2] if abs(n[2]) > 1e-6 else float("nan")
            tilt = np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1)))
            resid = np.abs(Q @ n + d)
            print("  %-3d %-9d [%+.3f %+.3f %+.3f]     %-9.4f %-8.1f %.2f"
                  % (i + 1, len(Q), n[0], n[1], n[2], rng_m, tilt, resid.mean() * 1000))
            planes.append(dict(n=n, d=d, count=len(Q), range=rng_m, tilt=tilt))
            remaining = remaining[~inl]

        if len(planes) >= 2:
            # Parallel planes in a wrist view of a table are the TABLETOP and the
            # FLOOR. Their perpendicular separation is the table height, and it
            # needs no extrinsics at all -- both surfaces are measured by the
            # same sensor in the same frame, so the camera pose cancels.
            ref = planes[0]["n"]
            para = [pl for pl in planes
                    if abs(float(pl["n"] @ ref)) > 0.995]
            if len(para) >= 2:
                lo = min(para, key=lambda pl: pl["range"])
                hi = max(para, key=lambda pl: pl["range"])
                sep = abs(float(lo["d"] - hi["d"]))
                print("\n  two parallel surfaces, %.4f m and %.4f m along the axis"
                      % (lo["range"], hi["range"]))
                print("  perpendicular separation: %.4f m" % sep)
                print("  -> the near one is the TABLETOP, the far one the FLOOR,")
                print("     so the table stands %.0f mm above the floor." % (sep * 1000))
                print("  This needs NO extrinsic calibration: both surfaces come")
                print("  from the same sensor, so the camera pose cancels out.")

        if self.a.ee_z is not None and planes:
            print("\n  end-effector z in base frame: %.4f m" % self.a.ee_z)
            print("  NOT converting range to a base-frame height: the plane is")
            print("  tilted off the optical axis, and the camera-to-flange")
            print("  transform is uncalibrated. Range along the optical axis is")
            print("  the honest measurement until the extrinsics exist.")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--tol", type=float, default=0.006)
    ap.add_argument("--planes", type=int, default=4)
    ap.add_argument("--ee-z", type=float, default=None)
    # kinova_vision currently publishes an EMPTY camera_info when handed a
    # custom *_camera_info_url (it logs "Failed to detect content in .ini file"
    # on a file that is byte-identical in format to its own shipped default).
    # Until that is chased down, pass this arm's real intrinsics directly --
    # measured from the device, see calibration/README.md.
    ap.add_argument("--fx", type=float, default=334.80999756)
    ap.add_argument("--fy", type=float, default=334.80999756)
    ap.add_argument("--cx", type=float, default=238.53599548)
    ap.add_argument("--cy", type=float, default=139.69400024)
    a = ap.parse_args()
    rclpy.init()
    n = PlaneFit(a)
    try:
        return n.run()
    finally:
        n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
