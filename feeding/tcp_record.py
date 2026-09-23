#!/usr/bin/env python3
"""Auto-capture poses for fork-tip (TCP) pivot calibration. Runs in the driver container.

You touch ONE fixed physical point with the tine tips and hold still; each time
the arm goes quiet in a pose meaningfully different from the ones already taken,
this records it and says so. When there are enough poses with enough orientation
spread, it solves and prints the tool offset.

Why "different orientation" is enforced rather than suggested: the solve is

    R_i @ p_tool + t_i = p_fixed

which is only determined if R varies. Move the arm around while keeping the wrist
pointing the same way and the equations are degenerate -- the fit converges, the
residual looks small, and the answer is wrong. So a pose that is merely
TRANSLATED from one already held is rejected, loudly.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from kinova_gen3_interfaces.msg import EeState
from sensor_msgs.msg import JointState

OUT = pathlib.Path("/tmp/tcp_poses.json")


def quat_to_R(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def angle_between(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def solve(poses):
    A, b = [], []
    for p in poses:
        R = quat_to_R(*p["quat"])
        A.append(np.hstack([R, -np.eye(3)]))
        b.append(-np.asarray(p["pos"], float))
    sol, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    p_tool, p_fixed = sol[:3], sol[3:]
    tips = np.array([quat_to_R(*p["quat"]) @ p_tool + np.asarray(p["pos"], float)
                     for p in poses])
    resid = np.linalg.norm(tips - p_fixed, axis=1)
    Rs = [quat_to_R(*p["quat"]) for p in poses]
    spread = max(angle_between(a, b) for a in Rs for b in Rs)
    return p_tool, p_fixed, resid, spread


class Recorder(Node):
    def __init__(self, a):
        super().__init__("tcp_recorder")
        self.a = a
        self.buf: deque = deque(maxlen=a.window)
        self.poses = []
        self.tau = None
        self.last_capture = 0.0
        self.create_subscription(EeState, "/ee_state", self._ee, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self._js, qos_profile_sensor_data)
        print("[ready] touch the SAME point with the tine tips and hold still ~2 s.")
        print("[ready] need %d poses; rotate the wrist a lot between them." % a.need)
        sys.stdout.flush()

    def _js(self, m):
        e = np.array(m.effort[:7], float)
        if np.isfinite(e).all():
            self.tau = e

    def _ee(self, m):
        p, o = m.pose.position, m.pose.orientation
        self.buf.append((np.array([p.x, p.y, p.z]),
                         np.array([o.x, o.y, o.z, o.w]), time.time()))
        self._maybe_capture()

    def _maybe_capture(self):
        if len(self.buf) < self.buf.maxlen:
            return
        now = time.time()
        if now - self.last_capture < self.a.cooldown:
            return
        P = np.array([b[0] for b in self.buf])
        # np.ptp(), not P.ptp() -- the method was removed in NumPy 2.0 and the
        # container ships 2.x while the host venv pins <2.
        if np.ptp(P, axis=0).max() > self.a.still_m:    # still?
            return
        pos, quat = self.buf[-1][0], self.buf[-1][1]
        R = quat_to_R(*quat)
        for p in self.poses:                            # novel orientation?
            if angle_between(R, quat_to_R(*p["quat"])) < self.a.min_sep_deg:
                return
        self.poses.append(dict(pos=pos.tolist(), quat=quat.tolist(),
                               tau=(self.tau.tolist() if self.tau is not None else None)))
        self.last_capture = now
        OUT.write_text(json.dumps(self.poses, indent=2))
        spread = 0.0
        if len(self.poses) > 1:
            Rs = [quat_to_R(*p["quat"]) for p in self.poses]
            spread = max(angle_between(x, y) for x in Rs for y in Rs)
        print("[pose %d/%d] ee %s  orientation spread so far %.0f deg"
              % (len(self.poses), self.a.need, np.round(pos, 4), spread))
        sys.stdout.flush()
        if len(self.poses) >= self.a.need and spread >= self.a.want_spread:
            self._report()

    def _report(self):
        p_tool, p_fixed, resid, spread = solve(self.poses)
        print("")
        print("[RESULT] fork tip relative to the flange frame:")
        print("[RESULT]   x %+.4f  y %+.4f  z %+.4f  m   |p| = %.1f mm"
              % (*p_tool, np.linalg.norm(p_tool) * 1000))
        print("[RESULT]   pivot point %s" % np.round(p_fixed, 4))
        print("[RESULT]   residual mean %.2f mm  max %.2f mm over %d poses"
              % (resid.mean() * 1000, resid.max() * 1000, len(self.poses)))
        print("[RESULT]   orientation spread %.0f deg" % spread)
        if spread < 45:
            print("[RESULT]   *** spread too small -- offset is not observable")
        elif resid.max() > 3.0:
            print("[RESULT]   *** residual too large to trust for 15 mm food")
        else:
            print("[RESULT]   usable.")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=5)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--still-m", type=float, default=0.0015)
    ap.add_argument("--min-sep-deg", type=float, default=20.0)
    ap.add_argument("--want-spread", type=float, default=45.0)
    ap.add_argument("--cooldown", type=float, default=3.0)
    a = ap.parse_args()
    rclpy.init()
    n = Recorder(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
