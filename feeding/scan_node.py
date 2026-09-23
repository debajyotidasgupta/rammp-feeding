#!/usr/bin/env python3
"""Retreat from the desk, then climb, under impedance control with a torque guard.

Runs INSIDE the driver container (it needs rclpy and kinova_gen3_interfaces).

Order of operations is dictated by the driver's own docs and is not optional:
create the setpoint publisher FIRST and let DDS discovery settle, THEN open the
stream -- otherwise the first setpoints go nowhere and the session expires.

The guard is a torque STEP detector, not the bite detector. A collision anywhere
on the arm (the fork catching the desk edge is the case we expect) shows up as a
jump in raw joint torque against the recent trend. We do not try to convert it
into a tip force: that mapping assumes contact is AT the tines and would be
nonsense for a forearm strike.

ATTENDED ONLY. e-stop in hand.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from kinova_gen3_interfaces.msg import PoseSetpoint
from kinova_gen3_interfaces.srv import OpenStream, CloseStream, AcquireControl, ReleaseControl


class Scanner(Node):
    def __init__(self, args):
        super().__init__("table_scanner")
        self.a = args
        # The token is a CAPABILITY issued by /acquire_control -- it is not ours
        # to invent. Every motion-commanding message must carry the one the
        # driver handed back, or the driver ignores it.
        self.token = None
        self.tau_hist: deque = deque(maxlen=args.window)
        self.tau = None
        self.q = None
        self.contact = None

        # 1. publisher FIRST, then let discovery settle.
        self.pub = self.create_publisher(PoseSetpoint, "/setpoint/pose", 10)
        # The driver publishes state as BEST_EFFORT sensor data; a default
        # RELIABLE subscription silently receives nothing.
        self.create_subscription(JointState, "/joint_states", self._js,
                                 qos_profile_sensor_data)
        self.ee = None
        from kinova_gen3_interfaces.msg import EeState
        self.create_subscription(EeState, "/ee_state", self._ee,
                                 qos_profile_sensor_data)

    def _js(self, m: JointState):
        e = np.array(m.effort[:7], dtype=float)
        if np.isfinite(e).all():
            self.tau = e
            self.q = np.array(m.position[:7], dtype=float)

    def _ee(self, m):
        p = m.pose.position
        o = m.pose.orientation
        self.ee = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))

    def spin_for(self, sec: float):
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.02)

    def check_contact(self) -> dict | None:
        """Flag a torque step against the EXTRAPOLATED trend, not the median.

        Gravity torque ramps monotonically as the arm changes pose -- joint_2
        swept -2.7 -> -6.7 Nm over a 73 mm retreat with nothing touched. A median
        of that window sits in the middle of the ramp, so the newest sample looks
        1.65 Nm "off trend" and a median detector cries contact on every step.

        Fitting a line over the window and predicting the next sample removes the
        ramp; what is left is the step a collision actually produces.
        """
        if self.tau is None:
            return None
        self.tau_hist.append(self.tau.copy())
        n = len(self.tau_hist)
        if n < max(4, self.tau_hist.maxlen // 2):
            return None
        hist = np.array(self.tau_hist)          # (n, 7)
        t = np.arange(n, dtype=float)
        # least-squares line per joint over the window, excluding the newest
        # sample, then predict it
        tf, hf = t[:-1], hist[:-1]
        A = np.vstack([tf, np.ones_like(tf)]).T
        coef, *_ = np.linalg.lstsq(A, hf, rcond=None)     # (2, 7)
        pred = coef[0] * t[-1] + coef[1]
        dev = hist[-1] - pred
        # residual scatter about the fit tells us what "normal" deviation is
        scatter = np.std(hf - (A @ coef), axis=0)
        j = int(np.argmax(np.abs(dev)))
        if abs(dev[j]) > max(self.a.threshold, 4.0 * scatter[j]):
            return dict(joint=j + 1, dev=float(dev[j]),
                        scatter=float(scatter[j]),
                        all=np.round(dev, 2).tolist())
        return None

    def send(self, pos, quat):
        m = PoseSetpoint()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = map(float, quat)
        m.token = self.token
        self.pub.publish(m)

    def run(self) -> int:
        self.spin_for(1.5)                      # DDS discovery
        if self.ee is None or self.tau is None:
            self.get_logger().error("no /ee_state or /joint_states yet")
            return 1
        start, quat = self.ee
        self.get_logger().info("start ee %s" % np.round(start, 4))

        acq = self.create_client(AcquireControl, "/acquire_control")
        if not acq.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/acquire_control not available")
            return 1
        areq = AcquireControl.Request()
        areq.owner_id = "table_scanner"
        af = acq.call_async(areq)
        rclpy.spin_until_future_complete(self, af, timeout_sec=10.0)
        ares = af.result()
        if ares is None or not ares.accepted:
            self.get_logger().error("control not granted: %s" % (ares and ares.message))
            return 1
        # ares.token is a numpy uint8 array; the .msg assertion demands
        # plain Python ints, and list() alone keeps numpy scalars.
        self.token = [int(b) for b in ares.token]
        self.get_logger().info("control acquired, generation %d" % ares.generation)

        cli = self.create_client(OpenStream, "/open_stream")
        if not cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/open_stream not available")
            return 1
        req = OpenStream.Request()
        req.controller = self.a.controller
        req.timeout_s = float(self.a.stream_timeout)
        req.token = self.token
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None or not res.accepted:
            self.get_logger().error("stream rejected: %s" % (res and res.message))
            return 1
        self.get_logger().info("stream open on %s, channels=%s"
                               % (self.a.controller, list(res.channels)))

        target = start.copy()
        ok = True
        try:
            # --- phase 1: back away from the desk ---------------------------
            ok = self._leg(target, quat, np.array([-1.0, 0, 0]),
                           self.a.retreat, "RETREAT (-X, away from desk)")
            # --- phase 2: climb ---------------------------------------------
            if ok:
                ok = self._leg(target, quat, np.array([0, 0, 1.0]),
                               self.a.climb, "CLIMB (+Z)")
        finally:
            c = self.create_client(CloseStream, "/close_stream")
            if c.wait_for_service(timeout_sec=3.0):
                r = CloseStream.Request()
                r.token = self.token
                f = c.call_async(r)
                rclpy.spin_until_future_complete(self, f, timeout_sec=5.0)
            self.get_logger().info("stream closed")
            rel = self.create_client(ReleaseControl, "/release_control")
            if rel.wait_for_service(timeout_sec=3.0):
                rr = ReleaseControl.Request()
                rr.token = self.token
                rf = rel.call_async(rr)
                rclpy.spin_until_future_complete(self, rf, timeout_sec=5.0)
                self.get_logger().info("control released")
        return 0 if ok else 2

    def _leg(self, target, quat, direction, distance, label) -> bool:
        n = max(1, int(round(distance / self.a.step)))
        self.get_logger().info("%s: %d steps of %.0f mm" % (label, n, self.a.step * 1000))
        for k in range(n):
            target += direction * self.a.step
            self.send(target, quat)
            self.spin_for(self.a.settle)
            hit = self.check_contact()
            cur = self.ee[0] if self.ee else target
            self.get_logger().info(
                "  step %2d/%d  cmd %s  ee %s  tau %s"
                % (k + 1, n, np.round(target, 3), np.round(cur, 3), np.round(self.tau, 1)))
            if hit:
                self.get_logger().warn(
                    "  CONTACT on joint_%d, %+.2f Nm off trend -- stopping this leg"
                    % (hit["joint"], hit["dev"]))
                self.get_logger().warn("  per-joint deviation: %s" % hit["all"])
                # hold where we are; impedance keeps it compliant
                self.send(target - direction * self.a.step, quat)
                self.spin_for(0.5)
                return False
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="ee_pose_impedance")
    ap.add_argument("--retreat", type=float, default=0.12)
    ap.add_argument("--climb", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--settle", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stream-timeout", type=float, default=120.0)
    a = ap.parse_args()
    rclpy.init()
    node = Scanner(a)
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
