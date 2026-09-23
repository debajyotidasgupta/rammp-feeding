#!/usr/bin/env python3
"""Raise the arm until the wrist camera can see the table, watching joint torque.

The move the user described: back away from the desk first, then climb, and stop
the instant something is touched -- because on the way up the fork can catch the
desk edge.

Contact detection here is deliberately NOT the bite detector. Two reasons:

  * The bite detector estimates a wrench AT THE TINE TIPS by pushing the residual
    through the Jacobian. That assumes the force is at the tip. If the forearm
    hits the desk, the assumption is false and the inferred "tip force" is
    meaningless -- while the raw joint torques still jump. So for collision we
    threshold the RAW per-joint signal.
  * We have no gravity model on this machine (the driver keeps its Pinocchio
    model to itself and publishes only measured effort). But gravity torque
    varies SMOOTHLY with pose and a collision is a STEP, so during a slow move a
    short-window deviation detector separates them without any model at all.

Measured on this arm at rest: effort is per-joint Nm, joint_4 sits near -8.4 Nm
carrying the arm's weight, the rest are under 2.5 Nm.

    python -m feeding.scan_for_table --dry-run      # prints the plan, moves nothing
    python -m feeding.scan_for_table --execute      # ATTENDED ONLY, e-stop in hand
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import deque

import numpy as np

ARM_JOINTS = 7
# Deviation from the short-window trend that counts as contact, per joint.
# Provisional: re-derive from a real touch trial before trusting it.
CONTACT_NM = 1.5
WINDOW = 12                 # samples in the trend window
STEP_M = 0.01               # 10 mm per climb step
RETREAT_M = 0.12            # back away from the desk before climbing
MAX_CLIMB_M = 0.35
SETTLE_S = 0.7


def ros(cmd: list[str], timeout: int = 20) -> str:
    """Run a ros2 CLI command inside the driver container."""
    full = ("source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; "
            + " ".join(cmd))
    return subprocess.run(
        ["docker", "exec", "kinova_gen3_real", "bash", "-lc", full],
        capture_output=True, text=True, timeout=timeout).stdout


def read_joints() -> tuple[np.ndarray, np.ndarray]:
    out = ros(["timeout", "8", "ros2", "topic", "echo", "--once", "/joint_states"])
    def block(name: str) -> list[float]:
        seg = out.split(name + ":", 1)[1].split("---", 1)[0]
        vals = []
        for line in seg.splitlines():
            line = line.strip()
            if line.startswith("- "):
                try:
                    vals.append(float(line[2:]))
                except ValueError:
                    return vals
            elif vals:
                break
        return vals
    return (np.array(block("position")[:ARM_JOINTS]),
            np.array(block("effort")[:ARM_JOINTS]))


def read_ee() -> np.ndarray:
    out = ros(["timeout", "8", "ros2", "topic", "echo", "--once", "/ee_state"])
    seg = out.split("position:", 1)[1]
    return np.array([float(seg.split(k + ":", 1)[1].split("\n", 1)[0]) for k in "xyz"])


class ContactWatch:
    """Flags a torque STEP against the recent trend. No dynamics model needed."""

    def __init__(self, threshold: float = CONTACT_NM, window: int = WINDOW):
        self.hist: deque = deque(maxlen=window)
        self.threshold = threshold

    def update(self, tau: np.ndarray):
        if len(self.hist) < self.hist.maxlen:
            self.hist.append(tau)
            return None
        trend = np.median(np.array(self.hist), axis=0)
        dev = tau - trend
        self.hist.append(tau)
        j = int(np.argmax(np.abs(dev)))
        if abs(dev[j]) > self.threshold:
            return dict(joint=j + 1, deviation_nm=float(dev[j]),
                        all_nm=np.round(dev, 2).tolist())
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threshold", type=float, default=CONTACT_NM)
    a = ap.parse_args()
    if not (a.execute or a.dry_run):
        print("choose --dry-run or --execute (ATTENDED ONLY)")
        return 2

    q, tau = read_joints()
    ee = read_ee()
    print("arm is at ee %s" % np.round(ee, 4))
    print("joint torque now: %s Nm" % np.round(tau, 2))
    print("\nplan:")
    print("  1. retreat %.0f mm in -X, away from the desk" % (RETREAT_M * 1000))
    print("  2. climb in +Z in %.0f mm steps, to at most %.0f mm"
          % (STEP_M * 1000, MAX_CLIMB_M * 1000))
    print("  3. after each step, sample torque; abort on a >%.1f Nm step change"
          % a.threshold)
    print("  4. at the top, grab a wrist frame and look for the table and bowl")
    print("  controller: ee_pose_impedance (compliant, so a contact is absorbed)")

    if a.dry_run:
        print("\n(dry run -- nothing was commanded)")
        return 0

    # --- execution is intentionally not wired up yet ------------------------
    # Commanding this motion needs the streaming controller opened and a live
    # abort path. It moves a pointed fork near a desk and a person, so it wants
    # a human watching it the first time, not a script that was written and run
    # in the same breath.
    print("\nEXECUTE path is deliberately not enabled in this revision.")
    print("Bring it up interactively with the e-stop in hand:")
    print("  ros2 service call /open_stream ... controller: ee_pose_impedance")
    print("  then step /setpoint/pose while watching /joint_states effort")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
