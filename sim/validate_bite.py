#!/usr/bin/env python3
"""How small a bite can the arm's own joint torques actually see?

This is the experiment behind docs/feeding-architecture.md Sec. 6. It holds the
fork at the mouth with a piece welded on, re-zeroes the detector, has the
simulated user tug with a known force, and reports whether the detector saw it
and how long it took. Then it sweeps the force down until detection fails, and
separately runs holds with NO bite to count false positives.

The answer depends almost entirely on how wrong the driver's dynamic model is,
not on torque-sensor noise -- so the sweep runs across model errors.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np

from world import World
from robot import Arm
from human import Human
from ik import IK, frame_from_axis
import skewer as SK
from feeding.bite_detector import JointTorqueBiteDetector, BiteDetectorConfig

STANDOFF = 0.09          # m in front of the mouth


def present_pose(w, ik, arm):
    """Joint angles holding the fork at the mouth, tines pointing at the user."""
    mouth = w.site_pos("mouth")
    axis = np.array([-1.0, 0.0, 0.0])          # tines point back toward the face
    q, pe, re = ik.solve(mouth + np.array([STANDOFF, 0, 0]),
                         frame_from_axis(axis), q0=Arm.RETRACT)
    return q, pe, re


def run_hold(seed, model_error, bite_force, det_kw, hold_s=3.0, settle_s=1.5,
             rezero=True):
    """One PRESENT hold. Returns (outcome, latency_s, peak_N, baseline_N)."""
    w = World(seed=seed, model_error=model_error)
    arm = Arm(w)
    human = Human(w, arm)
    ik = IK(w)

    q, _, _ = present_pose(w, ik, arm)
    arm.reset(q)
    arm.command(q)
    for _ in range(w.steps_for(settle_s)):
        arm.apply_feedforward()
        w.step()

    # Put a piece on the tines: move it to the tip, then pin the weld there.
    food_b, food_eq = int(w.food_bid[0]), int(w.food_eq[0])
    tip, R = arm.tcp()
    qadr = w.model.jnt_qposadr[w.model.body_jntadr[food_b]]
    w.data.qpos[qadr:qadr + 3] = tip - R[:, 2] * 0.004
    w.data.qvel[w.model.body_dofadr[food_b]:w.model.body_dofadr[food_b] + 6] = 0
    import mujoco
    mujoco.mj_forward(w.model, w.data)
    SK.weld_here(w.model, w.data, food_eq, w.bid("arm_fork"), food_b)

    for _ in range(w.steps_for(0.5)):
        arm.apply_feedforward()
        w.step()

    det = JointTorqueBiteDetector(BiteDetectorConfig(
        abs_min_nm=det_kw.get("threshold_n", 0.25),
        timeout_s=det_kw.get("timeout_s", 2.5),
        confirm_samples=3,
        # the sim runs at 1 kHz, so these windows hold far more samples than on
        # the 40 Hz high-level log
        settle_time=0.15, baseline_span=0.15, baseline_gap=0.12))
    det.reset(w.t)
    if not rezero:
        # No settling gate and no dead time: the version that looks simpler and
        # fails on hardware, kept as the counterfactual.
        det.cfg.settle_time = 0.0
        det.cfg.baseline_gap = 0.0
    # give the detector time to settle and take a baseline before the tug
    t_bite = w.t + det.cfg.settle_time + det.cfg.baseline_span + det.cfg.baseline_gap + 0.15
    started = False
    axis = arm.tcp()[1][:, 2]

    for _ in range(w.steps_for(hold_s)):
        if bite_force > 0 and not started and w.t >= t_bite:
            human.start_bite(food_b, peak=bite_force)
            started = True
        human.apply()
        arm.apply_feedforward()
        w.step()
        r = det.update(w.t, arm.qd, arm.measured_torque())
        if r is not None:
            lat = (r.t - t_bite) if started else float("nan")
            return r.outcome, lat, r.deviation_nm, r.baseline_nm
    return "timeout", float("nan"), 0.0, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=1.0)
    a = ap.parse_args()
    det_kw = dict(threshold_n=a.threshold, timeout_s=2.5)
    errs = (0.01, 0.03, 0.05)

    print("Bite detection from the joint-torque residual  (architecture Sec. 6)")
    print("  %.2f Nm absolute floor on joints 1-4, %d trials each\n"
          % (a.threshold, a.trials))

    print("  [1] with a fresh baseline at the start of the hold")
    print("      %-9s %-8s %-9s %-9s %s" % ("model err", "bite N", "detected", "latency", "baseline N"))
    for merr in errs:
        for force in (0.5, 1.0, 2.0, 5.0):
            outs = [run_hold(s, merr, force, det_kw) for s in range(a.trials)]
            hits = [o for o in outs if o[0] == "bite"]
            lat = np.nanmean([o[1] for o in hits]) if hits else float("nan")
            base = np.mean([o[3] for o in outs])
            print("      %-9s %-8.1f %-9s %-9s %+.2f"
                  % ("%.0f%%" % (merr * 100), force, "%d/%d" % (len(hits), len(outs)),
                     ("%.0f ms" % (lat * 1000)) if hits else "-", base))
    print()
    print("      no bite at all:")
    for merr in errs:
        outs = [run_hold(s, merr, 0.0, det_kw) for s in range(a.trials)]
        fp = sum(1 for o in outs if o[0] == "bite")
        print("      %-9s %-8s %-9s %-9s %+.2f"
              % ("%.0f%%" % (merr * 100), "-", "%d/%d" % (fp, len(outs)), "-",
                 np.mean([o[3] for o in outs])))

    print("\n  [2] same threshold, NO baseline -- absolute residual")
    print("      %-9s %-8s %-9s %s" % ("model err", "bite N", "triggered", "note"))
    for merr in errs:
        outs = [run_hold(s, merr, 0.0, det_kw, rezero=False) for s in range(a.trials)]
        fp = sum(1 for o in outs if o[0] == "bite")
        note = "FALSE BITE before the user moved" if fp else "quiet"
        print("      %-9s %-8s %-9s %s"
              % ("%.0f%%" % (merr * 100), "0.0", "%d/%d" % (fp, len(outs)), note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
