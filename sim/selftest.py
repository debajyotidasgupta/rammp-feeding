#!/usr/bin/env python3
"""Does the simulation environment still do what it claims?

Run after any change to the scene, the assets or the interfaces. Each check
prints the number it measured, not just a verdict, because most of these are
quantities the design depends on rather than booleans.

    python sim/selftest.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import mujoco
import numpy as np

from world import World
from robot import Arm
from human import Human
from ik import IK, frame_from_axis
from sensors import Cameras
from perception import Perception, project
from faults import Faults
from safety import ClearanceMonitor, plan_leg, sweep
import skewer as SK

FAILS: list[str] = []


CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    print("  [%s] %-42s %s" % ("ok" if ok else "FAIL", name, detail))
    if not ok:
        FAILS.append(name)


def main() -> int:
    w = World(seed=0)
    arm = Arm(w)
    human = Human(w, arm)
    ik = IK(w)

    print("\nmodel")
    check("scene compiles", True,
          "nq=%d nv=%d nu=%d nbody=%d ncam=%d" % (w.model.nq, w.model.nv, w.model.nu,
                                                  w.model.nbody, w.model.ncam))
    check("control rate is the arm's 1 kHz", abs(w.dt - 0.001) < 1e-9, "dt=%.4f s" % w.dt)

    # nothing should be interpenetrating before anything has moved
    worst, pair = 0.0, ""
    for i in range(w.data.ncon):
        c = w.data.contact[i]
        if c.dist < worst:
            g = lambda k: mujoco.mj_id2name(w.model, mujoco.mjtObj.mjOBJ_GEOM, k)
            worst, pair = c.dist, "%s/%s" % (g(c.geom1), g(c.geom2))
    check("no penetration at rest", worst > -1e-3,
          "worst %.2f mm %s" % (worst * 1000, pair))

    print("\narm")
    arm.reset(Arm.HOME); arm.command(Arm.HOME)
    for _ in range(w.steps_for(2.0)):
        arm.apply_feedforward(); w.step()
    err = np.degrees(np.abs(arm.q - Arm.HOME).max())
    check("holds HOME without droop", err < 0.5, "max joint error %.3f deg" % err)

    # the residual must recover a known external force, with the right sign
    w2 = World(seed=0, model_error=0.0)
    a2 = Arm(w2); a2.TORQUE_NOISE = 0.0
    a2.reset(Arm.HOME); a2.command(Arm.HOME)
    for _ in range(w2.steps_for(3.0)):
        a2.apply_feedforward(); w2.step()
    F = np.array([0.0, 3.0, -7.0])
    w2.data.xfrc_applied[w2.bid("arm_fork"), :3] = F
    for _ in range(w2.steps_for(2.0)):
        a2.apply_feedforward(); w2.step()
    est = a2.external_wrench()[:3]
    check("torque residual recovers an applied force",
          np.linalg.norm(est - F) < 0.05,
          "applied %s -> %s N" % (F, np.round(est, 3)))

    print("\nreach")
    targets = {
        "over the plate": ([0.69, -0.11, 0.95], [0, 0, -1]),
        "in the plate":   ([0.69, -0.13, 0.783], [0.5, 0, -0.866]),
        "at the mouth":   (w.site_pos("mouth") + [0.09, 0, 0], [-1, 0, 0]),
    }
    for name, (pos, ax) in targets.items():
        q, pe, re = ik.solve(np.asarray(pos, float), frame_from_axis(ax), q0=Arm.RETRACT)
        check("reaches %s" % name, pe < 2e-3 and re < 1.0,
              "pos %.2f mm, orient %.2f deg" % (pe * 1000, re))

    print("\nuser")
    m0 = human.mouth().copy()
    human.set_lean(0.35)
    for _ in range(w.steps_for(4.0)):
        arm.apply_feedforward(); w.step()
    d = human.mouth() - m0
    check("lean moves the mouth toward the arm", d[0] > 0.10,
          "+%.0f mm forward, %.0f mm down at %.0f deg" % (d[0] * 1000, -d[2] * 1000,
                                                          np.degrees(human.lean)))
    human.set_lean(0.0)
    for _ in range(w.steps_for(3.0)):
        arm.apply_feedforward(); w.step()

    print("\nsensors")
    q, _, _ = ik.solve(np.array([0.69, -0.11, 0.95]), frame_from_axis([0, 0, -1]), q0=Arm.RETRACT)
    arm.reset(q); arm.command(q)
    for _ in range(w.steps_for(1.5)):
        arm.apply_feedforward(); w.step()
    cam = Cameras(w)
    rgb, dep, face = cam.colour(), cam.depth(), cam.face()
    check("colour stream", rgb is not None and rgb.shape == (720, 1280, 3), str(rgb.shape))
    check("face camera sees the face", face is not None, str(face.shape))
    valid = 100 * (dep > 0).mean()
    check("depth valid fraction matches hardware", 85 <= valid <= 93,
          "%.1f%% (arm measured 87-91%%)" % valid)

    # a point we know the world position of must land where the render shows it
    pr = project(w, "arm_wrist_depth", cam.K["depth"], w.data.xpos[w.food_bid[0]])
    check("camera projection is consistent", pr is not None and 0 <= pr[0] < 480,
          "food0 -> pixel (%.0f, %.0f) at %.3f m" % pr if pr else "off frame")

    print("\nperception")
    per = Perception(w, cam, Faults())
    pieces = per.detect_food()
    if pieces:
        gt = np.array([w.data.xpos[w.food_bid[p.index]] for p in pieces])
        ec = np.linalg.norm(np.array([p.position for p in pieces]) - gt, axis=1)
        es = np.linalg.norm(np.array([p.surface for p in pieces]) - gt, axis=1)
        check("food detected through the depth image", len(pieces) >= 2 and ec.max() < 0.005,
              "%d/4 pieces, centre error %.1f-%.1f mm (raw surface %.1f mm)"
              % (len(pieces), ec.min() * 1000, ec.max() * 1000, es.mean() * 1000))
    else:
        check("food detected through the depth image", False, "none")

    plane = per.table_plane()
    check("table plane fit", plane is not None and abs(plane["height"] - 0.75) < 0.06,
          "z=%.3f m, %.0f%% inliers, normal %s"
          % (plane["height"], 100 * plane["fraction"], np.round(plane["normal"], 2))
          if plane else "no fit")

    ft = per.track_face()
    check("face track from the static camera", ft is not None,
          "mouth %s" % np.round(ft.mouth, 3) if ft else "no track")

    print("\nfaults")
    f = Faults(detection_blackout=True)
    check("detection blackout yields nothing",
          Perception(w, cam, f).detect_food() == [], "")
    f = Faults(phantom_food=True)
    ph = Perception(w, cam, f).detect_food()
    check("phantom food is injected", any(p.index == -1 for p in ph),
          "%d detections, one of which is not real" % len(ph))
    f = Faults(face_lost=True)
    check("face track can be lost", Perception(w, cam, f).track_face() is None, "")
    cam.stalls = True
    w.data.time = Cameras.STALL_PERIOD - 0.5
    check("depth stream stalls like the real one", cam.depth() is None,
          "silent for %.1f s every %.0f s" % (Cameras.STALL_RECOVERY, Cameras.STALL_PERIOD))

    print("\nsafety")
    mon = ClearanceMonitor(w)
    arm.reset(Arm.RETRACT); arm.command(Arm.RETRACT)
    for _ in range(w.steps_for(1.0)):
        arm.apply_feedforward(); w.step()
    check("monitor covers the whole arm", len(mon.arm) >= 10,
          "%d arm geoms x %d user volumes" % (len(mon.arm), len(mon.user)))

    # IK must not wind the continuous joints into a multi-turn path
    ax = np.array([0.5, 0.0, -0.866]); ax /= np.linalg.norm(ax)
    ret = SK.retention_axis(ax)
    qa, _, _ = ik.solve(np.array([0.74, -0.08, 0.83]), frame_from_axis(ret), q0=Arm.RETRACT)
    qb, _, _ = ik.solve(np.array([0.46, -0.12, 1.06]), frame_from_axis([-1, 0, 0]), q0=qa)
    travel = np.degrees(np.abs(qb - qa)).max()
    check("IK does not wind continuous joints", travel < 200,
          "max joint travel %.0f deg on the worst leg" % travel)

    # the elbow must stay out of the user
    legs = [("RETAIN", [0.74, -0.08, 0.83], ret),
            ("LIFT", [0.52, -0.16, 1.00], ret),
            ("REORIENT", [0.46, -0.12, 1.06], [-1, 0, 0]),
            ("PRESENT", [0.21, 0.0, 1.21], [-1, 0, 0])]
    q0, worst, reseeds = Arm.RETRACT, 1e9, 0
    for _, pos, a in legs:
        wps, cw, how = plan_leg(w, mon, ik, q0, np.asarray(pos, float), frame_from_axis(a))
        worst = min(worst, cw)
        reseeds += how.startswith("reseeded")
        q0 = wps[-1]
    check("planned meal path keeps the whole arm clear", worst > 0.02,
          "worst %+.0f mm across %d legs, %d needed a re-seeded IK branch"
          % (worst * 1000, len(legs), reseeds))

    # the same goal, solved from a bad seed, is what the planner is guarding against
    qbad, pe, re = ik.solve(np.array([0.74, -0.08, 0.83]), frame_from_axis(ret),
                            q0=Arm.RETRACT)
    w.data.qpos[w.arm_qadr] = qbad
    mujoco.mj_forward(w.model, w.data)
    bad, part, _ = mon.closest()
    check("redundancy resolution can put the elbow in the user", bad < 0.02,
          "an IK solution with %.2f mm residual clears by %+.0f mm (%s)"
          % (pe * 1000, bad * 1000, part))

    # and the gap a tip-only monitor would have missed
    w.data.qpos[w.arm_qadr] = q0
    mujoco.mj_forward(w.model, w.data)
    whole, part, _ = mon.closest()
    check("tip-only monitor is optimistic", mon.tip_only() > whole,
          "tip says %+.0f mm, whole arm says %+.0f mm (%s)"
          % (mon.tip_only() * 1000, whole * 1000, part))

    cam.close()
    print("\n%s  (%d checks, %d failed)\n"
          % ("ALL CHECKS PASSED" if not FAILS else "FAILURES: " + ", ".join(FAILS),
             CHECKS, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
