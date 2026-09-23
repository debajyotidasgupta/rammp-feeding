#!/usr/bin/env python3
"""A whole meal: detect, skewer, carry, present, wait for the bite, repeat.

This is an INTEGRATION TEST of the simulation environment, not the production
state machine -- that belongs in ROS 2 nodes against the driver's SimTransport
(docs/feeding-architecture.md Sec. 10). What it proves is narrower and is the
point of the last few days of work: every input the real stack will consume now
exists here, and the failure paths can be switched on.

Nothing is asserted that is not measured:
  * the piece is found through the DEPTH IMAGE, and rejected if its patch is
    too invalid, rather than read out of the simulator's state;
  * the bite ends PRESENT because the detector saw a torque residual, not
    because a timer expired;
  * the fork-to-head clearance is recorded every step, so "it was safe" is a
    number rather than an impression.

    python sim/demo_meal.py --bites 2
    python sim/demo_meal.py --bites 1 --lean-at 0.6   # user leans mid-transport
"""
from __future__ import annotations

import argparse
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
from perception import Perception
from faults import Faults
from safety import ClearanceMonitor, plan_leg
import skewer as SK
from feeding.bite_detector import JointTorqueBiteDetector, BiteDetectorConfig

STANDOFF = 0.09
TILT = np.array([0.5, 0.0, -0.866])          # 30 deg off vertical, into the plate
AT_MOUTH = np.array([-1.0, 0.0, 0.0])


class Runner:
    """Drives the arm and records what the safety monitor would care about."""

    def __init__(self, out: pathlib.Path | None, fps: int = 30):
        self.w = World(seed=3)
        self.arm = Arm(self.w)
        self.human = Human(self.w, self.arm)
        self.ik = IK(self.w)
        self.cam = Cameras(self.w)
        self.faults = Faults()
        self.per = Perception(self.w, self.cam, self.faults)
        self.mon = ClearanceMonitor(self.w)
        self.out = out
        self.fps = fps
        self.frame = 0
        self.min_clear = 1e9
        self.clear_at = ""
        self.clear_geom = ""
        self.max_push = 0.0
        self.push_at = ""
        self._reported = False
        self._r = None
        if out:
            out.mkdir(parents=True, exist_ok=True)
            for f in out.glob("*.png"):
                f.unlink()
            self._r = (mujoco.Renderer(self.w.model, 360, 640),
                       mujoco.Renderer(self.w.model, 360, 640))
        self.arm.reset(Arm.RETRACT)
        self.arm.command(Arm.RETRACT)

    # ------------------------------------------------------------------------
    def clearance(self):
        """Whole-arm clearance to the user: (metres, arm part, user part)."""
        return self.mon.closest()

    def user_contact(self) -> tuple[float, str]:
        """Largest normal force between the ARM and the user right now."""
        worst, pair = 0.0, ""
        for i in range(self.w.data.ncon):
            c = self.w.data.contact[i]
            n1 = mujoco.mj_id2name(self.w.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            n2 = mujoco.mj_id2name(self.w.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            hit = {n1, n2} & set(self.mon.names[g] for g in self.mon.user)
            if not hit:
                continue
            other = n2 if n1 in hit else n1
            if other is None:
                g = c.geom2 if n1 in hit else c.geom1
                other = mujoco.mj_id2name(self.w.model, mujoco.mjtObj.mjOBJ_BODY,
                                          self.w.model.geom_bodyid[g])
            f = np.zeros(6)
            mujoco.mj_contactForce(self.w.model, self.w.data, i, f)
            if abs(f[0]) > worst:
                worst, pair = float(abs(f[0])), "%s into %s" % (other, list(hit)[0])
        return worst, pair

    def tick(self, label: str) -> None:
        self.human.apply()
        self.arm.apply_feedforward()
        self.w.step()
        cl, part, who = self.clearance()
        if cl < self.min_clear:
            self.min_clear, self.clear_at = cl, label
            self.clear_geom = "%s/%s" % (part, who)
        push, pair = self.user_contact()
        if push > self.max_push:
            self.max_push, self.push_at = push, "%s during %s" % (pair, label)
        if push > 1.0 and not self._reported:
            self._reported = True
            print("  !! CONTACT WITH THE USER at t=%.1fs during %s: %s, %.0f N, "
                  "whole-arm clearance %.0f mm at %s"
                  % (self.w.t, label, pair, push, cl * 1000, self.clear_geom))

    def render(self) -> None:
        if not self._r:
            return
        from PIL import Image
        a, b = self._r
        a.update_scene(self.w.data, camera="user_view")
        b.update_scene(self.w.data, camera="side")
        Image.fromarray(np.hstack([a.render(), b.render()])).save(
            self.out / ("f%05d.png" % self.frame))
        self.frame += 1

    def move(self, q_goal, label: str, fraction: float = 0.35, settle: float = 1.0):
        """Servo to a joint goal on a raised-cosine ramp, then wait to converge."""
        q0 = self.arm.q_cmd.copy()
        secs = self.arm.duration_for(q0, q_goal, fraction=fraction)
        n = self.w.steps_for(secs)
        every = max(1, int(round(1.0 / (self.fps * self.w.dt))))
        for k in range(n):
            s = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            self.arm.command(q0 + (q_goal - q0) * s)
            self.tick(label)
            if k % every == 0:
                self.render()
        for k in range(self.w.steps_for(settle)):
            if np.abs(self.arm.q - q_goal).max() < np.radians(0.4):
                break
            self.tick(label)
            if k % every == 0:
                self.render()
        return secs, float(np.degrees(np.abs(self.arm.q - q_goal).max()))

    def hold(self, seconds: float, label: str, on_step=None):
        every = max(1, int(round(1.0 / (self.fps * self.w.dt))))
        for k in range(self.w.steps_for(seconds)):
            self.tick(label)
            if on_step is not None and on_step(k):
                return True
            if k % every == 0:
                self.render()
        return False

    def goto(self, pos, axis, label, fraction=0.35, settle=1.0):
        """Plan to a Cartesian goal with the WHOLE arm kept clear, then execute.

        Every leg is clearance-checked before it is driven. A straight line
        between two safe poses is not itself safe: the first version of this
        demo drove the upper arm 161 mm through the user's chest on exactly such
        a leg, at full position-control authority, with the fork tip 381 mm from
        anything.
        """
        R_des = frame_from_axis(axis)
        wps, worst, how = plan_leg(self.w, self.mon, self.ik, self.arm.q_cmd,
                                   np.asarray(pos, float), R_des)
        if not wps or how in ("NO SAFE PATH FOUND", "NO IK SOLUTION"):
            print("  !! %s: %s (worst %+.0f mm) -- refusing to move"
                  % (label, how, worst * 1000))
            return None, None, 0.0, 0.0, worst
        secs, err = 0.0, 0.0
        for q in wps:
            s, err = self.move(q, label, fraction=fraction, settle=settle)
            secs += s
        # residual of the final goal, for the record
        _, pe, re = self.ik.solve(np.asarray(pos, float), R_des, q0=self.arm.q_cmd)
        return pe, re, secs, err, worst


def one_bite(R: Runner, n: int, lean_at: float | None) -> str:
    w, arm, human = R.w, R.arm, R.human

    # The wrist camera has to be LOOKING at the plate before anything can be
    # detected in it. At RETRACT it points at the ceiling. This is not a detail
    # of the demo: it is why food detection is episodic and pose-gated rather
    # than a topic that is simply always publishing (architecture Sec. 2).
    R.goto([0.69, -0.11, 0.95], np.array([0.0, 0.0, -1.0]), "SCAN")
    pieces = R.per.detect_food()
    pieces = [p for p in pieces if p.index >= 0]
    if not pieces:
        print("  SCAN       no piece detected -> stop")
        return "no_food"
    # nearest to the plate centre is the least likely to be against the rim
    plate = w.data.xpos[w.bid("plate")]
    piece = min(pieces, key=lambda p: np.linalg.norm(p.position[:2] - plate[:2]))
    print("  SCAN       %d pieces, took #%d at %s (%.0f%% valid patch)"
          % (len(pieces), piece.index, np.round(piece.position, 3),
             100 * piece.valid_fraction))

    axis_in = TILT / np.linalg.norm(TILT)
    plate_z = float(plate[2]) + 0.006
    pre, ins, depth, clipped = SK.skewer_plan(piece.position, piece.half_extents,
                                              axis_in, plate_z)
    pe, re, s, e, cw = R.goto(pre, axis_in, "APPROACH")
    print("  APPROACH   %4.1fs  ik %.2fmm/%.1fdeg  path clearance %+.0f mm"
          % (s, pe * 1000, re, cw * 1000))

    pe, re, s, e, cw = R.goto(ins, axis_in, "SKEWER", fraction=0.10)
    print("  SKEWER     %4.1fs  depth %.1f mm, tip %.1f mm past centre%s"
          % (s, depth * 1000, float((ins - piece.position) @ axis_in) * 1000,
             ", plate-clipped" if clipped else ""))

    food_b = int(w.food_bid[piece.index])
    eq = int(w.food_eq[piece.index])
    SK.weld_here(w.model, w.data, eq, w.bid("arm_fork"), food_b)
    R.hold(0.3, "HOLD_IN")

    # The retention pose only has to be HORIZONTAL; its heading in plan is free.
    # Pinning it to the approach direction left two of the four pieces with no
    # reachable retention pose at all.
    retain, chosen = None, None
    for h in SK.RETENTION_HEADINGS:
        cand = SK.retention_axis(axis_in, heading_deg=h)
        wps, cw0, how = plan_leg(R.w, R.mon, R.ik, R.arm.q_cmd,
                                 ins + np.array([0, 0, 0.05]), frame_from_axis(cand))
        if wps and how not in ("NO IK SOLUTION", "NO SAFE PATH FOUND"):
            retain, chosen = cand, h
            break
    if retain is None:
        print("  RETAIN     no reachable retention pose -> drop this piece, try another")
        w.data.eq_active[eq] = 0
        return "unreachable"
    pe, re, s, e, cw = R.goto(ins + np.array([0, 0, 0.05]), retain, "RETAIN")
    tcp, Rm = arm.tcp()
    elev = np.degrees(np.arcsin(np.clip(Rm[2, 2], -1, 1)))
    print("  RETAIN     %4.1fs  heading %+.0f deg, fork elevation %+.1f deg (level = retained)"
          % (s, chosen, elev))

    # (0.52, -0.16, 1.00): of 60 candidate LIFT poses only 15 solved cleanly and
    # most of those put the forearm inside the user. This one clears by 50 mm.
    R.goto([0.52, -0.16, 1.00], retain, "LIFT")

    if lean_at is not None:
        human.set_lean(lean_at)
        print("  *** FAULT  user leans %.0f deg during transport" % np.degrees(lean_at))

    # Transport aims at the FACE TRACK, not at a constant, so a lean moves the goal.
    ft = R.per.track_face()
    if ft is None:
        print("  TRANSPORT  face track lost -> abort to a safe pose")
        R.goto([0.48, -0.14, 1.04], AT_MOUTH, "ABORT")
        return "face_lost"
    R.goto([0.46, -0.12, 1.06], AT_MOUTH, "REORIENT")
    ft = R.per.track_face()
    target = ft.nose + np.array([STANDOFF, 0, -0.029])
    pe, re, s, e, cw = R.goto(target, AT_MOUTH, "TRANSPORT", fraction=0.20)
    print("  TRANSPORT  %4.1fs  nose %s -> target %s  ik %.1fmm  clearance %.0f mm"
          % (s, np.round(ft.nose, 3), np.round(target, 3), pe * 1000, R.clearance()[0] * 1000))

    det = JointTorqueBiteDetector(BiteDetectorConfig(
        abs_min_nm=0.25, timeout_s=4.0, confirm_samples=3,
        settle_time=0.15, baseline_span=0.15, baseline_gap=0.12))
    det.reset(w.t)
    axis = arm.tcp()[1][:, 2]
    state = {"res": None, "bit": False}
    trace = []
    t_bite = w.t + 0.9

    def step(k):
        if not state["bit"] and w.t >= t_bite:
            human.start_bite(food_b)
            state["bit"] = True
        trace.append((w.t, float(arm.external_wrench()[:3] @ axis), R.clearance()[0]))
        r = det.update(w.t, arm.qd, arm.measured_torque())
        if r is not None:
            state["res"] = r
            return True
        return False

    R.hold(5.0, "PRESENT", on_step=step)
    r = state["res"]
    if trace:
        tr = np.array(trace)
        pre = tr[tr[:, 0] < t_bite]
        if len(pre):
            print("  PRESENT    before the tug: axial force %+.2f..%+.2f N, clearance %.0f mm"
                  % (pre[:, 1].min(), pre[:, 1].max(), pre[:, 2].min() * 1000))
    if r is None or r.outcome != "bite":
        print("  PRESENT    no bite detected -> timeout, hand back to the gamepad")
        w.data.eq_active[eq] = 0
        R.goto([0.48, -0.14, 1.04], AT_MOUTH, "RETRACT")
        return "timeout"

    print("  BITE       detected %.0f ms after the tug started, joint deviation %.2f Nm"
          % ((r.t - t_bite) * 1000, r.deviation_nm))
    w.data.eq_active[eq] = 0
    human.stop_bite()
    # Eaten. Without this the released cube falls into the user's lap and shows
    # up as arm-to-user contact, which is true of the model and false of the
    # world -- the piece went into the mouth.
    qadr = w.model.jnt_qposadr[w.model.body_jntadr[food_b]]
    # Park it well off to the side, ON the floor. Sending it BELOW the floor
    # plane instead creates a metre-deep penetration that the contact solver
    # ejects at speed -- the "eaten" cube then flies back and hits the user at
    # 26 N, which looks exactly like a safety failure and is purely self-inflicted.
    w.data.qpos[qadr:qadr + 3] = [0.0, 2.0, 0.02]
    dadr = w.model.body_dofadr[food_b]
    w.data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)
    R.hold(0.4, "BITE")
    R.goto([0.48, -0.14, 1.04], AT_MOUTH, "RETRACT")
    if lean_at is not None:
        human.set_lean(0.0)
        R.hold(1.0, "RETRACT")
    return "bite"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bites", type=int, default=2)
    ap.add_argument("--out", default="sim/out/meal")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--lean-at", type=float, default=None,
                    help="radians; user leans this far during the first transport")
    a = ap.parse_args()

    R = Runner(None if a.no_video else pathlib.Path(a.out), fps=a.fps)
    print("\nMeal: %d bite(s)\n" % a.bites)
    outcomes = []
    for i in range(a.bites):
        print("bite %d" % (i + 1))
        outcomes.append(one_bite(R, i, a.lean_at if i == 0 else None))
        print()
        if outcomes[-1] == "no_food":
            break

    print("outcomes: %s" % ", ".join(outcomes))
    print("closest approach to the user: %.0f mm to %s (during %s)"
          % (R.min_clear * 1000, R.clear_geom, R.clear_at))
    if R.max_push > 0:
        print("PEAK FORCE INTO THE USER: %.0f N  [%s]" % (R.max_push, R.push_at))
    else:
        print("no arm-to-user contact at any point")
    print("sim time %.1f s" % R.w.t)
    if R._r:
        print("%d frames -> %s" % (R.frame, a.out))
        for r in R._r:
            r.close()
    R.cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
