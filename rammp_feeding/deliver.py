"""Food delivery, from the scoop end.

  swing    one cuRobo path carrying the piece to the user: from the scoop end delivery.lift_m up, turned 90 deg toward
           the user with the utensil level (retracted along itself as far as the joints allow), then
           detect_sidestep_m to the user's right (the food out of the camera's view of the mouth)
  present  open loop: the mouth measured once, as it opens (the wrist camera, MediaPipe, its lips' depth), then one
           straight move -- the utensil turned toward the mouth by at most max_turn_deg -- to delivery.stop_m short of
           it, compliant (motion.impedance.approach). Exit 7: the user did not open their mouth (declined).
  bite     the force at the tool from the joint torques, against a baseline at rest: above bite_n for bite_hold_s,
           then until it settles (the user has let go), at most bite_settle_s
  retract  straight back along the utensil (back_off_m toward where the swing ended) and on in toward the arm's base
           in the same move, never stopped on torque (moving away is the safe direction); then one cuRobo path to
           food_detect_start that keeps the tip no further out from the base than where it starts or ends
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from rammp_feeding import record, scene
from rammp_feeding.arm import Arm, MoveError, dps_for
from rammp_feeding.config import cfg
from rammp_feeding.geometry import axis, rot

log = logging.getLogger("rammp")
DL, G = cfg.delivery, cfg.guards.torque_step_nm


def _swing_end_file(out: Path) -> Path:
    return out / "swing_end.json"


def swing_goal(arm: Arm, q0) -> list[float]:
    """Where the swing ends (the joints): continuous IK along the lift, the turn and the sidestep."""
    T0, R0 = arm.tool(q0)
    lift = float(DL.lift_m)
    up, err = arm.homotopy(q0, lambda s: (T0 + np.array([0, 0, lift * s]), R0), 35)
    if up is None:
        raise MoveError("the %.0f mm lift is not reachable (%s) -- nothing moved" % (1000 * lift, err))
    tip_t = arm.tip_tool
    tip0, ax0 = T0 + np.array([0, 0, lift]) + R0 @ tip_t, axis(R0)
    for back in (float(x) for x in DL.retract_tries_m):
        def at(s, r=back):
            Rs = rot("z", 90.0 * s) @ R0
            return rot("z", 90.0 * s) @ (tip0 - r * (1 - (1 - s) ** 2) * ax0) - Rs @ tip_t, Rs
        turn, err = arm.homotopy(up[-1], at, 60)
        if turn and min(arm.margin(q) for q in turn) >= min(3.0, arm.margin(up[-1]) - 0.5):
            break
    else:
        raise MoveError("no swing toward the user from here -- nothing moved")
    Ts, Rs = at(1.0)
    side = float(DL.detect_sidestep_m)
    step, _ = arm.homotopy(turn[-1], lambda s: (Ts - np.array([0, side * s, 0]), Rs), 8)
    return (step or turn)[-1]


def swing(arm: Arm, out: Path) -> None:
    q0 = arm.q()
    _, R0 = arm.tool(q0)
    if axis(R0)[1] < float(DL.pointing_tol_cos):
        raise MoveError("the utensil is not pointing at the user (+Y) -- run from the end of the scoop")
    arm.travel(swing_goal(arm, q0), "swing to the user", dps=float(DL.swing_dps), torque_limit=float(G.swing), max_tilt=95.0,
               carrying=True)
    _swing_end_file(out).write_text(json.dumps(dict(tool=[float(v) for v in arm.tool()[0]])))


def measure_mouth(robot, arm: Arm, face, t_frame: float, rgb, out: Path) -> np.ndarray:
    """The mouth in the base frame from a frame taken here (the arm still): its lips' depth, or one fresh look."""
    from PIL import Image
    from rammp_feeding.scene import cam_to_tool
    cam = robot.cameras["camera"]
    T, R = arm.tool()
    lo, hi = (float(v) for v in DL.mouth_depth_range_m)
    for look in range(2):
        D = cam.depth_median(5, t_frame - 0.2, wait_s=1.5, min_frames=3)
        m = face.mouth(rgb, D, arm.cam)
        why = ("no face" if m is None else "its range is only from the face's size (no depth on the lips)"
               if m["source"] != "depth" else "its depth %.0f mm is outside %.0f-%.0f" % (1000 * m["cam_xyz"][2], 1000 * lo, 1000 * hi)
               if not lo <= m["cam_xyz"][2] <= hi else None)
        (out / ("face_%d.json" % look)).write_text(json.dumps(dict(mouth=m, why=why), indent=1))
        Image.fromarray(rgb).save(out / ("look_%d.jpg" % look), quality=90)
        np.save(out / ("depth_%d.npy" % look), D)
        if why is None:
            mouth = T + R @ cam_to_tool(m["cam_xyz"])
            log.info("    the mouth at (%.3f %.3f %.3f), %.0f mm from the camera", *mouth, 1000 * m["cam_xyz"][2])
            return mouth
        log.info("    the mouth not usable (%s) -- looking again", why)
        t_frame, rgb = cam.frame(after=t_frame)
    raise MoveError("no usable mouth after two looks")


def approach(arm: Arm, mouth: np.ndarray) -> None:
    q0 = arm.q()
    T0, R = arm.tool(q0)
    ax = axis(R)
    if abs(ax[2]) > float(DL.level_tol):
        raise MoveError("the utensil is not level -- swing first")
    tip0 = T0 + R @ arm.tip_tool
    dh = np.array([mouth[0] - tip0[0], mouth[1] - tip0[1]])
    yaw = math.degrees(math.atan2(ax[0] * dh[1] - ax[1] * dh[0], ax[0] * dh[0] + ax[1] * dh[1]))
    yaw = float(np.clip(yaw, -float(DL.max_turn_deg), float(DL.max_turn_deg)))
    Rg = rot("z", yaw) @ R
    goal_tip = mouth - float(DL.stop_m) * axis(Rg) + np.array([0, 0, float(DL.aim_dz_m)])
    Tg = goal_tip - Rg @ arm.tip_tool
    L = float(np.linalg.norm(Tg - T0))
    record.event("approach_goal", mouth=mouth, goal_tip=goal_tip, distance_m=L, yaw_deg=yaw)
    log.info("    stop: the tip %.0f mm short of the mouth, turned %.0f deg; %.0f mm to go", 1000 * float(DL.stop_m), yaw, 1000 * L)
    if L > float(DL.max_approach_m):
        raise MoveError("an approach of %.0f mm is too long -- the user out of reach" % (1000 * L))
    n = max(40, int(math.ceil(L / float(DL.approach_step_m))))
    path, err = arm.homotopy(q0, lambda s: (T0 + (Tg - T0) * s, rot("z", yaw * s) @ R), n)
    if path is None:
        raise MoveError("the approach: %s -- the user out of reach from here" % err)
    arm.run(path, "approach", dps=dps_for(path, L, float(DL.approach_mps)), torque_limit=float(G.approach),
            max_tilt=95.0, impedance=DL.get("approach_impedance"))
    log.info("    presented: the tip %.0f mm from the mouth", 1000 * np.linalg.norm(mouth - arm.tip()))


def present(robot, arm: Arm, face, out: Path, wait_open: bool = True) -> int:
    cam = robot.cameras["camera"]
    if wait_open:
        got = face.wait_open(cam, float(DL.open_mouth_timeout_s), out)
        if got is None:
            return 7
    else:
        got = cam.frame()
    mouth = measure_mouth(robot, arm, face, got[0], got[1], out)
    scene.keep_mouth(mouth)                             # the user, where measured, in the planner's world from the retract
    approach(arm, mouth)
    return 0


def bite(robot, arm: Arm, timeout_s: float) -> int:
    """Wait (read-only) for the bite: the tip force above bite_n for bite_hold_s. 4 = no bite."""
    def sample(after):
        while True:
            s = robot.joint_sample()
            if s[0] > after:
                return s
            time.sleep(0.003)
    t, last = time.time(), 0.0
    base = []
    while len(base) < int(DL.bite_baseline_samples):
        last, _, tau = sample(last)
        base.append(tau)
    base = np.median(base, axis=0)
    q = arm.q()
    log.info("    waiting up to %.0f s for the bite (> %.1f N for %.2f s)", timeout_s, float(DL.bite_n), float(DL.bite_hold_s))
    above, peak, trace = None, 0.0, []
    while time.time() - t < timeout_s:
        last, _, tau = sample(last)
        f = float(np.linalg.norm(arm.tool_force(tau - base, q)))
        peak = max(peak, f)
        trace.append((last - t, f))
        if f <= float(DL.bite_n):
            above = None
        elif above is None:
            above = last
        elif last - above >= float(DL.bite_hold_s):
            log.info("    BITE (%.1f N)", peak)
            t_bite, n = time.time(), int(DL.bite_baseline_samples)
            while time.time() - t_bite < float(DL.bite_settle_s):     # then the force settles: the user has let go
                last, _, tau = sample(last)
                trace.append((last - t, float(np.linalg.norm(arm.tool_force(tau - base, q)))))
                if len(trace) > n and max(f for _, f in trace[-n:]) < float(DL.bite_n):
                    break
            log.info("    settled after %.1f s", time.time() - t_bite)
            record.save_arrays("bite", trace=np.array(trace))
            return 0
    record.save_arrays("bite", trace=np.array(trace))
    log.info("    no bite in %.0f s (peak %.1f N)", timeout_s, peak)
    return 4


def retract(robot, arm: Arm, out: Path, go: bool = True) -> None:
    R_ = DL.retract
    q = arm.q()
    T0, R0 = arm.tool(q)
    path, Tn = [q], T0
    if abs(axis(R0)[2]) <= float(DL.level_tol):          # at the user: straight back toward where the swing ended
        f = _swing_end_file(out)
        if not f.exists():
            raise MoveError("no swing recorded (%s) -- not guessing the way back from the user" % f)
        Te = np.array(json.loads(f.read_text())["tool"])
        L = float(np.linalg.norm(Te - T0))
        if L > float(R_.max_from_swing_end_m):
            raise MoveError("%.0f mm from the swing's end -- not backing off from here" % (1000 * L))
        if R_.back_off_m is not None and L > float(R_.back_off_m):
            Te, L = T0 + (Te - T0) * float(R_.back_off_m) / L, float(R_.back_off_m)
        if L > float(DL.arrived_m):
            back, err = arm.homotopy(q, lambda s: (T0 + (Te - T0) * s, R0), 40)
            if back is None:
                raise MoveError("back off: %s -- nothing moved" % err)
            path, Tn = back, Te
    tip = Tn + R0 @ arm.tip_tool                       # ... and on in toward the base's axis, level, not past the mouth
    r = math.hypot(tip[0], tip[1])
    mouth = np.array(scene.mouth_now())
    for tuck in (float(x) for x in R_.tuck_radii_m):
        if r <= tuck + float(DL.arrived_m):
            break
        Tt = Tn - np.array([tip[0], tip[1], 0.0]) * (r - tuck) / r
        line = [tip + (Tt - Tn) * k / 20 for k in range(21)]
        if min(np.linalg.norm(x - mouth) for x in line) < np.linalg.norm(tip - mouth) - float(DL.arrived_m):
            break
        more, _ = arm.homotopy(path[-1], lambda s, Tt=Tt: (Tn + (Tt - Tn) * s, R0), 40)
        if more is not None:
            path, r, Tn = path + more[1:], tuck, Tt
            break
    if len(path) > 1 and go:
        L = float(np.linalg.norm(Tn - T0))
        arm.run(path, "back off and in toward the arm", dps=dps_for(path, max(L, 1e-3), float(R_.back_off_mps)),
                torque_stop=False, max_tilt=95.0)
    goal = scene.overlook_joints(arm)
    r_goal = math.hypot(*arm.tip(goal)[:2])
    arm.travel(goal, "back to food_detect_start", dps=float(R_.home_dps), torque_limit=float(G.retract), go=go,
               start=None if go else path[-1], max_reach_m=max(r, r_goal) + float(R_.reach_margin_m))
    _swing_end_file(out).unlink(missing_ok=True)


def deliver(robot, arm: Arm, face, out: Path, stop_after: str = "", start_at: str = "swing", wait_open: bool = True) -> int:
    """swing -> present -> bite -> retract. 7: declined (retracted); 8: the presentation refused (retracted)."""
    stages = ["swing", "present", "bite", "retract"]
    run = stages[stages.index(start_at):]
    if "swing" in run:
        swing(arm, out)
        if stop_after == "swing":
            return 0
    if "present" in run:
        try:
            rc = present(robot, arm, face, out, wait_open)
        except MoveError as e:
            log.info("    the presentation stopped (%s) -- retracting", e)
            record.event("present_refused", why=str(e))
            retract(robot, arm, out)
            return 8
        if rc == 7:
            log.info("    no open mouth -- not approaching; retracting")
            retract(robot, arm, out)
            return 7
        if stop_after == "present":
            return 0
    if "bite" in run:
        bite(robot, arm, float(DL.bite_timeout_s))
        if stop_after == "bite":
            return 0
    retract(robot, arm, out)
    return 0
