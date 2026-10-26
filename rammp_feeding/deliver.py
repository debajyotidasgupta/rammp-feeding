"""Food delivery, from the scoop end.

  swing    one cuRobo path carrying the piece to the delivery look: a fixed pose (rammp capture deliver), the utensil
           level toward the user, the camera on the user's face, the food out of its view of the mouth
  present  open loop: the mouth measured once, as it opens (the wrist camera, MediaPipe, its lips' depth), from the
           delivery look -- or, with no face or no usable mouth there, from the second look, delivery.second_look_m
           further -- then one straight move (the utensil turned toward the mouth by at most max_turn_deg) to
           delivery.stop_m short of it, compliant (motion.impedance.approach). Exit 7: the user did not open their
           mouth (declined); no usable mouth from either look, or a mouth further than seated_tol_m from where the
           user is seated (user_mouth_m, the planner's user): the presentation refused (8).
  bite     the force at the tool from the joint torques, against a baseline at rest: above bite_n for bite_hold_s,
           then until it settles (the user has let go), at most bite_settle_s
  retract  straight back to the delivery look (the approach reversed), never stopped on torque (moving away is the
           safe direction); then one cuRobo path to food_detect_start that keeps the tip no further out from the base
           than where it starts or ends
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


def swing(arm: Arm) -> None:
    arm.travel(scene.captured("deliver"), "swing to the user", dps=float(DL.swing_dps), torque_limit=float(G.swing),
               max_tilt=95.0, carrying=True)


def measure_mouth(robot, arm: Arm, face, t_frame: float, rgb, out: Path, look: int) -> np.ndarray | None:
    """The mouth in the base frame from a frame taken here (the arm still), from its lips' depth; None if not usable."""
    from PIL import Image
    from rammp_feeding.scene import cam_to_tool
    D = robot.cameras["camera"].depth_median(5, t_frame - 0.2, wait_s=1.5, min_frames=3)
    m = face.mouth(rgb, D, arm.cam)
    lo, hi = (float(v) for v in DL.mouth_depth_range_m)
    why = ("no face" if m is None else "its range is only from the face's size (no depth on the lips)"
           if m["source"] != "depth" else "its depth %.0f mm is outside %.0f-%.0f" % (1000 * m["cam_xyz"][2], 1000 * lo, 1000 * hi)
           if not lo <= m["cam_xyz"][2] <= hi else None)
    (out / ("face_%d.json" % look)).write_text(json.dumps(dict(mouth=m, why=why), indent=1))
    Image.fromarray(rgb).save(out / ("look_%d.jpg" % look), quality=90)
    np.save(out / ("depth_%d.npy" % look), D)
    if why is not None:
        log.info("    the mouth not usable from look %d (%s)", look, why)
        return None
    T, R = arm.tool()
    mouth = T + R @ cam_to_tool(m["cam_xyz"])
    log.info("    the mouth at (%.3f %.3f %.3f), %.0f mm from the camera", *mouth, 1000 * m["cam_xyz"][2])
    return mouth


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
    for look in (1, 2):
        if look == 2:
            T, R = arm.tool()
            arm.straight(T + np.array(DL.second_look_m, float), R, float(DL.look_mps), "to the second look", max_tilt=95.0)
        if not face.find(cam, float(DL.face_find_s)):
            log.info("    no face in view from look %d", look)
            continue
        if wait_open:
            got = face.wait_open(cam, float(DL.open_mouth_timeout_s), out)
            if got is None:
                return 7
        else:
            got = cam.frame()
        mouth = measure_mouth(robot, arm, face, got[0], got[1], out, look)
        if mouth is not None:
            off = float(np.linalg.norm(mouth - scene.user_mouth()))
            if off > float(DL.seated_tol_m):                # the planner's world has the user where seated
                raise MoveError("the mouth is %.0f mm from where the user is seated (motion.free_space.user_mouth_m)"
                                % (1000 * off))
            approach(arm, mouth)
            return 0
    raise MoveError("no usable mouth from either look")


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


def retract(robot, arm: Arm) -> None:
    R_ = DL.retract
    T0, R0 = arm.tool()
    if abs(axis(R0)[2]) <= float(DL.level_tol):          # at the user or a look: straight back to the delivery look
        Te, Re = arm.tool(scene.captured("deliver"))
        L = float(np.linalg.norm(Te - T0))
        if L > float(R_.max_from_look_m):
            raise MoveError("%.0f mm from the delivery look -- not backing off from here" % (1000 * L))
        arm.straight(Te, Re, float(R_.back_off_mps), "back to the delivery look", torque_stop=False, max_tilt=95.0)
    goal = scene.overlook_joints(arm)
    reach = max(math.hypot(*arm.tip()[:2]), math.hypot(*arm.tip(goal)[:2])) + float(R_.reach_margin_m)
    arm.travel(goal, "back to food_detect_start", dps=float(R_.home_dps), torque_limit=float(G.retract), max_reach_m=reach)


def deliver(robot, arm: Arm, face, out: Path, stop_after: str = "", start_at: str = "swing", wait_open: bool = True) -> int:
    """swing -> present -> bite -> retract. 7: declined (retracted); 8: the presentation refused (retracted)."""
    stages = ["swing", "present", "bite", "retract"]
    run = stages[stages.index(start_at):]
    if "swing" in run:
        swing(arm)
        if stop_after == "swing":
            return 0
    if "present" in run:
        try:
            rc = present(robot, arm, face, out, wait_open)
        except MoveError as e:
            log.info("    the presentation stopped (%s) -- retracting", e)
            record.event("present_refused", why=str(e))
            retract(robot, arm)
            return 8
        if rc == 7:
            log.info("    no open mouth -- not approaching; retracting")
            retract(robot, arm)
            return 7
        if stop_after == "present":
            return 0
    if "bite" in run:
        bite(robot, arm, float(DL.bite_timeout_s))
        if stop_after == "bite":
            return 0
    retract(robot, arm)
    return 0
