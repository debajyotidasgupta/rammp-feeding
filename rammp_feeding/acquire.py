"""Food acquisition from food_detect_start, the wrist camera's target in hand.

  over     at the overlook height, a level slide until the tip's vertical line passes through the target piece
           (refused unless at food_detect_start, small and level: a correction inside the bowl, never a transit)
  plunge   straight down beside the piece to the bowl floor (never under the bowl profile's own floor: a clear bowl
           reads the table through it), in one move: fast, easing to slow above the piece; the contact guard strict
           above the food, a firm piece's piercing allowed once the tines are in it
  scoop    the piece rises while the utensil turns from vertical to level about it, backing off to the captured
           scoop_end (where the swing starts); stopped if the food point leaves its path or drops
  pick     over -> a fresh depth reading from there -> plunge -> scoop, planned before the plunge starts (no pause);
           a plunge that ran keeps the piece's height and where it ended with the bite (acquire/food.json): a scoop
           run on its own (--start-at scoop) starts from there
  descend / rise   the hover route for aim checks (tip_px): from OVER, down to a hover above the piece (never under
           the bowl's own floor), and straight back up
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rammp_feeding import record, scene
from rammp_feeding.arm import Arm, MoveError, TorqueGuard
from rammp_feeding.config import cfg
from rammp_feeding.geometry import max_step, rot
from rammp_feeding.perception import under_tip

log = logging.getLogger("rammp")
A = cfg.acquisition
G = cfg.guards.torque_step_nm


@dataclass
class Food:
    """The piece on the tines, as the plunge measured it: where its centre (the scoop's pivot) is."""
    height: float                       # m, the floor to its top
    floor_m: float                      # how far above the floor the tip stopped


def measure(robot, arm: Arm, out: Path, name: str) -> dict:
    """What is under the tip, from depth frames that arrived after the arm stopped here."""
    D = robot.cameras["camera"].depth_median(int(cfg.perception.depth_frames), time.time(),
                                             wait_s=float(cfg.perception.depth_wait_s))
    np.save(out / ("depth_%s.npy" % name), D)
    try:
        u = under_tip(D, arm)
    except ValueError as e:
        raise MoveError(str(e))
    (out / ("under_%s.json" % name)).write_text(json.dumps(u, indent=1))
    log.info("  under the tip: the top %.0f mm down%s", 1000 * u["gap_top"],
             ", the floor %.0f mm down, the food %.0f mm tall" % (1000 * u["gap_floor"], 1000 * u["height"])
             if "floor_z" in u else ", no floor seen")
    return u


def over(arm: Arm, target: dict) -> None:
    q = arm.q()
    off = max_step(q, scene.overlook_joints(arm))
    if off > float(cfg.motion.near_pose_deg):
        raise MoveError("not at food_detect_start (%.1f deg away) -- rammp detect" % off)
    T, R = arm.tool(q)
    goal = np.array(target["tool_goal"], float)
    d = math.hypot(*(goal[:2] - T[:2]))
    if abs(goal[2] - T[2]) > float(A.over_level_tol_m) or d > float(A.over_max_m):
        raise MoveError("the target is not a small level move (%.0f mm across) -- rammp perceive again" % (1000 * d))
    record.event("over", goal=goal, across_m=d)
    arm.straight(goal, R, float(A.over_speed_mps), "OVER")


def plunge_depth(u: dict, tip_z: float) -> tuple[float, Food]:
    """(m down from here, the piece as the scoop will see it): to the bowl floor beside the piece, never under the bowl's
    own floor (bowl.floor_above_table_m: a clear bowl's depth reads the table beneath it)."""
    if "floor_z" not in u:
        raise MoveError("no floor reading beside the piece -- not plunging")
    floor_m = float(cfg.utensil.punch_floor_margin_m)
    down, height = u["gap_floor"] - floor_m, u["height"]
    if not float(A.plunge.food_height_range_m[0]) < height < float(A.plunge.food_height_range_m[1]):
        raise MoveError("food %.0f mm tall is implausible -- re-measure" % (1000 * height))
    lowest = scene.workspace()["table_z"] + float(cfg.bowl.floor_above_table_m) + floor_m
    if tip_z - down < lowest:
        log.info("    the floor reads %.0f mm under the bowl's own floor (seen through it): stopping above that floor",
                 1000 * (lowest - (tip_z - down)))
        down, height = tip_z - lowest, (tip_z - u["gap_top"]) - (lowest - floor_m)
    lo, hi = (float(v) for v in A.plunge.sane_range_m)
    if not lo < down < hi:
        raise MoveError("a plunge of %.0f mm is outside the sane range (%.0f-%.0f) -- re-measure" % (1000 * down, 1000 * lo, 1000 * hi))
    return down, Food(height, floor_m)


def _speed(slow_at: float):
    """The plunge's tool speed at distance x: a ramp in, fast, a smoothstep to slow from slow_at."""
    P = A.plunge
    fast, slow = float(P.fast_mps), float(P.slow_mps)

    def v(x: float) -> float:
        w = min(1.0, max(0.0, (x - (slow_at - float(P.ease_m))) / float(P.ease_m)))
        return min(1.0, float(P.ramp_start_frac) + x / float(P.ramp_m)) * (fast + (slow - fast) * w * w * (3 - 2 * w))
    return v


def plunge(arm: Arm, u: dict, out: Path, scoop_after: bool = True) -> Food:
    q0 = arm.q()
    T0, R0 = arm.tool(q0)
    down, food = plunge_depth(u, float((T0 + R0 @ arm.tip_tool)[2]))
    n = int(math.ceil(down / float(A.plunge.ik_step_m)))
    path, err = arm.homotopy(q0, lambda s: (T0 - np.array([0, 0, down * s]), R0), n)
    if path is None:
        raise MoveError("no continuous plunge path (%s) -- nothing moved" % err)
    slow_at = max(0.0, u["gap_top"] - float(A.plunge.slow_above_food_m))
    v = _speed(slow_at)
    dist = [down * k / n for k in range(n + 1)]
    durations = [(dist[k] - dist[k - 1]) / v(0.5 * (dist[k] + dist[k - 1])) for k in range(1, n + 1)]
    scoop = plan_scoop(arm, path[-1], T0 - np.array([0, 0, down]), R0, food) if scoop_after else None
    log.info("    plunge: %.0f mm down, slow from %.0f mm; %.1f s", 1000 * down, 1000 * slow_at, sum(durations))
    record.event("plunge_plan", down_m=down, food_height_m=food.height, slow_at_m=slow_at, under=u)
    # the contact guard, zoned by depth: strict above the food (the rim, a wrong reading), a firm piece's piercing
    # allowed once the tines are in it; the bowl is kept off by where the plunge ends, not by the guard
    guard, in_food = TorqueGuard(float(G.default)), u["gap_top"] - float(A.plunge.in_food_from_above_m)

    def contact(q, tau, t):
        inside = T0[2] - arm.tool(q)[0][2] >= in_food
        guard.limit = float(G.plunge_in_food) if inside else float(G.default)
        hit = guard.update(tau)
        return ("joint %d torque step %s" % (hit[0] + 1, "in the food" if inside else "above the food")) if hit else None
    arm.run(path, "plunge", durations=durations, torque_stop=False, check=contact)
    (out / "food.json").write_text(json.dumps(dict(height=food.height, floor_m=food.floor_m, joints_deg=arm.q()), indent=1))
    if scoop is not None:
        run_scoop(arm, scoop)
    return food


# ------------------------------------------------------------------ the scoop
def plan_scoop(arm: Arm, q0, T0, R0, food: Food) -> dict:
    """From a given state (here, or where the plunge will end): the food point rises while the utensil turns 90 deg
    about it (the turn = 90 * s**rot_exp: it starts at once), backing off to the scoop end once past steep."""
    S = A.scoop
    tip = arm.tip_tool
    o_piv = tip - np.array([0, 0, max(0.0, food.height / 2.0 - food.floor_m)])
    piv0 = T0 + R0 @ o_piv
    se = scene.scoop_end(arm)
    pivE = np.array(se["tool_m"]) + rot("x", 90.0) @ R0 @ o_piv
    back = pivE - piv0
    rise = back[2]
    back[2] = 0.0
    if np.linalg.norm(back) > float(S.max_backoff_m):
        raise MoveError("backing off %.0f mm is too far -- not the bowl this was built for" % (1000 * np.linalg.norm(back)))
    frm = float(S.backoff_from)

    def piv_at(s):
        x = min(1.0, max(0.0, (s - frm) / (1 - frm)))
        return piv0 + np.array([0, 0, rise * s]) + x * x * (3 - 2 * x) * back

    def at(s):
        R = rot("x", 90.0 * s ** float(S.rot_exp)) @ R0
        return piv_at(s) - R @ o_piv, R
    n, jump = int(S.waypoints), float(S.max_joint_step_deg)
    path, _ = arm.homotopy(q0, at, n, toward=se["joints_deg"], max_jump=jump)
    if path is None:
        raise MoveError("no continuous scoop path from here -- nothing moved")
    log.info("    scoop: the food rises %.0f mm, backing off %.0f mm, turning 90 deg; joint margin >= %.1f deg",
             1000 * rise, 1000 * np.linalg.norm(back), min(arm.margin(q) for q in path))
    return dict(path=path, o_piv=o_piv, plan=np.array([piv_at(k / 400) for k in range(401)]), piv0=piv0)


def run_scoop(arm: Arm, plan: dict) -> None:
    path = [arm.q()] + plan["path"][1:]
    guard, worst, food_log = TorqueGuard(float(G.scoop)), [0.0], []

    def food_on_path(q, tau, t):
        T, R = arm.tool(q)
        pv = T + R @ plan["o_piv"]
        near = plan["plan"][np.argmin(np.linalg.norm(plan["plan"] - pv, axis=1))]
        off = math.hypot(*(pv[:2] - near[:2]))
        worst[0] = max(worst[0], off)
        food_log.append([t, *pv])
        if off > float(cfg.guards.scoop_food_off_path_m):
            return "the food point %.0f mm off its path" % (1000 * off)
        if pv[2] < near[2] - float(cfg.guards.scoop_food_drop_m):
            return "the food point %.0f mm under its path" % (1000 * (near[2] - pv[2]))
        hit = guard.update(tau)
        return "joint %d torque step" % (hit[0] + 1) if hit else None
    try:
        arm.run(path, "scoop", dps=float(A.scoop.speed_dps), torque_stop=False, max_tilt=95.0, check=food_on_path)
    finally:
        record.save_arrays("scoop_food_path", planned=plan["plan"], actual=np.array(food_log, float))
    log.info("    scoop: worst %.1f mm off the food's path", 1000 * worst[0])


# ------------------------------------------------------------------ the hover route
def descend(arm: Arm, u: dict, to_above: float) -> None:
    """Straight down, in one move, until the tip is to_above over the piece's top, or over the bowl's own floor where
    the depth reads lower (nothing there, or the table through a clear bowl)."""
    S = A.stepped
    floor = scene.workspace()["table_z"] + float(cfg.bowl.floor_above_table_m)
    down = float(arm.tip()[2]) - (max(u["top_z"], floor) + to_above)
    if down <= 0:
        return
    if down > float(S.descend_max_m):
        raise MoveError("a descent of %.0f mm is not an approach -- re-measure" % (1000 * down))
    T, R = arm.tool()
    arm.straight(T - np.array([0, 0, down]), R, float(S.descend_mps), "descend")


def rise(arm: Arm) -> None:
    """Straight back up to food_detect_start's height, where it is (after a hover)."""
    T, R = arm.tool()
    ws = scene.workspace()
    if math.hypot(T[0] - ws["bowl_xy"][0], T[1] - ws["bowl_xy"][1]) > ws["bowl_radius"] + float(A.over_max_m):
        raise MoveError("not over the bowl -- not a hover")
    top = scene.overlook_goal(arm)[0][2]
    if top - T[2] > 0.003:
        arm.straight(np.array([T[0], T[1], top]), R, float(A.stepped.rise_mps), "rise")


def acquire(robot, arm: Arm, target: dict, out: Path, stop_after: str = "", start_at: str = "over") -> int:
    """over -> measure -> plunge + scoop (the default), or the hover route (stop_after hover30 / hover10)."""
    H = A.stepped
    if start_at not in ("over", "plunge", "scoop"):
        raise ValueError("acquire starts at over, plunge or scoop, not %r" % start_at)
    if start_at == "scoop":                               # the tines in the piece, where this bite's plunge left them
        f = out / "food.json"
        if not f.exists():
            raise FileNotFoundError("no plunge in this bite (%s) -- the scoop starts where a plunge ends" % f)
        kept = json.loads(f.read_text())
        q = arm.q()
        if max_step(q, kept["joints_deg"]) > float(cfg.motion.near_pose_deg):
            raise MoveError("not where this bite's plunge ended (%.1f deg away) -- not scooping from here"
                            % max_step(q, kept["joints_deg"]))
        T, R = arm.tool(q)
        run_scoop(arm, plan_scoop(arm, q, T, R, Food(kept["height"], kept["floor_m"])))
        return 0
    if start_at == "over":
        over(arm, target)
        if stop_after == "over":
            return 0
    if stop_after in ("hover30", "hover10"):              # one reading from here, one move down
        descend(arm, measure(robot, arm, out, "to_" + stop_after), float(H.hover_m[stop_after]))
        return 0
    u = measure(robot, arm, out, "over")
    plunge(arm, u, out, scoop_after=stop_after != "plunge")
    return 0
