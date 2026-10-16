#!/usr/bin/env python3
"""Food acquisition, one step at a time.

    python -m kinova.scripts.acquire pick                       # the default: over + re-measure + plunge + scoop
    python -m kinova.scripts.acquire over    --target /tmp/target.json
    python -m kinova.scripts.acquire plunge  --under /tmp/under.json [--then-scoop] [--validate-only]
    python -m kinova.scripts.acquire scoop   [--speed 25] [--validate-only]
    python -m kinova.scripts.acquire descend --under /tmp/under.json --to-above 0.030   # the stepped route
    python -m kinova.scripts.acquire punch   --under /tmp/under.json

feeding/acquire.sh (`make acquire`) runs them. Every motion goes through safe_move / motion.

  over     at the overlook height, translate until the utensil tip's vertical
           line passes through the target food. Refuses unless at
           food_detect_start and the move is small and level.
  plunge   from OVER straight down through the food to the bowl floor (or, packed food with no floor seen,
           a fixed depth below its top) in one trajectory: fast, easing to slow above the food.
  scoop    the food rises while the utensil turns vertical -> horizontal about it, pointing +Y (toward the
           user), backing off to the bowl's scoop end. A continuous joint path (a single Cartesian move is
           unreachable: J6 hits its limit), validated by the controller, run as one trajectory.
  pick     over -> fresh depth -> plunge -> scoop in this one process.
  descend  (stepped route) straight down until the tip is --to-above over the food's top, from a depth
           reading taken where the arm is.
  punch    (stepped route) straight down to the bowl floor, slowly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

import runlog
from kinova import scene
from kinova.connection import kinova_connection
from kinova.frames import R_from_kortex, kortex_from_R, rot_x
from kinova.motion import (WATCH_EXTRA_S, _ik, _waypoints, margin, max_step, record_plan, tip_in_tool, validate,
                           waypoint_list)
from kinova.workspace import workspace
from kinova.scripts.safe_move import (Joints, MoveError, TorqueGuard, _events, check_tool, fold, goto_cartesian,
                                      joints, tilt_deg, tool_pose)
from settings import cfg

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

log = runlog.get("acquire")

A_ = cfg.acquisition
MAX_OVER_M = float(A_.over_max_m)             # OVER stays inside the bowl, never a transit
BACKOFF_FROM = float(A_.scoop.backoff_from)   # fraction of the scoop before the back-off starts
PUNCH_FLOOR_M = float(cfg.utensil.punch_floor_margin_m)   # m above the bowl floor the tip stops (nominal)
PLUNGE_FAST = float(A_.plunge.fast_mps)            # m/s down to SLOW_ABOVE the food top
PLUNGE_SLOW = float(A_.plunge.slow_mps)            # m/s through the food, to the floor
SLOW_ABOVE = float(A_.plunge.slow_above_food_m)    # slow from this far above the food top (the overlook reads it high)
SCOOP_TORQUE_NM = float(cfg.guards.torque_step_nm.scoop)   # looser: the scoop's own acceleration steps the torque

STATE_DIR = Path(os.environ.get("RAMMP_STATE_DIR", "/tmp"))   # what the steps hand each other (the shell copies in/out)
PUNCH_DONE = STATE_DIR / "punch_done.json"    # the food piece on the tip, for a scoop run as its own step
SCOOP_LOG = STATE_DIR / "scoop_log.json"      # the food point's path during the scoop (feeding/acquire.sh keeps it)


# ------------------------------------------------------------------ inputs and plans

@dataclass
class Under:
    """What is under the utensil tip (feeding/measure_under.py): distances from the camera, m."""
    raw: dict                          # the reading as measured, kept whole for the record
    gap_top: float                     # tip -> food top
    top: float | None = None
    floor: float | None = None         # the bowl floor beside the food; None = no reading
    gap_floor: float | None = None     # tip -> floor
    height: float | None = None        # food height (floor - top)
    depth_source: str | None = None    # "da3-raw" = raw Depth Anything 3, no sensor depth
    filled_top: float = 0.0            # fraction of each reading from the Depth Anything 3 fallback
    filled_floor: float = 0.0

    @classmethod
    def from_dict(cls, u: dict) -> "Under":
        return cls(raw=u, gap_top=u["gap_top"], top=u.get("top"), floor=u.get("floor"),
                   gap_floor=u.get("gap_floor"), height=u.get("height"), depth_source=u.get("depth_source"),
                   filled_top=u.get("filled_top", 0), filled_floor=u.get("filled_floor", 0))

    @classmethod
    def load(cls, path: str | Path) -> "Under":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def has_floor(self) -> bool:
        return "floor" in self.raw


@dataclass
class FoodOnTip:
    """The food piece the punch / plunge went through: sets where its centre (the scoop's pivot) is."""
    height: float = float(A_.scoop.default_food_height_m)   # m
    floor_m: float = PUNCH_FLOOR_M     # how far above the bowl floor the tip stopped

    def save(self) -> None:
        PUNCH_DONE.write_text(json.dumps(dict(height=self.height, floor_m=self.floor_m)))

    @classmethod
    def load(cls) -> "FoodOnTip":
        d = json.loads(PUNCH_DONE.read_text()) if PUNCH_DONE.exists() else {}
        return cls(height=d.get("height", float(A_.scoop.default_food_height_m)), floor_m=d.get("floor_m", PUNCH_FLOOR_M))


@dataclass
class ScoopSettings:
    speed_dps: float = float(A_.scoop.speed_dps)   # fast on purpose: food slides off a slowly tilting utensil
    rot_exp: float = float(A_.scoop.rot_exp)       # turn angle = 90 * s**rot_exp
    waypoints: int = int(A_.scoop.waypoints)


@dataclass
class ScoopPlan:
    path: list[Joints]                     # the joint path, from the start state it was planned for
    o_piv: np.ndarray                      # the food point (pivot) in the tool frame
    piv_at: Callable[[float], np.ndarray]  # s -> where the food point should be (base frame)
    piv0: np.ndarray                       # the food point at the start


class DepthUnavailable(Exception):
    """No usable sensor depth under the tip after OVER: the shell falls back to the filled measurement."""


# ------------------------------------------------------------------ the stepped route and OVER

def _vertical_move(conn: KinovaConnection, down_m: float, speed: float, label: str) -> None:
    p = tool_pose(conn)
    goto_cartesian(conn, (p[0], p[1], p[2] - down_m), tuple(p[3:]), speed=speed, label=label)


def over(conn: KinovaConnection, target_path: str) -> None:
    """At the overlook height, slide until the utensil tip's vertical line passes through the target food."""
    off = max_step(joints(conn), scene.overlook_joints(conn))
    if off > float(cfg.motion.near_pose_deg):
        raise MoveError("not at food_detect_start (%.1f deg away) -- run: make detect" % off)
    goal = list(json.loads(Path(target_path).read_text())["tool_goal"])
    aim = cfg.aim_offset()                       # the utensil's tine offset + the food's, base frame
    goal[0] += aim[0]
    goal[1] += aim[1]
    p = tool_pose(conn)
    d = math.hypot(goal[0] - p[0], goal[1] - p[1])
    if abs(goal[2] - p[2]) > float(A_.over_level_tol_m) or d > MAX_OVER_M:
        raise MoveError("target is not a small level move (%.0f mm across) -- re-run make perceive"
                        % (1000 * d))
    runlog.event("over", goal=goal, aim_offset_m=list(aim), tine_offset_tool_m=cfg.get("utensil.tine_offset_tool_m", None),
                 food_offset_m=cfg.get("food.aim_offset_base_m", None), across_m=d)
    goto_cartesian(conn, goal, tuple(p[3:]), speed=float(A_.over_speed_mps), label="OVER")


def descend(conn: KinovaConnection, under_path: str, to_above: float) -> None:
    """Straight down until the tip is `to_above` over the food's top (never closer than approach_stop_m)."""
    u = Under.load(under_path)
    stop = max(to_above, float(A_.stepped.approach_stop_m))
    down = u.gap_top - stop
    if down <= 0:
        log.info("    already %.1f mm above the food, nothing to do", 1000 * u.gap_top)
        return
    if down > float(A_.stepped.descend_max_m):
        raise MoveError("descent of %.0f mm is not an approach -- re-measure" % (1000 * down))
    # fast to within slow_zone_m of the stop, then slow
    fast = max(0.0, down - float(A_.stepped.slow_zone_m))
    if fast > 0:
        _vertical_move(conn, fast, float(A_.stepped.fast_mps), "descend")
    _vertical_move(conn, down - fast, float(A_.stepped.slow_mps), "descend, slow")
    log.info("    tip now ~%.1f mm above the food top", 1000 * (u.gap_top - down))


def rise(conn: KinovaConnection) -> None:
    """Straight back up to food_detect_start's height, where it is (after a hover check): the way out of the bowl
    (cuRobo refuses a start inside it -- the bowl grown by the scene camera's error)."""
    p = tool_pose(conn)
    top = float(scene.overlook_pose()["tool_pose"]["z_m"])
    ws = workspace()
    off = math.hypot(p[0] - ws["bowl_xy"][0], p[1] - ws["bowl_xy"][1])
    if off > ws["bowl_radius"] + float(A_.over_max_m):
        raise MoveError("%.0f mm from the bowl's centre -- not a hover over it" % (1000 * off))
    up = top - p[2]
    if up <= float(cfg.delivery.arrived_m):
        log.info("    already at the overlook height")
        return
    if up > float(A_.stepped.descend_max_m) + float(A_.stepped.slow_zone_m) + 0.05:
        raise MoveError("%.0f mm below the overlook height -- not a hover" % (1000 * up))
    _vertical_move(conn, -up, float(A_.stepped.rise_mps), "rise")


def punch(conn: KinovaConnection, under_path: str, floor_m: float) -> None:
    """Straight down, slowly, to `floor_m` above the bowl floor (the stepped route's last move)."""
    u = Under.load(under_path)
    if not u.has_floor:
        raise MoveError("no floor reading under the tip -- cannot punch safely")
    down = u.gap_floor - floor_m
    lo, hi = A_.stepped.punch_range_m
    if not lo < down < hi:
        raise MoveError("punch depth %.1f mm is outside the sane range" % (1000 * down))
    log.info("    food %.1f mm tall: stop %.1f mm above the bowl floor -> %.1f mm down",
             1000 * u.height, 1000 * floor_m, 1000 * down)
    first = min(float(A_.stepped.slow_zone_m), down)
    _vertical_move(conn, first, float(A_.stepped.slow_mps), "punch")
    if down > first:
        _vertical_move(conn, down - first, float(A_.stepped.slow_mps), "punch, finish")
    FoodOnTip(height=u.height, floor_m=floor_m).save()


# ------------------------------------------------------------------ the plunge

@dataclass
class PlungeDepth:
    down: float                        # m straight down from where the tip is
    food: FoodOnTip                    # the piece as the scoop will see it
    mode: str                          # "floor": to the measured bowl floor; "below top": packed food, no floor seen


NF = A_.plunge.no_floor


def _plunge_depth(u: Under, floor_m: float, tip_z: float) -> PlungeDepth:
    """How far to plunge: to the bowl floor when it is seen beside the food, else (packed food, config
    acquisition.plunge.no_floor) a fixed depth below the food's top, never closer to the table than allowed."""
    lo_h = float(A_.plunge.food_height_range_m[0])
    floor_above_table = (tip_z - u.gap_floor - scene.table_height()) if u.has_floor else None
    if not u.has_floor:
        why = "no floor reading beside it"
    elif u.height <= lo_h:
        why = "the floor ring reads the neighbours (%.0f mm \"tall\")" % (1000 * u.height)
    elif floor_above_table > float(NF.floor_max_above_table_m):
        why = "the \"floor\" beside it is %.0f mm above the table: food" % (1000 * floor_above_table)
    else:
        why = None
    if why is None or not NF.enabled:
        down, height = _check_plunge(u, floor_m), u.height
        bf = cfg.get("bowl.floor_above_table_m", None)
        if bf is not None:                     # the bowl's own floor: a reading below it is through the bowl (glass)
            lowest = scene.table_height() + float(bf) + floor_m
            if tip_z - down < lowest:
                height = (tip_z - u.gap_top) - (lowest - floor_m)
                log.info("    the floor reads %.0f mm below the bowl's own floor (seen through it?): stopping %.0f mm "
                         "above that floor instead", 1000 * (lowest - (tip_z - down)), 1000 * floor_m)
                down = tip_z - lowest
        return PlungeDepth(down, FoodOnTip(height=height, floor_m=floor_m), "floor")
    if u.depth_source == "da3-raw" or u.filled_top > float(NF.refuse_if_top_filled_over):
        raise MoveError("packed food (%s) and its top is the Depth Anything 3 fallback -- too uncertain to plunge "
                        "below it; nothing moved" % why)
    below = float(NF.depth_below_top_m)
    down = u.gap_top + below
    lowest = scene.table_height() + float(NF.min_tip_above_table_m)
    if tip_z - down < lowest:
        down = tip_z - lowest
        below = down - u.gap_top
        log.info("    (the plunge stops %.0f mm above the table: %.0f mm into the food instead of %.0f)",
                 1000 * float(NF.min_tip_above_table_m), 1000 * below, 1000 * float(NF.depth_below_top_m))
        if below < float(NF.min_into_food_m):
            raise MoveError("packed food, but the table is only %.0f mm below its top -- not a plunge; nothing moved"
                            % (1000 * (below + float(NF.min_tip_above_table_m))))
    lo, hi = A_.plunge.sane_range_m
    if not lo < down < hi:
        raise MoveError("plunge of %.0f mm is outside the sane range (%.0f-%.0f) -- re-measure"
                        % (1000 * down, 1000 * float(lo), 1000 * float(hi)))
    log.info("    packed food (%s): plunging %.0f mm below its top", why, 1000 * below)
    return PlungeDepth(down, FoodOnTip(height=below, floor_m=0.0), "below top")


def _check_plunge(u: Under, floor_m: float) -> float:
    """The plunge depth (m) if the reading is trustworthy enough to stop at the bowl floor; else MoveError."""
    if not u.has_floor:
        raise MoveError("no floor reading under the tip -- cannot punch safely")
    if u.depth_source == "da3-raw" and (u.filled_floor > 0 or u.filled_top > 0):
        raise MoveError("the floor under the tip was measured on RAW Depth Anything 3 (no sensor depth to fit it to) "
                        "-- not accurate enough to stop at the bowl floor; nothing moved")
    if u.filled_floor > float(A_.plunge.refuse_if_floor_filled_over):
        # the fallback gets the level right but its local shape can be tens of mm off: never stop a plunge on it
        raise MoveError("%.0f%% of the floor reading under the tip is the Depth Anything 3 fallback -- its local "
                        "error (up to ~60 mm) is too large to stop at the bowl floor; nothing moved" % (100 * u.filled_floor))
    down = u.gap_floor - floor_m
    lo, hi = A_.plunge.sane_range_m
    if not lo < down < hi:
        raise MoveError("plunge of %.0f mm is outside the sane range (%.0f-%.0f) -- re-measure"
                        % (1000 * down, 1000 * float(lo), 1000 * float(hi)))
    lo, hi = A_.plunge.food_height_range_m
    if not lo < u.height < hi:
        raise MoveError("food %.0f mm tall is implausible -- re-measure" % (1000 * u.height))
    return down


def _vertical_path(conn: KinovaConnection, q0: Joints, T0: np.ndarray, rpy: tuple, down: float) -> tuple[list[Joints], list[float]]:
    """A continuous IK path straight down `down` m, and the distance travelled at each point."""
    n = int(math.ceil(down / float(A_.plunge.ik_step_m)))   # fewer IK calls: planning is dead time
    path, prev, dist = [q0], q0, [0.0]
    for k in range(1, n + 1):
        dk = down * k / n
        q = _ik(conn, (T0[0], T0[1], T0[2] - dk), rpy, prev)
        if q is None or max_step(q, prev) > 8:
            raise MoveError("no continuous plunge path at %.0f mm -- nothing moved" % (1000 * dk))
        prev = q
        path.append(q)
        dist.append(dk)
    return path, dist


def _plunge_speed(slow_at: float) -> Callable[[float], float]:
    """Tool speed (m/s) at distance x travelled: ramp in, fast, then ease (smoothstep) to slow at `slow_at`."""
    start, ramp_m, ease_m = float(A_.plunge.ramp_start_frac), float(A_.plunge.ramp_m), float(A_.plunge.ease_m)

    def v(x: float) -> float:
        ramp = min(1.0, start + x / ramp_m)
        w = min(1.0, max(0.0, (x - (slow_at - ease_m)) / ease_m))
        w = w * w * (3 - 2 * w)
        return ramp * (PLUNGE_FAST + (PLUNGE_SLOW - PLUNGE_FAST) * w)
    return v


def _run_watched(conn: KinovaConnection, wl: Any, total: float, label: str, on_sample: Callable[[Any], str | None],
                 **meta: Any) -> str | None:
    """Execute a validated trajectory; stop the arm on the first sample on_sample(feedback) objects to.
    Returns that objection (None = it ran to the end). Every sample goes to the run record."""
    ev = _events(conn)
    ev.clear()
    tel = runlog.Telemetry(label)
    why = None
    check_tool(conn)
    conn.base.ExecuteWaypointTrajectory(wl)
    t0 = time.time()
    try:
        while time.time() - t0 < total + WATCH_EXTRA_S:
            time.sleep(0.02)
            fb = conn.base_cyclic.RefreshFeedback()
            tel.add(fb)
            why = on_sample(fb)
            if why is None and any(e[0] == "ACTION_ABORT" for e in ev):
                why = "controller aborted"
            if why:
                conn.base.Stop()
                break
            if any(e[0] == "ACTION_END" for e in ev):
                break
    finally:
        tel.save(outcome=why or "done", planned_s=total, **meta)
    return why


def plunge(conn: KinovaConnection, under: str | Under, floor_m: float, scoop_with: ScoopSettings | None = None,
           validate_only: bool = False) -> None:
    """From OVER straight down to the depth _plunge_depth picks, in one move with no stop on the way.

    One joint trajectory along the vertical, fast then easing to PLUNGE_SLOW from SLOW_ABOVE the food top,
    validated by the controller, watched for torque: strictly above the food, allowing a firm piece's piercing
    (guards.torque_step_nm.plunge_in_food) once the tines are in it. With `scoop_with`, the scoop is planned before the plunge
    starts and runs straight after it: no pause at the bottom."""
    t_plan = time.time()
    u = under if isinstance(under, Under) else Under.load(under)     # a path: measured by the shell
    p = tool_pose(conn)
    q0 = joints(conn)
    T0 = np.array(p[:3])
    plan = _plunge_depth(u, floor_m, T0[2] + (R_from_kortex(*p[3:]) @ tip_in_tool())[2])
    down = plan.down
    slow_at = max(0.0, u.gap_top - SLOW_ABOVE)          # distance travelled when the slow zone starts
    path, dist = _vertical_path(conn, q0, T0, tuple(p[3:]), down)
    v = _plunge_speed(slow_at)
    raw = [x.position for x in conn.base_cyclic.RefreshFeedback().actuators]
    for scale in [float(f) for f in A_.plunge.speed_fallbacks]:
        durations = [(dist[k] - dist[k - 1]) / (scale * v(0.5 * (dist[k] + dist[k - 1]))) for k in range(1, len(dist))]
        total = sum(durations)
        wl = waypoint_list(raw, path, durations, prefix="p")
        errs = validate(conn, wl)
        if not errs:
            break
        log.info("    x%.2f speed refused by the controller: %s", scale, " ".join(str(errs[0]).split()))
    else:
        raise MoveError("controller refused the plunge -- nothing moved")
    record_plan("plunge", wl, speed_scale=scale, down_m=down, slow_at_m=slow_at)
    if plan.mode == "floor":
        log.info("    food %.1f mm tall: %.0f mm down to the floor + %.0f mm, slow from %.0f mm; %.1f s",
                 1000 * u.height, 1000 * down, 1000 * floor_m, 1000 * slow_at, total)
    else:
        log.info("    %.0f mm down, %.0f mm below the food's top, slow from %.0f mm; %.1f s",
                 1000 * down, 1000 * plan.food.height, 1000 * slow_at, total)
    runlog.event("plunge_plan", mode=plan.mode, down_m=down, floor_m=floor_m, food_height_m=plan.food.height,
                 slow_at_m=slow_at, seconds=total, speed_scale=scale, under=u.raw)
    scoop_plan = None
    if scoop_with is not None:                                # plan the scoop NOW, from where the plunge will end
        Te = np.array([T0[0], T0[1], T0[2] - down])
        scoop_plan = plan_scoop(conn, scoop_with, path[-1], Te, R_from_kortex(*p[3:]), plan.food)
    log.info("    plunge%s planned in %.1f s", " + scoop" if scoop_plan else "", time.time() - t_plan)
    if validate_only:
        log.info("    validate-only: controller accepted it, nothing moved")
        return
    # the contact guard, zoned by depth: strict above the food (anything there is unexpected: the rim, a wrong
    # reading), firm food's piercing allowed once the tines are in it -- the glass is kept off by where the plunge
    # ends (punch_floor_margin_m above the bowl's floor), not by the guard
    guard = TorqueGuard()
    strict, firm = guard.limit, float(cfg.guards.torque_step_nm.plunge_in_food)
    in_food_at = u.gap_top - float(A_.plunge.in_food_from_above_m)   # tool travel (m) where the tines meet the food

    def contact(fb: Any) -> str | None:
        inside = T0[2] - fb.base.tool_pose_z >= in_food_at
        guard.limit = firm if inside else strict
        hit = guard.update([x.torque for x in fb.actuators])
        return ("joint %d torque step %s" % (hit[0] + 1, "in the food" if inside else "above the food")) if hit else None
    t0 = time.time()
    why = _run_watched(conn, wl, total, "plunge", contact)
    if why:
        raise MoveError("plunge stopped: %s" % why)
    log.info("    plunge done in %.1f s", time.time() - t0)
    plan.food.save()
    if scoop_plan is not None:
        run_scoop(conn, scoop_with, scoop_plan, validate_only)   # no pause at the bottom: already planned


def pick(conn: KinovaConnection, target_path: str, under_path: str, floor_m: float, scoop_with: ScoopSettings,
         validate_only: bool = False) -> None:
    """OVER -> fresh depth from the camera server -> plan plunge + scoop -> plunge -> scoop, all in this
    process (no restart or reconnection, so the arm does not stand still between them). The floor is
    re-measured from above the food after OVER: the overlook frame's reading is less reliable."""
    from feeding.depth_grab import fetch
    from feeding.measure_under import measure
    over(conn, target_path)
    t = time.time()                                  # the arm has stopped at OVER: frames from now on count
    d = fetch(int(cfg.perception.depth_frames), since=t)
    if d is None:
        raise DepthUnavailable("camera server: no fresh frames after %d tries" % int(cfg.perception.depth_tries))
    runlog.save_arrays("depth_at_over", depth_mm=d)
    try:
        u = measure(d, float(A_.over_depth_guess_m))
    except SystemExit as e:                    # measure_under's own "no depth return under the tip"
        raise DepthUnavailable(str(e))
    runlog.save_json("under_over", u)
    if "floor" not in u and not NF.enabled:
        raise DepthUnavailable("no floor reading beside the food")
    if "floor" in u:
        log.info("  under the tip: top %.1f mm (tip %.1f mm above it), floor %.1f mm (tip %.1f mm above it), food %.1f mm tall"
                 "   [fresh depth in %.1f s]", 1000 * u["top"], 1000 * u["gap_top"], 1000 * u["floor"], 1000 * u["gap_floor"],
                 1000 * u["height"], time.time() - t)
    else:
        log.info("  under the tip: top %.1f mm (tip %.1f mm above it), no floor visible beside it   [fresh depth in %.1f s]",
                 1000 * u["top"], 1000 * u["gap_top"], time.time() - t)
    # handed on in memory, not via under_path: that file is the shell's (host-owned; the container may not write it)
    try:
        (STATE_DIR / "under_over.json").write_text(json.dumps(u))       # feeding/acquire.sh copies it to the bite
    except OSError as e:
        log.info("    (could not keep the reading in %s: %s -- it is in the run record)", STATE_DIR, e)
    plunge(conn, Under.from_dict(u), floor_m, scoop_with, validate_only)


# ------------------------------------------------------------------ the scoop

def plan_scoop(conn: KinovaConnection, st: ScoopSettings, q0: Joints, T0: np.ndarray, R0: np.ndarray,
               food: FoodOnTip) -> ScoopPlan:
    """Plan the scoop from a given arm state (the current one, or where the plunge WILL end -- so the
    plunge can flow straight into it with no planning pause at the bottom)."""
    tip = tip_in_tool()
    # the punch leaves the tip floor_m above the floor; the food's centre is at h/2
    food_up = max(0.0, food.height / 2.0 - food.floor_m)
    o_piv = tip - np.array([0, 0, food_up])
    piv0 = T0 + R0 @ o_piv
    end_z = scene.overlook_pose()["tool_pose"]["z_m"] + (R0 @ tip)[2]        # tip height at the overlook
    rise = end_z - piv0[2]
    # With a calibrated scoop_end, the food also backs off to where that pose holds it: the rotation still
    # starts at once, the back-off mostly once the utensil is past steep (from BACKOFF_FROM of the scoop).
    se = scene.scoop_end()                       # moved with the bowl / table
    back = np.zeros(3)
    if se:
        pivE = np.array(se["tool_m"]) + rot_x(90.0) @ R0 @ o_piv
        back = pivE - piv0
        rise = back[2]
        back[2] = 0.0
        if np.linalg.norm(back) > float(A_.scoop.max_backoff_m):
            raise MoveError("backing off %.0f mm is too far -- not the bowl this was built for" % (1000 * np.linalg.norm(back)))

    def piv_at(s: float) -> np.ndarray:
        x = min(1.0, max(0.0, (s - BACKOFF_FROM) / (1 - BACKOFF_FROM)))
        return piv0 + np.array([0, 0, rise * s]) + x * x * (3 - 2 * x) * back

    def at(s: float) -> tuple[np.ndarray, tuple[float, float, float]]:
        R = rot_x(90.0 * s ** st.rot_exp) @ R0
        return piv_at(s) - R @ o_piv, kortex_from_R(R)

    def along(qe: Joints) -> list[Joints] | None:
        """A continuous joint path, the IK guess blended from the start toward the end shape qe."""
        prev, cand = q0, [q0]
        for k in range(1, st.waypoints + 1):
            s = k / st.waypoints
            T, r = at(s)
            q = _ik(conn, T, r, [fold(x + fold(y - x) * s) for x, y in zip(q0, qe)])
            if q is None or max_step(q, prev) > float(A_.scoop.max_joint_step_deg):
                return None
            prev = q
            cand.append(q)
        return cand

    # the calibrated end shape first (the one delivery was built from) -- the search only if it fails
    path = along(se["joints_deg"]) if se else None
    if path is None:
        path = _search_scoop(conn, q0, at, along)
    if path is None:
        raise MoveError("no continuous scoop path from here -- nothing moved")
    log.info("    scoop planned: food rises %.0f mm%s while turning 90 deg; %d waypoints, min joint margin %.1f deg",
             1000 * rise, ", backing off %.0f mm" % (1000 * np.linalg.norm(back)) if se else "", st.waypoints,
             min(margin(q) for q in path))
    runlog.event("scoop_plan", rise_m=rise, backoff_m=float(np.linalg.norm(back)), waypoints=st.waypoints,
                 min_margin_deg=min(margin(q) for q in path), food_height_m=food.height)
    return ScoopPlan(path=path, o_piv=o_piv, piv_at=piv_at, piv0=piv0)


def _search_scoop(conn: KinovaConnection, q0: Joints, at: Callable, along: Callable) -> list[Joints] | None:
    """No calibrated scoop end (or it failed): try IK end shapes from a grid of seeds, nearest first."""
    T, r = at(1.0)
    ends = []
    for j1, j2, j3, j4, j6, flip in itertools.product((q0[0], 150, 120, 100), (-30, 0, 30, 60),
                                                      (q0[2], 30, 60, 85), (-135, -110),
                                                      (-85, -50, 50, 85), (0, 1)):
        g = [j1, j2, j3, j4, q0[4], j6, q0[6]]
        if flip:
            g[4], g[6] = fold(g[4] + 180), fold(g[6] + 180)
        q = _ik(conn, T, r, g)
        if q and margin(q) > 8:
            ends.append(q)
    ends.sort(key=lambda q: max_step(q, q0))
    for qe in ends[:10]:
        path = along(qe)
        if path:
            return path
    return None


def scoop(conn: KinovaConnection, st: ScoopSettings, validate_only: bool = False) -> None:
    """The scoop as its own step, from where the arm is (after punch / plunge)."""
    p = tool_pose(conn)
    run_scoop(conn, st, plan_scoop(conn, st, joints(conn), np.array(p[:3]), R_from_kortex(*p[3:]), FoodOnTip.load()),
              validate_only)


def run_scoop(conn: KinovaConnection, st: ScoopSettings, plan: ScoopPlan, validate_only: bool = False) -> None:
    """Validate and run a planned scoop, watching the food point: off its path or dropping = stop."""
    path = [joints(conn)] + plan.path[1:]                    # start from where the arm actually is
    # one smooth trajectory, validated by the controller; slower only if refused
    wl, speed = _waypoints(conn, path, st.speed_dps, "scoop")
    total = sum(w.angular_waypoint.duration for w in wl.waypoints)
    log.info("    scoop at %.0f deg/s: %.1f s", speed, total)
    if validate_only:
        log.info("    validate-only: controller accepted it, nothing moved")
        return
    # watch: the food must stay on its planned path (its vertical line, if not backing off) and not drop
    plan_piv = np.array([plan.piv_at(k / 400) for k in range(401)])
    guard = TorqueGuard(limit=SCOOP_TORQUE_NM)
    food_log: list[dict] = []
    worst = 0.0
    pv = plan.piv0
    t0 = time.time()

    def food_on_path(fb: Any) -> str | None:
        nonlocal worst, pv
        b = fb.base
        pv = np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z]) + \
            R_from_kortex(b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z) @ plan.o_piv
        near = plan_piv[np.argmin(np.linalg.norm(plan_piv - pv, axis=1))]
        off = math.hypot(*(pv[:2] - near[:2]))
        worst = max(worst, off)
        food_log.append(dict(t=time.time() - t0, piv=pv.tolist(), off=off))
        why = None
        if off > float(cfg.guards.scoop_food_off_path_m):
            why = "food point %.1f mm off its planned path" % (1000 * off)
        elif pv[2] < near[2] - float(cfg.guards.scoop_food_drop_m):
            why = "food point %.1f mm below its planned path" % (1000 * (near[2] - pv[2]))
        hit = guard.update([x.torque for x in fb.actuators])
        if hit and not why:
            why = "joint %d torque step" % (hit[0] + 1)
        return why
    why = _run_watched(conn, wl, total, "scoop", food_on_path, speed_dps=speed)
    SCOOP_LOG.write_text(json.dumps(food_log))
    runlog.save_json("scoop_food_path", dict(planned=plan_piv, actual=food_log))
    p = tool_pose(conn)
    ax = R_from_kortex(*p[3:]) @ np.array([0, 0, 1.0])
    if why:
        raise MoveError("scoop stopped: " + why)
    runlog.event("scoop", rose_m=float(pv[2] - plan.piv0[2]), worst_off_path_m=worst, axis=ax,
                 seconds=time.time() - t0)
    log.info("    food rose %.1f mm, worst %.1f mm off its path, utensil axis (%.2f %.2f %.2f), %.1f s",
             1000 * (pv[2] - plan.piv0[2]), 1000 * worst, *ax, time.time() - t0)


# ------------------------------------------------------------------ command line

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["over", "descend", "rise", "punch", "plunge", "scoop", "pick"])
    ap.add_argument("--target", default=str(STATE_DIR / "target.json"))
    ap.add_argument("--under", default=str(STATE_DIR / "under.json"), help="measure_under.py output")
    ap.add_argument("--to-above", type=float, default=float(A_.stepped.approach_stop_m), help="descend: stop this far above the food top")
    ap.add_argument("--floor-m", type=float, default=PUNCH_FLOOR_M, help="punch: stop this far above the bowl floor")
    ap.add_argument("--speed", type=float, default=float(A_.scoop.speed_dps),
                    help="scoop: deg/s. Fast on purpose: food slides off a slowly tilting utensil")
    ap.add_argument("--rot-exp", type=float, default=float(A_.scoop.rot_exp),
                    help="scoop: turn angle = 90 * progress**rot_exp. <1 starts turning the moment the food lifts")
    ap.add_argument("--n", type=int, default=int(A_.scoop.waypoints), help="scoop waypoints")
    ap.add_argument("--validate-only", action="store_true", help="scoop: plan and let the controller validate; do not move")
    ap.add_argument("--then-scoop", action="store_true", help="plunge: plan the scoop first and run it straight after")
    a = ap.parse_args()
    log_ = runlog.setup("acquire")
    st = ScoopSettings(speed_dps=a.speed, rot_exp=a.rot_exp, waypoints=a.n)
    steps: dict[str, Callable[[KinovaConnection], None]] = dict(
        over=lambda c: over(c, a.target),
        descend=lambda c: descend(c, a.under, a.to_above),
        rise=rise,
        punch=lambda c: punch(c, a.under, a.floor_m),
        plunge=lambda c: plunge(c, a.under, a.floor_m, st if a.then_scoop else None, a.validate_only),
        scoop=lambda c: scoop(c, st, a.validate_only),
        pick=lambda c: pick(c, a.target, a.under, a.floor_m, st, a.validate_only),
    )
    runlog.event("start", command=a.step, args=vars(a))
    with kinova_connection() as conn:
        try:
            steps[a.step](conn)
            p = tool_pose(conn)
            tilt = tilt_deg(conn)
            log_.info("  %s done: tool (%.4f %.4f %.4f), utensil %.1f deg off vertical", a.step, p[0], p[1], p[2], tilt)
            runlog.event("done", command=a.step, tool=p, tilt_deg=tilt)
        except DepthUnavailable as e:
            log_.error("\n  no sensor depth under the tip (%s) -- the shell falls back to the filled measurement", e)
            runlog.event("stopped", command=a.step, reason="depth unavailable: %s" % e, rc=6)
            return 6
        except MoveError as e:
            log_.error("\n  STOPPED: %s", e)
            runlog.event("stopped", command=a.step, reason=str(e), rc=1)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
