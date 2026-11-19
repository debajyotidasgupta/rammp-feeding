"""The fork on its stand (utensil.pick): it hangs tines down, a block on its handle with a rectangular opening on top.
The fingers go into the opening closed and open there, holding the block from inside; lifting it lifts the fork off
the stand. The gripper is commanded here and nowhere else. Two kinds of holder (utensil.pick.holder): side -- the fork
leaves and comes back through an open side, its seat kept against a taped strip on the stand; top -- straight up out of
it and straight back down into the same opening.

The pick (rammp pick-fork), from feeding_start with the gripper empty, step by step (--stop-after / --start-at):
  look      the fingers closed; a planned path to fork_look (the wrist camera over the stand, pointing down)
  hover     the opening found (find); the fingers pointing straight down, the axis they open along
            (fingers_open_along_tool) along the opening's long side (refused past max_turn_deg from fork_look's; of the
            two ways round, the one whose way in keeps the joints furthest from their limits); a planned path to that
            pose above_m over the rim, the way in and up checked from the arm shape it ends in (min_room_deg)
  in        straight down into the opening, insert_m under the rim
  hold      the fingers opened: where the arm node serves their position, refused unless the opening's end walls stopped
            them lock_margin short of open (else not in it: closed again)
  lift      straight up lift_m, the block off its seat
  out       a side holder: straight out through its open side (slide_m, the way open_side points), then fork_look again,
            the empty stand's strip found (strip) and the seat kept in its frame, with where the fork hangs in the
            camera's frame (fork_px). Both: the fingers' firm hold for the meal (hold_force). Then a planned path home
The put-back (rammp put-fork), holding the fork:
  back      the fingers soft again. A top holder: a planned path back to the joints the lift ended at. A side holder: a
            planned path to fork_look, the strip found and the seat placed from it (refused when the stand moved more
            than max_stand_move_m or turned more than max_stand_turn_deg, or the fork hangs more than grip_shift_tol_m
            from where it hung at the pick), a planned path beside the open side, straight in through it
  lower     straight down onto the seat (the guard's stop within seat_tol_m of its depth: seated, the holder taking the
            fork's weight)
  release   the fingers closed: refused unless they moved lock_margin from where the walls held them
  up        straight up out of the opening, above_m over the rim. Then a planned path home
recover   after a stop with the fingers in the opening: straight up out of it (holding the fork: to where the lift ends;
          empty: above_m over the rim) -- the first thing a pick or put-back from its first step does

find      from fork_look: one colour frame and a depth median from frames after the arm stopped; SAM3 on
          utensil.pick.prompts (the opening, whatever colour the block is); for the best-scored mask, the rim -- the
          block's top: the median height of the depth points in a ring just outside the mask (rim_ring_px), those within
          inlier_m of it -- or, where the depth cannot read the block, the height that gives the mask the opening's
          ruler size (opening_m); the mask's pixels laid on the rim's height: their centroid the opening's centre, their
          principal axis its long side, their extent its size (outside min_opening_m .. max_opening_m: the next mask);
          refused when the finger ends, insert_m under the rim, would come within floor_clear_m of the opening's floor.
          The centre shifted by look_offset_m and the long side turned by look_turn_deg: the site's correction for the
          camera's mount as seen from fork_look, where every find is made
strip     the stand's marker, from fork_look: SAM3 on stand_prompts (the tape along the stand's arm to its seat); its
          depth points between the table and the fingers' ends, outside the utensil's image region; the best-scored
          with stand_min_points: its post end and its direction toward the seat, on the base's level; the median of
          stand_reads readings (looked at again while they spread)

The utensil's state (state_dir/utensil_state.json): holding (the fork in the gripper), in_holder (the fingers in the
opening: written before going in, cleared once out), the opening and the fingers' orientation (hole, R: what a later
step and recover need), where the walls held the fingers (held_at), the joints where the lift ended (lifted_joints),
the seat in the strip's frame (seat) and where the stand was last seen (stand: the collision world keeps its block).
With utensil.pick.use false the fork is put in by hand and is always held.

While the routine runs the arm holds AtStand: the empty gripper (its fingertips the tip) or the fork (its measured
tip), kept above the table but in the stand's own column. The planner's world is the meal's without the stand's block
and with the table at its own height (the gripper goes down to the stand: our checks keep the fingertips and the fork
over the table); the moves into and out of the holder are ours.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from rammp_feeding import record, scene, sensing
from rammp_feeding.arm import Arm, Held, MoveError
from rammp_feeding.config import cfg, read_json, state_dir
from rammp_feeding.geometry import max_step

log = logging.getLogger("rammp")
Step = Callable[[str, float, str], None]
DOWN = np.array([0.0, 0.0, -1.0])


def P():
    return cfg.utensil.pick


def used() -> bool:
    """A stand to pick the fork from (utensil.pick.use), read live."""
    return bool(cfg.get("utensil.pick.use", False))


def top() -> bool:
    """The holder lets the fork out straight up (utensil.pick.holder top), not through an open side (side)."""
    return str(P().get("holder", "side")) == "top"


# ------------------------------------------------------------------ the utensil's state
def _state_file() -> Path:
    return state_dir() / "utensil_state.json"


def state() -> dict:
    """utensil_state.json ({} when there is none: the fork on its stand)."""
    f = _state_file()
    return read_json(f) if f.exists() else {}


def save_state(s: dict) -> None:
    _state_file().write_text(json.dumps(dict(s, t=time.time()), indent=1))


def holding() -> bool:
    """The fork in the gripper, as the utensil's state says."""
    return bool(state().get("holding"))


def need() -> None:
    """The routine's own values set (RuntimeError naming what is not): a stand (utensil.pick.use), the table it keeps
    above (workspace.table_height_m), the finger ends' ruler and the wrist camera's mount (fingertips_in_tool)."""
    missing = [k for k in ("workspace.table_height_m", "utensil.pick.glass_to_fingertips_m") +
               (() if top() else ("utensil.pick.open_side",)) if cfg.get(k, None) is None]
    if not used():
        missing.insert(0, "utensil.pick.use (no stand to pick the fork from here)")
    if scene.CAM_IN_TOOL is None:
        missing.append("the wrist camera's mount (rammp calibrate-wrist)")
    if missing:
        raise RuntimeError("the fork's stand: not set up on this robot yet (not measured: %s; docs/porting.md)"
                           % ", ".join(missing))


def fingertips_in_tool() -> np.ndarray:
    """The closed fingers' ends in the tool frame: on the tool's axis (the 2F-85 closes them on it), as far along the
    wrist camera's axis from its glass as utensil.pick.glass_to_fingertips_m says (glass_to_tip_m's ruler: the camera's
    glass_to_optical_m added, through the camera's mount)."""
    need()
    d = float(P().glass_to_fingertips_m) + float(cfg.camera.get("glass_to_optical_m", 0.0))
    a = scene.CAM_ROT[:, 2]                             # the camera's axis, tool frame
    return np.array([0.0, 0.0, (d + float(scene.CAM_IN_TOOL @ a)) / a[2]])


# ------------------------------------------------------------------ what the arm holds here
class AtStand(Held):
    """The empty gripper (tip None: the closed fingertips lead, any tilt) or the fork (tip: its measured tip in the tool
    frame; its tilt the fork's rule) during the stand routine: kept above the table (motion.free_space's floor over
    workspace.table_extent_m) but in the stand's own column round the opening (at), where the fingers go in and the fork
    hangs; the planner's world the meal's but the stand, the table at its own height (the gripper goes down to the
    stand: these checks keep its fingertips and the fork above the table)."""
    camera = True

    def __init__(self, at=None, tip=None):
        self.at = None if at is None else np.asarray(at, float)
        self.fork = tip is not None
        self.name = "the fork" if self.fork else "the gripper"
        self.tip_in_tool = fingertips_in_tool() if tip is None else np.asarray(tip, float)
        self.max_tilt_deg = float(cfg.motion.max_tilt_deg) if self.fork else 180.0

    def floor(self, x: float, y: float) -> float:
        th, ext = cfg.workspace.get("table_height_m", None), cfg.workspace.table_extent_m
        if self.at is not None and math.hypot(x - self.at[0], y - self.at[1]) < float(P().stand_radius_m):
            return -math.inf
        if th is not None and ext.x[0] <= x <= ext.x[1] and ext.y[0] <= y <= ext.y[1]:
            return float(th) + float(cfg.motion.free_space.min_tip_above_table_m)
        return -math.inf

    def push_world(self, robot) -> None:
        scene.write_world(robot, scene.world(stand=False, raise_table=False))


def _fork(arm: Arm, at) -> AtStand:
    from rammp_feeding.scene import tip_in_tool
    return AtStand(at, tip_in_tool(arm.cam))


# ------------------------------------------------------------------ the gripper
@dataclass
class Gripper:
    """The gripper as the routine uses it: move(position, speed, force) the fingers there (0 open .. 1 closed; speed a
    fraction of the fastest, force a fraction: a current ceiling), back once they stop; position() where they are now,
    None where the arm node serves none."""
    move: Callable[[float, float, float], None]
    position: Callable[[], Optional[float]]


def _grip(g: Gripper, position: float, force: float | None = None) -> Optional[float]:
    """The fingers to position, slow and soft (gripper_speed, gripper_force: the stand is light; force: another
    ceiling), settle_s for the block to settle round them; where they stopped (None: not served)."""
    g.move(float(position), float(P().gripper_speed), float(P().gripper_force if force is None else force))
    time.sleep(float(P().settle_s))
    return g.position()


def _straight(arm: Arm, T, R, mps: float, label: str, check: str, away: bool = False,
              impedance: str | None = None) -> None:
    """A straight move at the stand, slow, our contact guard at guards.torque_step_nm.fork (low: the stand is light);
    a stop says what to check. Its IK steps short enough that a turn about the vertical (the fingers' other way round)
    stays a continuous path. away: backing out after a stop, the guard off -- the press a stop leaves on the stand
    lets go as it rises, and reads as a new contact. impedance: a motion.impedance entry, the move run compliant."""
    from rammp_feeding.geometry import rot_log
    T0, R0 = arm.tool()
    turn = math.degrees(float(np.linalg.norm(rot_log(np.asarray(R) @ R0.T))))
    L = float(np.linalg.norm(np.asarray(T) - T0))
    step = max(1e-4, min(0.005, L / max(1, math.ceil(turn / 6.0))))
    try:
        arm.straight(T, R, mps, label, step_m=step, inside=("stand",), torque_stop=not away,
                     torque_limit=float(cfg.guards.torque_step_nm.fork), impedance=impedance)
    except MoveError as e:
        if "torque" in str(e):
            raise MoveError("%s -- %s; nothing else moved" % (e, check))
        raise


# ------------------------------------------------------------------ the opening
@dataclass
class Hole:
    centre: np.ndarray                  # the opening's centre at the rim's height (base frame)
    long_axis: np.ndarray               # level, unit: along its long side
    size_m: tuple[float, float]         # its short side, its long side
    depth_m: float = float("nan")       # the rim down to its floor, as the depth reads it

    def to_json(self) -> dict:
        return dict(centre=[float(v) for v in self.centre], long_axis=[float(v) for v in self.long_axis],
                    size_m=[round(float(v), 4) for v in self.size_m], depth_m=round(float(self.depth_m), 4))

    @classmethod
    def from_json(cls, d: dict) -> "Hole":
        return cls(np.array(d["centre"], float), np.array(d["long_axis"], float), tuple(d["size_m"]),
                   float(d.get("depth_m", float("nan"))))


def opening(mask: np.ndarray, rim_z: float, cam, arm, q) -> Hole:
    """The opening from its mask: each mask pixel's ray laid on the rim's height; their centroid, principal axis and
    extent."""
    o, Rc = scene.camera_pose(arm, q)
    v, u = np.nonzero(mask)
    c = cam.colour
    d = np.stack([(u - c.cx) / c.fx, (v - c.cy) / c.fy, np.ones(len(u))], 1) @ Rc.T
    pts = o[:2] + d[:, :2] * ((rim_z - o[2]) / d[:, 2])[:, None]
    centre = pts.mean(0)
    w, V = np.linalg.eigh(np.cov((pts - centre).T))
    long = V[:, 1] / np.linalg.norm(V[:, 1])
    short = np.array([-long[1], long[0]])
    a, b = (pts - centre) @ long, (pts - centre) @ short
    return Hole(np.array([*centre, rim_z]), np.array([*long, 0.0]), (float(np.ptp(b)), float(np.ptp(a))))


def find(robot, arm: Arm, out: Path | None = None) -> Hole:
    """The opening on the fork's block, as the wrist camera sees it from here (the arm still)."""
    from scipy.ndimage import binary_dilation
    from rammp_feeding.perception import _in, world_points
    p = P()
    q = arm.q()
    cam = robot.cameras["camera"]
    since = time.time()
    _, rgb = cam.frame(after=since)
    D = cam.depth_median(int(p.depth_frames), since, wait_s=float(p.depth_wait_s))
    record.save_image(out, "colour.jpg", rgb, quality=95)
    record.save_npy(out, "depth.npy", D)
    dets = robot.detect("camera", [str(w) for w in p.prompts], float(p.score_threshold))
    W, uv, _ = world_points(arm.cam, arm, D, q)
    lo, hi = (int(v) for v in p.rim_ring_px)
    why = "SAM3 found no opening (%s)" % ", ".join(str(w) for w in p.prompts)
    for d in sorted(dets, key=lambda d: -d.score):
        ring = binary_dilation(d.mask, iterations=hi) & ~binary_dilation(d.mask, iterations=lo)
        z = W[_in(ring, uv), 2]
        if len(z):
            z = z[np.abs(z - np.median(z)) < float(p.inlier_m)]
        if len(z) < int(p.min_points):
            why = "the rim round the opening has %d depth points (at least %d)" % (len(z), int(p.min_points))
            continue
        h = opening(d.mask, float(np.median(z)), arm.cam, arm, q)
        known = p.get("opening_m", None)               # [short, long, deep] by ruler: the rim's height from its size
        if known is not None:                           # in the image (a stereo camera too near the block misreads it)
            o = scene.camera_pose(arm, q)[0]
            h = opening(d.mask, float(o[2] - (o[2] - h.centre[2]) * float(known[1]) / h.size_m[1]), arm.cam, arm, q)
        h.centre = h.centre + np.asarray(p.look_offset_m, float)
        h.long_axis = _turn(h.long_axis, float(p.get("look_turn_deg", 0.0)))
        inside = W[_in(d.mask, uv), 2]
        h.depth_m = (float(known[2]) if known is not None else
                     float(h.centre[2] - np.median(inside)) if len(inside) >= int(p.min_points) else float("nan"))
        record.save_json(out, "opening.json", dict(h.to_json(), rim_points=len(z), inside_points=len(inside),
                                                   score=d.score, label=d.label, box=d.box))
        log.info("  the opening at (%.3f %.3f %.3f), %.1f x %.1f mm, %.0f mm deep, its long side %.0f deg round from "
                 "x; the rim %d points, SAM3 %.2f (%s)", *h.centre, *(1000 * np.array(h.size_m)), 1000 * h.depth_m,
                 math.degrees(math.atan2(h.long_axis[1], h.long_axis[0])), len(z), d.score, d.label)
        lo_m, hi_m = (np.asarray(v, float) for v in (p.min_opening_m, p.max_opening_m))
        if np.any(np.asarray(h.size_m) < lo_m) or np.any(np.asarray(h.size_m) > hi_m):
            why = ("the opening reads %.1f x %.1f mm, not within %.0f x %.0f .. %.0f x %.0f mm (utensil.pick."
                   "min_opening_m .. max_opening_m): not the holder's, or the closed fingers do not fit"
                   % (*(1000 * np.array(h.size_m)), *(1000 * lo_m), *(1000 * hi_m)))
            continue                                    # (a mask of the whole block: the next one)
        need = float(p.insert_m) + float(p.floor_clear_m)
        if not h.depth_m >= need:
            deep = "no depth inside it: not known how" if math.isnan(h.depth_m) else "%.0f mm" % (1000 * h.depth_m)
            raise MoveError("the opening reads %s deep: the finger ends would reach %.0f mm under the rim, and need "
                            "%.0f mm clear of its floor (utensil.pick.insert_m, floor_clear_m)"
                            % (deep, 1000 * float(p.insert_m), 1000 * float(p.floor_clear_m)))
        return h
    record.save_json(out, "opening.json", dict(centre=None, why=why, detections=[
        dict(label=d.label, score=d.score, box=d.box) for d in dets]))
    raise MoveError(why)


def insertion(arm: Arm, R_look: np.ndarray, h: Hole) -> tuple[np.ndarray, list]:
    """(R, the joints over the opening): the fingers pointing straight down (tool z down), the tool axis they open along
    (fingers_open_along_tool, x or y) along the opening's long side; MoveError when that turns them more than
    max_turn_deg from fork_look's (R_look), either way round. The gripper is the same turned 180 deg about its axis: of
    the two, the one whose way in (over the opening -- by IK from the arm's joints, home's and a fixed spread -- then
    the straight lines in and up, as they run) keeps every joint furthest from its limit, of the arm shapes within
    near_shape_deg of the arm's own where there is one (the planner has no wrist to turn over on the way); the arm is planned to it over the opening (a stand beside the
    table can be out of a straight line's reach from fork_look)."""
    k = {"x": 0, "y": 1}[str(P().fingers_open_along_tool)]
    a = np.array([R_look[0, k], R_look[1, k], 0.0])
    a /= max(np.linalg.norm(a), 1e-9)
    turn = math.degrees(math.acos(min(1.0, abs(float(h.long_axis @ a)))))
    if turn > float(P().max_turn_deg):
        raise MoveError("the opening's long side is %.0f deg round from fork_look's fingers (at most %.0f): the stand "
                        "turned, or not the holder's opening -- capture fork_look again"
                        % (turn, float(P().max_turn_deg)))
    best, why = None, "no arm shape reaches over, into and up out of the opening either way round"
    for long in (h.long_axis, -h.long_axis):
        x, y = (long, np.cross(DOWN, long)) if k == 0 else (np.cross(long, DOWN), long)
        R = np.stack([x, y, DOWN], 1)
        seeds = []                                     # (seeds: the arm's joints and home's, the last joint turned half
        for q0 in (arm.q(), scene.captured("home")):   # round, and a fixed spread: a stand at the edge of reach)
            seeds += [list(q0)] + [list(np.asarray(q0) + [0, 0, 0, 0, 0, 0, t]) for t in (180.0, -180.0)]
        seeds += [list(v) for v in np.random.default_rng(0).uniform(-170.0, 170.0, (60, 7))]
        for over in (v for v in (arm.ik(tool_at(h, R, float(P().above_m)), R, sd) for sd in seeds) if v is not None):
            q, m = over, arm.margin(over)              # (each arm shape over it: in and up from there along the
            for a_, b_ in ((P().above_m, -P().insert_m), (-P().insert_m, P().lift_m - P().insert_m)):   # straight
                A_, B_ = tool_at(h, R, float(a_)), tool_at(h, R, float(b_))                           # lines run)
                n = max(4, int(math.ceil(float(np.linalg.norm(B_ - A_)) / 0.005)))
                path, _ = arm.homotopy(q, lambda s, A_=A_, B_=B_: (A_ + (B_ - A_) * s, R), n)
                if path is None:
                    break
                q, m = path[-1], min(m, min(arm.margin(p) for p in path))
            else:
                near = max_step(over, arm.q()) <= float(P().near_shape_deg)
                if best is None or (near, m) > (best[3], best[0]):
                    best = (m, R, over, near)
    if best is None:
        raise MoveError(why)
    return best[1], best[2]


def room_in(arm: Arm, h: Hole, R: np.ndarray) -> None:
    """From the arm's own joints over the opening (the planner may end in its own shape there): the straight lines in
    and up as they will run keep every joint min_room_deg from its limit, else MoveError (nothing moved): a line run to
    a joint's limit leaves the arm where the driver refuses any move."""
    q, worst = arm.q(), math.inf
    for a_, b_ in ((P().above_m, -P().insert_m), (-P().insert_m, P().lift_m - P().insert_m)):
        A_, B_ = tool_at(h, R, float(a_)), tool_at(h, R, float(b_))
        n = max(4, int(math.ceil(float(np.linalg.norm(B_ - A_)) / 0.005)))
        path, why = arm.homotopy(q, lambda s, A_=A_, B_=B_: (A_ + (B_ - A_) * s, R), n)
        if path is None:
            raise MoveError("the way into the opening from this arm shape: %s -- not going in" % why)
        q, worst = path[-1], min(worst, min(arm.margin(p) for p in path))
    if worst < float(P().min_room_deg):
        raise MoveError("the way into the opening from this arm shape takes a joint %.1f deg from its limit (at least "
                        "%.1f) -- not going in" % (worst, float(P().min_room_deg)))


def _level(v) -> np.ndarray:
    v = np.array([v[0], v[1], 0.0], float)
    return v / max(np.linalg.norm(v), 1e-9)


def _yaw(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _turn(v, deg: float) -> np.ndarray:
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1], 0.0])


def open_side(h: Hole) -> np.ndarray:
    """Out through the holder's open side: across the opening's long side, the way utensil.pick.open_side (the site's
    direction, base frame) points -- the stand turned a little turns it with it."""
    short = _turn(h.long_axis, 90.0)
    return short if float(short[:2] @ np.asarray(P().open_side, float)[:2]) >= 0 else -short


def _strip_once(dets, W, uv, lo: float, hi: float, toward=None, like=None):
    """One reading of the strip from SAM3's detections and the base-frame points: (its post end, its direction toward
    its seat end, level; points, score, label), or (None, why)."""
    from rammp_feeding.perception import _in
    p = P()
    x0, y0, x1, y1 = (int(v) for v in cfg.utensil.image_region_px)
    why = "SAM3 found no strip on the stand (%s)" % ", ".join(str(w) for w in p.stand_prompts)
    for d in sorted(dets, key=lambda d: -d.score):
        mask = d.mask.copy()
        mask[y0:y1 + 1, x0:x1 + 1] = False             # (not the utensil's own pixels)
        P3 = W[_in(mask, uv)]
        P3 = P3[np.isfinite(P3).all(1) & (P3[:, 2] > lo) & (P3[:, 2] < hi)]   # (not the table's, not the gripper's)
        if len(P3):
            P3 = P3[np.abs(P3[:, 2] - np.median(P3[:, 2])) < float(p.inlier_m)]
        if len(P3) < int(p.stand_min_points):
            why = "the strip on the stand has %d depth points (at least %d)" % (len(P3), int(p.stand_min_points))
            continue
        c = P3[:, :2].mean(0)
        a = np.linalg.eigh(np.cov((P3[:, :2] - c).T))[1][:, 1]
        if (toward is not None and float((np.asarray(toward)[:2] - c) @ a) < 0) or \
                (like is not None and float(np.asarray(like)[:2] @ a) < 0):
            a = -a
        # its post end, on its centreline: the tape's clean cut (at the seat it wraps round, the fork hangs beside it)
        s_ = (P3[:, :2] - c) @ a
        E = np.array([*(c + a * np.percentile(s_, 2)), float(np.median(P3[:, 2]))])
        return (E + np.asarray(p.look_offset_m, float), _level(a), len(P3), d.score, d.label), None
    return None, why


def fork_px(robot) -> tuple[float, float] | None:
    """Where the fork hangs in the wrist camera's frame now (from fork_look): the centre of SAM3's best fork mask
    (pick.fork_prompts) on the fork's box (utensil.image_region_px), in pixels; None when none is seen. The camera
    rides on the gripper: its place in the frame is its place in the fingers."""
    x0, y0, x1, y1 = (int(v) for v in cfg.utensil.image_region_px)
    best = None
    for d in robot.detect("camera", [str(w) for w in P().fork_prompts], float(P().score_threshold)):
        m = np.asarray(d.mask, bool)
        on = m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if on.sum() >= int(P().min_points) and (best is None or d.score > best[0]):
            best = (d.score, m)
    if best is None:
        return None
    v, u = np.nonzero(best[1])
    return float(u.mean()), float(v.mean())


def strip(robot, arm: Arm, out: Path | None = None, toward=None, like=None) -> tuple[np.ndarray, np.ndarray]:
    """The stand's strip as the wrist camera sees it from here (the arm still): (its post end, its direction toward
    its seat end, level), the median of stand_reads readings (each SAM3 on a new colour frame, one depth median) that
    agree to stand_spread_m -- most of them must (a stray one dropped), else read again, up to stand_looks times. toward: a point the seat end is nearer
    (the opening, at the pick); like: a direction it keeps (the pick's, at the put-back)."""
    from rammp_feeding.perception import world_points
    p = P()
    q = arm.q()
    cam = robot.cameras["camera"]
    T, R = arm.tool(q)
    lo, hi = float(cfg.workspace.table_height_m) + 0.03, (T + R @ fingertips_in_tool())[2] - 0.02
    why = "no look at the strip"
    for look in range(int(p.stand_looks)):
        since = time.time()
        _, rgb = cam.frame(after=since)
        D = cam.depth_median(int(p.depth_frames), since, wait_s=float(p.depth_wait_s))
        record.save_image(out, "strip%d.jpg" % look, rgb, quality=95)
        W, uv, _ = world_points(arm.cam, arm, D, q)
        reads = []
        for _ in range(int(p.stand_reads)):
            r, why = _strip_once(robot.detect("camera", [str(w) for w in p.stand_prompts], float(p.score_threshold)),
                                 W, uv, lo, hi, toward, like)
            if r is not None:
                reads.append(r)
        if len(reads) < max(2, int(p.stand_reads) - 1):
            why = "the strip read in %d of %d (%s)" % (len(reads), int(p.stand_reads), why)
            continue
        Es = np.array([r[0] for r in reads])
        off = np.linalg.norm(Es[:, :2] - np.median(Es, axis=0)[:2], axis=1)
        ok = off <= float(p.stand_spread_m)            # a stray reading dropped; most must agree
        if ok.sum() < len(reads) // 2 + 1:
            why = "the strip's readings spread %.1f mm (at most %.1f)" % (1000 * float(off.max()),
                                                                          1000 * float(p.stand_spread_m))
            log.info("    %s: looking again", why)
            continue
        E = np.median(Es[ok], axis=0)
        a = _level(np.median([r[1] for r, k in zip(reads, ok) if k], axis=0))
        spread = float(off[ok].max())
        reads = [r for r, k in zip(reads, ok) if k]
        log.info("  the stand's strip: its post end (%.3f %.3f %.3f), toward its seat %.0f deg; %d readings within "
                 "%.1f mm, %d points, SAM3 %.2f (%s)", *E, _yaw(a), len(reads), 1000 * spread, reads[0][2],
                 reads[0][3], reads[0][4])
        return E, a
    raise MoveError("the stand's strip: %s -- put the fork back by hand" % why)


def seat_record(h: Hole, R: np.ndarray, E: np.ndarray, a: np.ndarray) -> dict:
    """The seat in the strip's frame: the opening's centre from the strip's post end (along it, across it, up), and
    the fingers' turn and the open side's, from the strip's direction."""
    n = _turn(a, 90.0)
    d = h.centre - E
    return dict(E=[float(v) for v in E], a=[float(v) for v in a], along=float(d[:2] @ a[:2]),
                across=float(d[:2] @ n[:2]), up=float(d[2]), fingers_deg=_yaw(R[:, 0]) - _yaw(a),
                open_deg=_yaw(open_side(h)) - _yaw(a), hole=h.to_json())


def seat_now(rec: dict, E: np.ndarray, a: np.ndarray) -> tuple[Hole, np.ndarray, np.ndarray]:
    """The seat placed from the strip as found now: (the opening where it is now, the fingers' orientation, the open
    side's direction); MoveError when the stand moved or turned past max_stand_move_m / max_stand_turn_deg."""
    p = P()
    moved = float(np.linalg.norm(np.asarray(E)[:2] - np.asarray(rec["E"])[:2]))
    turned = (_yaw(a) - _yaw(rec["a"]) + 180.0) % 360.0 - 180.0
    log.info("  the stand moved %.1f mm and turned %.1f deg since the pick", 1000 * moved, turned)
    if moved > float(p.max_stand_move_m) or abs(turned) > float(p.max_stand_turn_deg):
        raise MoveError("the stand moved %.0f mm and turned %.0f deg since the pick (at most %.0f mm, %.0f deg): "
                        "put the fork back by hand" % (1000 * moved, turned, 1000 * float(p.max_stand_move_m),
                                               float(p.max_stand_turn_deg)))
    n = _turn(a, 90.0)
    off = seat_in_strip(rec)
    c = np.asarray(E) + off[0] * a + off[1] * n + np.array([0.0, 0.0, off[2]])
    h0 = Hole.from_json(rec["hole"])
    long = _turn(h0.long_axis, turned)
    x = _turn(a, rec["fingers_deg"])
    k = {"x": 0, "y": 1}[str(p.fingers_open_along_tool)]
    R = np.stack([x, np.cross(DOWN, x), DOWN], 1) if k == 0 else np.stack([np.cross(x, DOWN), x, DOWN], 1)
    return Hole(c, long, h0.size_m, h0.depth_m), R, _turn(a, rec["open_deg"])


def seat_in_strip(rec: dict) -> np.ndarray:
    """The seat from the strip's post end (along, across, up): the stand's own (utensil.pick.seat_in_strip_m, the site's
    when set: the stand is rigid), else the pick's reading of it; the pick's logged against the stand's."""
    mine = np.array([rec["along"], rec["across"], rec["up"]])
    site = P().get("seat_in_strip_m", None)
    if site is None:
        return mine
    site = np.asarray(site, float)
    off = float(np.linalg.norm((mine - site)[:2]))
    (log.warning if off > float(P().seat_check_m) else log.info)(
        "    the pick read the seat %.1f mm from the stand's own (utensil.pick.seat_in_strip_m): the stand's taken",
        1000 * off)
    return site


def look(robot, arm: Arm, out: Path | None = None) -> None:
    """rammp find-fork: the opening from here, and the way in (never moves)."""
    need()
    h = sensing.now().opening(robot, arm, out)
    R, _ = insertion(arm, arm.tool()[1], h)
    for k, dz in (("over", float(P().above_m)), ("in", -float(P().insert_m)),
                  ("lifted", float(P().lift_m) - float(P().insert_m))):
        q = arm.ik(tool_at(h, R, dz), R, arm.q())
        log.info("    %-7s tool (%.3f %.3f %.3f)%s", k, *tool_at(h, R, dz),
                 "" if q is None else "  joints %.0f deg from their limits" % arm.margin(q))


def tool_at(h: Hole, R: np.ndarray, dz: float) -> np.ndarray:
    """The tool point that puts the fingertips dz above the opening's centre at the rim (negative: into it),
    orientation R."""
    return h.centre + np.array([0.0, 0.0, dz]) - R @ fingertips_in_tool()


# ------------------------------------------------------------------ the routine, step by step
PICK_STEPS = ("look", "hover", "in", "hold", "lift", "out")   # rammp pick-fork [--start-at S] [--stop-after S]
PUT_STEPS = ("back", "lower", "release", "up")                # rammp put-fork [--start-at S] [--stop-after S]


def _no_step(name: str, fraction: float, message: str) -> None:
    log.info("  [%s] %s", name, message)


def _home(robot, arm: Arm) -> None:
    from rammp_feeding.cli import home
    home(robot, arm)


def record_dir(name: str) -> Path | None:
    """name/ in this session's record (None with the record off)."""
    s = record.session(name)
    if s is None:
        return None
    (s / name).mkdir(parents=True, exist_ok=True)
    return s / name


def by_hand(g: Gripper, hold: bool) -> None:
    """rammp gripper open|close: open -- the fork held; close -- not held. With the fingers in the opening (in_holder:
    a pick or put-back stopped there) a step taken by hand, its place kept (pick-fork / put-fork --start-at go on from
    it); elsewhere, for setting up (the block hung on the closed fingers by hand, or taken off), where it stood is not
    known: put-fork refuses it."""
    at = _grip(g, float(P().gripper_open if hold else P().gripper_closed))
    log.info("    the fingers %s: at %s", "opened" if hold else "closed", "?" if at is None else "%.2f" % at)
    s = state()
    if s.get("in_holder"):
        save_state(dict(s, holding=hold, **({"held_at": at} if hold else {})))
        return
    stand = s.get("stand") or s.get("hole")             # (where the stand was last seen: the world keeps it)
    save_state(dict(holding=hold, **({"stand": stand} if stand else {})))


@dataclass
class Run:
    """What the steps share: the robot, the arm, the gripper, the record, the progress callback, and the opening and
    the fingers' orientation once found (from the utensil's state when a run starts at a later step)."""
    robot: object
    arm: Arm
    g: Gripper
    out: Optional[Path]
    step: Step
    h: Optional[Hole] = None
    R: Optional[np.ndarray] = None

    def load(self) -> None:
        s = state()
        if not s.get("hole") or s.get("R") is None:
            raise MoveError("no opening found yet in this pick -- start at look (or hover)")
        self.h, self.R = Hole.from_json(s["hole"]), np.asarray(s["R"], float)

    def save(self, **kw) -> None:
        save_state(dict(state(), hole=self.h.to_json(), R=self.R.round(6).tolist(), stand=self.h.to_json(), **kw))


def look_step(r: Run) -> None:
    """The fingers closed (they go into the opening closed); a planned path to fork_look."""
    r.arm.hold(AtStand())
    r.step("look", 0.05, "Looking for the fork")
    _grip(r.g, float(P().gripper_closed))
    r.arm.travel(scene.captured("fork_look"), "to fork_look")


def hover_step(r: Run) -> None:
    """The opening found from here; a planned path to its pose above_m over it (the planner's arm shape, inside its
    margins), the way in checked from the shape it ends in (room_in)."""
    r.step("find", 0.2, "Finding the fork's holder")
    r.h = sensing.now().opening(r.robot, r.arm, r.out)
    r.R, over = insertion(r.arm, r.arm.tool()[1], r.h)
    r.arm.hold(AtStand(r.h.centre))
    r.step("grasp", 0.35, "Over the fork")
    r.arm.travel(over, "over the fork", pose=True)
    room_in(r.arm, r.h, r.R)
    tips = r.arm.tool()[0] + r.R @ fingertips_in_tool()
    log.info("    the opening's centre (%.4f %.4f %.4f); the finger ends at (%.4f %.4f %.4f), %.1f mm off it across, "
             "going in %.0f mm under the rim", *r.h.centre, *tips, 1000 * math.hypot(*(tips[:2] - r.h.centre[:2])),
             1000 * float(P().insert_m))
    save_state(dict(holding=False, in_holder=False, hole=r.h.to_json(), R=r.R.round(6).tolist(), stand=r.h.to_json()))


def in_step(r: Run) -> None:
    """Straight down into the opening, insert_m under its rim (written first: a stop from here comes back out)."""
    r.save(in_holder=True)
    _straight(r.arm, tool_at(r.h, r.R, -float(P().insert_m)), r.R, float(P().insert_mps), "into the opening",
              "the fingers met the block: off its opening? check the stand and the opening's centre (find-fork)",
              impedance=P().get("impedance", None))


def hold_step(r: Run) -> None:
    """The fingers opened in the opening: its walls stop them -- the hold (where the arm node serves their position,
    refused unless they stopped lock_margin short of open: else nothing held them, closed again first)."""
    p = P()
    at = _grip(r.g, float(p.gripper_open))
    if at is None:
        log.info("    the arm node serves no gripper position: the hold is not checked")
    elif at < float(p.gripper_open) + float(p.lock_margin):
        _grip(r.g, float(p.gripper_closed))
        raise MoveError("the fingers opened fully (%.2f): nothing held them -- not in the opening; closed again, the "
                        "arm where it is (the next pick or put comes straight back up first)" % at)
    else:
        log.info("    the fingers stopped at %.2f (open: %.2f): held by the opening's walls", at, float(p.gripper_open))
    save_state(dict(state(), holding=True, held_at=at))


def lift_step(r: Run) -> None:
    """Straight up lift_m from where the fingers held it: the block off its seat."""
    p = P()
    r.arm.hold(_fork(r.arm, r.h.centre))
    _straight(r.arm, tool_at(r.h, r.R, float(p.lift_m) - float(p.insert_m)), r.R, float(p.lift_mps), "lift the fork",
              "the block snug in the stand, or the lift dragging the stand: check it")
    save_state(dict(state(), holding=True, in_holder=False, lifted_joints=[float(v) for v in r.arm.q()]))


def out_step(r: Run) -> None:
    """Clear of the holder, a firm hold for the meal (hold_force). A side holder: first straight out through its open
    side (slide_m), then fork_look again and the seat kept in its strip's frame (and where the fork hangs in the
    camera's frame), for the put-back to find wherever the stand is then. A top holder keeps the opening it was
    picked from as its seat."""
    p = P()
    if not top():
        _straight(r.arm, r.arm.tool()[0] + float(p.slide_m) * open_side(r.h), r.R, float(p.lift_mps),
                  "out of the holder's open side", "the fork caught on the holder: check the stand")
    _grip(r.g, float(p.gripper_open), float(p.hold_force))   # a firm hold: the meal's moves don't shift it
    if top():
        return
    r.step("seat", 0.7, "Looking at the empty holder")
    r.arm.travel(scene.captured("fork_look"), "to fork_look")
    E, a = sensing.now().strip(r.robot, r.arm, r.out, toward=r.h.centre)
    rec = seat_record(r.h, r.R, E, a)
    log.info("    the seat from the strip's post end: along %.1f, across %.1f, up %.1f mm (as utensil.pick."
             "seat_in_strip_m)", *(1000 * np.array([rec["along"], rec["across"], rec["up"]])))
    px = sensing.now().fork_px(r.robot)
    if px is None:
        log.info("    the fork in the fingers: not seen (%s) -- the put-back cannot check it", ", ".join(P().fork_prompts))
    save_state(dict(state(), seat=dict(rec, fork_px=px)))


def back_step(r: Run) -> None:
    """Over the seat at the lift's height, the fingers soft again (a firm hold drags the holder). A top holder: the
    planned path back to where the lift ended over the opening it was picked from. A side holder: fork_look, the
    strip found and the seat placed from it (refused when the stand moved or turned past max_stand_move_m /
    max_stand_turn_deg, or the fork hangs more than grip_shift_tol_m from where it hung at the pick), a planned path
    beside the open side, straight in through it."""
    p, s = P(), state()
    if s.get("lifted_joints") and max_step(r.arm.q(), s["lifted_joints"]) <= float(cfg.motion.near_pose_deg):
        r.load()                                       # (lifted, never moved on: straight back down)
        r.arm.hold(_fork(r.arm, r.h.centre))
        _grip(r.g, float(p.gripper_open))
        return
    if top():
        if not s.get("lifted_joints"):
            raise MoveError("where the fork was picked from is not known -- put it back by hand (rammp gripper close)")
        r.load()
        r.arm.hold(_fork(r.arm, r.h.centre))
        r.step("place", 0.2, "Taking the fork back to its holder")
        r.arm.travel(s["lifted_joints"], "back over the holder")
        _grip(r.g, float(p.gripper_open))
        return
    if not s.get("seat"):
        raise MoveError("where the fork's seat is on its stand is not known -- put it back by hand "
                        "(rammp gripper close)")
    r.arm.hold(_fork(r.arm, Hole.from_json(s["seat"]["hole"]).centre))
    r.step("place", 0.1, "Finding the fork's stand")
    from rammp_feeding.cli import home_tilt_cap
    r.arm.travel(scene.captured("fork_look"), "to fork_look", tilt_cap=home_tilt_cap(r.arm.q()))
    E, a = sensing.now().strip(r.robot, r.arm, record_dir("put_fork"), like=s["seat"]["a"])
    then, now = s["seat"].get("fork_px"), sensing.now().fork_px(r.robot)
    if then is None or now is None:
        log.info("    the fork in the fingers: not seen %s -- not checked", "at the pick" if then is None else "now")
    else:
        moved = (math.hypot(now[0] - then[0], now[1] - then[1])               # (pixels, at the tines' depth)
                 * float(cfg.utensil.glass_to_tip_m) / float(r.arm.cam.colour.fx))
        log.info("    the fork in the fingers: %.1f mm from where it hung at the pick (its tines' depth)", 1000 * moved)
        if moved > float(p.grip_shift_tol_m):
            raise MoveError("the fork has moved %.1f mm in the fingers since the pick (its tines, at most %.1f): the "
                            "block would land off its seat -- seat the fork in the fingers by hand, or put it back by "
                            "hand (rammp gripper close)" % (1000 * moved, 1000 * float(p.grip_shift_tol_m)))
    r.h, r.R, open_dir = seat_now(s["seat"], E, a)
    r.arm.hold(_fork(r.arm, r.h.centre))
    over = tool_at(r.h, r.R, float(p.lift_m) - float(p.insert_m))
    q0 = r.arm.q()
    seeds = [s.get("lifted_joints", q0), q0] + [list(np.asarray(q0) + [0, 0, 0, 0, 0, 0, t]) for t in (180.0, -180.0)]
    q_in = max((v for v in (r.arm.ik(over + float(p.slide_m) * open_dir, r.R, sd) for sd in seeds) if v is not None),
               key=r.arm.margin, default=None)
    if q_in is None:
        raise MoveError("beside the holder's open side is out of the arm's reach -- put the fork back by hand")
    r.step("place", 0.3, "Taking the fork back to its stand")
    r.arm.travel(q_in, "beside the holder's open side")
    _grip(r.g, float(p.gripper_open))
    _straight(r.arm, over, r.R, float(p.lift_mps), "in through the open side",
              "the fork met the holder: check the stand")
    log.info("    over the seat: the opening's centre (%.4f %.4f %.4f)", *r.h.centre)
    r.save(lifted_joints=[float(v) for v in r.arm.q()])


def lower_step(r: Run) -> None:
    """Straight down onto the seat: the guard's stop within seat_tol_m of the seat's depth is the holder taking the
    fork's weight (seated); higher, the block is on something else."""
    p = P()
    r.save(in_holder=True)
    seat = tool_at(r.h, r.R, -float(p.insert_m))
    try:
        _straight(r.arm, seat, r.R, float(p.insert_mps), "into the stand",
                  "the block met the stand: off its place? check the stand", impedance=p.get("impedance", None))
    except MoveError as e:
        short = float(r.arm.tool()[0][2] - seat[2])
        if "torque" not in str(e) or short > float(p.seat_tol_m):
            raise
        log.info("    seated %.0f mm above the seat's depth (the holder took the fork's weight)", 1000 * short)


def release_step(r: Run) -> None:
    """The fingers closed: they let go -- refused unless they closed at least lock_margin from where the walls held
    them (held_at; or from open): in some blocks the fork's handle between them stops them short of closed."""
    p = P()
    at = _grip(r.g, float(p.gripper_closed))
    held_at = state().get("held_at")
    least = (float(p.gripper_open) if held_at is None else float(held_at)) + float(p.lock_margin)
    if at is None:
        log.info("    the arm node serves no gripper position: the release is not checked")
    elif at < least:
        raise MoveError("the fingers did not close (%.2f, at least %.2f): still in the block -- the arm where it is"
                        % (at, least))
    else:
        log.info("    the fingers closed to %.2f: let go", at)
    save_state(dict(state(), holding=False))
    r.arm.hold(AtStand(r.h.centre))


def up_step(r: Run) -> None:
    """Straight up out of the opening, above_m over the rim (the guard off: away from the stand)."""
    _straight(r.arm, tool_at(r.h, r.R, float(P().above_m)), r.R, float(P().lift_mps), "out of the opening",
              "the fingers caught the block: check the stand", away=True)
    save_state(dict(holding=False, stand=r.h.to_json()))   # (back on its stand: the next pick finds it anew; the
    record.event("fork_put_down", hole=r.h.to_json())     # world keeps the stand where it was seen)


PICK = dict(look=look_step, hover=hover_step, **{"in": in_step}, hold=hold_step, lift=lift_step, out=out_step)
PUT = dict(back=back_step, lower=lower_step, release=release_step, up=up_step)


def _steps(r: Run, steps: dict, order: tuple, start_at: str, stop_after: str) -> str:
    """Run order's steps from start_at, up to and including stop_after: the last one run."""
    run = order[order.index(start_at or order[0]):]
    if stop_after and stop_after not in run:
        raise ValueError("--stop-after %s comes before --start-at %s" % (stop_after, start_at))
    for name in run:
        steps[name](r)
        if name == stop_after:
            return name
    return run[-1]


def pick_up(robot, arm: Arm, g: Gripper, step: Step = _no_step, out: Path | None = None, stop_after: str = "",
            start_at: str = "") -> dict:
    """The fork picked up: PICK_STEPS from start_at to stop_after (the whole routine: then home)."""
    need()
    r = Run(robot, arm, g, out, step)
    if start_at in ("", "look", "hover"):
        if holding():
            raise MoveError("the gripper already holds the fork -- rammp put-fork, or rammp gripper close to take it "
                            "off")
        recover(robot, arm)
        arm.hold(AtStand())
    else:                                              # (on from a step that stopped: its opening and orientation)
        r.load()
        arm.hold(_fork(arm, r.h.centre) if holding() else AtStand(r.h.centre))
    last = _steps(r, PICK, PICK_STEPS, start_at, stop_after)
    if stop_after:
        return dict(stopped_after=last, **({"opening_m": [round(float(v), 4) for v in r.h.centre]} if r.h else {}))
    record.event("fork_picked_up", hole=r.h.to_json())
    step("home", 0.8, "Going home with the fork")
    _home(robot, arm)                                  # (the arm holding AtStand's fork: a fresh Arm is the meal's)
    return dict(holding=True, opening_m=[round(float(v), 3) for v in r.h.centre])


def put_down(robot, arm: Arm, g: Gripper, step: Step = _no_step, stop_after: str = "", start_at: str = "") -> dict:
    """The fork hung back on its stand: PUT_STEPS from start_at to stop_after (the whole routine: then home)."""
    need()
    r = Run(robot, arm, g, None, step)
    if start_at in ("", "back"):
        if not holding():
            raise MoveError("the fork is not in the gripper")
        recover(robot, arm)                            # (a pick stopped in the opening: its lift finished first)
    else:                                              # (on from a step that stopped: its opening and orientation)
        r.load()
        arm.hold(_fork(arm, r.h.centre) if holding() else AtStand(r.h.centre))
    last = _steps(r, PUT, PUT_STEPS, start_at, stop_after)
    if stop_after:
        return dict(stopped_after=last, opening_m=[round(float(v), 4) for v in r.h.centre])
    step("home", 0.8, "Going home")
    _home(robot, arm)
    return dict(put_down=True)


def recover(robot, arm: Arm) -> None:
    """The fingers in the opening after a stop (in_holder): straight up out of it, holding the fork to where the lift
    ends, empty to above_m over the rim (the guard off: away from the stand)."""
    s = state()
    if not s.get("in_holder"):
        return
    p = P()
    h, R = Hole.from_json(s["hole"]), np.asarray(s["R"], float)
    held = bool(s.get("holding"))
    arm.hold(_fork(arm, h.centre) if held else AtStand(h.centre))
    T, Rn = arm.tool()
    up = tool_at(h, R, float(p.lift_m) - float(p.insert_m) if held else float(p.above_m))
    if up[2] - T[2] > float(cfg.motion.arrived_m):
        _straight(arm, np.array([T[0], T[1], up[2]]), Rn, float(p.lift_mps), "lift the fork" if held else
                  "out of the opening", "check the stand", away=True)
    save_state(dict(s, in_holder=False, **({"lifted_joints": [float(v) for v in arm.q()]} if held else {})))
    arm.hold(_fork(arm, h.centre) if held else AtStand(h.centre))
