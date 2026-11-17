"""The fork on its stand (utensil.pick): it hangs tines down, a block on its handle with a rectangular hole on top. The
fingers go into the hole closed and open there, holding the block from inside; lifting it lifts the fork off the
stand. The gripper is commanded here and nowhere else.

  find      from fork_look (the wrist camera over the stand, the gripper empty and pointing down): one colour frame and a
            depth median from frames after the arm stopped; SAM3 on utensil.pick.prompts (the opening, whatever colour
            the block is); for the best-scored mask, the rim -- the block's top: the median height of the depth points in
            a ring just outside the mask (rim_ring_px), those within inlier_m of it (the opening's own depth is its
            floor) -- then the mask's pixels laid on the rim's height: their centroid the opening's centre, their
            principal axis its long side, their extent its size (refused outside min_opening_m .. max_opening_m); its
            floor the median height of the depth inside the mask (refused when the finger ends, insert_m under the
            rim, would come within floor_clear_m of it); the centre shifted by look_offset_m (the site's correction
            for what the camera sees from fork_look, where every find is made; the strip's end too)
  pick_up   from feeding_start, the gripper empty: the fingers closed; a planned path to fork_look; the opening found;
            the fingers pointing straight down, the axis they open along (fingers_open_along_tool) along the opening's
            long side (refused when that turns them more than max_turn_deg from fork_look's; of the two ways round, the
            one whose way in keeps the joints furthest from their limits); straight over the opening (above_m),
            straight down into it (insert_m under the rim);
            the fingers open (the hold) -- where the arm node serves their position, refused unless the opening's end
            walls stopped them short of open (lock_margin: else they are not in it), closed again first; straight up
            (lift_m), the block off its seat; straight out through the holder's open side (slide_m across the
            opening's long side, the way open_side points: away from the stand's post); the fingers opened again at
            hold_force (a firm hold for the meal); a planned path to fork_look, the holder now empty, and the stand's
            strip found there (strip): the seat -- the opening's centre and the fingers' turn -- kept in the strip's
            frame, and where the fork hangs in the camera's frame (fork_px: SAM3's fork); a planned path to
            feeding_start
  strip     the stand's marker, from fork_look: SAM3 on stand_prompts (the tape along the stand's arm to its seat); its
            depth points between the table and the fingers' ends, outside the utensil's image region; the best-scored
            with stand_min_points: its post end (the tape's clean cut, on its centreline) and its direction toward
            the seat, on the base's level; the median of stand_reads readings (looked at again while they spread)
  put_down  from feeding_start holding the fork: a planned path to fork_look; the strip found and the seat placed from
            it (refused when the stand moved more than max_stand_move_m or turned more than max_stand_turn_deg since the
            pick, or no strip: put the fork back by hand; refused, too, when the fork hangs more than grip_shift_tol_m
            (at its tines' depth) from where it hung at the pick: it moved in the fingers); a planned path beside the open
            side at the lift's height, the fingers back to gripper_force;
            straight in through it; straight down onto the seat (the guard's stop within seat_tol_m of it: seated, the
            holder taking the fork's weight); the fingers close (they let go; refused unless they
            closed); straight up out of the opening (above_m over the rim); a planned path to feeding_start
  stop_after  rammp pick-fork / put-fork stop at a stage for a first try: above (over the opening), insert (in it),
            open / release (the fingers moved; the lift, or the way out, not yet)
  recover   after a stop with the fingers in the opening: straight up out of it (holding the fork: to where the lift
            ends; empty: above_m over the rim)

The utensil's state (state_dir/utensil_state.json): holding (the fork in the gripper), in_holder (the fingers in the
opening: written before going in, cleared once out), where it was picked up (the opening, the fingers' orientation, the
joints where the lift ended: what recover needs) and the seat in the stand's strip's frame (what put_down places it by,
wherever the stand is then). The place is cleared once the fork is back: the next pick finds it anew. With
utensil.pick.use false the fork is put in by hand and is always held.

While the routine runs the arm holds AtStand: the empty gripper (its fingertips the tip) or the fork (its measured
tip), checked against the table (workspace.table_height_m) and kept above it but in the stand's own column; the
planner's world is the table. The bowl is not in it: the stand stands beside the bowl, clear of the way out.
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

from rammp_feeding import record, scene
from rammp_feeding.arm import Arm, Held, MoveError
from rammp_feeding.config import cfg, read_json, state_dir

log = logging.getLogger("rammp")
Step = Callable[[str, float, str], None]
DOWN = np.array([0.0, 0.0, -1.0])


def P():
    return cfg.utensil.pick


def used() -> bool:
    """A stand to pick the fork from (utensil.pick.use), read live."""
    return bool(cfg.get("utensil.pick.use", False))


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
    missing = [k for k in ("workspace.table_height_m", "utensil.pick.glass_to_fingertips_m", "utensil.pick.open_side")
               if cfg.get(k, None) is None]
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
    hangs; the planner's world the table."""
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
        scene.write_world(robot, world())


def world() -> dict:
    """RAMMP-CuRobo's world for the stand routine: the pedestal and the table (workspace.table_height_m) raised to our
    floor (the planner knows the arm, not what hangs below it), its near edge moved back as a meal's first look has it
    (the scene camera's error_m; with none, workspace.bowl_centre_error_m): the planner's padded table clear of the
    arm's own base."""
    from rammp_feeding.scene_camera import in_use
    obs = [dict(name="pedestal", position=[0.0, 0.0, -0.05], dims=[0.14, 0.14, 0.04])]   # (the planner's own: unpadded)
    th = cfg.workspace.get("table_height_m", None)
    if th is not None:
        ext = cfg.workspace.table_extent_m
        tx, ty = [float(v) for v in ext.x], [float(v) for v in ext.y]
        tx[0] += float(cfg.scene_camera.error_m if in_use() else cfg.workspace.bowl_centre_error_m)
        top = float(th) + float(cfg.motion.free_space.min_tip_above_table_m)
        obs.append(dict(name="table", position=[sum(tx) / 2, sum(ty) / 2, top - 0.025],
                        dims=[tx[1] - tx[0], ty[1] - ty[0], 0.05]))
    for o in obs:
        o["position"], o["dims"] = [float(v) for v in o["position"]], [float(v) for v in o["dims"]]
        o["rpy_deg"] = [0.0, 0.0, 0.0]
    return dict(base_frame=str(cfg.arm.base_frame), obstacles=obs, objects=[], targets=[])


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


def _straight(arm: Arm, T, R, mps: float, label: str, check: str) -> None:
    """A straight move at the stand, slow, our contact guard at guards.torque_step_nm.fork (low: the stand is light);
    a stop says what to check. Its IK steps short enough that a turn about the vertical (the fingers' other way round)
    stays a continuous path."""
    from rammp_feeding.geometry import rot_log
    T0, R0 = arm.tool()
    turn = math.degrees(float(np.linalg.norm(rot_log(np.asarray(R) @ R0.T))))
    L = float(np.linalg.norm(np.asarray(T) - T0))
    step = max(1e-4, min(0.005, L / max(1, math.ceil(turn / 6.0))))
    try:
        arm.straight(T, R, mps, label, step_m=step, torque_limit=float(cfg.guards.torque_step_nm.fork))
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
        h.centre = h.centre + np.asarray(p.look_offset_m, float)
        inside = W[_in(d.mask, uv), 2]
        h.depth_m = float(h.centre[2] - np.median(inside)) if len(inside) >= int(p.min_points) else float("nan")
        record.save_json(out, "opening.json", dict(h.to_json(), rim_points=len(z), inside_points=len(inside),
                                                   score=d.score, label=d.label, box=d.box))
        log.info("  the opening at (%.3f %.3f %.3f), %.1f x %.1f mm, %.0f mm deep, its long side %.0f deg round from "
                 "x; the rim %d points, SAM3 %.2f (%s)", *h.centre, *(1000 * np.array(h.size_m)), 1000 * h.depth_m,
                 math.degrees(math.atan2(h.long_axis[1], h.long_axis[0])), len(z), d.score, d.label)
        lo_m, hi_m = (np.asarray(v, float) for v in (p.min_opening_m, p.max_opening_m))
        if np.any(np.asarray(h.size_m) < lo_m) or np.any(np.asarray(h.size_m) > hi_m):
            raise MoveError("the opening reads %.1f x %.1f mm, not within %.0f x %.0f .. %.0f x %.0f mm "
                            "(utensil.pick.min_opening_m .. max_opening_m): not the holder's, or the closed fingers do "
                            "not fit" % (*(1000 * np.array(h.size_m)), *(1000 * lo_m), *(1000 * hi_m)))
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


def insertion(arm: Arm, R_look: np.ndarray, h: Hole) -> np.ndarray:
    """The fingers pointing straight down (tool z down), the tool axis they open along (fingers_open_along_tool, x or
    y) along the opening's long side; MoveError when that turns them more than max_turn_deg from fork_look's (R_look),
    either way round. The gripper is the same turned 180 deg about its axis: of the two, the one whose way in (over the
    opening, in it, the lift, by IK from the arm's joints) keeps every joint furthest from its limit."""
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
        q0 = arm.q()                                   # (seeds: the arm's joints, and the last joint turned half round)
        seeds = [q0] + [list(np.asarray(q0) + [0, 0, 0, 0, 0, 0, t]) for t in (180.0, -180.0)]
        q = max((v for v in (arm.ik(tool_at(h, R, float(P().above_m)), R, sd) for sd in seeds) if v is not None),
                key=arm.margin, default=None)
        m = math.inf if q is None else arm.margin(q)
        for dz in (-float(P().insert_m), float(P().lift_m) - float(P().insert_m)):
            q = None if q is None else arm.ik(tool_at(h, R, dz), R, q)
            if q is None:
                break
            m = min(m, arm.margin(q))
        else:
            if best is None or m > best[0]:
                best = (m, R)
    if best is None:
        raise MoveError(why)
    return best[1]


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
    h = find(robot, arm, out)
    R = insertion(arm, arm.tool()[1], h)
    for k, dz in (("over", float(P().above_m)), ("in", -float(P().insert_m)),
                  ("lifted", float(P().lift_m) - float(P().insert_m))):
        q = arm.ik(tool_at(h, R, dz), R, arm.q())
        log.info("    %-7s tool (%.3f %.3f %.3f)%s", k, *tool_at(h, R, dz),
                 "" if q is None else "  joints %.0f deg from their limits" % arm.margin(q))


def tool_at(h: Hole, R: np.ndarray, dz: float) -> np.ndarray:
    """The tool point that puts the fingertips dz above the opening's centre at the rim (negative: into it),
    orientation R."""
    return h.centre + np.array([0.0, 0.0, dz]) - R @ fingertips_in_tool()


# ------------------------------------------------------------------ the routine
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
    """rammp gripper open|close, for setting up: open, the block hung on the closed fingers by hand -- the fork held
    (calibrate-wrist can run before a fork_look exists); close, the fork taken off -- not held. Where it stood is not
    known: put-fork refuses it."""
    _grip(g, float(P().gripper_open if hold else P().gripper_closed))
    save_state(dict(holding=hold))


def pick_up(robot, arm: Arm, g: Gripper, step: Step = _no_step, out: Path | None = None, stop_after: str = "") -> dict:
    p = P()
    need()
    if holding():
        raise MoveError("the gripper already holds the fork -- rammp put-fork, or rammp gripper close to take it off")
    recover(robot, arm)
    arm.hold(AtStand())
    step("look", 0.05, "Looking for the fork")
    _grip(g, float(p.gripper_closed))                  # closed: they go into the opening
    arm.travel(scene.captured("fork_look"), "to fork_look")
    step("find", 0.2, "Finding the fork's holder")
    h = find(robot, arm, out)
    R = insertion(arm, arm.tool()[1], h)
    arm.hold(AtStand(h.centre))
    step("grasp", 0.35, "Picking up the fork")
    _straight(arm, tool_at(h, R, float(p.above_m)), R, float(p.lift_mps), "over the fork", "check the stand")
    tips = arm.tool()[0] + R @ fingertips_in_tool()
    log.info("    the opening's centre (%.4f %.4f %.4f); the finger ends at (%.4f %.4f %.4f), %.1f mm off it across, "
             "going in %.0f mm under the rim", *h.centre, *tips, 1000 * math.hypot(*(tips[:2] - h.centre[:2])),
             1000 * float(p.insert_m))
    if stop_after == "above":
        return dict(stopped_after="above", opening_m=[round(float(v), 4) for v in h.centre],
                    finger_ends_m=[round(float(v), 4) for v in tips])
    s = dict(holding=False, in_holder=True, hole=h.to_json(), R=R.round(6).tolist())
    save_state(s)                                      # (the fingers going in: a stop from here comes back out)
    _straight(arm, tool_at(h, R, -float(p.insert_m)), R, float(p.insert_mps), "into the opening",
              "the fingers met the block: off its opening? check the stand and the opening's centre (find-fork)")
    if stop_after == "insert":
        return dict(stopped_after="insert")
    at = _grip(g, float(p.gripper_open))               # open in the opening: its end walls stop them, the hold
    if at is None:
        log.info("    the arm node serves no gripper position: the hold is not checked")
    elif at < float(p.gripper_open) + float(p.lock_margin):
        _grip(g, float(p.gripper_closed))
        raise MoveError("the fingers opened fully (%.2f): nothing held them -- not in the opening; closed again, the "
                        "arm where it is (the next pick or put comes straight back up first)" % at)
    else:
        log.info("    the fingers stopped at %.2f (open: %.2f): held by the opening's walls", at, float(p.gripper_open))
    save_state(dict(s, holding=True))
    if stop_after == "open":
        return dict(stopped_after="open")
    arm.hold(_fork(arm, h.centre))
    _straight(arm, tool_at(h, R, float(p.lift_m) - float(p.insert_m)), R, float(p.lift_mps), "lift the fork",
              "the block snug in the stand, or the lift dragging the stand: check it")
    s = dict(s, holding=True, in_holder=False, lifted_joints=[float(v) for v in arm.q()])
    save_state(s)
    _straight(arm, arm.tool()[0] + float(p.slide_m) * open_side(h), R, float(p.lift_mps),
              "out of the holder's open side", "the fork caught on the holder: check the stand")
    _grip(g, float(p.gripper_open), float(p.hold_force))   # clear of the stand: a firm hold, the meal's moves don't shift it
    step("seat", 0.7, "Looking at the empty holder")
    arm.travel(scene.captured("fork_look"), "to fork_look")
    E, a = strip(robot, arm, out, toward=h.centre)
    rec = seat_record(h, R, E, a)
    log.info("    the seat from the strip's post end: along %.1f, across %.1f, up %.1f mm (as utensil.pick."
             "seat_in_strip_m)", *(1000 * np.array([rec["along"], rec["across"], rec["up"]])))
    px = fork_px(robot)
    if px is None:
        log.info("    the fork in the fingers: not seen (%s) -- the put-back cannot check it", ", ".join(P().fork_prompts))
    save_state(dict(s, seat=dict(rec, fork_px=px)))
    record.event("fork_picked_up", hole=h.to_json())
    step("home", 0.8, "Going home with the fork")
    _home(robot, arm)                                  # (the arm holding AtStand's fork: a fresh Arm is the meal's)
    return dict(holding=True, opening_m=[round(float(v), 3) for v in h.centre])


def put_down(robot, arm: Arm, g: Gripper, step: Step = _no_step, stop_after: str = "") -> dict:
    p = P()
    need()
    if not state().get("holding"):
        raise MoveError("the fork is not in the gripper")
    recover(robot, arm)                                # (a pick stopped in the opening: its lift finished first)
    s = state()
    from rammp_feeding.geometry import max_step
    if s.get("lifted_joints") and max_step(arm.q(), s["lifted_joints"]) <= float(cfg.motion.near_pose_deg):
        h, R = Hole.from_json(s["hole"]), np.asarray(s["R"], float)    # lifted, never out: straight back down
        arm.hold(_fork(arm, h.centre))
        return _lower(robot, arm, g, s, h, R, step, stop_after)
    if not s.get("seat"):
        raise MoveError("where the fork's seat is on its stand is not known -- put it back by hand "
                        "(rammp gripper close)")
    arm.hold(_fork(arm, Hole.from_json(s["seat"]["hole"]).centre))
    step("place", 0.1, "Finding the fork's stand")
    from rammp_feeding.cli import home_tilt_cap
    arm.travel(scene.captured("fork_look"), "to fork_look", tilt_cap=home_tilt_cap(arm.q()))
    E, a = strip(robot, arm, record_dir("put_fork"), like=s["seat"]["a"])
    then, now = s["seat"].get("fork_px"), fork_px(robot)
    if then is None or now is None:
        log.info("    the fork in the fingers: not seen %s -- not checked", "at the pick" if then is None else "now")
    else:
        moved = (math.hypot(now[0] - then[0], now[1] - then[1])               # (pixels, at the tines' depth)
                 * float(cfg.utensil.glass_to_tip_m) / float(arm.cam.colour.fx))
        log.info("    the fork in the fingers: %.1f mm from where it hung at the pick (its tines' depth)", 1000 * moved)
        if moved > float(p.grip_shift_tol_m):
            raise MoveError("the fork has moved %.1f mm in the fingers since the pick (its tines, at most %.1f): the "
                            "block would land off its seat -- seat the fork in the fingers by hand, or put it back by "
                            "hand (rammp gripper close)" % (1000 * moved, 1000 * float(p.grip_shift_tol_m)))
    h, R, open_dir = seat_now(s["seat"], E, a)
    arm.hold(_fork(arm, h.centre))
    over = tool_at(h, R, float(p.lift_m) - float(p.insert_m))
    q0 = arm.q()
    seeds = [s.get("lifted_joints", q0), q0] + [list(np.asarray(q0) + [0, 0, 0, 0, 0, 0, t]) for t in (180.0, -180.0)]
    q_in = max((v for v in (arm.ik(over + float(p.slide_m) * open_dir, R, sd) for sd in seeds) if v is not None),
               key=arm.margin, default=None)
    if q_in is None:
        raise MoveError("beside the holder's open side is out of the arm's reach -- put the fork back by hand")
    step("place", 0.3, "Taking the fork back to its stand")
    arm.travel(q_in, "beside the holder's open side")
    _grip(g, float(p.gripper_open))                    # soft again before the stand: a firm hold drags the holder
    _straight(arm, over, R, float(p.lift_mps), "in through the open side", "the fork met the holder: check the stand")
    log.info("    over the seat: the opening's centre (%.4f %.4f %.4f)", *h.centre)
    if stop_after == "above":                         # (over the seat: the next put-back goes straight down)
        save_state(dict(s, hole=h.to_json(), R=R.round(6).tolist(), lifted_joints=[float(v) for v in arm.q()]))
        return dict(stopped_after="above", opening_m=[round(float(v), 4) for v in h.centre])
    return _lower(robot, arm, g, dict(s, hole=h.to_json(), R=R.round(6).tolist()), h, R, step, stop_after)


def _lower(robot, arm: Arm, g: Gripper, s: dict, h: Hole, R: np.ndarray, step: Step, stop_after: str) -> dict:
    """From over the seat (the lift's height): down onto it, the fingers closed (it lets go), up out, home."""
    p = P()
    s = dict(s, in_holder=True)
    save_state(s)
    seat = tool_at(h, R, -float(p.insert_m))
    try:
        _straight(arm, seat, R, float(p.insert_mps), "into the stand",
                  "the block met the stand: off its place? check the stand")
    except MoveError as e:                             # the holder taking the fork's weight trips the guard: seated
        short = float(arm.tool()[0][2] - seat[2])      # within seat_tol_m of the seat's depth, else the block is on
        if "torque" not in str(e) or short > float(p.seat_tol_m):    # something above it
            raise
        log.info("    seated %.0f mm above the seat's depth (the holder took the fork's weight)", 1000 * short)
    if stop_after == "insert":
        return dict(stopped_after="insert")
    at = _grip(g, float(p.gripper_closed))             # closed: it lets go
    if at is None:
        log.info("    the arm node serves no gripper position: the release is not checked")
    elif at < float(p.gripper_closed) - float(p.lock_margin):
        raise MoveError("the fingers did not close (%.2f, closed: %.2f): still in the block -- the arm where it is"
                        % (at, float(p.gripper_closed)))
    save_state(dict(s, holding=False))
    arm.hold(AtStand(h.centre))
    if stop_after == "release":
        return dict(stopped_after="release")
    _straight(arm, tool_at(h, R, float(p.above_m)), R, float(p.lift_mps), "out of the opening",
              "the fingers caught the block: check the stand")
    save_state(dict(holding=False))                    # (back on its stand: the next pick finds it anew)
    record.event("fork_put_down", hole=h.to_json())
    step("home", 0.8, "Going home")
    _home(robot, arm)
    return dict(put_down=True)


def recover(robot, arm: Arm) -> None:
    """The fingers in the opening after a stop (in_holder): straight up out of it, holding the fork to where the lift
    ends, empty to above_m over the rim."""
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
                  "out of the opening", "check the stand")
    save_state(dict(s, in_holder=False, **({"lifted_joints": [float(v) for v in arm.q()]} if held else {})))
    arm.hold(_fork(arm, h.centre) if held else AtStand(h.centre))
