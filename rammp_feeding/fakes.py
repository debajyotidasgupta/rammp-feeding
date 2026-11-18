"""Stand-ins for what the robot senses, each chosen in config (fakes.*; null: the real one). The arm, the planner and the
gripper stay real: a run with fakes moves the arm to the fake things, through the same moves, checks and records.

  food     the target (perception.perceive): the pieces at fakes.food.pieces_m lying in the bowl, taken in turn nearest
           the bowl's centre first (state_dir/fakes_state.json keeps which are gone); none left: the bowl is empty
  under    what is under the tip (acquire.measure): the fake piece's top where the tip is over one, the bowl's floor
           round it
  mouth    the user's face (face.Face): the mouth at fakes.mouth.at_m, seen from a look that has it in the wrist camera's
           image, open fakes.open_after_s into the wait; the bite (deliver.bite) felt fakes.bite_after_s after the approach
  holder   the fork's opening (fork.find) at fakes.holder, and the stand's strip (fork.strip) read from it

sensing.py picks the real or the fake one for each.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from rammp_feeding import record, scene
from rammp_feeding.config import cfg, state_dir

log = logging.getLogger("rammp")


def on(kind: str) -> bool:
    """Whether fakes.<kind> stands in for the real sensing (food, mouth, holder)."""
    return cfg.get("fakes." + kind, None) is not None


# ------------------------------------------------------------------ food
def _state() -> dict:
    f = state_dir() / "fakes_state.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _pieces() -> list[np.ndarray]:
    """The fake pieces still in the bowl."""
    gone = _state().get("gone", [])
    return [np.asarray(p, float) for i, p in enumerate(cfg.fakes.food.pieces_m) if i not in gone]


def taken(xy) -> None:
    """The fake piece at xy is on the fork: gone from the bowl for the next target."""
    pieces = [np.asarray(p, float) for p in cfg.fakes.food.pieces_m]
    i = int(np.argmin([np.linalg.norm(p - np.asarray(xy, float)[:2]) for p in pieces]))
    s = _state()
    s["gone"] = sorted(set(s.get("gone", [])) | {i})
    (state_dir() / "fakes_state.json").write_text(json.dumps(s))


def refill() -> None:
    """Every fake piece back in the bowl (a new meal)."""
    (state_dir() / "fakes_state.json").unlink(missing_ok=True)


def _floor_z() -> float:
    return scene.workspace()["table_z"] + float(cfg.bowl.floor_above_table_m)


def perceive(robot, arm, out: Path | None) -> dict:
    """As perception.perceive, from food_detect_start: the fake piece nearest the bowl's centre, the tool's goal the level
    slide that puts the tip over it; NoTarget(3) when none is left."""
    from rammp_feeding.perception import NoTarget
    q = arm.q()
    scene.at_food_detect_start(arm, q)
    ws = scene.workspace()
    left = _pieces()
    if not left:
        raise NoTarget("no food in the bowl (the fake pieces are all taken)", 3)
    p = min(left, key=lambda p: math.hypot(p[0] - ws["bowl_xy"][0], p[1] - ws["bowl_xy"][1]))
    T, _ = arm.tool(q)
    tip = arm.tip(q)
    goal = [float(T[0] + p[0] - tip[0]), float(T[1] + p[1] - tip[1]), float(T[2])]
    g = dict(xy=[float(v) for v in p], top_z=_floor_z() + float(cfg.fakes.piece_height_m))
    tgt = dict(piece=dict(g, fake=True), tool_now=[float(v) for v in T], tool_goal=goal, t=time.time())
    record.save_json(out, "target.json", tgt)
    log.info("  target (fake): the piece at (%.3f, %.3f), its top z %.3f; %d left", *g["xy"], g["top_z"], len(left))
    record.event("perceived", target=tgt, pieces=len(left), fake=True)
    return tgt


def under_tip(arm) -> dict:
    """As perception.under_tip: the fake piece's top straight under the tip (within fakes.piece_radius_m of one), else
    the bowl's floor; the floor round it."""
    tip = arm.tip()
    floor = _floor_z()
    over = [p for p in _pieces() if math.hypot(p[0] - tip[0], p[1] - tip[1]) < float(cfg.fakes.piece_radius_m)]
    top = floor + float(cfg.fakes.piece_height_m) if over else floor
    return dict(top_z=top, gap_top=float(tip[2] - top), tip=[float(v) for v in tip], floor_z=floor,
                gap_floor=float(tip[2] - floor), height=top - floor, fake=True)


# ------------------------------------------------------------------ the user
class Face:
    """face.Face's interface for a mouth at fakes.mouth.at_m: where the wrist camera would see it from the arm's pose."""

    def __init__(self, arm):
        self.arm = arm

    def orient(self, R) -> None:
        pass

    def _cam(self) -> np.ndarray:
        o, Rc = scene.camera_pose(self.arm)
        return Rc.T @ (np.asarray(cfg.fakes.mouth.at_m, float) - o)

    def mouth(self, rgb, depth, cam) -> dict | None:
        """The mouth in the camera's optical frame; None when it falls outside the image."""
        c = self._cam()
        u, v = cam.pixel(c) if c[2] > 0 else (-1.0, -1.0)
        if not (0 <= u < cam.colour.width and 0 <= v < cam.colour.height):
            return None
        return dict(cam_xyz=[float(x) for x in c], source="depth", open=1.0, px=[float(u), float(v)], fake=True)

    def find(self, cam, wait_s: float) -> bool:
        return self.mouth(None, None, self.arm.cam) is not None

    def wait_open(self, cam, timeout_s: float, out: Path | None):
        wait = float(cfg.fakes.open_after_s)
        if wait > timeout_s:
            time.sleep(timeout_s)
            return None
        time.sleep(wait)
        log.info("    mouth open (fake) after %.1f s", wait)
        return cam.frame()


def bite(robot, arm, timeout_s: float, settle_s: float | None = None) -> int:
    """As deliver.bite: felt fakes.bite_after_s after it starts waiting (4 past timeout_s)."""
    wait = float(cfg.fakes.bite_after_s)
    time.sleep(min(wait, timeout_s))
    if wait > timeout_s:
        log.info("    no bite in %.0f s (fake)", timeout_s)
        return 4
    log.info("    BITE (fake, after %.1f s)", wait)
    return 0


# ------------------------------------------------------------------ the fork's stand
def holder():
    """The fake holder's opening (fork.Hole): fakes.holder's centre and long side, utensil.pick.opening_m's size."""
    from rammp_feeding.fork import Hole
    H = cfg.fakes.holder
    a = math.radians(float(H.long_deg))
    short, long, deep = (float(v) for v in (cfg.utensil.pick.get("opening_m", None) or (0.024, 0.058, 0.036)))
    return Hole(np.asarray(H.centre_m, float), np.array([math.cos(a), math.sin(a), 0.0]), (short, long), deep)


def find(robot, arm, out: Path | None = None):
    """As fork.find: the fake holder's opening."""
    h = holder()
    record.save_json(out, "opening.json", dict(h.to_json(), fake=True))
    log.info("  the opening (fake) at (%.3f %.3f %.3f), its long side %.0f deg round from x", *h.centre,
             math.degrees(math.atan2(h.long_axis[1], h.long_axis[0])))
    return h


def strip(robot, arm, out: Path | None = None, toward=None, like=None) -> tuple[np.ndarray, np.ndarray]:
    """As fork.strip: the fake holder's own frame -- its opening's centre and its long side (the seat is kept in it)."""
    h = holder()
    return np.asarray(h.centre, float), np.asarray(h.long_axis, float)


def fork_px(robot):
    """As fork.fork_px: not seen (the put-back does not check the fork in the fingers)."""
    return None
