"""What the robot senses, one function each, real or fake (fakes.py, chosen in config: fakes.*). Every caller goes
through here, so a stand-in replaces the real sensing everywhere at once:

    from rammp_feeding import sensing
    target = sensing.now().target(robot, arm, out)

  target(robot, arm, out)              the piece to pick, from food_detect_start (perception.perceive)
  under(robot, arm, out, name)         what is under the utensil tip (acquire.measure's reading)
  face(arm)                            the user's face: orient, find, wait_open, mouth (face.Face)
  bite(robot, arm, timeout_s, settle)  the bite at the mouth: 0, or 4 none (deliver.bite)
  opening(robot, arm, out)             the fork's opening on its stand (fork.find)
  strip(robot, arm, out, toward, like) the stand's strip (fork.strip)
  fork_px(robot)                       the fork in the fingers, in the wrist image (fork.fork_px)

use(Sensing(...)) puts any set of them in place (the tests' own); use(None) goes back to the config's choice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from rammp_feeding import fakes


@dataclass
class Sensing:
    target: Callable
    under: Callable
    face: Callable
    bite: Callable
    opening: Callable
    strip: Callable
    fork_px: Callable


_FACE: dict = {}
_USE: dict = {"sensing": None}


def _real_face(arm):
    """MediaPipe's landmarker, loaded once (about a second) and kept."""
    if "face" not in _FACE:
        from rammp_feeding.face import Face
        _FACE["face"] = Face()
    return _FACE["face"]


def _real_under(robot, arm, out, name):
    from rammp_feeding import record
    from rammp_feeding.arm import MoveError
    from rammp_feeding.config import cfg
    from rammp_feeding.perception import under_tip
    D = robot.cameras["camera"].depth_median(int(cfg.perception.depth_frames), time.time(),
                                             wait_s=float(cfg.perception.depth_wait_s))
    record.save_npy(out, "depth_%s.npy" % name, D)
    try:
        return under_tip(D, arm)
    except ValueError as e:
        raise MoveError(str(e))


def configured() -> Sensing:
    """The config's choice: each fake where fakes.<kind> is set, the real sensing otherwise."""
    from rammp_feeding import deliver, fork, perception
    food, mouth, holder = fakes.on("food"), fakes.on("mouth"), fakes.on("holder")
    return Sensing(
        target=fakes.perceive if food else perception.perceive,
        under=(lambda robot, arm, out, name: fakes.under_tip(arm)) if food else _real_under,
        face=fakes.Face if mouth else _real_face,
        bite=fakes.bite if mouth else deliver.bite,
        opening=fakes.find if holder else fork.find,
        strip=fakes.strip if holder else fork.strip,
        fork_px=fakes.fork_px if holder else fork.fork_px)


def now() -> Sensing:
    return _USE["sensing"] or configured()


def use(s: Sensing | None) -> None:
    _USE["sensing"] = s
