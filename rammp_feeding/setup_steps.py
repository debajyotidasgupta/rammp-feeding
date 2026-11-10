"""rammp setup: what changes when the arm moves against the table, the bowl and the user, step by step. Each step says
what to do first; Enter runs it, s skips it, q stops. `rammp setup <step> ...` runs only those.

  site      the tape-measured values in local.yaml (the table's edges, where the bowl is laid): shown to confirm
  home      feeding_start, posed by hand, captured
  scene     the scene camera's pose through the ChArUco board (rammp calibrate-scene), from feeding_start
  overlook  food_detect_start's arm shape, posed by hand over the bowl, captured
  bowl      the scene camera's reading, then the wrist camera's from over the bowl (rammp detect); its centre can be
            written to local.yaml as workspace.bowl_centre_m
  deliver   the delivery look, posed by hand facing the seated user, captured; checked for a face in view
  home      at the end, back to feeding_start (a planned path), then rammp check

Not here: what is fixed to the arm (the wrist camera's mount, the utensil: rammp calibrate-wrist), and the mouth
(measured on every delivery). Posing by hand: stop the arm node, move the arm with the Kinova Web App, restart it.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable

import numpy as np

from rammp_feeding import record, scene
from rammp_feeding.config import cfg, state_dir
from rammp_feeding.geometry import fold

log = logging.getLogger("rammp")
STEPS = ("site", "home", "scene", "overlook", "bowl", "deliver")
POSE = {"home": "feeding_start: the utensil straight down, the arm parked beside the table and clear of the user, "
                "with room above it (calibrate-scene rises 0.10 m and moves 0.15 m out from there)",
        "overlook": "over the bowl: the utensil straight down, the wrist camera looking down at the bowl",
        "deliver": "facing the seated user: the utensil level and pointing at their mouth, the wrist camera on their "
                   "face, the food out of its view of the mouth"}
BY_HAND = ("Stop the arm node (sheppy stop arm), move the arm with the Kinova Web App, then restart it "
           "(sheppy restart arm).")


def capture(arm, name: str) -> list[float]:
    """The arm's joints now -> state_dir/<name>.json: the median of 30 readings, folded to (-180, 180]."""
    qs = []
    for _ in range(30):
        qs.append(arm.q())
        time.sleep(0.02)
    q = [round(float(v), 3) for v in fold(np.asarray(qs[0]) + np.median(fold(np.asarray(qs) - qs[0]), axis=0))]
    (state_dir() / (name + ".json")).write_text(json.dumps(dict(joints_deg=q, captured=time.strftime("%F %T")), indent=1))
    log.info("  %s <- %s", state_dir() / (name + ".json"), q)
    return q


def write_bowl_centre(xy) -> bool:
    """workspace.bowl_centre_m in local.yaml set to xy, in place (the line must exist: the template has it)."""
    p = state_dir() / "local.yaml"
    text = p.read_text() if p.exists() else ""
    line = re.compile(r"^(\s*)bowl_centre_m:.*$", re.M)
    if not line.search(text):
        return False
    p.write_text(line.sub(lambda m: "%sbowl_centre_m: [%.3f, %.3f]    # rammp setup: the wrist camera's reading, %s"
                          % (m.group(1), xy[0], xy[1], time.strftime("%F")), text, count=1))
    return True


def _fresh_joints(robot, since: float, wait_s: float = 120.0) -> None:
    """Wait for joint states newer than `since`: the arm node is back after a pose by hand."""
    t_end = time.time() + wait_s
    while robot.joint_sample()[0] <= since:
        if time.time() > t_end:
            raise RuntimeError("no joint states from the arm node in %.0f s -- is it running?" % wait_s)
        time.sleep(0.1)


def run(robot, arm, steps=STEPS, ask: Callable[[str], str] = input, face=None) -> int:
    from rammp_feeding.cli import check, detect, home

    def go(step: str, what: str) -> bool:
        a = ask("\n[%s] %s\n  Enter: go   s: skip   q: stop > " % (step, what)).strip().lower()
        if a == "q":
            raise KeyboardInterrupt
        return a != "s"

    def posed(name: str) -> None:
        t = time.time()
        if go(name, "Pose the arm at %s. %s Then Enter to capture it." % (POSE[name], BY_HAND)):
            _fresh_joints(robot, t)
            capture(arm, name)

    for step in [s for s in STEPS if s in steps]:
        if step == "site":
            ws = cfg.workspace
            log.info("  workspace.table_extent_m: %s\n  workspace.bowl_centre_m:  %s",
                     dict(x=list(ws.table_extent_m.x), y=list(ws.table_extent_m.y)), list(ws.bowl_centre_m))
            go("site", "The table's edges and where the bowl is laid, from the arm's base (local.yaml), as above. "
                       "If the arm moved, tape-measure and correct them, then run rammp setup again (q). Enter: they "
                       "are right")
        elif step in ("home", "overlook", "deliver"):
            posed(step)
            if step == "deliver" and (state_dir() / "deliver.json").exists():
                if face is None:
                    from rammp_feeding.face import Face
                    face = Face()
                seen = face.find(robot.cameras["camera"], float(cfg.delivery.face_find_s))
                log.info("  the wrist camera %s from the delivery look", "sees a face" if seen else
                         "sees NO face -- pose it again with the user seated (rammp setup deliver)")
        elif step == "scene":
            if go("scene", "Lay the ChArUco board on the table where the scene camera sees it and the arm can reach "
                           "over it. The arm must be at feeding_start. Enter: rammp calibrate-scene"):
                from rammp_feeding.scene_camera import calibrate
                if not calibrate(robot, arm):
                    return 1
        elif step == "bowl":
            if go("bowl", "Lay the bowl where it goes. Enter: the scene camera reads it, then the arm goes over it "
                          "and the wrist camera measures it (rammp detect)"):
                from rammp_feeding.scene_camera import locate
                locate(robot, record.session("scene"))
                detect(robot, arm)
                xy = scene.workspace()["bowl_xy"]
                if go("bowl", "The bowl's centre is (%.3f, %.3f). Enter: write it to local.yaml as "
                              "workspace.bowl_centre_m" % tuple(xy)):
                    log.info("  %s", "local.yaml: workspace.bowl_centre_m set" if write_bowl_centre(xy) else
                             "local.yaml has no bowl_centre_m line: add workspace.bowl_centre_m: [%.3f, %.3f]" % tuple(xy))
    if not go("home", "Enter: back to feeding_start (a planned path), then rammp check"):
        return 0
    home(robot, arm)
    return check(robot, arm)
