"""The arm side of motion.planner: curobo -- ask the planner server (kinova/planner_server.py) for a free-space path.

    from kinova.curobo_client import enabled, plan
    if enabled():
        path = plan(start_deg, goal_deg, carrying=False)     # [[7 deg], ...] from start to goal; MoveError if none

The path is only geometry here: it is run through kinova.motion.run_path like every planned path (the arm's speed
limits and the controller's validation set its timing; the torque watch and telemetry as always), and is
thinned to the joint step the controller is happy with.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np

from kinova.scripts.safe_move import MoveError, fold
from settings import cfg


def enabled() -> bool:
    return str(cfg.motion.get("planner", "scripted")) == "curobo"


def _post(path: str, body: dict, timeout: float) -> dict:
    url = "http://127.0.0.1:%d%s" % (int(cfg.deployment.servers.planner_port), path)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except OSError as e:
        raise MoveError("the planner server is not answering (%s) -- make planner, or motion.planner: scripted" % e)


def refresh_world() -> None:
    """The planner re-reads the workspace and the user's mouth (after make locate-scene / a delivery look)."""
    _post("/world", {}, 60.0)


def check_points(conn, path) -> tuple[float, float]:
    """(highest, lowest) utensil tip z at a planned path's waypoints; refuses one below scene.tip_floor() -- cuRobo's
    world is a model: this is our own floor (above the bowl's top over it, above the table elsewhere), checked."""
    from kinova import scene
    from kinova.motion import fk_tip
    tips = [fk_tip(conn, q) for q in path]
    for t in tips:
        if t[2] < scene.tip_floor(t[0], t[1]) - 1e-6:
            raise MoveError("the planned path takes the utensil tip to z %.3f at (%.3f, %.3f), below the floor there "
                            "(z %.3f) -- not moving" % (t[2], t[0], t[1], scene.tip_floor(t[0], t[1])))
    zs = [float(t[2]) for t in tips]
    return max(zs), min(zs)


def planned(conn, start, goal, label: str, dps: float, torque_limit: float, carrying: bool = False,
            go: bool = True) -> list[list[float]]:
    """A cuRobo path from start to goal, floor-checked, then validated and run by kinova.motion.run_path (the arm's
    speed limits, the controller's validation, the torque watch, the telemetry)."""
    import logging
    from kinova.motion import run_path
    path = plan(start, goal, carrying=carrying)
    hi, lo = check_points(conn, path)
    logging.getLogger("rammp.curobo").info("    %-26s cuRobo: %d waypoints, utensil tip z %.3f .. %.3f", label, len(path), lo, hi)
    run_path(conn, path, dps, label, go, torque_limit=torque_limit)
    return path


def plan(start_deg, goal_deg, carrying: bool = False) -> list[list[float]]:
    start = [fold(v) for v in start_deg]
    goal = [s + fold(g - s) for s, g in zip(start, (fold(v) for v in goal_deg))]   # the short way, as goto does
    r = _post("/plan", dict(start=start, goal=goal, carrying=bool(carrying)), float(cfg.motion.planner_curobo.timeout_s))
    if not r.get("ok"):
        raise MoveError("cuRobo found no collision-free path (%s): %s" % (r.get("status"), r.get("error")))
    path = np.array(r["path"], float)
    path[0] = start                                          # exactly from here (the planner starts within 1e-4)
    keep = [0]
    step = float(cfg.motion.planner_curobo.path_step_deg)
    for i in range(1, len(path) - 1):                        # thin to ~step deg on the furthest joint
        if np.abs(path[i] - path[keep[-1]]).max() >= step:
            keep.append(i)
    keep.append(len(path) - 1)
    return [[float(v) for v in path[i]] for i in keep]
