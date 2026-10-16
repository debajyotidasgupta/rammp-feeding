"""Free-space moves: a collision-free path from the planner server (kinova/planner_server.py), run through our guards.

    from kinova.curobo_client import travel
    travel(conn, goal_deg, "to food_detect_start")          # plan, floor-check, run, settle onto the goal
    travel(conn, goal_deg, "swing", carrying=True)          # with a piece on the utensil

The path is geometry only: kinova.motion.run_path times it under the arm's speed limits, has the controller validate
it, and watches the torque on the way. Every waypoint's utensil tip must also clear our own floor
(scene.tip_floor: above the table, above the bowl's top over the bowl) -- cuRobo's world is a model of the scene.
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np

import runlog
from kinova.scripts.safe_move import MoveError, fold, goto, joints
from settings import cfg

log = runlog.get("curobo")
FS = cfg.motion.free_space


def _post(path: str, body: dict, timeout: float) -> dict:
    url = "http://127.0.0.1:%d%s" % (int(cfg.deployment.servers.planner_port), path)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except OSError as e:
        raise MoveError("the planner server is not answering (%s) -- make planner" % e)


def refresh_world() -> None:
    """The planner re-reads the workspace and the user's mouth (after make locate-scene / a delivery look)."""
    _post("/world", {}, 60.0)


def plan(start_deg, goal_deg, carrying: bool = False) -> list[list[float]]:
    """A collision-free joint path from start to goal (deg), thinned to motion.free_space.path_step_deg."""
    start = [fold(v) for v in start_deg]
    goal = [s + fold(g - s) for s, g in zip(start, (fold(v) for v in goal_deg))]   # each joint the short way
    r = _post("/plan", dict(start=start, goal=goal, carrying=bool(carrying)), float(FS.timeout_s))
    if not r.get("ok"):
        raise MoveError("cuRobo found no collision-free path (%s): %s" % (r.get("status"), r.get("error")))
    path = np.array(r["path"], float)
    path[0] = start                                          # exactly from here (the planner starts within 1e-4)
    keep, step = [0], float(FS.path_step_deg)
    for i in range(1, len(path) - 1):
        if np.abs(path[i] - path[keep[-1]]).max() >= step:
            keep.append(i)
    keep.append(len(path) - 1)
    return [[float(v) for v in path[i]] for i in keep]


def check_floor(conn, path) -> tuple[float, float]:
    """(lowest, highest) utensil tip z at the path's waypoints; refuses one below scene.tip_floor()."""
    from kinova import scene
    from kinova.motion import fk_tip
    tips = [fk_tip(conn, q) for q in path]
    for t in tips:
        floor = scene.tip_floor(t[0], t[1])
        if t[2] < floor - 1e-6:
            raise MoveError("the planned path takes the utensil tip to z %.3f at (%.3f, %.3f), below the floor there "
                            "(z %.3f) -- not moving" % (t[2], t[0], t[1], floor))
    zs = [float(t[2]) for t in tips]
    return min(zs), max(zs)


def check_reach(conn, path, max_r: float) -> float:
    """The utensil tip's furthest horizontal distance from the arm's base axis on the path; refuses past max_r."""
    from kinova.motion import fk_tip
    r = max(float(np.hypot(*fk_tip(conn, q)[:2])) for q in path)
    if r > max_r + 1e-6:
        raise MoveError("the planned path swings the utensil tip %.0f mm out from the arm's base (at most %.0f) -- "
                        "not moving" % (1000 * r, 1000 * max_r))
    return r


def travel(conn, goal, label: str, carrying: bool = False, go: bool = True, dps: float | None = None,
           torque_limit: float | None = None, max_tilt: float | None = None, max_reach_m: float | None = None,
           start=None) -> list[list[float]]:
    """Plan from where the arm is (or `start`, plan only) to goal, check it, run it, settle exactly onto goal.
    go=False: plan and check only. max_reach_m: the tip stays within this of the base axis (re-planned, else refused)."""
    from kinova.motion import max_step, run_path
    q = list(start) if start is not None else joints(conn)
    if start is not None and go:
        raise MoveError("travel from a given start is plan-only")
    settle = dict(speed=float(cfg.motion.settle_dps), label="settle %s" % label)
    if max_tilt is not None:
        settle["max_tilt"] = max_tilt
    if max_step(q, goal) <= float(cfg.motion.near_pose_deg):
        if go:
            goto(conn, goal, **settle)
        return [q, list(goal)]
    for k in range(int(FS.floor_retries) + 1):            # cuRobo does not know our floor: another plan may clear it
        path = plan(q, goal, carrying=carrying)
        try:
            lo, hi = check_floor(conn, path)
            if max_reach_m is not None:
                check_reach(conn, path, max_reach_m)
            break
        except MoveError as e:
            if k == int(FS.floor_retries):
                raise
            log.info("    %-26s re-planning (%d/%d): %s", label, k + 1, int(FS.floor_retries), str(e).split(" -- ")[0])
    log.info("    %-26s planned: %d waypoints, utensil tip z %.3f .. %.3f%s", label, len(path), lo, hi,
             "" if go else "  (plan only)")
    runlog.event("planned", label=label, waypoints=len(path), tip_z=[lo, hi], carrying=carrying)
    if not go:
        return path
    run_path(conn, path, float(dps or FS.dps), label, True,
             torque_limit=float(torque_limit or cfg.guards.torque_step_nm.free_space))
    goto(conn, goal, **settle)
    return path
