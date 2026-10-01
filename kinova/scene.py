"""Where the table and the bowl are, and the poses that follow from them.

The workspace: the fixed scene camera's latest reading when one is configured (make locate-scene ->
<runs_dir>/scene_state.json), else rammp.yaml's `workspace`. The scene camera is coarse: its reading is trusted
to its extrinsic.error_m, which the planner's world and the utensil's floor keep as extra clearance; the wrist
camera then measures the food itself, from food_detect_start.

food_detect_start is computed, not stored: the wrist camera workspace.overlook_camera_height_m above the table,
its lens axis through the bowl, in the captured overlook's orientation (utensil down, the image oriented as
calibrated). Its joints come from continuous IK out of the captured overlook, so the arm keeps that shape.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from feeding.camera import CAM_IN_TOOL
from kinova.motion import R_from_kortex, homotopy, margin
from kinova.scripts.safe_move import Joints, MoveError
from kinova.workspace import workspace
from settings import cfg

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

_RESOLVED: dict[tuple, Joints] = {}


def table_height() -> float:
    return workspace()["table_z"]


def tip_floor(x: float, y: float) -> float:
    """The lowest the utensil tip may pass at (x, y) on a free-space move: above the bowl's top over the bowl (and
    a margin round it), above the table over its footprint (workspace.table_extent_m) -- each plus the workspace's
    error; anywhere else, no floor of ours (cuRobo's world still holds)."""
    w, fs, ext = workspace(), cfg.motion.free_space, cfg.workspace.table_extent_m
    e = w["error_m"]
    if np.hypot(x - w["bowl_xy"][0], y - w["bowl_xy"][1]) < w["bowl_radius"] + float(fs.bowl_margin_m) + e:
        return w["bowl_top_z"] + float(fs.min_tip_above_bowl_m) + e
    if ext.x[0] <= x <= ext.x[1] and ext.y[0] <= y <= ext.y[1]:
        return w["table_z"] + float(fs.min_tip_above_table_m) + e
    return -np.inf


def shift() -> np.ndarray:
    """(dx, dy, dz) m: how far the bowl (x, y) and the table (z) are from where the bowl profile was captured."""
    now, then = workspace(), cfg.bowl.captured_at
    return np.array([now["bowl_xy"][0] - float(then.bowl_centre_m[0]),
                     now["bowl_xy"][1] - float(then.bowl_centre_m[1]),
                     now["table_z"] - float(then.table_height_m)])


def overlook_pose() -> dict:
    """food_detect_start: tool_pose computed over the bowl; joints_deg the captured ones (an IK seed)."""
    pose = copy.deepcopy(dict(cfg.overlook_pose()))
    ws, t = workspace(), dict(pose["tool_pose"])
    R = R_from_kortex(t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"])
    cam = R @ CAM_IN_TOOL                                       # the lens, from the tool point, in the base frame
    cam_z = ws["table_z"] + float(cfg.workspace.overlook_camera_height_m)
    t["x_m"], t["y_m"], t["z_m"] = ws["bowl_xy"][0] - cam[0], ws["bowl_xy"][1] - cam[1], cam_z - cam[2]
    pose["tool_pose"] = t
    return pose


def overlook_joints(conn: KinovaConnection) -> Joints:
    """food_detect_start's joints: continuous IK from the captured overlook, the orientation held (a single seeded
    IK can jump to another arm shape)."""
    captured = cfg.overlook_pose()
    seed = [float(v) for v in captured["joints_deg"]]
    t = overlook_pose()["tool_pose"]
    key = tuple(round(t[k], 6) for k in ("x_m", "y_m", "z_m"))
    if key in _RESOLVED:
        return list(_RESOLVED[key])
    c = captured["tool_pose"]
    T0, T1 = np.array([c["x_m"], c["y_m"], c["z_m"]]), np.array([t["x_m"], t["y_m"], t["z_m"]])
    R = R_from_kortex(t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"])
    n = max(4, int(np.ceil(np.linalg.norm(T1 - T0) / 0.015)))
    path, why = homotopy(conn, seed, lambda s: (T0 + (T1 - T0) * s, R), n)
    if path is None:
        raise MoveError("food_detect_start over the bowl at (%.3f, %.3f) m is not reachable from the captured one "
                        "without changing the arm's shape (%s)" % (*workspace()["bowl_xy"], why))
    q = path[-1]
    if margin(q) < float(cfg.motion.plan_min_margin_deg):
        raise MoveError("food_detect_start comes within %.1f deg of a joint limit" % margin(q))
    _RESOLVED[key] = q
    return list(q)


def scoop_end() -> dict:
    """Where the scoop carries the food (the swing to the user starts there): the bowl profile's scoop end moved by
    the table's height (workspace.scoop_end_follows: table) or by the bowl and the table (bowl). joints_deg: an IK
    seed (the arm's shape)."""
    se = dict(cfg.bowl.scoop_end)
    d = shift()
    if str(cfg.workspace.scoop_end_follows) == "table":
        d = np.array([0.0, 0.0, d[2]])
    se["tool_m"] = [float(se["tool_m"][i]) + float(d[i]) for i in range(3)]
    return se
