"""Where the table and the bowl are, and the poses that follow from them.

The bowl profile's poses (food_detect_start, the scoop end) were captured with the table and bowl at the
profile's `captured_at`. The deployment says where they are now (rammp.yaml: workspace). Every bowl-dependent
pose is the captured one shifted by the difference -- the camera keeps its orientation and its height above the
bowl, so every orientation-only calibration (vertical_vp, tip pixel, aim offset, camera-to-tip) still holds:

    shift = (workspace.bowl_centre_m - captured_at.bowl_centre_m,  workspace.table_height_m - captured_at.table_height_m)

With no shift the captured joints are used exactly, as before. With one, food_detect_start's joints come from
the controller's IK, seeded with the captured joints (so the arm keeps the same shape), and are refused if
the arm would have to change shape or come near a joint limit -- then re-capture the pose at the new place.

Table-relative heights: the retract's lowest utensil tip (delivery.retract.min_tip_above_table_m) and the
lift's clearance over the table (motion.sequence.min_lift_above_table_m).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from settings import cfg
from kinova.motion import _ik, margin, max_step
from kinova.scripts.safe_move import Joints, MoveError

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

_RESOLVED: dict[tuple, Joints] = {}


def table_height() -> float:
    """The table surface, base-frame z (m)."""
    return float(cfg.workspace.table_height_m)


def shift() -> np.ndarray:
    """(dx, dy, dz) m: how far the bowl (x, y) and the table (z) are from where the bowl's poses were captured."""
    now, then = cfg.workspace, cfg.bowl.captured_at
    return np.array([float(now.bowl_centre_m[0]) - float(then.bowl_centre_m[0]),
                     float(now.bowl_centre_m[1]) - float(then.bowl_centre_m[1]),
                     float(now.table_height_m) - float(then.table_height_m)])


def shifted() -> bool:
    return bool(np.any(np.abs(shift()) > 1e-6))


def overlook_pose() -> dict:
    """food_detect_start as captured, with its tool pose moved by shift(). joints_deg is the CAPTURED joints
    (exact when there is no shift; only an IK seed otherwise -- use overlook_joints())."""
    pose = copy.deepcopy(dict(cfg.overlook_pose()))
    if shifted():
        d = shift()
        t = dict(pose["tool_pose"])
        t["x_m"], t["y_m"], t["z_m"] = t["x_m"] + d[0], t["y_m"] + d[1], t["z_m"] + d[2]
        pose["tool_pose"] = t
    return pose


def overlook_joints(conn: KinovaConnection) -> Joints:
    """food_detect_start's joints for the bowl and table where they are now."""
    captured = cfg.overlook_pose()
    seed = [float(v) for v in captured["joints_deg"]]
    if not shifted():
        return seed
    key = tuple(np.round(shift(), 6))
    if key in _RESOLVED:
        return list(_RESOLVED[key])
    t = overlook_pose()["tool_pose"]
    q = _ik(conn, (t["x_m"], t["y_m"], t["z_m"]), (t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"]), seed)
    d = 1000 * shift()
    where = "the bowl moved (%+.0f, %+.0f) mm and the table %+.0f mm from where food_detect_start was captured" % tuple(d)
    if q is None:
        raise MoveError("food_detect_start is not reachable: %s -- re-capture it there" % where)
    if max_step(q, seed) > float(cfg.workspace.max_shape_change_deg):
        raise MoveError("food_detect_start needs a different arm shape (%.0f deg on one joint): %s -- re-capture it"
                        % (max_step(q, seed), where))
    if margin(q) < float(cfg.motion.plan_min_margin_deg):
        raise MoveError("food_detect_start comes within %.1f deg of a joint limit: %s -- re-capture it" % (margin(q), where))
    _RESOLVED[key] = q
    return list(q)


def scoop_end() -> dict:
    """Where the scoop carries the food to: tool_m moved by shift(); joints_deg is an IK seed (the arm's shape)."""
    se = dict(cfg.bowl.scoop_end)
    d = shift()
    se["tool_m"] = [float(se["tool_m"][i]) + float(d[i]) for i in range(3)]
    return se


def min_tip_z() -> float:
    """The lowest the utensil tip may go on the retract's joint move home (base z, m)."""
    return round(table_height() + float(cfg.delivery.retract.min_tip_above_table_m), 6)
