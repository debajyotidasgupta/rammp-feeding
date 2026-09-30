"""Where the table and the bowl are, and the poses that follow from them.

The bowl profile's poses (food_detect_start, the scoop end) were captured with the table and bowl at the
profile's `captured_at`. The deployment says where they are now (rammp.yaml: workspace). Every bowl-dependent
pose is the captured one shifted by the difference -- the camera keeps its orientation and its height above the
bowl, so every orientation-only calibration (vertical_vp, tip pixel, aim offset, camera-to-tip) still holds:

    shift = (workspace.bowl_centre_m - captured_at.bowl_centre_m,  workspace.table_height_m - captured_at.table_height_m)

With no shift the captured joints are used exactly, as before. With one, food_detect_start's joints come from
the controller's IK, seeded with the captured joints (so the arm keeps the same shape), and are refused if
the arm would have to change shape or come near a joint limit -- then re-capture the pose at the new place.

Where the bowl and table are ("the workspace"): from the fixed scene camera when one is configured
(make locate-scene -> <runs_dir>/scene_state.json, coarse, ~5 cm), else rammp.yaml's `workspace`.

With workspace.overlook_camera_height_m set, food_detect_start is COMPUTED, not shifted: the wrist camera's lens
axis through the bowl centre at that height above the table, the captured orientation (utensil down, the same
yaw: the image stays oriented as calibrated), joints from the controller's IK seeded with the captured shape. The
lift from feeding_start then goes no higher than that overlook (lift_z()), not to the captured lift top.

Table-relative heights: the retract's lowest utensil tip (delivery.retract.min_tip_above_table_m) and the
lift's clearance over the table (motion.sequence.min_lift_above_table_m).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

import json
import os
import time
from pathlib import Path

from feeding.camera import CAM_IN_TOOL
from settings import cfg
from kinova.motion import R_from_kortex, _ik, homotopy, margin, max_step
from kinova.scripts.safe_move import Joints, MoveError

if TYPE_CHECKING:
    from kinova.connection import KinovaConnection

_RESOLVED: dict[tuple, Joints] = {}


def workspace() -> dict:
    """{"bowl_xy", "table_z", "source"}: the scene camera's latest (fresh) reading, else rammp.yaml's workspace."""
    w = cfg.workspace
    if cfg.profiles.get("scene_camera"):
        f = Path(os.path.expanduser(str(cfg.deployment.runs_dir))) / "scene_state.json"
        try:
            st = json.loads(f.read_text())
            if time.time() - float(st["t"]) < float(w.scene_max_age_s):
                return dict(bowl_xy=[float(v) for v in st["bowl_xy"]], table_z=float(st["table_z"]),
                            bowl_top_z=float(st.get("bowl_top_z", st["table_z"])),
                            source="scene camera (%s)" % time.strftime("%H:%M:%S", time.localtime(st["t"])))
        except (OSError, ValueError, KeyError):
            pass
    return dict(bowl_xy=[float(v) for v in w.bowl_centre_m], table_z=float(w.table_height_m),
                bowl_top_z=float(w.table_height_m) + float(cfg.workspace.get("bowl_height_m", 0.06)), source="config")


def table_height() -> float:
    """The table surface, base-frame z (m)."""
    return workspace()["table_z"]


def shift() -> np.ndarray:
    """(dx, dy, dz) m: how far the bowl (x, y) and the table (z) are from where the bowl's poses were captured."""
    now, then = workspace(), cfg.bowl.captured_at
    return np.array([now["bowl_xy"][0] - float(then.bowl_centre_m[0]),
                     now["bowl_xy"][1] - float(then.bowl_centre_m[1]),
                     now["table_z"] - float(then.table_height_m)])


def computed() -> bool:
    """food_detect_start computed from the workspace (a set camera height), not the captured pose shifted."""
    return cfg.workspace.get("overlook_camera_height_m") is not None


def shifted() -> bool:
    return bool(np.any(np.abs(shift()) > 1e-6))


def overlook_pose() -> dict:
    """food_detect_start: computed over the bowl (computed()), or as captured with its tool pose moved by shift().
    joints_deg is the CAPTURED joints (exact when captured and unshifted; else only an IK seed -- overlook_joints())."""
    pose = copy.deepcopy(dict(cfg.overlook_pose()))
    if computed():
        ws, t = workspace(), dict(pose["tool_pose"])
        R = R_from_kortex(t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"])
        cam = R @ CAM_IN_TOOL                                   # the lens, from the tool point, in the base frame
        cam_z = ws["table_z"] + float(cfg.workspace.overlook_camera_height_m)
        t["x_m"], t["y_m"], t["z_m"] = ws["bowl_xy"][0] - cam[0], ws["bowl_xy"][1] - cam[1], cam_z - cam[2]
        pose["tool_pose"] = t
        return pose
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
    if not shifted() and not computed():
        return seed
    key = tuple(np.round(shift(), 6)) + (computed(),)
    if key in _RESOLVED:
        return list(_RESOLVED[key])
    t = overlook_pose()["tool_pose"]
    if computed():
        # walked there from the captured overlook by continuous IK (the orientation held): the arm keeps its shape
        # family however far the computed pose is -- a single seeded IK jumped branches (161 deg on a joint, 2026-09-30)
        c = captured["tool_pose"]
        T0 = np.array([c["x_m"], c["y_m"], c["z_m"]]); T1 = np.array([t["x_m"], t["y_m"], t["z_m"]])
        R = R_from_kortex(t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"])
        n = max(4, int(np.ceil(np.linalg.norm(T1 - T0) / 0.015)))
        path, why = homotopy(conn, seed, lambda s: (T0 + (T1 - T0) * s, R), n)
        if path is None:
            raise MoveError("food_detect_start over the bowl at (%.3f, %.3f) m is not reachable from the captured "
                            "one without changing the arm's shape (%s)" % (*workspace()["bowl_xy"], why))
        q = path[-1]
        if margin(q) < float(cfg.motion.plan_min_margin_deg):
            raise MoveError("the computed food_detect_start comes within %.1f deg of a joint limit" % margin(q))
        _RESOLVED[key] = q
        return list(q)
    q = _ik(conn, (t["x_m"], t["y_m"], t["z_m"]), (t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"]), seed)
    d = 1000 * shift()
    where = "the bowl moved (%+.0f, %+.0f) mm and the table %+.0f mm from where food_detect_start was captured" % tuple(d)
    if computed():
        where = ("over the bowl at (%.3f, %.3f) m, the camera %.0f mm above the table (%s)"
                 % (*workspace()["bowl_xy"], 1000 * float(cfg.workspace.overlook_camera_height_m), workspace()["source"]))
    if q is None:
        raise MoveError("food_detect_start is not reachable: %s -- re-capture it there" % where)
    if max_step(q, seed) > float(cfg.workspace.max_shape_change_deg):
        raise MoveError("food_detect_start needs a different arm shape (%.0f deg on one joint): %s -- re-capture it"
                        % (max_step(q, seed), where))
    if margin(q) < float(cfg.motion.plan_min_margin_deg):
        raise MoveError("food_detect_start comes within %.1f deg of a joint limit: %s -- re-capture it" % (margin(q), where))
    _RESOLVED[key] = q
    return list(q)


def lift_z() -> float:
    """The top of the lift from feeding_start (tool z): with a computed overlook, no higher than it (plus
    motion.sequence.lift_margin_m) and at least the table clearance; else the configured lift top, as it is (a
    table too high for it is refused by the lift's own clearance check)."""
    if computed():
        low = table_height() + float(cfg.motion.sequence.min_lift_above_table_m)
        return max(overlook_pose()["tool_pose"]["z_m"] + float(cfg.motion.sequence.lift_margin_m), low)
    return float(cfg.motion.sequence.lift_z_m)


def scoop_end() -> dict:
    """Where the scoop carries the food to: tool_m moved by shift(); joints_deg is an IK seed (the arm's shape)."""
    se = dict(cfg.bowl.scoop_end)
    d = shift()
    se["tool_m"] = [float(se["tool_m"][i]) + float(d[i]) for i in range(3)]
    return se


def min_tip_z() -> float:
    """The lowest the utensil tip may go on the retract's joint move home (base z, m)."""
    return round(table_height() + float(cfg.delivery.retract.min_tip_above_table_m), 6)
