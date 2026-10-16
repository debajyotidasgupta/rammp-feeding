"""cuRobo's view of the scene: the collision world and the utensil on the wrist. Plain data, no torch / cuRobo.

    from kinova.curobo_world import world, spoon_spheres
    w = world(workspace, mouth, wheelchair_boxes)        # RAMMP-CuRobo's world schema (obstacles: cuboids)

Used by kinova/planner_server.py (the planning service) and tests/scene_error_study.py.
  table      the workspace's table height over its footprint (workspace.table_extent_m), raised to our tip floor
  bowl       HOLLOW: a floor and 12 wall boxes round its rim (a solid box would put the overlook, the utensil over
             its opening, inside it)
  Both grown by the workspace's error_m (the scene camera's reading is trusted to that): the table and the rim
  that much higher (the table's near edge, beside the arm's base, as much further out), the walls that much
  thicker either side.
  user       head and torso boxes round the measured mouth (the user faces +x, the arm's side is -y)
  wheelchair the deployment's boxes (sim/assets/wheelchair/wheelchair_boxes.yaml, if present)
  pedestal   the arm's own mount (unpadded)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WHEELCHAIR = ROOT / "sim" / "assets" / "wheelchair" / "wheelchair_boxes.yaml"
TOOL_Z = 0.12                       # the flange -> tool frame (ours and RAMMP-CuRobo's tool_frame alike)


def _f(v) -> list[float]:
    return [float(x) for x in v]


def wheelchair_boxes(path: Path = WHEELCHAIR) -> list[dict]:
    if not path.exists():
        return []
    import yaml
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    boxes = doc.get("boxes", doc) if isinstance(doc, dict) else doc
    return [dict(name="chair_" + str(b["name"]), position=_f(b["position"]), dims=_f(b["dims"]),
                 rpy_deg=_f(b.get("rpy_deg", [0, 0, 0]))) for b in (boxes or [])]


def world(workspace: dict, mouth, chair: list[dict] | None = None, bowl_radius: float | None = None,
          table_extent: dict | None = None, floor_m: float = 0.0) -> dict:
    """RAMMP-CuRobo's world (base_link frame): table, hollow bowl, user, wheelchair, pedestal. floor_m: the table's
    box this much higher over its footprint -- our own floor for the utensil tip (scene.tip_floor), so the planner's
    paths respect it rather than being refused after planning."""
    e = float(workspace.get("error_m", 0.0))         # the reading may be this far off: the table and bowl grow by it
    tz, top = float(workspace["table_z"]) + e, float(workspace["bowl_top_z"]) + e
    bx, by = (float(v) for v in workspace["bowl_xy"])
    r = float(bowl_radius or workspace.get("bowl_radius") or 0.06)
    m = np.asarray(mouth, float)
    ext = table_extent or dict(x=[0.08, 0.76], y=[-0.30, 0.80])
    tx, ty = [float(v) for v in ext["x"]], [float(v) for v in ext["y"]]
    tx[0] += e                                       # the raised table stops short of the arm's own base by as much
    obs = [dict(name="pedestal", position=[0.0, 0.0, -0.05], dims=[0.14, 0.14, 0.04]),
           dict(name="table", position=[sum(tx) / 2, sum(ty) / 2, tz + floor_m - 0.025], dims=[tx[1] - tx[0], ty[1] - ty[0], 0.05]),
           dict(name="bowl_floor", position=[bx, by, tz + 0.004], dims=[2 * (r + e), 2 * (r + e), 0.008]),
           dict(name="user_head", position=[m[0] - 0.09, m[1], m[2] + 0.05], dims=[0.20, 0.18, 0.26]),
           dict(name="user_torso", position=[m[0] - 0.12, m[1], m[2] - 0.30], dims=[0.28, 0.42, 0.45])]
    for k in range(12):
        a = 2 * math.pi * k / 12
        obs.append(dict(name="bowl_wall_%d" % k, position=[bx + r * math.cos(a), by + r * math.sin(a), (tz + top) / 2],
                        dims=[0.006 + 2 * e, 2 * (r + e) * math.tan(math.pi / 12) + 0.004, top - tz],
                        rpy_deg=[0.0, 0.0, math.degrees(a)]))
    obs += chair if chair is not None else wheelchair_boxes()
    for o in obs:
        o["position"], o["dims"] = _f(o["position"]), _f(o["dims"])
        o["rpy_deg"] = _f(o.get("rpy_deg", [0.0, 0.0, 0.0]))
    return dict(base_frame="base_link", obstacles=obs, objects=[], targets=[])


def no_pad_names(w: dict) -> list[str]:
    """Obstacles never padded: the arm's own pedestal (padding it puts every start state in collision) and the bowl's
    pieces (the utensil works over and into it; its clearance is our own tip floor)."""
    return [o["name"] for o in w["obstacles"] if o["name"] == "pedestal" or o["name"].startswith("bowl_")]


def spoon_spheres(tip_tool, food: bool) -> list[dict]:
    """The utensil (and a carried piece) as spheres in bracelet_link's frame: the flange is 61.5 mm out along -z,
    turned 180 deg about x, so flange (x, y, z) = bracelet (x, -y, -0.061525 - z). From the gripper pads to the tip."""
    tip_f = np.asarray(tip_tool, float) + np.array([0.0, 0.0, TOOL_Z])
    out = []
    for s in np.linspace(0.0, 1.0, 5):
        p = np.array([0.0, 0.0, 0.17]) + s * (tip_f - np.array([0.0, 0.0, 0.17]))
        out.append(dict(center=[float(p[0]), float(-p[1]), float(-0.061525 - p[2])], radius=0.012 if s < 0.8 else 0.02))
    if food:                                                 # a ~25 mm piece riding the tip
        p = tip_f - np.array([0.0, 0.0, 0.01])
        out.append(dict(center=[float(p[0]), float(-p[1]), float(-0.061525 - p[2])], radius=0.02))
    return out


GRIPPER_LINKS = ["robotiq_arg2f_base_link", "left_outer_finger", "right_outer_finger", "left_inner_finger_pad",
                 "right_inner_finger_pad"]


def robot_config(base: dict, tip_tool, food: bool) -> dict:
    """RAMMP-CuRobo's Gen3 + 2F-85 robot config with the utensil (and food) spheres on bracelet_link."""
    import copy
    robot = copy.deepcopy(base)
    k = robot["robot_cfg"]["kinematics"]
    k["collision_spheres"]["bracelet_link"] = list(k["collision_spheres"].get("bracelet_link", [])) + spoon_spheres(tip_tool, food)
    ign = k.setdefault("self_collision_ignore", {})
    ign["bracelet_link"] = sorted(set(ign.get("bracelet_link", [])) | set(GRIPPER_LINKS))
    return robot
