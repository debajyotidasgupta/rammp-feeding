"""Where things are: the wrist camera and the utensil on the arm, the table and the bowl, the user, and the poses that
follow from them (base frame, m).

  the wrist camera   its driver's intrinsics; its mount in the tool frame (rammp calibrate-wrist), pixel mappings
  the utensil tip    camera + camera_to_tip along the axis + the tip's sideways offset read off the image (tip_px)
  the workspace      the bowl and the table (state_dir/scene_state.json: rammp locate-scene, the scene camera's reading,
                     once per rammp feed; the bowl's centre and radius and the table then the wrist camera's, rammp
                     detect), trusted to the camera that read them (scene_camera.error_m, then
                     perception.wrist_bowl_error_m; the bowl's top stays the scene camera's), which the planner's world
                     and our tip floor keep as extra clearance
  the user           where the wrist camera measures the mouth from the delivery look, on every delivery: in the
                     planner's world from then on (none before)
  food_detect_start  computed: the wrist camera overlook_camera_height_m above the table, the bowl's centre on its
                     optical axis (the middle of its image), in the captured overlook's orientation, by continuous IK
                     from it -- first on the scene camera's coarse reading, then on the wrist camera's own (rammp
                     detect)
  the planner world  the table (raised to our tip floor), a hollow bowl, the user round the measured mouth, the
                     pedestal -- RAMMP-CuRobo's world schema, cuboids only
"""

from __future__ import annotations

import math
import time

import numpy as np
import yaml

from rammp_feeding.config import calibration, cfg, read_json, state_dir
from rammp_feeding.geometry import quat_matrix, rpy_matrix

# ------------------------------------------------------------------ the wrist camera on the arm
# Its intrinsics and depth -> colour transform are its driver's (camera_info, /tf_static: rammp_feeding.camera); where
# it sits on the arm is ours (the camera profile's mount: the colour optical frame in the tool frame; CAM_ROT: the optical
# axes in tool coordinates). None until it is measured (config.unmeasured: the arm is not built without it).
_MOUNT = cfg.camera.mount
CAM_IN_TOOL = None if _MOUNT.camera_in_tool_m is None else np.array(_MOUNT.camera_in_tool_m, float)
CAM_ROT = None if _MOUNT.rotation_quat_xyzw is None else np.round(quat_matrix(_MOUNT.rotation_quat_xyzw), 12) + 0.0


def cam_to_tool(p_cam) -> np.ndarray:
    """A point in the colour camera's optical frame (x right, y down, z out of the lens) -> the tool frame."""
    return CAM_IN_TOOL + CAM_ROT @ np.asarray(p_cam, float)


def tip_in_tool(cam) -> np.ndarray:
    """The utensil tip in the tool frame, from calibration: the point camera_to_tip_m out along the camera's axis that
    images at tip_px."""
    u, v = (float(x) for x in cfg.utensil.tip_px)
    return cam_to_tool(cam.ray(u, v, float(cfg.utensil.camera_to_tip_m)))


def camera_pose(arm, q=None) -> tuple[np.ndarray, np.ndarray]:
    """The colour camera in the base frame: (its optical centre, R base <- optical frame)."""
    T, R = arm.tool(q)
    return T + R @ CAM_IN_TOOL, R @ CAM_ROT


# ------------------------------------------------------------------ the workspace
def workspace() -> dict:
    """The bowl and the table: bowl_xy, table_z, bowl_top_z, bowl_radius, when; error_m, how far the bowl's place and
    the table's height may be off (the scene camera's error_m, the wrist camera's once rammp detect has read them),
    and scene_error_m, the scene camera's (the bowl's top is always its reading). table_z is the site's
    workspace.table_height_m where it is set (one table, every module on the robot), else the reading."""
    f = state_dir() / "scene_state.json"
    if not f.exists():
        raise FileNotFoundError("no scene camera reading (%s) -- rammp locate-scene" % f)
    st = read_json(f)
    scene_e = float(cfg.scene_camera.error_m)
    th = cfg.workspace.get("table_height_m", None)
    return dict(bowl_xy=[float(v) for v in st["bowl_xy"]], table_z=float(st["table_z"] if th is None else th),
                bowl_top_z=float(st["bowl_top_z"]),
                bowl_radius=float(st["rim_radius_m"]), scene_error_m=scene_e,
                error_m=float(cfg.perception.wrist_bowl_error_m) if st.get("refined_by") else scene_e,
                when=time.strftime("%Y-%m-%d %H:%M", time.localtime(st["t"])))


_MOUTH: dict = {"at": None}


def user_mouth() -> np.ndarray | None:
    """The user's mouth as the wrist camera measured it from the delivery look in this command (None before: the
    planner's world has no user until then)."""
    return _MOUTH["at"]


def measured_mouth(m) -> None:
    _MOUTH["at"] = None if m is None else np.asarray(m, float)


def tip_floor(x: float, y: float) -> float:
    """The lowest the utensil tip may pass at (x, y) on a free-space move: above the bowl's top over the bowl (and a
    margin round it), above the table over its footprint -- each plus how far that reading may be off; elsewhere none
    of ours."""
    w, fs, ext = workspace(), cfg.motion.free_space, cfg.workspace.table_extent_m
    e = w["error_m"]
    if math.hypot(x - w["bowl_xy"][0], y - w["bowl_xy"][1]) < w["bowl_radius"] + float(fs.bowl_margin_m) + e:
        return w["bowl_top_z"] + float(fs.min_tip_above_bowl_m) + w["scene_error_m"]
    if ext.x[0] <= x <= ext.x[1] and ext.y[0] <= y <= ext.y[1]:
        return w["table_z"] + float(fs.min_tip_above_table_m) + e
    return -math.inf


# ------------------------------------------------------------------ poses
def captured(name: str) -> list[float]:
    return [float(v) for v in calibration(name)["joints_deg"]]


def overlook_goal(arm) -> tuple[np.ndarray, np.ndarray]:
    """food_detect_start's tool pose: the captured overlook's orientation, the wrist camera overlook_camera_height_m
    above the table with the bowl's centre on its optical axis -- the whole bowl in the middle of its image, which a
    bowl reading a few cm off still leaves in view (the tip over the bowl instead left it at the image's edge)."""
    from rammp_feeding.arm import MoveError
    _, R = arm.tool(captured("overlook"))
    ws = workspace()
    h = float(cfg.workspace.overlook_camera_height_m)
    cam, a = R @ CAM_IN_TOOL, R @ CAM_ROT[:, 2]                 # the camera, and its optical axis, base frame
    if -a[2] < 0.5:
        raise MoveError("the wrist camera does not look down from the overlook (its axis %.0f deg off vertical)"
                        % np.degrees(np.arccos(min(1.0, -a[2]))))
    off = cam[:2] + a[:2] * h / -a[2]                           # where the axis meets the table, from the tool point
    return np.array([ws["bowl_xy"][0] - off[0], ws["bowl_xy"][1] - off[1], ws["table_z"] + h - cam[2]]), R


_RESOLVED: dict[tuple, list[float]] = {}


def overlook_joints(arm) -> list[float]:
    """food_detect_start's joints: continuous IK from the captured overlook, its orientation held (one seeded IK can
    land on another arm shape)."""
    from rammp_feeding.arm import MoveError
    T1, R = overlook_goal(arm)
    name = "food_detect_start"
    seed = captured("overlook")
    key = tuple(np.round(T1, 5)) + tuple(np.round(seed, 3))      # (a new capture is a new shape)
    if key not in _RESOLVED:
        T0, _ = arm.tool(seed)
        n = max(4, int(math.ceil(np.linalg.norm(T1 - T0) / 0.015)))
        path, why = arm.homotopy(seed, lambda s: (T0 + (T1 - T0) * s, R), n)
        if path is None:
            raise MoveError("%s over the bowl at (%.3f, %.3f) is not reachable from the captured overlook without "
                            "changing the arm's shape (%s)" % (name, *workspace()["bowl_xy"], why))
        _RESOLVED[key] = path[-1]
    return list(_RESOLVED[key])


def at_food_detect_start(arm, q=None) -> None:
    """MoveError unless the arm is at food_detect_start (within near_pose_deg): where a target is read and aimed from."""
    from rammp_feeding.arm import MoveError
    from rammp_feeding.geometry import max_step
    off = max_step(arm.q() if q is None else q, overlook_joints(arm))
    if off > float(cfg.motion.near_pose_deg):
        raise MoveError("not at food_detect_start (%.1f deg away) -- rammp detect" % off)


# ------------------------------------------------------------------ the planner's world
def user_obstacles(m) -> list[dict]:
    """The user's head and torso round their measured mouth m, as the planner's world takes them."""
    return [dict(name="user_head", position=[m[0] - 0.09, m[1], m[2] + 0.05], dims=[0.20, 0.18, 0.26]),
            dict(name="user_torso", position=[m[0] - 0.12, m[1], m[2] - 0.30], dims=[0.28, 0.42, 0.45])]


def world() -> dict:
    """RAMMP-CuRobo's world (arm.base_frame): the table raised to our tip floor, a hollow bowl (a floor up to
    bowl.floor_above_table_m and 12 wall boxes: a solid box would put the utensil over its opening inside it), the
    user's head and torso round the measured mouth (once measured), the pedestal. The bowl grows by the workspace's
    error_m and its top by scene_error_m; the table's top rises by error_m, and its near edge (toward the arm) moves
    back by scene_error_m (the planner's padded table clear of the arm's own base, whichever camera read the rest)."""
    ws = workspace()
    e, es = float(ws["error_m"]), float(ws["scene_error_m"])
    tz, top = ws["table_z"] + e, ws["bowl_top_z"] + es
    bx, by = ws["bowl_xy"]
    r = float(ws["bowl_radius"])
    m = user_mouth()
    ext = cfg.workspace.table_extent_m
    tx, ty = [float(v) for v in ext.x], [float(v) for v in ext.y]
    tx[0] += es
    floor = float(cfg.motion.free_space.min_tip_above_table_m)
    f = float(cfg.bowl.floor_above_table_m)                     # the bowl's floor: the same height the plunge stops above
    obs = [dict(name="pedestal", position=[0.0, 0.0, -0.05], dims=[0.14, 0.14, 0.04]),   # (the planner's own: unpadded)
           dict(name="table", position=[sum(tx) / 2, sum(ty) / 2, tz + floor - 0.025], dims=[tx[1] - tx[0], ty[1] - ty[0], 0.05]),
           dict(name="bowl_floor", position=[bx, by, tz + f / 2], dims=[2 * (r + e), 2 * (r + e), f])]
    if m is not None:
        obs += user_obstacles(m)
    for k in range(12):
        a = 2 * math.pi * k / 12
        obs.append(dict(name="bowl_wall_%d" % k, position=[bx + r * math.cos(a), by + r * math.sin(a), (tz + top) / 2],
                        dims=[0.006 + 2 * e, 2 * (r + e) * math.tan(math.pi / 12) + 0.004, top - tz],
                        rpy_deg=[0.0, 0.0, math.degrees(a)]))
    for o in obs:
        o["position"], o["dims"] = [float(v) for v in o["position"]], [float(v) for v in o["dims"]]
        o["rpy_deg"] = [float(v) for v in o.get("rpy_deg", [0.0, 0.0, 0.0])]
    return dict(base_frame=str(cfg.arm.base_frame), obstacles=obs, objects=[], targets=[])


def utensil_spheres(arm, food: bool = False) -> list[list[float]]:
    """The utensil (and, carrying, the piece on it) as spheres [x, y, z, r] in the tool frame, as the utensil check takes
    them: evenly from the gripper's pads to the tip, the last one round the tines larger."""
    U = cfg.planner.utensil_spheres
    a, tip, n = np.array(U.from_tool_m, float), np.asarray(arm.tip_tool, float), int(U.count)
    out = [[*(a + s * (tip - a)), float(U.tip_radius_m if k == n - 1 else U.radius_m)]
           for k, s in enumerate(np.linspace(0.0, 1.0, n))]
    if food:
        out.append([*(tip - [0.0, 0.0, float(U.food_back_m)]), float(U.food_radius_m)])
    return [[float(v) for v in s] for s in out]


def push_world(robot) -> None:
    """The latest workspace and mouth into the shared planner: a YAML file both containers see, then set_world (padded
    as the planner pads every box). Done before each planned move (Arm.travel), and undone when the command ends
    (restore_world: the planner's /reset): other clients plan against the planner's own world."""
    from pathlib import Path
    p = Path(str(cfg.planner.world_file)).expanduser()
    text = yaml.safe_dump(world())
    if not p.exists() or p.read_text() != text:          # (written only when it changes: a planned move is frequent)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    robot.set_world(str(cfg.planner.world_file_in_planner or p))
    robot.world_pushed = True


def restore_world(robot) -> None:
    """The planner's own world back (its boot world and configured layers)."""
    robot.reset_world()
    robot.world_pushed = False


def utensil_boxes() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Our boxes as the utensil check takes them (centres, rotations base <- box, half extents, names): world() but the
    pedestal and the table (raised for the planner, which does not see the utensil; the tip floor keeps the utensil
    above the real one), the bowl's pieces exact, the user grown by utensil_padding_m per side."""
    pad = float(cfg.motion.free_space.utensil_padding_m)
    boxes = [o for o in world()["obstacles"] if o["name"] not in ("pedestal", "table")]
    grow = [0.0 if o["name"].startswith("bowl_") else pad for o in boxes]
    return (np.array([o["position"] for o in boxes]),
            np.array([rpy_matrix(*np.radians(o["rpy_deg"])) for o in boxes]),
            np.array([np.asarray(o["dims"]) / 2 + g for o, g in zip(boxes, grow)]),
            [o["name"] for o in boxes])
