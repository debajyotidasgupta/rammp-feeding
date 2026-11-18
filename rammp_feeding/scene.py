"""Where things are: the wrist camera and the utensil on the arm, the table and the bowl, the user, and the poses that
follow from them (base frame, m).

  the wrist camera   its driver's intrinsics; its mount in the tool frame (rammp calibrate-wrist), pixel mappings
  the utensil tip    camera + camera_to_tip along the axis + the tip's sideways offset read off the image (tip_px)
  the workspace      the bowl and the table (state_dir/scene_state.json: rammp locate-scene, the scene camera's reading
                     -- with no scene camera, the site's own bowl -- once per rammp feed; the bowl's centre and radius
                     and the table then the wrist camera's, rammp detect), trusted to what read them
                     (scene_camera.error_m or workspace.bowl_centre_error_m, then perception.wrist_bowl_error_m; the
                     bowl's top stays the first look's), which the planner's world and our tip floor keep as extra
                     clearance
  the user           where the wrist camera measures the mouth from the delivery look, on every delivery: in the
                     planner's world from then on (none before)
  food_detect_start  computed: the wrist camera overlook_camera_height_m above the table, the bowl's centre on its
                     optical axis (the middle of its image), in the captured overlook's orientation, by continuous IK
                     from it -- first on the scene camera's coarse reading, then on the wrist camera's own (rammp
                     detect); "at" it by the tool's pose
  the world          the table (raised to our tip floor), a hollow bowl, the user round the measured mouth, the fork's
                     stand where it is known, the pedestal -- RAMMP-CuRobo's world schema, cuboids only: the planner
                     keeps the arm's links off it, and our check (Arm.check_world) the utensil on every path
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


def shifted_tip_px(arm, shift_m) -> tuple[float, float]:
    """The tip pixel (tip_px) that moves the tip model by shift_m [dx, dy] (base frame, level) at the arm's pose now: a
    correction from where the tip is seen to be against where the model puts it (camera_to_tip_m, the ruler's, kept)."""
    T, R = arm.tool()
    want = T + R @ arm.tip_tool + np.array([float(shift_m[0]), float(shift_m[1]), 0.0])
    o, Rc = camera_pose(arm)
    u, v = arm.cam.pixel(Rc.T @ (want - o))
    return float(u), float(v)


def camera_pose(arm, q=None) -> tuple[np.ndarray, np.ndarray]:
    """The colour camera in the base frame: (its optical centre, R base <- optical frame)."""
    T, R = arm.tool(q)
    return T + R @ CAM_IN_TOOL, R @ CAM_ROT


# ------------------------------------------------------------------ the workspace
def workspace() -> dict:
    """The bowl and the table: bowl_xy, table_z, bowl_top_z, bowl_radius, when; error_m, how far the bowl's place and
    the table's height may be off (the first look's, the wrist camera's once rammp detect has read them), and
    scene_error_m, how far the bowl's top may be (always the first look's reading): the scene camera's error_m; with
    no scene camera the top is the site's ruler, off as far as the bowl's place -- workspace.bowl_centre_error_m until
    the wrist camera has placed it, its own error from then on. table_z is the site's workspace.table_height_m where
    it is set (one table, every module on the robot), else the reading."""
    f = state_dir() / "scene_state.json"
    if not f.exists():
        raise FileNotFoundError("no reading of the bowl (%s) -- rammp locate-scene" % f)
    st = read_json(f)
    wrist_e = float(cfg.perception.wrist_bowl_error_m)
    scene_e = (float(cfg.scene_camera.error_m) if st.get("source") != "site" else
               wrist_e if st.get("refined_by") else float(cfg.workspace.bowl_centre_error_m))
    th = cfg.workspace.get("table_height_m", None)
    return dict(bowl_xy=[float(v) for v in st["bowl_xy"]], table_z=float(st["table_z"] if th is None else th),
                bowl_top_z=float(st["bowl_top_z"]),
                bowl_radius=float(st["rim_radius_m"]), scene_error_m=scene_e,
                error_m=wrist_e if st.get("refined_by") else scene_e,
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
    of ours. With no reading of the bowl yet: above the site's table (workspace.table_height_m)."""
    fs, ext = cfg.motion.free_space, cfg.workspace.table_extent_m
    try:
        w = workspace()
    except FileNotFoundError:
        th = cfg.workspace.get("table_height_m", None)
        on = th is not None and ext.x[0] <= x <= ext.x[1] and ext.y[0] <= y <= ext.y[1]
        return float(th) + float(fs.min_tip_above_table_m) if on else -math.inf
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


def off_pose(arm, q, goal) -> tuple[float, float]:
    """How far the tool at joints q is from where it is at joints goal: (m, deg), whatever the arm's shape."""
    from rammp_feeding.geometry import rot_log
    (T, R), (Tg, Rg) = arm.tool(q), arm.tool(goal)
    return float(np.linalg.norm(Tg - T)), float(np.degrees(np.linalg.norm(rot_log(Rg @ R.T))))


def is_at_food_detect_start(arm, q=None) -> bool:
    """The tool at food_detect_start's pose (within motion.near_pose_m and near_pose_deg), in any arm shape."""
    m, deg = off_pose(arm, arm.q() if q is None else q, overlook_joints(arm))
    return m <= float(cfg.motion.near_pose_m) and deg <= float(cfg.motion.near_pose_deg)


def at_food_detect_start(arm, q=None) -> None:
    """MoveError unless the tool is at food_detect_start's pose: where a target is read and aimed from."""
    from rammp_feeding.arm import MoveError
    q = arm.q() if q is None else q
    if not is_at_food_detect_start(arm, q):
        m, deg = off_pose(arm, q, overlook_joints(arm))
        raise MoveError("not at food_detect_start (the tool %.0f mm, %.1f deg from it) -- rammp detect"
                        % (1000 * m, deg))


# ------------------------------------------------------------------ the planner's world
def user_obstacles(m) -> list[dict]:
    """The user's head and torso round their measured mouth m, as the planner's world takes them."""
    return [dict(name="user_head", position=[m[0] - 0.09, m[1], m[2] + 0.05], dims=[0.20, 0.18, 0.26]),
            dict(name="user_torso", position=[m[0] - 0.12, m[1], m[2] - 0.30], dims=[0.28, 0.42, 0.45])]


def table_x(error_m: float) -> list[float]:
    """The planner's table along x (workspace.table_extent_m.x): its near edge (toward the arm) moved back by error_m,
    and never nearer the arm's base than motion.free_space.table_clear_of_base_m (the planner's padded table clear of
    the arm's own links at any pose, whatever the site's edge and the camera's error)."""
    near, far = [float(v) for v in cfg.workspace.table_extent_m.x]
    return [max(near + float(error_m), float(cfg.motion.free_space.table_clear_of_base_m)), far]


def stand_box() -> dict | None:
    """The fork's stand as the collision world takes it: the block round its opening (utensil.pick.block_m, its top at
    the rim), where the stand is known -- the fake holder's (fakes.holder), else where the utensil's state last placed
    it (the pick's opening, or the seat it went back to); None without one."""
    if not cfg.get("utensil.pick.use", False):
        return None
    from rammp_feeding import fakes, fork
    h = fakes.holder() if fakes.on("holder") else None
    if h is None:
        s = fork.state()
        known = s.get("hole") or s.get("stand")
        if not known:
            return None
        h = fork.Hole.from_json(known)
    across, along, down = (float(v) for v in cfg.utensil.pick.block_m)
    yaw = math.degrees(math.atan2(h.long_axis[1], h.long_axis[0]))
    return dict(name="fork_stand", position=[h.centre[0], h.centre[1], h.centre[2] - down / 2],
                dims=[along, across, down], rpy_deg=[0.0, 0.0, yaw])


def world() -> dict:
    """RAMMP-CuRobo's world (arm.base_frame): the table raised to our tip floor, a hollow bowl (a floor up to
    bowl.floor_above_table_m and 12 wall boxes: a solid box would put the utensil over its opening inside it), the
    user's head and torso round the measured mouth (once measured), the fork's stand (once known: stand_box), the
    pedestal. The bowl grows by the workspace's error_m and its top by scene_error_m; the table's top rises by error_m,
    and its near edge (toward the arm) moves back by scene_error_m, at least table_clear_of_base_m from the base
    (table_x). With no reading of the bowl yet (the fork's routine before a meal): the site's table, its edge moved back
    as a meal's first look has it, and no bowl."""
    from rammp_feeding.scene_camera import in_use
    try:
        ws = workspace()
    except FileNotFoundError:
        ws = None
    floor = float(cfg.motion.free_space.min_tip_above_table_m)
    ty = [float(v) for v in cfg.workspace.table_extent_m.y]
    obs = [dict(name="pedestal", position=[0.0, 0.0, -0.05], dims=[0.14, 0.14, 0.04])]   # (the planner's own: unpadded)
    if ws is None:
        th = cfg.workspace.get("table_height_m", None)
        if th is not None:
            tx = table_x(cfg.scene_camera.error_m if in_use() else cfg.workspace.bowl_centre_error_m)
            obs.append(dict(name="table", position=[sum(tx) / 2, sum(ty) / 2, float(th) + floor - 0.025],
                            dims=[tx[1] - tx[0], ty[1] - ty[0], 0.05]))
    else:
        e, es = float(ws["error_m"]), float(ws["scene_error_m"])
        tz, top = ws["table_z"] + e, ws["bowl_top_z"] + es
        bx, by = ws["bowl_xy"]
        r = float(ws["bowl_radius"])
        tx = table_x(es)
        f = float(cfg.bowl.floor_above_table_m)         # the bowl's floor: the same height the plunge stops above
        obs += [dict(name="table", position=[sum(tx) / 2, sum(ty) / 2, tz + floor - 0.025],
                     dims=[tx[1] - tx[0], ty[1] - ty[0], 0.05]),
                dict(name="bowl_floor", position=[bx, by, tz + f / 2], dims=[2 * (r + e), 2 * (r + e), f])]
        for k in range(12):
            a = 2 * math.pi * k / 12
            obs.append(dict(name="bowl_wall_%d" % k,
                            position=[bx + r * math.cos(a), by + r * math.sin(a), (tz + top) / 2],
                            dims=[0.006 + 2 * e, 2 * (r + e) * math.tan(math.pi / 12) + 0.004, top - tz],
                            rpy_deg=[0.0, 0.0, math.degrees(a)]))
    m = user_mouth()
    if m is not None:
        obs += user_obstacles(m)
    stand = stand_box()
    if stand is not None:
        obs.append(stand)
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
    """The latest workspace and mouth into the shared planner (write_world). Done before each planned move
    (Arm.travel), and undone when the command ends (restore_world: the planner's /reset): other clients plan against
    the planner's own world."""
    write_world(robot, world())


def write_world(robot, w: dict) -> None:
    """A world into the shared planner: a YAML file both containers see, then set_world (padded as the planner pads
    every box)."""
    from pathlib import Path
    p = Path(str(cfg.planner.world_file)).expanduser()
    text = yaml.safe_dump(w)
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
