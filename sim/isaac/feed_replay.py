"""Thousands of bites at once on one GPU (Isaac Lab): each arm replays a candidate's joint path, the food is physics.

    /workspace/isaaclab/isaaclab.sh -p sim/isaac/feed_replay.py --batch b.npz --usd USD_DIR --out DIR [--rows 0:4096]
        [--utensil fork|spoon] [--video --hero K] [--headless --enable_cameras]

The batch (sim/isaac/make_batch.py) holds one joint path per candidate (what the stack's own stages commanded, from
tests/site_check.py with cuRobo) and, per arm, which candidate it replays, its scenario (table, bowl, mouth) and its
food draw (where the piece really is, size, mass, friction). The scene (sim/isaac/build_usd.py): cuRobo's Gen3 +
2F-85 (the fingers closed on the handle), the wheelchair, the SMPL-X person with the mouth on the row's mouth, a
ceramic bowl (its colliders: 12 staves and a floor) with three more pieces in it, the table, the spoon. The arm
follows its path exactly (a stiff position-controlled robot). --utensil fork: the tines pierce (a piece 5 mm onto
them rides there until the pull along them beats its hold); spoon: the head is a velocity-driven tray the piece must
ride on. Per arm: picked, still on the utensil at the mouth, when it fell, the tip's closest approach to the mouth.
Results: DIR/results_<rows>.npz; --video: DIR/frames_<rows>/ grid_*.jpg (long lens over the rows of arms) and
hero_*.jpg (arm K close up), plus each frame's picked / on-utensil / dropped counts (sim/isaac/compose.py films them).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--batch", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rows", default=":", help="a slice of the batch's rows (another GPU takes the next slice)")
ap.add_argument("--video", action="store_true")
ap.add_argument("--spacing", type=float, default=1.8)
ap.add_argument("--frame-every", type=int, default=4, help="control steps per video frame (DT = 1/120 s -> 30 fps)")
ap.add_argument("--max-seconds", type=float, default=0.0, help="stop the replay after this much bite time (0: all)")
ap.add_argument("--utensil", choices=["fork", "spoon"], default="fork",
                help="fork: the tines pierce (no contact with the food; a piece 5 mm onto the tines is held there until "
                     "the pull along them beats its hold); spoon: the head is a rigid tray the piece must ride on")
ap.add_argument("--hold-n", default="0.2:1.5", help="fork: the hold of a skewered piece along the tines, N (drawn per arm)")
ap.add_argument("--usd", required=True, help="sim/isaac/build_usd.py's output: the arm, the person, the chair, the bowl, the spoon")
ap.add_argument("--hero", type=int, default=0, help="--video: the arm the close-up follows")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul  # noqa: E402

B = np.load(args.batch)
sl = slice(*[int(x) if x else None for x in args.rows.split(":")])
ROWS = np.arange(len(B["candidate"]))[sl]
N = len(ROWS)
DT = float(B["dt"])
dev = "cuda:0"
Q = torch.tensor(B["q"], device=dev)                         # (candidates, T, 7) rad
CAND = torch.tensor(B["candidate"][ROWS], device=dev, dtype=torch.long)
T = Q.shape[1] if not args.max_seconds else min(Q.shape[1], int(args.max_seconds / DT))
EV = torch.tensor(B["events"][ROWS], device=dev, dtype=torch.long)
TABLE_Z = float(np.median(B["table_z"]))
BOWL = torch.tensor(B["bowl"][ROWS], device=dev)
MOUTH = torch.tensor(B["mouth"][ROWS], device=dev)
TIP = torch.tensor(B["tip_in_tool"][ROWS], device=dev)
NORMAL = torch.tensor(B["tray_normal_tool"][ROWS], device=dev)
FOOD0 = torch.tensor(B["food_pos"][ROWS], device=dev)
SIZE = float(np.median(B["food_size"]))                     # one spawn size per run; mass and friction vary per arm
TZ = torch.tensor(B["table_z"][ROWS], device=dev)            # each arm's own table height
USD = Path(args.usd)
GRIP = 0.70                                                  # the 2F-85 closed on the utensil's handle (0 open, 0.8 shut)
FINGERS = {"left_outer_knuckle_joint": GRIP, "right_outer_knuckle_joint": GRIP, "left_inner_knuckle_joint": GRIP,
           "right_inner_knuckle_joint": GRIP, "left_inner_finger_joint": -GRIP, "right_inner_finger_joint": -GRIP}
EXTRA = [(0.033, 1.0), (0.033, 3.1), (0.033, 5.2)]           # the bowl's other pieces (radius m, angle rad), spaced
FRUIT = [(0.98, 0.80, 0.20), (0.55, 0.80, 0.30), (0.90, 0.25, 0.30)]
WOOD = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.18), roughness=0.6)
WHITE = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.95), roughness=0.3)
STEEL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.82, 0.86), metallic=0.8, roughness=0.25)
FOOD = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.62, 0.25), roughness=0.5)
KIN = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
DRIVEN = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=0.5, linear_damping=0.0,
                                         angular_damping=0.0)
COL = sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)


def mesh(name, usd, pos=(0.0, 0.0, 0.0)):
    """A visual-only mesh asset (build_usd.py) posed per arm by the replay."""
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=sim_utils.UsdFileCfg(usd_path=str(USD / usd)),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=pos))


def cuboid(name, size, mat, pos, kinematic=True, mass=None, visual_only=False, driven=False, visible=True):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(size=size, visual_material=mat, visible=visible,
                                  rigid_props=DRIVEN if driven else (KIN if kinematic else sim_utils.RigidBodyPropertiesCfg()),
                                  collision_props=None if visual_only else COL,
                                  mass_props=sim_utils.MassPropertiesCfg(mass=mass) if mass else None,
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6,
                                                                                   dynamic_friction=0.5)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos))


# the spoon's head as a shallow tray (a plate and four low rims) and its shaft, in the TRAY frame:
# x along the utensil toward the tip, z the food side; one kinematic body per piece
TRAY = [("plate", (0.040, 0.026, 0.0015), (0.0, 0.0, 0.0)), ("rim_l", (0.040, 0.0015, 0.005), (0.0, 0.0135, 0.0025)),
        ("rim_r", (0.040, 0.0015, 0.005), (0.0, -0.0135, 0.0025)), ("rim_t", (0.0015, 0.026, 0.005), (0.0205, 0.0, 0.0025)),
        ("rim_b", (0.0015, 0.026, 0.004), (-0.0205, 0.0, 0.002))]


@configclass
class FeedSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(color=(0.10, 0.28, 0.75)),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_Z - 0.74)))   # the room's floor
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1600.0, color=(0.95, 0.96, 1.0)))
    sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(intensity=2200.0, angle=1.5, color=(1.0, 0.97, 0.92)),
                       init_state=AssetBaseCfg.InitialStateCfg(rot=(0.88, 0.28, -0.12, 0.36)))
    # the arm the planner plans for: cuRobo's Gen3 + 2F-85 URDF, every joint written each step (no drive acts)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(USD / "gen3_2f85.usd"),
                                   rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                                   articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=4000.0, damping=40.0)})
    wheelchair = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Wheelchair", spawn=sim_utils.UsdFileCfg(usd_path=str(USD / "wheelchair.usd")))
    person = mesh("Person", "person.usd")                    # the mouth at the row's mouth
    bowl = mesh("Bowl", "bowl.usd")                          # the ceramic you see; the colliders are the staves below
    spoon = mesh("Spoon", "spoon.usd")                       # the head's centre on the tray frame
    table = cuboid("Table", (0.68, 1.10, 0.03), WOOD, (0.42, 0.25, TABLE_Z - 0.015))
    bowl_floor = cuboid("BowlFloor", (0.10, 0.10, 0.006), WHITE, (0.29, 0.158, TABLE_Z + 0.003), visible=False)
    food = cuboid("Food", (SIZE, SIZE, SIZE), FOOD, (0.29, 0.158, TABLE_Z + 0.03), kinematic=False, mass=0.012)


for k in range(4):                                           # the table's legs, to the floor
    setattr(FeedSceneCfg, "leg_%d" % k, cuboid("Leg%d" % k, (0.04, 0.04, 0.70), WOOD,
                                               (0.42 + 0.30 * (1 if k % 2 else -1), 0.25 + 0.50 * (1 if k // 2 else -1),
                                                TABLE_Z - 0.38), visual_only=True))
for k, (r, a) in enumerate(EXTRA):
    setattr(FeedSceneCfg, "extra_%d" % k, cuboid("Extra%d" % k, (SIZE, SIZE, SIZE), sim_utils.PreviewSurfaceCfg(
        diffuse_color=FRUIT[k], roughness=0.5), (0.29, 0.158, TABLE_Z + 0.03), kinematic=False, mass=0.012))
for k in range(12):                                          # the bowl's wall: 12 staves round the rim (colliders)
    a = 2 * math.pi * k / 12
    setattr(FeedSceneCfg, "bowl_wall_%d" % k,
            cuboid("BowlWall%d" % k, (0.006, 0.030, 0.070), WHITE, (0.29 + 0.054 * math.cos(a), 0.158 + 0.054 * math.sin(a), TABLE_Z + 0.035),
                   visible=False))
for name, size, _ in TRAY:
    setattr(FeedSceneCfg, "spoon_" + name, cuboid("Spoon_" + name, size, STEEL, (0.0, 0.0, 1.0), kinematic=False,
                                                  mass=5.0, driven=True, visual_only=args.utensil == "fork", visible=False))


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT / 2, device=dev, gravity=(0.0, 0.0, -9.81)))
    scene = InteractiveScene(FeedSceneCfg(num_envs=N, env_spacing=args.spacing, replicate_physics=True))
    cams = {}
    if args.video:                    # the grid, long lens from the table side; the close-up on one arm
        from isaaclab.sensors import Camera, CameraCfg
        for name, (w, h, f) in {"grid": (1920, 1080, 30.0), "hero": (1280, 720, 22.0)}.items():
            cams[name] = Camera(CameraCfg(prim_path="/World/Cam_" + name, update_period=0, height=h, width=w, data_types=["rgb"],
                                          spawn=sim_utils.PinholeCameraCfg(focal_length=f, clipping_range=(0.02, 400.0))))
    sim.reset()
    robot: Articulation = scene["robot"]
    food: RigidObject = scene["food"]
    origins = scene.env_origins                                  # (N, 3)
    ee = robot.find_bodies("end_effector_link")[0][0]
    # per-arm scenario: the table, the bowl, the food and the person where this row says
    one = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)

    def place(name, xyz, quat=None):
        scene[name].write_root_pose_to_sim(torch.cat([xyz + origins, one if quat is None else quat], 1))
        scene[name].write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))

    col = lambda v: torch.full((N,), float(v), device=dev)  # noqa: E731
    place("table", torch.stack([col(0.42), col(0.25), TZ - 0.015], 1))
    for k in range(4):
        place("leg_%d" % k, torch.stack([col(0.42 + 0.30 * (1 if k % 2 else -1)), col(0.25 + 0.50 * (1 if k // 2 else -1)),
                                         TZ - 0.38], 1))
    for k in range(12):
        a = 2 * math.pi * k / 12
        qz = torch.tensor([math.cos(a / 2), 0, 0, math.sin(a / 2)], device=dev).repeat(N, 1)
        place("bowl_wall_%d" % k, torch.stack([BOWL[:, 0] + 0.054 * math.cos(a), BOWL[:, 1] + 0.054 * math.sin(a), TZ + 0.035], 1), qz)
    place("bowl_floor", torch.stack([BOWL[:, 0], BOWL[:, 1], TZ + 0.003], 1))
    place("bowl", torch.stack([BOWL[:, 0], BOWL[:, 1], TZ], 1))
    place("person", MOUTH)
    place("food", FOOD0)
    for k, (r, a) in enumerate(EXTRA):
        place("extra_%d" % k, torch.stack([BOWL[:, 0] + r * math.cos(a), BOWL[:, 1] + r * math.sin(a), TZ + 0.008 + SIZE / 2], 1))
    # the arm's joints in the articulation's order: the path's seven, the fingers closed on the handle
    arm_j = robot.find_joints(["joint_%d" % i for i in range(1, 8)], preserve_order=True)[0]
    q_full = robot.data.default_joint_pos.clone()
    for name, v in FINGERS.items():
        q_full[:, robot.find_joints(name)[0][0]] = v
    zero_full = torch.zeros_like(q_full)
    view = food.root_physx_view
    idx = torch.arange(N, device="cpu")
    view.set_masses(torch.tensor(B["mass"][ROWS]).reshape(N, 1), idx)
    mat = view.get_material_properties()
    mu = torch.tensor(B["mu"][ROWS])
    mat[:, :, 0] = mu[:, None]; mat[:, :, 1] = 0.85 * mu[:, None]
    view.set_material_properties(mat, idx)

    # the tray frame in the tool frame: x toward the tip (the utensil's axis from the gripper pads), z the food side
    axis_t = TIP / TIP.norm(dim=1, keepdim=True)
    z_t = NORMAL - (NORMAL * axis_t).sum(1, keepdim=True) * axis_t
    z_t = z_t / z_t.norm(dim=1, keepdim=True)
    y_t = torch.cross(z_t, axis_t, dim=1)
    R_tray = torch.stack([axis_t, y_t, z_t], 2)                  # tool <- tray
    q_tray = quat_from_matrix(R_tray)
    c_tray = TIP - 0.022 * axis_t                                # the head's centre, 22 mm back from the tip

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = "%d_%d" % (ROWS[0], ROWS[-1] + 1)
    frames = out / ("frames_" + tag)
    if cams:
        frames.mkdir(exist_ok=True)
        lo, hi = origins.min(0).values, origins.max(0).values
        mid, ext = (lo + hi) / 2, (hi - lo).max().item() + args.spacing
        floor = TABLE_Z - 0.74
        # the grid: from beyond the table side's corner, low, so the rows of arms run to the frame's top edge
        eye = torch.tensor([[hi[0].item() + 0.9 + 0.10 * ext, lo[1].item() - 0.6 - 0.10 * ext, floor + 1.6 + 0.12 * ext]], device=dev)
        target = torch.tensor([[mid[0].item() - 0.05 * ext, mid[1].item() + 0.05 * ext, floor + 0.6]], device=dev)
        cams["grid"].set_world_poses_from_view(eye, target)
        o = origins[args.hero]
        cams["hero"].set_world_poses_from_view((o + torch.tensor([1.10, -0.62, TABLE_Z + 0.42], device=dev))[None],
                                               (o + torch.tensor([0.02, 0.22, TABLE_Z + 0.10], device=dev))[None])
        np.savez(out / ("camera_" + tag + ".npz"), eye=eye.cpu().numpy(), target=target.cpu().numpy(),
                 origins=origins.cpu().numpy(), hero=args.hero)

    on_spoon = torch.zeros(N, dtype=torch.bool, device=dev)
    picked = torch.zeros(N, dtype=torch.bool, device=dev)
    at_mouth = torch.zeros(N, dtype=torch.bool, device=dev)
    dropped_at = torch.full((N,), -1, dtype=torch.long, device=dev)
    min_clear = torch.full((N,), 1e3, device=dev)
    peak_acc = torch.zeros(N, device=dev)
    prev_v = torch.zeros(N, 3, device=dev)
    tray_names = ["spoon_" + n for n, _, _ in TRAY]
    stuck = torch.zeros(N, dtype=torch.bool, device=dev)          # fork: the piece on the tines
    released = torch.zeros(N, dtype=torch.bool, device=dev)
    rel_p = torch.zeros(N, 3, device=dev); rel_q = torch.zeros(N, 4, device=dev); rel_q[:, 0] = 1
    lo_h, hi_h = (float(x) for x in args.hold_n.split(":"))
    g_ = torch.Generator(device=dev); g_.manual_seed(int(ROWS[0]))
    hold = lo_h + (hi_h - lo_h) * torch.rand(N, device=dev, generator=g_)
    mass_t = torch.tensor(B["mass"][ROWS], device=dev)
    prev_tool_v = torch.zeros(N, 3, device=dev); prev_tool_p = None
    grav = torch.tensor([0.0, 0.0, -9.81], device=dev)
    qinv = lambda q_: torch.cat([q_[:, :1], -q_[:, 1:]], 1)  # noqa: E731
    tray_off = {("spoon_" + n): torch.tensor(o, device=dev) for n, _, o in TRAY}
    for t in range(T):
        q_full[:, arm_j] = Q[CAND, t]
        robot.write_joint_state_to_sim(q_full, zero_full)
        scene.update(0.0)
        p_ee, q_ee = robot.data.body_pos_w[:, ee], robot.data.body_quat_w[:, ee]   # the flange (end_effector_link)
        q_tool = q_ee
        p_tool = p_ee + quat_apply(q_ee, torch.tensor([0.0, 0.0, 0.12], device=dev).repeat(N, 1))
        tip_w = p_tool + quat_apply(q_tool, TIP)
        q_t = quat_mul(q_tool, q_tray)
        c_w = p_tool + quat_apply(q_tool, c_tray)
        targets = {name: c_w + quat_apply(q_t, tray_off[name].repeat(N, 1)) for name in tray_names}
        if t == 0:                                   # start exactly on the path
            for name in tray_names:
                scene[name].write_root_pose_to_sim(torch.cat([targets[name], q_t], 1))
                scene[name].write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))
        scene["spoon"].write_root_pose_to_sim(torch.cat([c_w, q_t], 1))
        for _ in range(2):                           # each spoon piece driven by velocity onto its pose
            for name in tray_names:
                b: RigidObject = scene[name]
                p, qb = b.data.root_pos_w, b.data.root_quat_w
                v = (targets[name] - p) / (DT / 2)
                dq = quat_mul(q_t, torch.cat([qb[:, :1], -qb[:, 1:]], 1))
                dq = torch.where(dq[:, :1] < 0, -dq, dq)
                w = 2.0 * dq[:, 1:] / (DT / 2)
                b.write_root_velocity_to_sim(torch.cat([v, w], 1))
            scene.write_data_to_sim(); sim.step(render=False); scene.update(DT / 2)
        tt = torch.full((N,), t, device=dev)
        f = food.data.root_pos_w
        if args.utensil == "fork":
            # the tool's motion (for the pull on a skewered piece)
            tool_v = torch.zeros_like(p_tool) if prev_tool_p is None else (p_tool - prev_tool_p) / DT
            tool_a = (tool_v - prev_tool_v) / DT
            prev_tool_p, prev_tool_v = p_tool.clone(), tool_v
            # skewering: the tip at least 5 mm inside the piece during the pick
            qf = food.data.root_quat_w
            tip_f = quat_apply(qinv(qf), tip_w - f)
            h = SIZE / 2
            inside = (tip_f[:, 0].abs() < h) & (tip_f[:, 1].abs() < h) & (tip_f[:, 2] < h - 0.005) & (tip_f[:, 2] > -h)
            new = inside & ~stuck & ~released & (tt >= EV[:, 0]) & (tt <= EV[:, 1])
            if new.any():
                rel_p = torch.where(new[:, None], quat_apply(qinv(q_tool), f - p_tool), rel_p)
                rel_q = torch.where(new[:, None], quat_mul(qinv(q_tool), qf), rel_q)
            stuck = stuck | new
            # the pull along the tines (toward the tip) the piece needs to keep up with the tool, against its hold
            axis_w = quat_apply(q_tool, (TIP / TIP.norm(dim=1, keepdim=True)))
            F = mass_t[:, None] * (tool_a - grav)
            pull = -(F * axis_w).sum(1)                       # >0: the piece is pulled off past the tip
            bitten = tt >= EV[:, 3]
            off = stuck & ((pull > hold) | bitten)
            if off.any():
                stuck = stuck & ~off; released |= off
                food.write_root_velocity_to_sim(torch.where(off[:, None], torch.cat([tool_v, torch.zeros_like(tool_v)], 1),
                                                            food.data.root_vel_w))
            if stuck.any():                                   # the skewered piece rides the tines
                pose = torch.cat([p_tool + quat_apply(q_tool, rel_p), quat_mul(q_tool, rel_q)], 1)
                cur = torch.cat([food.data.root_pos_w, food.data.root_quat_w], 1)
                food.write_root_pose_to_sim(torch.where(stuck[:, None], pose, cur))
                food.write_root_velocity_to_sim(torch.where(stuck[:, None], torch.cat([tool_v, torch.zeros_like(tool_v)], 1),
                                                            food.data.root_vel_w))
            holding = stuck
        else:
            # where the food is against the spoon's head, in the tray frame
            rel = quat_apply(torch.cat([q_t[:, :1], -q_t[:, 1:]], 1), f - c_w)
            on = (rel[:, 0].abs() < 0.026) & (rel[:, 1].abs() < 0.018) & (rel[:, 2] > -0.004) & (rel[:, 2] < 0.035)
            lifted = f[:, 2] - origins[:, 2] > BOWL[:, 2] + 0.01
            holding = on & lifted
        picked |= holding & (tt >= EV[:, 1])
        was = on_spoon.clone()
        on_spoon = torch.where(tt >= EV[:, 1], holding, on_spoon)
        fell = was & ~on_spoon & (tt < EV[:, 2]) & (dropped_at < 0)
        dropped_at = torch.where(fell, tt, dropped_at)
        at_mouth |= (tt == EV[:, 2]) & holding
        v = food.data.root_lin_vel_w
        acc = (v - prev_v).norm(dim=1) / DT; prev_v = v
        peak_acc = torch.where(on_spoon, torch.maximum(peak_acc, acc), peak_acc)
        tip_w = p_tool + quat_apply(q_tool, TIP)
        d_head = (tip_w - (MOUTH + origins)).norm(dim=1)
        min_clear = torch.minimum(min_clear, d_head)
        if cams and t % args.frame_every == 0:
            sim.render()
            from PIL import Image
            for name, c in cams.items():
                c.update(DT)
                Image.fromarray(c.data.output["rgb"][0, ..., :3].cpu().numpy()).save(
                    frames / ("%s_%05d.jpg" % (name, t // args.frame_every)), quality=90)
            np.save(frames / ("%05d_state.npy" % (t // args.frame_every)),
                    np.stack([picked.cpu().numpy(), on_spoon.cpu().numpy(), (dropped_at >= 0).cpu().numpy()]))
        if t == 0 or t == int(EV[0, 0]) + 600 or t == int(EV[0, 0]) + 840 or t == int(EV[0, 1]):
            tip_e = tip_w - origins
            print("  [check t %.1f s] tool (env 0) %s vs MuJoCo at t=0 %s | tip %s | food %s | tray centre %s" % (
                t * DT, (p_tool - origins)[0].cpu().numpy().round(4), B["tool_at_start"][ROWS[0]][:3].round(4),
                tip_e[0].cpu().numpy().round(4), (food.data.root_pos_w - origins)[0].cpu().numpy().round(4),
                (c_w - origins)[0].cpu().numpy().round(4)), flush=True)
        if t % 600 == 0:
            print("  t %5.1f s: picked %d/%d, on the spoon %d" % (t * DT, int(picked.sum()), N, int(on_spoon.sum())),
                  flush=True)
    np.savez(out / ("results_" + tag + ".npz"), rows=ROWS, candidate=B["candidate"][ROWS], picked=picked.cpu().numpy(),
             at_mouth=at_mouth.cpu().numpy(), dropped_at_s=(dropped_at.float() * DT).cpu().numpy(),
             min_tip_to_mouth_m=min_clear.cpu().numpy(), peak_food_acc=peak_acc.cpu().numpy(),
             duration_s=(torch.tensor(B["length"][ROWS]) * DT).numpy(), utensil=args.utensil,
             hold_n=hold.cpu().numpy() if args.utensil == "fork" else 0)
    print("DONE %s: picked %d/%d, at the mouth %d/%d" % (tag, int(picked.sum()), N, int(at_mouth.sum()), N), flush=True)


main()
import os  # noqa: E402
os._exit(0)                                         # app.close() can hang headless; the results are written
