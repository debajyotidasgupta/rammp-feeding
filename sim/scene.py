"""The twin scene, in the ARM BASE FRAME (Kortex base, metres, Z-up), built from the deployment's calibration.

    from sim.scene import Layout, build
    model = build(Layout.from_config(mouth=(-0.277, 0.306, 0.635), foods=[Food(...)]))

What is real and what is set dressing:
  real   the arm (Kinova's Gen3 model; joints replayed from the run record), the fork (tip and tines where
         kinova/motion.tip_in_tool puts them), the wrist camera (config/camera: mount and intrinsics, so its
         render lines up with the real frame), the table height (rammp.yaml `workspace`), the food (the run's
         detections), the mouth (the delivery's measured mouth_base: the person is placed so it lands there)
         and the food container (the clear clamshell on a blue mat: its detection box and the overlook depth)
  dress  the room, the wheelchair and the arm's mount, the laptop, the person's body (SMPL-X, seated)

The user faces +X with the table in front of them; the arm is on their right, at the base origin (so the user
sits at +Y, as on the real chair). Nothing here is a collision model: the twin replays, it does not simulate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from settings import cfg
from sim import procedural as P

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
HUMAN = ASSETS / "human"
GRIP_CLOSED = {"right_driver_joint": 0.7235, "right_spring_link_joint": 0.7235, "right_follower_joint": -0.6947,
               "left_driver_joint": 0.7235, "left_spring_link_joint": 0.7235, "left_follower_joint": -0.6947}
PRINCIPAL_V_SIGN = -1.0      # MuJoCo's principal offset is y-up; the camera's is image-down (checked: tine tips at v 425)
TUB_HEIGHT_M = 0.055         # the clear clamshell: its rim ~55 mm above the table in the overlook depth
UPPER = ("skin", "hair", "eyes", "sclera", "shirt")          # parts that ride the lean hinge (pose_human.py)
LOWER = ("trousers", "shoes")


@dataclass
class Food:
    """One piece: its centre on the bowl floor (base frame), edge length, colour, yaw."""
    xy: tuple[float, float]
    size: float = 0.024
    rgb: tuple[float, float, float] = (0.95, 0.6, 0.3)
    yaw: float = 0.0
    aspect: tuple[float, float, float] = (1.0, 1.0, 1.0)   # a chunk, not a die: per-axis scale


@dataclass
class Layout:
    table_z: float
    bowl_xy: tuple[float, float]
    mouth: tuple[float, float, float]
    foods: list[Food] = field(default_factory=list)
    bowl_half: tuple[float, float] = (0.062, 0.074)     # the container's half extents (x, y) at its rim
    bowl_round: tuple[float, float] | None = None       # (rim radius, height): a round bowl in place of the tub
    width: int = 1920
    height: int = 1080
    laptop: bool = True                                  # the 2026-09-25 table's laptop
    tip_in_tool: tuple[float, float, float] | None = None   # a run's own utensil tip (else this machine's model)
    hair: bool = True                                    # False: the scalp's shape in skin (a bald head)

    @classmethod
    def from_config(cls, mouth: tuple[float, float, float], foods: list[Food] | None = None, **kw) -> "Layout":
        w = cfg.workspace
        kw.setdefault("bowl_xy", tuple(map(float, w.bowl_centre_m)))
        return cls(float(w.table_height_m), mouth=tuple(mouth), foods=foods or [], **kw)

    @property
    def bowl_floor_z(self) -> float:
        return self.table_z + 0.004


def tip_in_flange(tip_tool=None) -> np.ndarray:
    """The fork's tine tips in the flange frame (the tool frame is the flange + 120 mm along z); tip_tool: a run's
    own tip in the tool frame (a dump's tip_in_tool), else this machine's tip model."""
    from kinova.motion import tip_in_tool
    return np.asarray(tip_in_tool() if tip_tool is None else tip_tool, float) + np.array([0.0, 0.0, 0.12])


def camera_in_flange() -> np.ndarray:
    from feeding.camera import CAM_IN_TOOL                  # the profile's mount, or the file it names
    return np.asarray(CAM_IN_TOOL, float) + np.array([0.0, 0.0, 0.12])


def _quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _quat_between(a: np.ndarray, b: np.ndarray) -> list[float]:
    """The shortest rotation taking direction a to b (w, x, y, z)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = np.cross(a, b)
    w = 1.0 + float(a @ b)
    if w < 1e-8:
        return [0.0, 1.0, 0.0, 0.0]
    q = np.array([w, *c])
    return list(q / np.linalg.norm(q))


def _materials(s: mujoco.MjSpec) -> None:
    def tex(name, path, **kw):
        s.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path), **kw)

    tex("wood", P.wood_png())
    tex("concrete", P.floor_png())
    tex("fabric", P.fabric_png())
    s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                  rgb1=[0.10, 0.12, 0.16], rgb2=[0.01, 0.01, 0.02], width=512, height=3072)

    def mat(name, rgba, texture=None, **kw):
        m = s.add_material(name=name, rgba=rgba, **kw)
        if texture:
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture
        return m

    mat("floor", [1, 1, 1, 1], "concrete", texrepeat=[3, 3], reflectance=0.22, specular=0.4, shininess=0.6)
    mat("wood", [1, 1, 1, 1], "wood", texrepeat=[1.2, 1.2], texuniform=True, reflectance=0.06, specular=0.35,
        shininess=0.55)
    mat("plastic_clear", [0.86, 0.93, 1.0, 0.22], specular=1.0, shininess=1.0)
    mat("mat_blue", [0.05, 0.12, 0.62, 1], specular=0.7, shininess=0.8, reflectance=0.05)
    mat("alu", [0.62, 0.64, 0.68, 1], specular=0.8, shininess=0.7, reflectance=0.05)
    mat("keys", [0.08, 0.08, 0.09, 1], specular=0.3, shininess=0.3)
    s.add_texture(name="screen", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(P.screen_png()))
    mat("screen", [1, 1, 1, 1], "screen", emission=0.9, specular=0.6, shininess=0.9)
    mat("arm", [0.78, 0.80, 0.83, 1], specular=0.7, shininess=0.75, reflectance=0.04)
    mat("arm_dark", [0.12, 0.13, 0.15, 1], specular=0.5, shininess=0.6)
    mat("steel", [0.86, 0.87, 0.90, 1], specular=1.0, shininess=1.0, reflectance=0.35)
    mat("rubber", [0.05, 0.05, 0.055, 1], specular=0.15, shininess=0.2)
    mat("chrome", [0.72, 0.74, 0.78, 1], specular=1.0, shininess=0.95, reflectance=0.3)
    mat("cushion", [0.16, 0.18, 0.22, 1], "fabric", texrepeat=[6, 6], specular=0.1, shininess=0.1)
    mat("frame", [0.20, 0.22, 0.26, 1], specular=0.8, shininess=0.8, reflectance=0.08)
    mat("skin", [0.87, 0.70, 0.58, 1], specular=0.25, shininess=0.35)
    mat("hair", [0.10, 0.07, 0.05, 1], specular=0.35, shininess=0.4)
    mat("eyes", [0.16, 0.10, 0.06, 1], specular=1.0, shininess=1.0)
    mat("sclera", [0.93, 0.92, 0.90, 1], specular=0.8, shininess=0.9)
    mat("shirt", [0.07, 0.07, 0.08, 1], specular=0.12, shininess=0.2)        # a black T-shirt
    mat("trousers", [0.16, 0.20, 0.30, 1], specular=0.1, shininess=0.15)
    mat("shoes", [0.93, 0.93, 0.94, 1], specular=0.35, shininess=0.4)


def _visual(s: mujoco.MjSpec, lay: Layout) -> None:
    v = s.visual
    v.global_.offwidth, v.global_.offheight = lay.width, lay.height
    v.global_.fovy = 38
    v.quality.shadowsize = 8192
    v.quality.offsamples = 8
    v.headlight.ambient = [0.22, 0.22, 0.25]
    v.headlight.diffuse = [0.28, 0.28, 0.30]
    v.headlight.specular = [0.1, 0.1, 0.1]
    v.map.znear = 0.005
    v.map.haze = 0.25
    v.rgba.haze = [0.06, 0.07, 0.09, 1]


def _arm(s: mujoco.MjSpec, lay: Layout) -> None:
    """Kinova's Gen3 + the 2F-85 (closed on the fork) + the fork and the wrist camera, at the base origin."""
    arm = mujoco.MjSpec.from_file(str(ASSETS / "kinova_gen3" / "gen3.xml"))
    grip = mujoco.MjSpec.from_file(str(ASSETS / "robotiq_2f85" / "2f85.xml"))
    for sp in (arm, grip):
        # the stock models set solver options; match the scene's defaults so attaching does not warn (a replay
        # never steps the solver)
        sp.option.impratio, sp.option.cone, sp.option.integrator = s.option.impratio, s.option.cone, s.option.integrator
        for g in sp.geoms:
            if g.contype == 0 and g.conaffinity == 0:        # the visual meshes
                g.group = 2
            else:
                g.group = 3
    # materials live in the spec that uses them (attaching prefixes their names)
    arm.add_material(name="arm", rgba=[0.78, 0.80, 0.83, 1], specular=0.7, shininess=0.75, reflectance=0.04)
    arm.add_material(name="arm_dark", rgba=[0.12, 0.13, 0.15, 1], specular=0.5, shininess=0.6)
    arm.add_material(name="fork", rgba=[0.96, 0.96, 0.95, 1], specular=0.6, shininess=0.7)   # white plastic
    for g in arm.geoms:
        if g.group == 2:
            g.material = "arm_dark" if g.meshname in ("bracelet_with_vision_link", "base_link") else "arm"
            g.rgba = [1, 1, 1, 1]
    for c in list(arm.cameras):
        arm.delete(c)
    for key in list(arm.keys):
        arm.delete(key)
    for a in list(arm.actuators):
        arm.delete(a)
    for key in list(grip.keys):
        grip.delete(key)
    brace = arm.body("bracelet_link")
    flange = brace.add_body(name="flange", pos=[0, 0, -0.061525], quat=[0, 1, 0, 0])
    gf = flange.add_frame()                                 # fingers close along y: on the handle's flat faces
    gf.attach_body(grip.body("base_mount"), "grip_", "")
    _fork(flange, lay.tip_in_tool)
    cam = camera_in_flange()
    c = cfg.camera.colour                                    # the measured intrinsics (principal point too)
    wrist = flange.add_camera(name="wrist", pos=list(cam), quat=[0, 0, 1, 0])   # looks along the flange's +z;
    wrist.resolution = [int(c.width), int(c.height)]                            # image +u = flange -x
    wrist.sensor_size = [c.width * 3e-6, c.height * 3e-6]                     # any pitch: only ratios matter
    wrist.focal_pixel = [float(c.fx), float(c.fy)]
    wrist.principal_pixel = [float(c.cx) - c.width / 2, PRINCIPAL_V_SIGN * (float(c.cy) - c.height / 2)]
    flange.add_site(name="tip", pos=list(tip_in_flange(lay.tip_in_tool)), size=[0.002, 0, 0], rgba=[1, 0.8, 0.1, 0], group=5)
    flange.add_site(name="cam", pos=list(cam), size=[0.002, 0, 0], group=5)
    frame = s.worldbody.add_frame()
    frame.attach_body(arm.body("base_link"), "arm_", "")


def _fork(flange: mujoco.MjsBody, tip_tool=None) -> None:
    """The fork as calibrated (white plastic): a flat handle in the pads, a neck, a 23 mm head curving to the tines."""
    tip = tip_in_flange(tip_tool)
    w = float(cfg.utensil.head_width_m)
    fork = flange.add_body(name="fork")
    fork.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0055, 0.0017, 0.042], pos=[0, 0, 0.158],
                  material="fork", group=2)
    base = np.array([tip[0] * 0.55, tip[1] * 0.45, 0.212])
    fork.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0028, 0, 0], fromto=[0, 0, 0.198, *base],
                  material="fork", group=2)
    head0 = np.array([tip[0], tip[1] * 0.72, 0.236])
    fork.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.003, 0, 0], fromto=[*base, *head0],
                  material="fork", group=2)
    across = np.linspace(-w / 2 + 0.0015, w / 2 - 0.0015, 4)
    d = head0 - base
    fork.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[w / 2, 0.0011, 0.004], pos=list(head0),
                  quat=_quat_between(np.array([0, 0, 1.0]), d), material="fork", group=2)
    for dx in across:
        a = head0 + np.array([dx, 0, 0.003])
        mid = np.array([tip[0] + dx, (head0[1] + tip[1]) / 2 + 0.0012, (head0[2] + tip[2]) / 2])
        b = np.array([tip[0] + dx, tip[1], tip[2]])
        fork.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0010, 0, 0], fromto=[*a, *mid],
                      material="fork", group=2)
        fork.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0009, 0, 0], fromto=[*mid, *b],
                      material="fork", group=2)


def _room(s: mujoco.MjSpec, lay: Layout, floor_z: float) -> None:
    wb = s.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.05], pos=[0, 0, floor_z],
                material="floor", group=2)
    # the table: a walnut top on slim dark legs, its surface at the calibrated height
    tz, th = lay.table_z, 0.028
    x0, x1, y0, y1 = 0.075, 0.70, -0.30, 0.95
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    table = wb.add_body(name="table", pos=[cx, cy, 0])
    table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, th / 2],
                   pos=[0, 0, tz - th / 2], material="wood", group=2)
    leg_h = tz - th - floor_z
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.018, 0.018, leg_h / 2],
                           pos=[sx * ((x1 - x0) / 2 - 0.05), sy * ((y1 - y0) / 2 - 0.05), floor_z + leg_h / 2],
                           material="frame", group=2)
    # the food's clear clamshell on a blue mat (and, on 2026-09-25, a laptop)
    bx, by = lay.bowl_xy
    wb.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.13, 0.12, 0.0008], pos=[bx + 0.03, by + 0.01, tz + 0.0008],
                material="mat_blue", group=2)
    if lay.bowl_round:
        _round_bowl(wb, bx, by, tz, *lay.bowl_round)
    else:
        hx, hy = lay.bowl_half
        tub = s.add_mesh(name="tub", file=str(P.tub_obj(hx, hy, TUB_HEIGHT_M)))
        tub.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        wb.add_geom(name="tub", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tub", pos=[bx, by, tz + 0.0016],
                    material="plastic_clear", group=2)
    if lay.laptop:
        _laptop(wb, bx, by, tz)
    # lights: a warm key over the table (shadows), a cool rim from behind the user, a soft fill
    wb.add_light(name="key", pos=[0.9, -0.6, 1.9], dir=[-0.45, 0.35, -1], diffuse=[0.95, 0.9, 0.82],
                 specular=[0.5, 0.5, 0.45], castshadow=True, cutoff=48, exponent=6, attenuation=[1, 0, 0])
    wb.add_light(name="rim", pos=[-1.2, 1.1, 1.5], dir=[1, -0.6, -0.55], diffuse=[0.35, 0.55, 0.85],
                 specular=[0.3, 0.4, 0.6], castshadow=True, cutoff=50, exponent=4)
    wb.add_light(name="fill", pos=[0.4, 1.6, 1.2], dir=[0, -1, -0.6], diffuse=[0.25, 0.25, 0.28], castshadow=False)
    wb.add_light(name="table_glow", pos=[*lay.bowl_xy, tz + 0.9], dir=[0, 0, -1], diffuse=[0.35, 0.3, 0.25],
                 castshadow=False, cutoff=22, exponent=12)


def _round_bowl(wb, bx: float, by: float, tz: float, r: float, h: float, n: int = 36) -> None:
    """An opaque round bowl: a floor and n wall staves round the rim (what a depth camera sees of a ceramic one)."""
    bowl = wb.add_body(name="bowl", pos=[bx, by, tz])
    bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.75 * r, 0.004, 0], pos=[0, 0, 0.004], material="shoes",
                  group=2)
    for k in range(n):
        a = 2 * math.pi * k / n
        rr = r * (0.75 + 0.25 * 0.5)                           # the staves lean out: centre between base and rim
        bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.003, math.pi * r / n * 1.05, h / 2],
                      pos=[rr * math.cos(a), rr * math.sin(a), h / 2], quat=_quat_y_about(a, math.atan2(0.25 * r, h)),
                      material="shoes", group=2)


def _quat_y_about(yaw: float, lean: float) -> list[float]:
    """A stave at angle yaw round the bowl, leaning outward by `lean` (rad)."""
    cz, sz, cy, sy = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(lean / 2), math.sin(lean / 2)
    # q = Rz(yaw) * Ry(lean): the stave's top tips out along +x before the turn
    return [cz * cy, -sz * sy, cz * sy, sz * cy]


def _laptop(wb, bx: float, by: float, tz: float) -> None:
    """A laptop to the user's left (as in the 2026-09-25 wrist camera's frame)."""
    lx, ly = bx + 0.02, by + 0.30
    lap = wb.add_body(name="laptop", pos=[lx, ly, tz], quat=_quat_z(math.radians(-8)))
    lap.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.105, 0.155, 0.007], pos=[0, 0, 0.007], material="alu", group=2)
    lap.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.13, 0.0006], pos=[0.02, 0, 0.0142], material="keys",
                 group=2)
    # the hinge on the far (+x) edge, the lid open ~110 deg, the screen toward the user (-x)
    lid = lap.add_body(pos=[0.105, 0, 0.014], quat=[math.cos(0.175), 0, math.sin(0.175), 0])
    lid.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.155, 0.105], pos=[0.004, 0, 0.105], material="alu",
                 group=2)
    lid.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0005, 0.145, 0.095], pos=[-0.0006, 0, 0.107],
                 material="screen", group=2)


def _wheelchair(s: mujoco.MjSpec, pelvis: np.ndarray, floor_z: float) -> None:
    """A manual wheelchair under the person, and the arm's mount on its front right (the base origin)."""
    wb = s.worldbody
    px, py, pz = pelvis
    seat_z = pz - 0.075
    ch = wb.add_body(name="wheelchair", pos=[0, 0, 0])
    ch.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.21, 0.225, 0.035], pos=[px + 0.06, py, seat_z],
                material="cushion", group=2)
    ch.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.225, 0.24], pos=[px - 0.17, py, seat_z + 0.29],
                quat=[math.cos(-0.05), 0, math.sin(-0.05), 0], material="cushion", group=2)
    R, r = 0.29, 0.016
    tyre = s.add_mesh(name="tyre", file=str(P.torus_obj(R, r)))
    rim = s.add_mesh(name="pushrim", file=str(P.torus_obj(R - 0.035, 0.006, name="pushrim")))
    for tm in (tyre, rim):
        tm.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    wz = floor_z + R + r
    for side in (-1, 1):
        wy = py + side * 0.265
        q = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tyre", pos=[px - 0.03, wy, wz], quat=q,
                    material="rubber", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pushrim", pos=[px - 0.03, wy + side * 0.025, wz],
                    quat=q, material="chrome", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[R - 0.01, 0.002, 0], pos=[px - 0.03, wy, wz],
                    quat=q, rgba=[0.5, 0.55, 0.6, 0.18], group=2)
        for k in range(12):                                  # spokes
            a = k * math.pi / 6
            d = np.array([math.cos(a), 0, math.sin(a)]) * (R - 0.012)
            c = np.array([px - 0.03, wy, wz])
            ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.0012, 0, 0], fromto=[*(c - d), *(c + d)],
                        material="chrome", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.02, 0], pos=[px - 0.03, wy, wz],
                    quat=q, material="frame", group=2)
        # frame: side rail, front post, footrest hanger, a caster
        sy = py + side * 0.235
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011, 0, 0],
                    fromto=[px - 0.18, sy, seat_z - 0.05, px + 0.26, sy, seat_z - 0.05], material="frame", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011, 0, 0],
                    fromto=[px + 0.26, sy, seat_z - 0.05, px + 0.30, sy, floor_z + 0.10], material="frame", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011, 0, 0],
                    fromto=[px + 0.26, sy, seat_z - 0.05, px + 0.46, sy, floor_z + 0.09], material="frame", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.045, 0.012, 0],
                    pos=[px + 0.31, sy, floor_z + 0.045], quat=q, material="rubber", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011, 0, 0],
                    fromto=[px - 0.18, sy, seat_z - 0.05, px - 0.20, sy, seat_z + 0.55], material="frame", group=2)
        ch.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.13, 0.03, 0.012],
                    pos=[px + 0.02, py + side * 0.25, seat_z + 0.20], material="cushion", group=2)
    ch.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.08, 0.20, 0.008], pos=[px + 0.48, py, floor_z + 0.09],
                material="frame", group=2)
    # the arm's mount: a bracket from the chair's right rail to a column under the base
    ry = py - 0.235
    ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.014, 0, 0],
                fromto=[px + 0.20, ry, seat_z - 0.05, 0.0, 0.0, seat_z - 0.05], material="frame", group=2)
    ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.048, (0.0 - (seat_z - 0.06)) / 2, 0],
                pos=[0, 0, (seat_z - 0.06) / 2], material="arm_dark", group=2)
    ch.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.062, 0.004, 0], pos=[0, 0, -0.004],
                material="chrome", group=2)


def _person(s: mujoco.MjSpec, lay: Layout) -> tuple[np.ndarray, float]:
    """The seated person, moved so their (closed) mouth is the measured one. Returns (pelvis, floor z)."""
    if not (HUMAN / "landmarks.json").exists():
        raise FileNotFoundError("no person meshes: run  .venv/bin/python sim/pose_human.py  once (needs SMPL-X)")
    lm = json.loads((HUMAN / "landmarks.json").read_text())
    off = np.array(lay.mouth) - np.array(lm["closed"]["mouth"])
    hinge = np.array(lm["closed"]["lean_hinge"]) + off
    lower = s.worldbody.add_body(name="user_lower", pos=list(off))
    upper = s.worldbody.add_body(name="user_upper", pos=list(hinge))
    upper.add_joint(name="lean", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
    for variant, group in (("closed", 1), ("open", 4)):
        for part in LOWER + UPPER:
            if part in LOWER and variant == "open":
                continue
            mesh = s.add_mesh(name="%s_%s" % (part, variant), file=str(HUMAN / ("%s_%s.obj" % (part, variant))))
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            body, pos = (lower, [0, 0, 0]) if part in LOWER else (upper, list(off - hinge))
            body.add_geom(name="%s_%s" % (part, variant), type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=mesh.name, pos=pos, material="skin" if part == "hair" and not lay.hair else part,
                          group=1 if part in LOWER else group)
    upper.add_site(name="mouth", pos=list(np.array(lay.mouth) - hinge), size=[0.004, 0, 0], group=5)
    return np.array(lm["closed"]["pelvis"]) + off, float(off[2])


def _food(s: mujoco.MjSpec, lay: Layout) -> None:
    mesh = s.add_mesh(name="food", file=str(P.rounded_cube_obj()))
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    for i, f in enumerate(lay.foods):
        h = f.size / 2
        b = s.worldbody.add_body(name="food%d" % i, mocap=True, pos=[*f.xy, lay.bowl_floor_z + h * f.aspect[2]],
                                 quat=_quat_z(f.yaw))
        mat = s.add_material(name="food%d" % i, rgba=[*f.rgb, 1], specular=0.55, shininess=0.7, reflectance=0.02)
        mesh_i = s.add_mesh(name="food%d" % i, file=mesh.file, scale=[h * a for a in f.aspect])
        mesh_i.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        b.add_geom(name="food%d" % i, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_i.name, material=mat.name,
                   group=2)


def build(lay: Layout, extra=None) -> mujoco.MjModel:
    s = mujoco.MjSpec()
    s.modelname = "rammp_twin"
    s.compiler.degree = False
    _visual(s, lay)
    _materials(s)
    pelvis, floor_z = _person(s, lay)
    _room(s, lay, floor_z)
    _wheelchair(s, pelvis, floor_z)
    _arm(s, lay)
    _food(s, lay)
    if extra is not None:                                   # a caller's additions (a scene camera, ...)
        extra(s)
    return s.compile()


def new_data(m: mujoco.MjModel) -> mujoco.MjData:
    """The scene's data with the fingers closed on the fork. (Not via qpos0: hinges are measured from qpos0, so a
    closed qpos0 would still draw the model's open pose.)"""
    d = mujoco.MjData(m)
    for name, q in GRIP_CLOSED.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_grip_" + name)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = q
    return d


def arm_qpos_adr(m: mujoco.MjModel) -> np.ndarray:
    return np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint_%d" % i)]
                     for i in range(1, 8)])
