"""rammp_feeding.ros.Robot without ROS, for the offline tests: a kinematic arm that plays the trajectories it is sent
(sampled at 100 Hz), a straight-line "planner", and cameras rendering the scene from where the arm actually holds them.

The scene: a table at table_z, a clear bowl (its depth reads the table through it, its rim rim_height up) round bowl_xy,
one food piece (a cylinder) at piece_xy, the user's mouth at mouth. SAM3's boxes are the bowl's and the piece's
projections. On the level site, the fork's stand beside the bowl: the block on the fork's handle, its level top (the
rim) at hole_at with a rectangular opening in it (the lab's: 24.1 x 57.9 mm, hole_depth deep), and the stand's taped
strip running from the block's side away across its long side, strip_drop under the rim, both rendered in depth; SAM3's
masks for the opening's words the opening, for the stand's words the strip. The gripper's commands are kept (grips: the
position, and where the closed fingertips were). The site (tests/fixture.py's RAMMP_TEST_SITE): level -- the arm level
on its mount, a D405 on the wrist, the user to the arm's left; tilt30 -- the arm leaning 30 deg forward on its mount
(that robot's own description render), an OAK-D Pro on the wrist, a narrow deep cup on a low table, the user to the
arm's right.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import yaml

from rammp_feeding import scene
from rammp_feeding.camera import CameraModel, Intrinsics
from rammp_feeding.config import cfg
from rammp_feeding.geometry import fold, joint_delta
from rammp_feeding.ros import Detection

TILT = os.environ.get("RAMMP_TEST_SITE", "level") == "tilt30"
# the wrist camera's camera_info: an OAK-D Pro's 640x400 colour (tilt30), the D405's 848x480 (one sensor for both)
INTR = (Intrinsics(398.443, 398.350, 337.992, 203.154, 640, 400) if TILT else
        Intrinsics(434.884, 434.330, 423.960, 237.798, 848, 480))
CAM = CameraModel(INTR, INTR, np.eye(4))
MODEL = open(os.path.join(os.path.dirname(__file__), "arm_tilt30.urdf" if TILT else "arm_level.urdf")).read()


@dataclass
class Scene:
    table_z: float = 0.095 if TILT else 0.0072      # the test site's table, in the arm's base frame
    bowl_xy: tuple = (0.639, -0.223) if TILT else (0.290, 0.158)
    bowl_radius: float = 0.045 if TILT else 0.07
    rim_height: float = 0.08 if TILT else 0.06      # the rim over the table
    piece_xy: tuple = (0.645, -0.217) if TILT else (0.296, 0.152)
    piece_radius: float = 0.013
    piece_height: float = 0.025
    floor_lift: float = 0.0                         # the surface the depth reads under the bowl, above the table
    mouth: tuple = (-0.27, -0.20, 0.60) if TILT else (-0.277, 0.306, 0.428)
    hole_at: tuple = (0.55, -0.05, 0.095 + 0.20) if TILT else (0.42, 0.02, 0.0072 + 0.20)   # the opening's centre, at
                                                    # the rim (on tilt30 beside its bowl, in the 30 deg arm's reach)
    hole_yaw_deg: float = 64.0                      # its long side's direction, round from the base's x (fork_look's
                                                    # fingers: 60)
    hole_size: tuple = (0.0241, 0.0579)             # the opening: short side, long side
    block_size: tuple = (0.045, 0.085)              # the block's top round it: short side, long side
    hole_depth: float = 0.04
    strip_gap: float = 0.03                         # the strip's seat end this far from the block's side, across its long
    strip_size: tuple = (0.03, 0.10)                # side; the strip's width, its length
    strip_drop: float = 0.066                       # its top under the block's rim

    def move_stand(self, dx: float, dy: float, turn_deg: float) -> None:
        """The stand (the block and its strip) moved and turned about the opening's centre."""
        self.hole_at = (self.hole_at[0] + dx, self.hole_at[1] + dy, self.hole_at[2])
        self.hole_yaw_deg += turn_deg


class FakeTrajectory:
    def __init__(self, robot, positions, times):
        self.robot, self.p, self.t = robot, np.asarray(positions, float), np.asarray(times, float)
        self.k, self.cancelled = 0, False
        self.n = max(2, int(np.ceil(self.t[-1] * 100)))

    def step(self) -> None:
        """One 10 ms sample further along the trajectory."""
        if self.done():
            return
        self.k += 1
        tt = self.t[-1] * self.k / self.n
        self.robot.raw = np.array([np.interp(tt, self.t, self.p[:, j]) for j in range(7)])
        got = self.robot.inject.get("now")                 # (at this fraction of the move, a torque step)
        if got is not None and self.k / self.n > got[0]:
            self.robot.effort = np.array(got[1], float)

    def done(self) -> bool:
        return self.cancelled or self.k >= self.n

    def result(self):
        return (-6, "PREEMPTED") if self.cancelled else (0, "")

    def cancel(self, wait_s: float = 2.0) -> None:
        self.cancelled = True


class FakeCamera:
    def __init__(self, robot, role):
        self.robot, self.role = robot, role
        self.colour_topic = "/fake/%s/color" % role

    def frame(self, after: float = 0.0, wait_s: float = 3.0):
        t = max(time.time(), after + 1e-3)
        return t, np.full((INTR.height, INTR.width, 3), 120, np.uint8)

    def depth_median(self, frames, since, wait_s=3.0, min_frames=5):
        return self.robot.render_depth()


class FakeRobot:
    def __init__(self, sc: Scene, q0):
        self.sc = sc
        self.raw = np.radians(np.asarray(q0, float))
        self.effort = np.zeros(7)
        self.estopped = False
        self.inject: dict = {}
        self.effort_fn = None                           # t (unix s) -> joint torques: a bite, say
        self.traj: FakeTrajectory | None = None
        self.cameras = {"camera": FakeCamera(self, "camera")}
        self.moves: list[str] = []
        self.worlds: list = []                          # set_world: a file's content; "reset": the planner's own back
        self.grips: list[tuple[float, np.ndarray]] = []   # gripper(): the position, where the fingertips were
        self.fork_slip_px = (0, 0)                       # the fork in the fingers, moved this far in the wrist image
        self.fingers = 0.0                              # where they are: 0 open .. 1 closed (None: not served)
        self.grip_settings: list[tuple] = []            # gripper(): (speed, force)
        from rammp_feeding.arm import chain
        self.kin = chain(self)

    # --- the models the stack serves
    def robot_description(self, wait_s: float = 3.0):
        return MODEL                                    # the 2.0 description's level render, rooted at arm_mount

    def camera_model(self, role: str, wait_s: float = 5.0) -> CameraModel:
        return CAM

    def publish_static(self, parent, child, T):
        pass

    # --- the arm
    def joint_sample(self, wait_s: float = 2.0):
        if self.traj is not None:
            self.traj.step()
        if self.effort_fn is not None:
            time.sleep(0.01)                            # 100 Hz, in real time (the bite is timed)
            self.effort = np.asarray(self.effort_fn(time.time()), float)
        return time.time() + 1e-6 * np.random.rand(), self.raw.copy(), self.effort.copy()

    def joints(self):
        return [float(v) for v in fold(np.degrees(self.raw))]

    def execute(self, positions_rad, times_s, impedance=None, path_tolerance_rad=None, goal_tolerance_rad=None):
        assert np.allclose(positions_rad[0], self.raw, atol=1e-6), "a trajectory not starting where the arm is"
        assert all(b > a for a, b in zip(times_s, times_s[1:])), "times not increasing"
        steps = np.abs(np.diff(np.asarray(positions_rad), axis=0)).max(1) / np.diff(times_s)
        assert steps.max() < np.radians(81), "a joint over its speed limit (%.0f deg/s)" % np.degrees(steps.max())
        self.effort = np.zeros(7)
        self.inject["now"] = self.inject.pop(len(self.moves), None)
        self.moves.append("impedance" if impedance else "position")
        self.traj = FakeTrajectory(self, positions_rad, times_s)
        return self.traj

    # --- the planner: a straight joint line
    def plan_joints(self, start_deg, goal_deg, timeout_s):
        s = np.asarray(start_deg, float)
        g = s + joint_delta(s, goal_deg)
        n = 60
        return np.array([s + (g - s) * k / n for k in range(n + 1)]), np.linspace(0, 3, n + 1)

    def plan_pose(self, start_deg, T, R, timeout_s, hold=1):
        """A path to the tool pose (T, R), as the planner picks an arm shape for a pose: a straight joint line onto the
        IK solution nearest the start (from it, and from seeds round it)."""
        s, rng, best = np.asarray(start_deg, float), np.random.default_rng(0), None
        for seed in [s] + [s + rng.normal(0.0, 45.0, 7) for _ in range(40)]:
            g = self.kin.ik(np.asarray(T, float), np.asarray(R, float), list(seed), iters=200)
            if g is not None:
                g = s + joint_delta(s, g)
                if best is None or np.abs(g - s).max() < np.abs(best - s).max():
                    best = g
        if best is None:
            raise RuntimeError("the fake planner: no IK at the goal")
        n = 60
        return np.array([s + (best - s) * k / n for k in range(n + 1)]), np.linspace(0, 3, n + 1)

    def set_world(self, path):
        self.worlds.append(yaml.safe_load(open(path)) if os.path.isfile(path) else path)

    def reset_world(self):
        self.worlds.append("reset")

    def planner_listens(self):
        return True

    # --- the gripper: opened in the block's opening, its end walls stop the fingers (a 2F-85: 85 mm open)
    def gripper(self, position: float, speed: float | None = None, force: float | None = None) -> None:
        from rammp_feeding.fork import fingertips_in_tool
        T, R = self.kin.fk(self.joints())
        tips = T + R @ fingertips_in_tool()
        self.grips.append((float(position), tips))
        self.grip_settings.append((speed, force))
        sc, at = self.sc, float(position)
        if sc.hole_at is not None:
            c, a = np.asarray(sc.hole_at, float), np.radians(sc.hole_yaw_deg)
            d = tips[:2] - c[:2]
            inside = (abs(d @ [-np.sin(a), np.cos(a)]) < sc.hole_size[0] / 2 and abs(d @ [np.cos(a), np.sin(a)])
                      < sc.hole_size[1] / 2 and c[2] - sc.hole_depth < tips[2] < c[2])
            if inside:
                at = max(at, 1.0 - sc.hole_size[1] / 0.085)
        self.fingers = at if self.fingers is not None else None

    def gripper_position(self):
        return self.fingers

    # --- the wrist camera: depth rendered from the arm's pose, SAM3's boxes projected
    def camera_pose(self):
        T, R = self.kin.fk(self.joints())
        return T + R @ scene.CAM_IN_TOOL, R @ scene.CAM_ROT          # (independently of scene.camera_pose)

    def _rays(self):
        o, Rc = self.camera_pose()
        vv, uu = np.mgrid[0:INTR.height, 0:INTR.width]
        return o, np.stack([(uu - INTR.cx) / INTR.fx, (vv - INTR.cy) / INTR.fy, np.ones_like(uu, float)], -1) @ Rc.T

    def _disc(self, centre, r, z):
        """The pixels whose ray meets the horizontal disc (centre, r) at height z."""
        o, d = self._rays()
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - o[2]) / d[..., 2]
        p = o + t[..., None] * d
        return (np.hypot(p[..., 0] - centre[0], p[..., 1] - centre[1]) < r) & (t > 0)

    def render_depth(self) -> np.ndarray:
        """mm along the optical axis: the bowl's rim (6 mm wide, rim_height up), the piece's top where its disc is, else
        the table (the bowl's walls and floor read as it)."""
        o, d = self._rays()
        sc = self.sc
        floor, rim = sc.table_z + sc.floor_lift, sc.table_z + sc.rim_height
        with np.errstate(divide="ignore", invalid="ignore"):
            t_tab = (floor - o[2]) / d[..., 2]
            t_top = (floor + sc.piece_height - o[2]) / d[..., 2]
            t_rim = (rim - o[2]) / d[..., 2]
        z = np.where(t_tab > 0, t_tab, np.nan)
        z = np.where(self._disc(sc.piece_xy, sc.piece_radius, floor + sc.piece_height), t_top, z)
        ring = self._disc(sc.bowl_xy, sc.bowl_radius, rim) & ~self._disc(sc.bowl_xy, sc.bowl_radius - 0.006, rim)
        z = np.where(ring, t_rim, z)
        strip, t_strip = self._strip()
        z = np.where(strip, t_strip, z)
        block, hole, t_rim, t_floor = self._block()
        z = np.where(block & ~hole, t_rim, z)
        z = np.where(hole, t_floor, z)
        return np.nan_to_num(1000 * z, nan=0.0).astype(np.uint16)

    def _block(self):
        """The fork's block seen from the camera: (its top's pixels, the opening's, the depth along each ray to its top,
        and into the opening: to its floor, or to the wall the ray meets first)."""
        o, d = self._rays()
        sc = self.sc
        none = np.zeros(d.shape[:2], bool)
        if sc.hole_at is None:
            return none, none, np.zeros(d.shape[:2]), np.zeros(d.shape[:2])
        c = np.asarray(sc.hole_at, float)
        a = np.radians(sc.hole_yaw_deg)
        axes = (np.array([-np.sin(a), np.cos(a)]), np.array([np.cos(a), np.sin(a)]))     # short, long
        with np.errstate(divide="ignore", invalid="ignore"):
            t_rim = (c[2] - o[2]) / d[..., 2]
            t_in = (c[2] - sc.hole_depth - o[2]) / d[..., 2]
            p = (o + t_rim[..., None] * d)[..., :2] - c[:2]
            block = (np.abs(p @ axes[0]) <= sc.block_size[0] / 2) & (np.abs(p @ axes[1]) <= sc.block_size[1] / 2)
            hole = (np.abs(p @ axes[0]) <= sc.hole_size[0] / 2) & (np.abs(p @ axes[1]) <= sc.hole_size[1] / 2)
            for ax, half in zip(axes, np.asarray(sc.hole_size) / 2):    # a wall: where the ray leaves the opening
                u0, du = p @ ax, d[..., :2] @ ax
                t_in = np.minimum(t_in, np.where(du > 0, t_rim + (half - u0) / du,
                                                 np.where(du < 0, t_rim + (-half - u0) / du, np.inf)))
        return block & (t_rim > 0), hole & (t_rim > 0), t_rim, t_in

    def _strip(self):
        """The stand's strip seen from the camera (where the block above does not hide it): its pixels, the depth along
        each ray to its top."""
        o, d = self._rays()
        sc = self.sc
        if sc.hole_at is None:
            return np.zeros(d.shape[:2], bool), np.zeros(d.shape[:2])
        c, a = np.asarray(sc.hole_at, float), np.radians(sc.hole_yaw_deg)
        short, long = np.array([-np.sin(a), np.cos(a)]), np.array([np.cos(a), np.sin(a)])
        mid = c[:2] - short * (sc.block_size[0] / 2 + sc.strip_gap + sc.strip_size[1] / 2)   # (the far side, as the lab's)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c[2] - sc.strip_drop - o[2]) / d[..., 2]
            q = (o + t[..., None] * d)[..., :2] - mid
            on = (np.abs(q @ long) <= sc.strip_size[0] / 2) & (np.abs(q @ short) <= sc.strip_size[1] / 2) & (t > 0)
        return on & ~self._block()[0], t

    def project(self, p) -> np.ndarray:
        o, Rc = self.camera_pose()
        c = Rc.T @ (np.asarray(p, float) - o)
        return np.array(CAM.pixel(c))

    def _box(self, centre, r, z) -> list[float]:
        pts = [self.project([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), z]) for a in np.linspace(0, 2 * np.pi, 16)]
        pts = np.array(pts)
        return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]

    def detect(self, role, prompts, threshold):
        """SAM3's boxes and masks: the bowl's rim and the piece's top, projected."""
        sc = self.sc
        out = []
        if any(p in cfg.perception.bowl_prompts for p in prompts):          # its mask stops at the piece on it, as SAM3's
            piece = self._disc(sc.piece_xy, sc.piece_radius, sc.table_z + sc.piece_height)
            rim = sc.table_z + sc.rim_height
            out.append(Detection(cfg.perception.bowl_prompts[0], 0.9, self._box(sc.bowl_xy, sc.bowl_radius, rim),
                                 self._disc(sc.bowl_xy, sc.bowl_radius, rim) & ~piece))
        if cfg.food.prompt in prompts:
            top = sc.table_z + sc.piece_height
            out.append(Detection(cfg.food.prompt, 0.8, self._box(sc.piece_xy, sc.piece_radius, top),
                                 self._disc(sc.piece_xy, sc.piece_radius, top)))
        if any(p in cfg.utensil.get("pick.stand_prompts", []) for p in prompts):
            on = self._strip()[0]
            if on.any():
                v, u = np.nonzero(on)
                out.append(Detection(str(prompts[0]), 0.5, [float(u.min()), float(v.min()), float(u.max()),
                                                            float(v.max())], on))
        if any(p in cfg.utensil.get("pick.fork_prompts", []) for p in prompts):   # the fork in the fingers: rides with the
            x0, y0, x1, y1 = (int(v) for v in cfg.utensil.image_region_px)            # camera (fork_slip_px: it moved)
            du, dv = self.fork_slip_px
            m = np.zeros((480, 640), bool)
            m[max(0, y0 + dv):max(0, y1 + dv), max(0, x0 + du):max(0, x1 + du)] = True
            out.append(Detection(str(prompts[0]), 0.9, [x0 + du, y0 + dv, x1 + du, y1 + dv], m))
        if any(p in cfg.utensil.get("pick.prompts", []) for p in prompts):
            hole = self._block()[1]
            if hole.any():
                v, u = np.nonzero(hole)
                out.append(Detection(str(prompts[0]), 0.92, [float(u.min()), float(v.min()), float(u.max()),
                                                             float(v.max())], hole))
        return out

    def close(self):
        pass


class FakeFace:
    """The user's mouth where the scene says, seen through the camera at its depth (the lips' own)."""

    def __init__(self, robot):
        self.robot = robot

    def orient(self, R):
        pass

    def find(self, cam, wait_s):
        return self.mouth(None, None, cam) is not None

    def wait_open(self, cam, timeout_s, out):
        return cam.frame()

    def mouth(self, rgb, depth, cam):
        """None when the mouth is outside the image."""
        o, Rc = self.robot.camera_pose()
        c = Rc.T @ (np.asarray(self.robot.sc.mouth, float) - o)
        u, v = CAM.pixel(c) if c[2] > 0 else (-1.0, -1.0)
        if not (0 <= u < INTR.width and 0 <= v < INTR.height):
            return None
        return dict(cam_xyz=[float(v) for v in c], source="depth", open=0.4)
