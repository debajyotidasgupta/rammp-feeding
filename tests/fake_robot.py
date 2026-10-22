"""rammp_feeding.ros.Robot without ROS, for the offline tests: a kinematic arm that plays the trajectories it is sent
(sampled at 100 Hz), a straight-line "planner", and cameras rendering the scene from where the arm actually holds them.

The scene: a table at table_z, a clear bowl (its depth reads the table through it) round bowl_xy, one food piece (a
cylinder) at piece_xy, the user's mouth at mouth. SAM3's boxes are the bowl's and the piece's projections.
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
from rammp_feeding.geometry import fold
from rammp_feeding.ros import Detection

# the site D405's camera_info (848x480, 2026-09-30): colour and depth one sensor
D405 = Intrinsics(434.884, 434.330, 423.960, 237.798, 848, 480)
CAM = CameraModel(D405, D405, np.eye(4))


@dataclass
class Scene:
    table_z: float = -0.024                         # the abra site (2026-09-30): the scene camera's reading
    bowl_xy: tuple = (0.290, 0.158)
    bowl_radius: float = 0.07
    piece_xy: tuple = (0.296, 0.152)
    piece_radius: float = 0.013
    piece_height: float = 0.025
    floor_lift: float = 0.0                         # the surface the depth reads under the bowl, above the table
    mouth: tuple = (-0.277, 0.306, 0.397)


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
        return t, np.full((D405.height, D405.width, 3), 120, np.uint8)

    def depth_median(self, frames, since, wait_s=3.0, min_frames=5):
        return self.robot.render_depth()

    def latest_depth(self):
        return time.time(), self.robot.render_depth()


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
        self.worlds: list = []                          # set_world: a file's content, or the scene's name
        self.world_param = None                         # the planner's `world` parameter (None: it has none)
        from rammp_feeding.arm import chain
        self.kin = chain()

    # --- the models the stack serves
    def robot_description(self, wait_s: float = 3.0):
        return None                                     # the bundled model (the driver's)

    def camera_model(self, role: str, wait_s: float = 5.0) -> CameraModel:
        return CAM

    def segmented_objects(self, max_age_s):
        return None

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

    def estop(self, reason):
        self.estopped = True

    # --- the planner: a straight joint line
    def plan_joints(self, start_deg, goal_deg, timeout_s):
        s = np.asarray(start_deg, float)
        g = s + fold(np.asarray(goal_deg, float) - s)
        n = 60
        return np.array([s + (g - s) * k / n for k in range(n + 1)]), np.linspace(0, 3, n + 1)

    def set_world(self, path):
        self.worlds.append(yaml.safe_load(open(path)) if os.path.isfile(path) else path)

    def planner_parameter(self, name):
        return self.world_param if name == "world" else None

    # --- the wrist camera: depth rendered from the arm's pose, SAM3's boxes projected
    def camera_pose(self):
        T, R = self.kin.fk(self.joints())
        return T + R @ scene.CAM_IN_TOOL, R @ scene.CAM_ROT          # (independently of scene.camera_pose)

    def _rays(self):
        o, Rc = self.camera_pose()
        vv, uu = np.mgrid[0:D405.height, 0:D405.width]
        return o, np.stack([(uu - D405.cx) / D405.fx, (vv - D405.cy) / D405.fy, np.ones_like(uu, float)], -1) @ Rc.T

    def _disc(self, centre, r, z):
        """The pixels whose ray meets the horizontal disc (centre, r) at height z."""
        o, d = self._rays()
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - o[2]) / d[..., 2]
        p = o + t[..., None] * d
        return (np.hypot(p[..., 0] - centre[0], p[..., 1] - centre[1]) < r) & (t > 0)

    def render_depth(self) -> np.ndarray:
        """mm along the optical axis: the piece's top where its disc is, else the table (a clear bowl reads it)."""
        o, d = self._rays()
        sc = self.sc
        floor = sc.table_z + sc.floor_lift
        with np.errstate(divide="ignore", invalid="ignore"):
            t_tab = (floor - o[2]) / d[..., 2]
            t_top = (floor + sc.piece_height - o[2]) / d[..., 2]
        z = np.where(t_tab > 0, t_tab, np.nan)
        z = np.where(self._disc(sc.piece_xy, sc.piece_radius, floor + sc.piece_height), t_top, z)
        return np.nan_to_num(1000 * z, nan=0.0).astype(np.uint16)

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
        if any(p in cfg.perception.bowl_prompts for p in prompts):
            out.append(Detection(cfg.perception.bowl_prompts[0], 0.9, self._box(sc.bowl_xy, sc.bowl_radius, sc.table_z + 0.06),
                                 self._disc(sc.bowl_xy, sc.bowl_radius, sc.table_z + 0.03)))
        if cfg.food.prompt in prompts:
            top = sc.table_z + sc.piece_height
            out.append(Detection(cfg.food.prompt, 0.8, self._box(sc.piece_xy, sc.piece_radius, top),
                                 self._disc(sc.piece_xy, sc.piece_radius, top)))
        return out

    def close(self):
        pass


class FakeFace:
    """The user's mouth where the scene says, seen through the camera at its depth (the lips' own)."""

    def __init__(self, robot):
        self.robot = robot

    def wait_open(self, cam, timeout_s, out):
        return cam.frame()

    def mouth(self, rgb, depth, cam):
        o, Rc = self.robot.camera_pose()
        c = Rc.T @ (np.asarray(self.robot.sc.mouth, float) - o)
        return dict(cam_xyz=[float(v) for v in c], source="depth", open=0.4)
