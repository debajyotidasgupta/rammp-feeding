"""The robot, as the RAMMP stack serves it -- the only module that speaks ROS 2. One node, spun on its own thread:

  arm       kinova-gen3-ros2: /robot_description (its model), /joint_states (joints, torques), /control_status (the
            e-stop, arbitration), /execute_joint_trajectory (position or impedance)
  planner   RAMMP-CuRobo, shared: /rammp_curobo/plan_to_joints, set_world, its `world` parameter (it plans; we
            execute)
  cameras   rammp-suite-software's drivers: colour and depth images, their camera_info, their /tf_static
  detection rammp-perception: SAM3's /perception/sam3/detect (boxes and masks) on either camera
  tf        ours: the scene camera's measured pose, base_link -> its link frame (rammp calibrate-scene), static

Joint vectors are degrees (folded, joint_1..7) through joints(); joint_sample() and execute() carry the arm's raw
(unwrapped) radians, so a trajectory continues from the arm's own reading. tests/fake_robot.py implements the same
interface without ROS.
"""

from __future__ import annotations

import math
import os
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from rammp_feeding.camera import CameraModel, Intrinsics
from rammp_feeding.config import cfg
from rammp_feeding.geometry import fold, matrix_quat, quat_matrix

JOINTS = ["joint_%d" % i for i in range(1, 8)]


class RobotError(RuntimeError):
    """A node is missing, or answered with a failure."""


@dataclass
class Detection:
    label: str
    score: float
    box: list[float]                  # x1 y1 x2 y2, source-image pixels
    mask: np.ndarray                  # bool, the source image's size


def image_array(msg) -> np.ndarray:
    """sensor_msgs/Image -> numpy: 16-bit as uint16 (depth mm, or a label image), colour as RGB uint8."""
    enc = msg.encoding.lower()
    if "16" in enc:
        return np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width).copy()
    if enc == "32fc1":
        return (np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width) * 1000).astype(np.uint16)
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)
    return (a[..., ::-1] if enc.startswith("bgr") else a)[..., :3].copy()


def detections(msg) -> list[Detection]:
    """rammp_perception_interfaces/ObjectDetections -> the visible objects that carry a mask (SAM3 sends one for each)."""
    if not msg.label_image.height:
        return []
    labels = image_array(msg.label_image)
    out = []
    for o in msg.objects:
        mask = labels == o.mask_id
        if o.state == 0 and o.mask_id and mask.any():
            out.append(Detection(o.perception_class, float(o.score), [float(v) for v in o.box], mask))
    return out


def median_depth(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of depth frames (mm), ignoring no-returns (0)."""
    s = np.stack(frames)
    m = np.where(s > 0, s, np.nan).astype(np.float32)
    with warnings.catch_warnings():                 # (a pixel with no return in any frame: 0, as the sensor's own)
        warnings.simplefilter("ignore", RuntimeWarning)
        d = np.nanmedian(m, axis=0)
    return np.nan_to_num(d, nan=0.0).astype(np.uint16)


def _info_topic(image_topic: str) -> str:
    return image_topic.rsplit("/", 1)[0] + "/camera_info"


class Camera:
    """One camera: its latest colour frame, a few seconds of depth frames, and its model (camera_info + tf)."""

    def __init__(self, colour_topic: str, depth_topic: str):
        self.colour_topic, self.depth_topic = colour_topic, depth_topic
        self.colour: tuple[float, np.ndarray] | None = None
        self.depth: deque = deque(maxlen=150)
        self.info: dict = {}                       # "colour" / "depth" -> (Intrinsics, frame_id)
        self.model: CameraModel | None = None
        self.cv = threading.Condition()

    def on_colour(self, msg) -> None:
        with self.cv:
            self.colour = (time.time(), image_array(msg))
            self.cv.notify_all()

    def on_depth(self, msg) -> None:
        with self.cv:
            self.depth.append((time.time(), image_array(msg)))
            self.cv.notify_all()

    def frame(self, after: float = 0.0, wait_s: float = 3.0) -> tuple[float, np.ndarray]:
        """The newest colour frame that arrived after `after` (unix s): (arrival, RGB)."""
        t_end = time.time() + wait_s
        with self.cv:
            while self.colour is None or self.colour[0] <= after:
                if not self.cv.wait(max(0.0, t_end - time.time())):
                    raise RobotError("no colour frame on %s in %.0f s" % (self.colour_topic, wait_s))
            return self.colour

    def depth_median(self, frames: int, since: float, wait_s: float = 3.0, min_frames: int = 5) -> np.ndarray:
        """The median of the first `frames` depth frames that arrived after `since` (when the arm stopped): after
        wait_s, of what there is if at least min_frames."""
        t_end = time.time() + wait_s
        with self.cv:
            while True:
                got = [f for t, f in self.depth if t >= since]
                if len(got) >= frames:
                    return median_depth(got[:frames])
                if not self.cv.wait(max(0.0, t_end - time.time())):
                    if len(got) >= min_frames:
                        return median_depth(got)
                    raise RobotError("only %d depth frames on %s in %.0f s" % (len(got), self.depth_topic, wait_s))


class Trajectory:
    """An executing ExecuteJointTrajectory goal."""

    def __init__(self, handle):
        self.handle = handle
        self.result_future = handle.get_result_async()

    def done(self) -> bool:
        return self.result_future.done()

    def result(self) -> tuple[int, str]:
        r = self.result_future.result().result
        return int(r.error_code), str(r.error_string)

    def cancel(self, wait_s: float = 2.0) -> None:
        f = self.handle.cancel_goal_async()
        t_end = time.time() + wait_s
        while not (f.done() and self.done()) and time.time() < t_end:
            time.sleep(0.01)


class Robot:
    def __init__(self, name: str = "rammp_feeding"):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from rammp_arm_interfaces.action import ExecuteJointTrajectory
        from rammp_common_interfaces.msg import ControlStatus
        from rammp_curobo_interfaces.action import PlanToJoints
        from rammp_curobo_interfaces.srv import SetWorld
        from rcl_interfaces.srv import GetParameters
        from rammp_perception_interfaces.srv import Detect
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import String
        from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

        if not rclpy.ok():                 # no rclpy signal handlers: Ctrl-C must reach the move in flight (it cancels)
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = n = rclpy.create_node(name)
        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._lock = threading.Lock()
        self._joint: tuple[float, np.ndarray, np.ndarray] | None = None     # (time, raw q rad, effort N.m)
        self.estopped, self.arbitration, self.description = False, False, None
        self.token = list(bytes.fromhex(os.environ.get("RAMMP_ARM_TOKEN", "00" * 16)))   # from the orchestrator

        def status(m):
            self.estopped, self.arbitration = bool(m.estopped), bool(m.arbitration_enabled)
        n.create_subscription(JointState, "/joint_states", self._on_joints, qos_profile_sensor_data,
                              callback_group=MutuallyExclusiveCallbackGroup())          # (in order: one at a time)
        n.create_subscription(ControlStatus, "/control_status", status, latched, callback_group=cb)
        n.create_subscription(String, "/robot_description", lambda m: setattr(self, "description", m.data), latched,
                              callback_group=cb)
        self.tf = Buffer()
        self._tf_listener = TransformListener(self.tf, n, spin_thread=False)
        self._static_tf = StaticTransformBroadcaster(n)
        self.cameras: dict[str, Camera] = {}
        for role in ("camera", "scene_camera"):
            s = cfg.get(role + ".streams", None)
            if not s:
                continue
            cam = self.cameras[role] = Camera(str(s.colour_topic), str(s.depth_topic))
            n.create_subscription(Image, cam.colour_topic, cam.on_colour, qos_profile_sensor_data, callback_group=cb)
            n.create_subscription(Image, cam.depth_topic, cam.on_depth, qos_profile_sensor_data, callback_group=cb)
            for kind, topic in (("colour", cam.colour_topic), ("depth", cam.depth_topic)):
                n.create_subscription(CameraInfo, _info_topic(topic),
                                      lambda m, c=cam, k=kind: c.info.__setitem__(k, (Intrinsics.from_camera_info(m),
                                                                                      m.header.frame_id)),
                                      qos_profile_sensor_data, callback_group=cb)
        R = cfg.ros
        self._exec = ActionClient(n, ExecuteJointTrajectory, str(R.execute_action), callback_group=cb)
        self._plan = ActionClient(n, PlanToJoints, str(R.plan_action), callback_group=cb)
        self._world = n.create_client(SetWorld, str(R.set_world_service), callback_group=cb)
        self._params = n.create_client(GetParameters, str(R.planner_node) + "/get_parameters", callback_group=cb)
        self._detect = n.create_client(Detect, str(R.detect_service), callback_group=cb)
        self._types = dict(Execute=ExecuteJointTrajectory, Plan=PlanToJoints, SetWorld=SetWorld, Detect=Detect,
                           GetParameters=GetParameters)
        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(n)
        self._spinner = threading.Thread(target=self._executor.spin, daemon=True)
        self._spinner.start()

    # ------------------------------------------------------------------ the models the stack serves
    def robot_description(self, wait_s: float = 3.0) -> str | None:
        """The arm's URDF as its node publishes it (/robot_description, latched), or None."""
        t_end = time.time() + wait_s
        while self.description is None and time.time() < t_end:
            time.sleep(0.05)
        return self.description

    def transform(self, target: str, source: str, wait_s: float = 3.0) -> np.ndarray:
        """target <- source (4 x 4), from /tf and /tf_static."""
        from rclpy.duration import Duration
        from rclpy.time import Time
        try:
            t = self.tf.lookup_transform(target, source, Time(), timeout=Duration(seconds=wait_s))
        except Exception as e:  # noqa: BLE001
            raise RobotError("no transform %s <- %s (%s)" % (target, source, e))
        T = np.eye(4)
        T[:3, :3] = quat_matrix([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])
        T[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        return T

    def camera_model(self, role: str, wait_s: float = 5.0) -> CameraModel:
        """The camera as its driver describes it: both camera_info, and depth -> colour from its /tf_static."""
        cam = self.cameras[role]
        if cam.model is None:
            t_end = time.time() + wait_s
            while len(cam.info) < 2:
                if time.time() > t_end:
                    raise RobotError("no camera_info on %s / %s" % (_info_topic(cam.colour_topic), _info_topic(cam.depth_topic)))
                time.sleep(0.05)
            (ci, cf), (di, df) = cam.info["colour"], cam.info["depth"]
            cam.colour_frame, cam.depth_frame = cf, df
            cam.model = CameraModel(ci, di, np.eye(4) if cf == df else self.transform(cf, df, wait_s))
        return cam.model

    def publish_static(self, parent: str, child: str, T: np.ndarray) -> None:
        """A static transform parent -> child (ours: the scene camera's measured pose)."""
        from geometry_msgs.msg import TransformStamped
        m = TransformStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in T[:3, 3])
        q = matrix_quat(T[:3, :3])
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = q
        self._static_tf.sendTransform(m)

    # ------------------------------------------------------------------ state
    def _on_joints(self, m) -> None:
        idx = {name: i for i, name in enumerate(m.name)}
        if not all(j in idx for j in JOINTS):
            return
        q = np.array([m.position[idx[j]] for j in JOINTS])
        e = np.array([m.effort[idx[j]] if len(m.effort) > idx[j] else 0.0 for j in JOINTS])
        with self._lock:
            self._joint = (time.time(), q, e)

    def joint_sample(self, wait_s: float = 2.0) -> tuple[float, np.ndarray, np.ndarray]:
        """(time, raw joints rad as the driver reports them, joint torques N.m)."""
        t_end = time.time() + wait_s
        while True:
            with self._lock:
                s = self._joint
            if s is not None and time.time() - s[0] < 0.5:
                return s
            if time.time() > t_end:
                raise RobotError("no fresh /joint_states in %.0f s -- is the arm node running?" % wait_s)
            time.sleep(0.003)

    def joints(self) -> list[float]:
        return [float(v) for v in fold(np.degrees(self.joint_sample()[1]))]

    # ------------------------------------------------------------------ the arm
    def execute(self, positions_rad: np.ndarray, times_s: list[float], impedance: dict | None = None,
                path_tolerance_rad: float | None = None, goal_tolerance_rad: float | None = None) -> Trajectory:
        """Start a joint trajectory (raw radians, continuous from the arm's own reading; times from now)."""
        from builtin_interfaces.msg import Duration
        from control_msgs.msg import JointTolerance
        from trajectory_msgs.msg import JointTrajectoryPoint
        g = self._types["Execute"].Goal()
        g.trajectory.joint_names = JOINTS
        for p, t in zip(positions_rad, times_s):
            ns = int(round(t * 1e9))
            g.trajectory.points.append(JointTrajectoryPoint(positions=[float(v) for v in p],
                                                            time_from_start=Duration(sec=ns // 10**9, nanosec=ns % 10**9)))
        if path_tolerance_rad is not None:
            g.path_tolerance = [JointTolerance(name=j, position=float(path_tolerance_rad)) for j in JOINTS]
        if goal_tolerance_rad is not None:
            g.goal_tolerance = [JointTolerance(name=j, position=float(goal_tolerance_rad)) for j in JOINTS]
        g.control_mode = 1 if impedance else 0
        g.preemption = 0                             # queue: never preempt a move in flight
        if impedance:
            g.gains.kq = [float(v) for v in impedance["kq"]]
            g.gains.zeta = float(impedance["zeta"])
            g.gains.torque_limit = [float(v) for v in impedance["torque_limit"]]
        g.sender_id, g.token = self.node.get_name(), self.token
        if self.arbitration and not any(self.token):
            raise RobotError("the arm enforces arbitration and this module holds no token (RAMMP_ARM_TOKEN, from the "
                             "orchestrator's /acquire_control)")
        if not self._exec.wait_for_server(timeout_sec=3.0):
            raise RobotError("the arm's %s is not up -- is the arm node running?" % cfg.ros.execute_action)
        sent = self._exec.send_goal_async(g)
        try:
            handle = self._wait(sent, 5.0, "the arm did not answer the trajectory goal")
        except BaseException:                        # given up on (a timeout, Ctrl-C): never let it run unwatched
            t_end = time.time() + 2.0
            while not sent.done() and time.time() < t_end:
                time.sleep(0.01)
            if sent.done():
                if sent.result().accepted:
                    Trajectory(sent.result()).cancel()
            else:
                sent.add_done_callback(lambda f: f.result().accepted and f.result().cancel_goal_async())
            raise
        if not handle.accepted:
            raise RobotError("the arm rejected the trajectory goal (its log says why)")
        return Trajectory(handle)

    # ------------------------------------------------------------------ the planner
    def plan_joints(self, start_deg, goal_deg, timeout_s: float) -> tuple[np.ndarray, np.ndarray]:
        """A collision-free path (cuRobo): (positions N x 7 deg, unwrapped from start; times s)."""
        g = self._types["Plan"].Goal()
        g.start_joints = [math.radians(float(v)) for v in start_deg]
        g.target_joints = [math.radians(float(s) + float(fold(float(e) - float(s)))) for s, e in zip(start_deg, goal_deg)]
        if not self._plan.wait_for_server(timeout_sec=3.0):
            raise RobotError("the planner (%s) is not up" % cfg.ros.plan_action)
        handle = self._wait(self._plan.send_goal_async(g), 5.0, "the planner did not answer")
        if not handle.accepted:
            raise RobotError("the planner rejected the goal")
        r = self._wait(handle.get_result_async(), timeout_s, "the planner did not finish in %.0f s" % timeout_s).result
        if not r.success:
            raise RobotError("cuRobo found no collision-free path: %s" % r.message)
        pts = r.trajectory.points
        q = np.unwrap(np.array([p.positions for p in pts], float), axis=0)
        t = np.array([p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec for p in pts])
        return np.degrees(q), t

    def set_world(self, path: str) -> None:
        if not self._world.wait_for_service(timeout_sec=3.0):
            raise RobotError("the planner's %s is not up" % cfg.ros.set_world_service)
        r = self._wait(self._world.call_async(self._types["SetWorld"].Request(world=path)), 30.0, "set_world timed out")
        if not r.success:
            raise RobotError("the planner refused the world %s: %s" % (path, r.message))

    def planner_parameter(self, name: str) -> str | None:
        """One of the planner node's own string parameters, or None where it has none, it is empty, or the planner does
        not answer."""
        if not self._params.wait_for_service(timeout_sec=2.0):
            return None
        r = self._wait(self._params.call_async(self._types["GetParameters"].Request(names=[name])), 5.0, "no parameters")
        v = r.values[0] if r.values else None
        if v is None:
            return None
        return (v.string_value or None) if v.type == 4 else None     # (4: PARAMETER_STRING)

    # ------------------------------------------------------------------ perception
    def detect(self, role: str, prompts: list[str], threshold: float) -> list[Detection]:
        """SAM3 on the camera's next colour frame: every prompt's objects (label = the prompt), with masks."""
        if not self._detect.wait_for_service(timeout_sec=3.0):
            raise RobotError("SAM3 (%s) is not up" % cfg.ros.detect_service)
        Req = self._types["Detect"].Request
        r = self._wait(self._detect.call_async(Req(prompts=list(prompts), image_topic=self.cameras[role].colour_topic,
                                                   score_threshold=float(threshold))), 30.0, "SAM3 did not answer")
        if not r.success:
            raise RobotError("SAM3: %s" % r.message)
        return detections(r.detections)

    # ------------------------------------------------------------------
    def _wait(self, fut, timeout_s: float, why: str):
        t_end = time.time() + timeout_s
        while not fut.done():
            if time.time() > t_end:
                raise RobotError(why)
            time.sleep(0.005)
        return fut.result()

    def close(self) -> None:
        """Stop spinning before anything is destroyed (a thread left in the executor crashes the exit)."""
        import rclpy
        try:
            self._executor.shutdown(timeout_sec=2.0)
            self._spinner.join(timeout=2.0)
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        rclpy.try_shutdown()
