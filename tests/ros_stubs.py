"""Stand-ins for the RAMMP nodes this module calls, for the ROS wiring test (tests/test_ros.sh): the planner's
plan_to_joints (a straight joint line), set_world (a file it can open, or a packaged world's name) and its `world`
parameter (world_real_bench.yaml); SAM3's detect (one box and its mask), the scene segmenter's topic, and the two
cameras as their drivers serve them (flat frames, camera_info, /tf_static between their frames). The worlds the
planner was set to go to /share/planner_worlds.json. The arm is the real kinova-gen3-ros2 in its --sim mode."""

from __future__ import annotations

import json
import os

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rammp_curobo_interfaces.action import PlanToJoints
from rammp_curobo_interfaces.srv import SetWorld
from geometry_msgs.msg import TransformStamped
from rammp_perception_interfaces.msg import DetectedObject, ObjectDetections
from rammp_perception_interfaces.srv import Detect
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster
from trajectory_msgs.msg import JointTrajectoryPoint


def main() -> None:
    rclpy.init()
    n = rclpy.create_node("rammp_stubs")

    def plan(goal):
        s, e = np.array(goal.request.start_joints), np.array(goal.request.target_joints)
        goal.succeed()
        r = PlanToJoints.Result(success=True, message="stub")
        for k in range(21):
            p = JointTrajectoryPoint(positions=list(s + (e - s) * k / 20))
            t = 2.0 * k / 20
            p.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            r.trajectory.points.append(p)
        return r

    worlds = []

    def world(req, resp):
        resp.success = os.path.exists(req.world) or req.world.endswith(".yaml") and "/" not in req.world
        resp.message = "" if resp.success else "no such file"
        if resp.success:
            worlds.append(req.world)
            open("/share/planner_worlds.json", "w").write(json.dumps(worlds))
        return resp

    def detect(req, resp):
        resp.success = True
        resp.detections.objects = [DetectedObject(perception_class=req.prompts[-1], score=0.9, box=[400.0, 200.0, 440.0, 240.0],
                                                  mask_id=1)]
        lab = np.zeros((480, 848), np.uint16)
        lab[200:241, 400:441] = 1
        resp.detections.label_image = Image(height=480, width=848, encoding="mono16", step=848 * 2, data=lab.tobytes())
        return resp

    ActionServer(n, PlanToJoints, "/rammp_curobo/plan_to_joints", plan)
    n.create_service(SetWorld, "/rammp_curobo/set_world", world)
    p = rclpy.create_node("rammp_curobo")              # the planner node's own parameters
    p.declare_parameter("world", "world_real_bench.yaml")
    nodes = [n, p]
    n.create_service(Detect, "/perception/sam3/detect", detect)
    pubs = []
    for ns, frame, kind, enc, shape in (("/wrist_camera/color", "wrist_camera_color_optical_frame", "image_raw", "rgb8", (480, 848, 3)),
                                        ("/wrist_camera/depth", "wrist_camera_depth_optical_frame", "image_rect_raw", "16UC1", (480, 848)),
                                        ("/scene_camera/color", "scene_camera_color_optical_frame", "image_raw", "rgb8", (360, 640, 3)),
                                        ("/scene_camera/depth", "scene_camera_depth_optical_frame", "image_raw", "16UC1", (240, 424))):   # (small: Python publishes slowly)
        a = np.full(shape, 400, np.uint16) if enc == "16UC1" else np.full(shape, 120, np.uint8)
        img = Image(height=shape[0], width=shape[1], encoding=enc, step=a.strides[0], data=a.tobytes())
        img.header.frame_id = frame
        h, w = shape[:2]
        info = CameraInfo(height=h, width=w, k=[0.8 * w, 0.0, w / 2, 0.0, 0.8 * w, h / 2, 0.0, 0.0, 1.0])
        info.header.frame_id = frame
        pubs += [(n.create_publisher(Image, ns + "/" + kind, qos_profile_sensor_data), img),
                 (n.create_publisher(CameraInfo, ns + "/camera_info", qos_profile_sensor_data), info)]
    seg = ObjectDetections(objects=[DetectedObject(perception_class="food container", score=0.7, box=[500.0, 300.0, 700.0, 450.0])])
    pubs.append((n.create_publisher(ObjectDetections, "/perception/object_detections", 10), seg))
    n.create_timer(1 / 15, lambda: [p.publish(m) for p, m in pubs])
    tf = StaticTransformBroadcaster(n)                 # what the camera drivers put on /tf_static
    msgs = []
    for parent, child, x in (("wrist_camera_color_optical_frame", "wrist_camera_depth_optical_frame", 0.0),
                             ("scene_camera_link", "scene_camera_color_optical_frame", 0.0),
                             ("scene_camera_color_optical_frame", "scene_camera_depth_optical_frame", 0.0235)):
        m = TransformStamped()
        m.header.frame_id, m.child_frame_id = parent, child
        m.transform.translation.x, m.transform.rotation.w = x, 1.0
        msgs.append(m)
    tf.sendTransform(msgs)
    ex = MultiThreadedExecutor()
    for x in nodes:
        ex.add_node(x)
    print("stubs up", flush=True)
    ex.spin()


if __name__ == "__main__":
    main()
