"""The module's ROS interface against the real arm driver (kinova-gen3-ros2 --sim) and the stand-ins (tests/ros_stubs.py):
joint states, a trajectory that runs to the end, one cancelled mid-way, the planner, set_world (our world by file, then
the planner's own back, read from its `world` parameter), SAM3 (its masks), the cameras, our static transform.
Run inside the module's image (tests/test_ros.sh)."""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("RAMMP_NO_LOCAL", "1")
os.environ.setdefault("RAMMP_STATE_DIR", "/tmp/rammp_state")
os.environ.setdefault("RAMMP_SET", "deployment.runs_dir=/tmp/rammp_runs,planner.world_file=/tmp/rammp_state/world.yaml,"
                                   "planner.world_file_in_planner=")

import numpy as np  # noqa: E402

from rammp_feeding.ros import Robot  # noqa: E402


def main() -> int:
    r = Robot("rammp_wiring")
    t, raw, tau = r.joint_sample(wait_s=10)
    print("joint states: q", np.round(np.degrees(raw), 2), "tau", np.round(tau, 2))
    q = r.joints()
    # a tiny move that runs to the end (the sim arm echoes its command, so keep it inside the goal tolerance)
    pos = np.array([raw, raw + np.radians([0, 0, 0, 0, 0, 0, 0.3])])
    tr = r.execute(pos, [0.0, 1.0], path_tolerance_rad=np.radians(20), goal_tolerance_rad=np.radians(1.0))
    while not tr.done():
        time.sleep(0.05)
    print("trajectory:", tr.result())
    assert tr.result()[0] == 0, tr.result()
    # a long one, cancelled after a second: settles PREEMPTED
    pos = np.array([raw, raw + np.radians([0, 0, 0, 0, 0, 0, 5])])
    tr = r.execute(pos, [0.0, 20.0], impedance=dict(kq=[80] * 4 + [30] * 3, zeta=0.7, torque_limit=[39] * 4 + [9] * 3))
    time.sleep(1.0)
    tr.cancel()
    print("cancelled:", tr.done(), tr.result())
    assert tr.done() and tr.result()[0] == -6, tr.result()
    pos, times = r.plan_joints(q, np.asarray(q) + 5, 10.0)
    print("planner: %d points, %.1f s" % (len(pos), times[-1]))
    assert np.allclose(pos[-1], np.asarray(q) + 5, atol=1e-6)
    open("/share/world.yaml", "w").write("base_frame: base_link\nobstacles: []\n")
    r.set_world("/share/world.yaml")      # the planner reads it: a shared mount (/config on the robot)
    r.set_world(r.planner_parameter("world"))
    got = json.load(open("/share/planner_worlds.json"))
    print("set_world:", got)
    assert got == ["/share/world.yaml", "world_real_bench.yaml"]
    d = r.detect("camera", ["bowl", "piece of food"], 0.5)
    print("sam3:", [(x.label, x.box, int(x.mask.sum())) for x in d])
    assert d and d[0].label == "piece of food" and d[0].mask.sum() == 41 * 41
    for role, cam in r.cameras.items():
        t, rgb = cam.frame(wait_s=5)
        D = cam.depth_median(3, time.time() - 1, wait_s=5, min_frames=1)   # (the stand-ins publish slowly: Python)
        m = r.camera_model(role)
        print("%s: colour %s, depth %s median %d mm; camera_info fx %.1f; depth -> colour %s" % (
            role, rgb.shape, D.shape, int(np.median(D)), m.colour.fx, np.round(m.d2c[:3, 3], 4)))
    assert abs(r.camera_model("scene_camera").d2c[0, 3] - 0.0235) < 1e-6
    r.publish_static("base_link", "scene_camera_link", np.eye(4))
    T = r.transform("base_link", "scene_camera_depth_optical_frame")
    print("tf base_link <- the scene camera's depth frame, through our static transform:", np.round(T[:3, 3], 4))
    print("ROS WIRING OK")
    r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
