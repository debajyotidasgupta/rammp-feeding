"""One whole bite end to end on the real arm stack (tests/test_e2e.sh): the arm's model from kinova-gen3-ros2
2.0.0's description node; every trajectory of the bite sent to its driver (--sim), which must accept it (joint names
and order, tolerances, timing, gains) -- its simulator does not move the joints, so each accepted goal is cancelled and
the motion is tests/fake_robot.py's; every free-space path from RAMMP-CuRobo 2.0.1's node, our world pushed by file
(/feeding) and its own put back by /reset; the wrist camera, SAM3, the face and the bite's force rendered from the
fake scene. From feeding_start: detect, perceive, acquire (OVER, plunge, scoop, lift), deliver (swing, present, the
bite, retract), home -- each stage's result checked."""

from __future__ import annotations

import json
import os
import sys
import time

import fixture  # noqa: F401  (the test site: before rammp_feeding is imported)

os.environ["RAMMP_SET"] += (",planner.world_file=/share/planner_world.yaml,"
                            "planner.world_file_in_planner=/feeding/planner_world.yaml")

import numpy as np  # noqa: E402

from fake_robot import FakeFace, FakeRobot, Scene  # noqa: E402
from rammp_feeding import record, scene  # noqa: E402
from rammp_feeding.acquire import acquire  # noqa: E402
from rammp_feeding.arm import Arm  # noqa: E402
from rammp_feeding.cli import _target, detect, home, perceive  # noqa: E402
from rammp_feeding.config import cfg, state_dir  # noqa: E402
from rammp_feeding.deliver import deliver  # noqa: E402
from rammp_feeding.ros import Robot  # noqa: E402


class SimArmRobot(FakeRobot):
    """tests/fake_robot.py's arm and scene, with the real arm stack's model, driver (each goal accepted, then
    cancelled) and planner over ROS."""

    def __init__(self, sc: Scene, ros: Robot, q0):
        self.ros, self.accepted = ros, 0
        super().__init__(sc, q0)

    def robot_description(self, wait_s: float = 3.0):
        return self.ros.robot_description(wait_s)

    def execute(self, positions_rad, times_s, impedance=None, path_tolerance_rad=None, goal_tolerance_rad=None):
        sim = self.ros.joint_sample()[1]                # the driver's simulator, where it is (it never moves)
        tr = self.ros.execute(np.asarray(positions_rad) - positions_rad[0] + sim, times_s, impedance,
                              path_tolerance_rad, goal_tolerance_rad)     # RobotError if it rejects the goal
        tr.cancel()
        assert tr.result()[0] == -6, "the driver ended an accepted goal with %s, not PREEMPTED" % (tr.result(),)
        self.accepted += 1
        return super().execute(positions_rad, times_s, impedance, path_tolerance_rad, goal_tolerance_rad)

    def plan_joints(self, start_deg, goal_deg, timeout_s):
        return self.ros.plan_joints(start_deg, goal_deg, timeout_s)

    def set_world(self, path):
        self.worlds.append(path)
        self.ros.set_world(path)

    def reset_world(self):
        self.worlds.append("reset")
        self.ros.reset_world()

    def planner_listens(self):
        return self.ros.planner_listens()


def main() -> int:
    ros = Robot("rammp_e2e")
    t_end = time.time() + 600                           # the planner compiles its kernels on a first run
    while not ros._plan.wait_for_server(timeout_sec=5.0):
        assert time.time() < t_end, "the planner did not come up"
    sc = Scene()
    (state_dir() / "scene_state.json").write_text(json.dumps(dict(
        bowl_xy=list(sc.bowl_xy), table_z=sc.table_z, bowl_top_z=sc.table_z + 0.06, rim_radius_m=sc.bowl_radius,
        t=time.time())))
    robot = SimArmRobot(sc, ros, scene.captured("home"))
    arm = Arm(robot)
    assert arm.kin.base == "arm_mount", arm.kin.base
    assert robot.planner_listens(), "the planner does not listen on /reset"
    record.setup_logging()
    record.session("e2e")
    t0 = time.time()
    detect(robot, arm)                                  # planned, centred, our world pushed
    assert str(cfg.planner.world_file_in_planner) in robot.worlds, robot.worlds
    assert perceive(robot, arm) == 0
    acquire(robot, arm, _target(), record.stage("acquire"))   # OVER, the depth under the tip, plunge, scoop, lift
    T, R = arm.tool()
    assert abs(R[2, 2]) < 0.2 and R[:, 2][1] > 0.95, "the scoop did not end level, pointing at the user: %s" % R[:, 2]
    top = scene.overlook_goal(arm)[0][2]
    assert abs(T[2] - top) < 0.005, "the lift ended at tool z %.3f, not %.3f" % (T[2], top)
    food = json.loads((record.stage("acquire") / "food.json").read_text())
    assert (record.stage("acquire") / "plunge_end.jpg").exists()
    want = sc.table_z + float(cfg.bowl.floor_above_table_m) + float(cfg.utensil.punch_floor_margin_m)
    tip_at_plunge = arm.tip(food["joints_deg"])[2]
    assert abs(tip_at_plunge - want) < 0.003, "the plunge stopped at z %.4f, not %.4f" % (tip_at_plunge, want)
    face = FakeFace(robot)
    assert deliver(robot, arm, face, record.stage("deliver"), stop_after="present") == 0
    gap = float(np.linalg.norm(arm.tip() - (np.array(sc.mouth) + [0, 0, float(cfg.delivery.aim_dz_m)])))
    assert abs(gap - float(cfg.delivery.stop_m)) < 0.01, "the tip %.0f mm from the mouth" % (1000 * gap)
    assert robot.moves[-1] == "position", "the approach did not run in position control"
    J = arm.kin.jacobian(arm.q())[:3]                   # the bite: 3 N at the tool, a moment after the wait starts
    t_bite = time.time() + 1.5
    robot.effort_fn = lambda t: J.T @ np.array([0, 0, 3.0]) if t_bite < t < t_bite + 0.5 else np.zeros(7)
    assert deliver(robot, arm, face, record.stage("deliver"), start_at="bite") == 0, "no bite"
    robot.effort_fn = None
    assert max(abs(a - b) for a, b in zip(arm.q(), scene.overlook_joints(arm))) < 1.0, "not back at food_detect_start"
    home(robot, arm)
    assert max(abs(a - b) for a, b in zip(arm.q(), scene.captured("home"))) < 1.0, "not home"
    scene.restore_world(robot)                          # (as the command ends: the planner's own world back)
    assert robot.worlds[-1] == "reset"
    assert robot.accepted == len(robot.moves), (robot.accepted, len(robot.moves))
    print("E2E BITE THROUGH THE REAL ARM STACK (%d moves, each goal accepted by the driver; %d planned moves; %.0f s; "
          "the plunge at %.1f mm of its floor; the tip %.0f mm short of the mouth)" % (
              len(robot.moves), sum(1 for w in robot.worlds if w != "reset"), time.time() - t0,
              1000 * abs(tip_at_plunge - want), 1000 * gap))
    ros.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
