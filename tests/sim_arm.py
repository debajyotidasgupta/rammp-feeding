"""tests/fake_robot.py's arm and scene, with the real arm stack over ROS (tests/test_e2e.sh): the arm's model from its
description node, every trajectory sent to its driver (--sim), which must accept it -- its simulator does not move the
joints, so each accepted goal is cancelled and the motion is the fake's -- and every free-space path from the planner
node. The cameras, SAM3, the face and the bite's force are the fake scene's."""

from __future__ import annotations

import time

import numpy as np

from fake_robot import FakeRobot


class SimArmRobot(FakeRobot):
    def __init__(self, sc, ros, q0):
        self.ros, self.accepted, self.realtime = ros, 0, False
        super().__init__(sc, q0)

    def robot_description(self, wait_s: float = 3.0):
        return self.ros.robot_description(wait_s)

    def check(self):
        self.ros.check()

    def joint_sample(self, wait_s: float = 2.0):
        self.ros.check()                                # the action's cancel, as ros.Robot.joint_sample polls it
        if self.realtime and self.traj is not None and not self.traj.done():
            time.sleep(0.01)                            # (a move at its own pace: each sample is 10 ms of it)
        return super().joint_sample(wait_s)

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

    def plan_pose(self, start_deg, T, R, timeout_s, hold=1):
        return self.ros.plan_pose(start_deg, T, R, timeout_s, hold)

    def set_world(self, path):
        self.worlds.append(path)
        self.ros.set_world(path)

    def reset_world(self):
        self.worlds.append("reset")
        self.ros.reset_world()

    def planner_listens(self):
        return self.ros.planner_listens()

    def close(self):
        self.ros.close()
