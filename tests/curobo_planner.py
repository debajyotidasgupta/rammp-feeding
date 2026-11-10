"""The real planner in place of fake_robot's straight line: RAMMP-CuRobo 2.0's core (rammp_curobo), on a CUDA box, as
its ROS node would run it for every client -- plan_to_joints and set_world exactly as the module calls them over ROS."""

from __future__ import annotations

import numpy as np

from rammp_feeding.ros import RobotError


class CuroboPlanner:
    def __init__(self, config: str, urdf_path: str):
        """config: a robot's curobo/planner.yaml (rammp-deployments config/robots/<robot>/curobo/); urdf_path: the arm's
        model as a file (as /arm/robot_description serves it)."""
        import tempfile
        from rammp_curobo import CuRoboPlanner
        from rammp_curobo.config import load_planner_config
        from rammp_curobo.contract import check_robot_dir
        from rammp_curobo.urdf_model import write_urdf
        cfg = load_planner_config(config)
        xml = check_robot_dir(*cfg, open(urdf_path).read())        # as the planner node prepares the served model
        self._dir = tempfile.TemporaryDirectory(prefix="rammp_curobo_")
        self.p = CuRoboPlanner(*cfg, urdf_path=write_urdf(xml, self._dir.name))
        self.calls = []

    def plan_joints(self, start_deg, goal_deg, timeout_s):
        s = np.asarray(start_deg, float)
        g = s + ((np.asarray(goal_deg, float) - s + 180.0) % 360.0 - 180.0)
        r = self.p.plan_to_joints(list(np.radians(g)), start=list(np.radians(s)))
        self.calls.append(dict(start=list(s), goal=list(g), success=bool(r.success), status=str(r.status)))
        if not r.success:
            raise RobotError("cuRobo found no collision-free path: %s %s" % (r.status, r.error))
        q = np.degrees(np.unwrap(np.vstack([np.radians(s), r.joint_traj.positions]), axis=0))
        return q, np.concatenate([[0.0], r.joint_traj.times()])

    def set_world(self, path):
        self.p.update_world(path)
