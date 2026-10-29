"""The whole bite offline, through the module's own code, against tests/fake_robot.py: from feeding_start to
food_detect_start, perceive, acquire (OVER, plunge, then the scoop on its own from the plunge's record), deliver
(swing, present, bite, retract), home.

    python3 tests/test_bite.py [--user-offset dx,dy,dz] [--piece dx,dy] [--expect 8] [--planner curobo]
        curobo: RAMMP-CuRobo's real planner (a CUDA box) as the deployment shares it: its default config, our world
        pushed before each planned move

What it checks: every stage's refusals and the geometry (reachability, joint limits, continuity, our tip floor, the
plunge's stop above the bowl's own floor, where the tip ends at the mouth). What it cannot: contact, real food,
images (SAM3 and the face are stand-ins; the depth is rendered from the arm's pose).
"""

from __future__ import annotations

import json
import sys
import time

import fixture  # noqa: F401  (the test site: before rammp_feeding is imported)
import numpy as np  # noqa: E402

from fake_robot import FakeFace, FakeRobot, Scene  # noqa: E402
from rammp_feeding import record, scene  # noqa: E402
from rammp_feeding.acquire import acquire  # noqa: E402
from rammp_feeding.arm import Arm  # noqa: E402
from rammp_feeding.cli import _target, detect, home, perceive  # noqa: E402
from rammp_feeding.config import state_dir  # noqa: E402
from rammp_feeding.deliver import deliver  # noqa: E402


def arg(name, default):
    return [float(v) for v in sys.argv[sys.argv.index(name) + 1].split(",")] if name in sys.argv else default


def main() -> int:
    sc = Scene()
    sc.mouth = tuple(np.array(sc.mouth) + arg("--user-offset", [0, 0, 0]))
    sc.piece_xy = tuple(np.array(sc.piece_xy) + arg("--piece", [0, 0]))
    (state_dir() / "scene_state.json").write_text(json.dumps(dict(
        bowl_xy=list(sc.bowl_xy), table_z=sc.table_z, bowl_top_z=sc.table_z + 0.06, rim_radius_m=sc.bowl_radius, t=time.time())))
    robot = FakeRobot(sc, scene.captured("home"))
    if "--planner" in sys.argv and sys.argv[sys.argv.index("--planner") + 1] == "curobo":
        from curobo_planner import CuroboPlanner
        real, fake_world = CuroboPlanner(), robot.set_world
        robot.plan_joints = real.plan_joints
        robot.set_world = lambda path: (fake_world(path), real.set_world(path))   # recorded, and to the planner
    arm = Arm(robot)
    record.setup_logging()
    record.session("test")
    t0 = time.time()
    detect(robot, arm)
    assert robot.worlds and robot.worlds[-1]["obstacles"], "no planner world pushed"
    rc = perceive(robot, arm)
    assert rc == 0, "perceive: %d" % rc
    tgt = _target()
    acquire(robot, arm, tgt, record.stage("acquire"), stop_after="over")
    off = float(np.linalg.norm(arm.tip()[:2] - np.array(sc.piece_xy)))
    assert off < 0.003, "OVER put the tip %.1f mm from the piece" % (1000 * off)
    acquire(robot, arm, tgt, record.stage("acquire"), start_at="plunge", stop_after="plunge")
    want = sc.table_z + float(scene.cfg.bowl.floor_above_table_m) + float(scene.cfg.utensil.punch_floor_margin_m)
    assert abs(arm.tip()[2] - want) < 0.002, "the plunge stopped at z %.4f, not %.4f" % (arm.tip()[2], want)
    acquire(robot, arm, tgt, record.stage("acquire"), start_at="scoop")
    T, R = arm.tool()
    assert abs(R[2, 2]) < 0.2 and R[:, 2][1] > 0.95, "the scoop did not end level, pointing at the user: axis %s" % R[:, 2]
    rc = deliver(robot, arm, FakeFace(robot), record.stage("deliver"), stop_after="present")
    expect = int(arg("--expect", [0])[0])
    if expect:                                       # refused (out of reach): retracted to food_detect_start
        assert rc == expect, "present: %d, expected %d" % (rc, expect)
        assert max(abs(a - b) for a, b in zip(arm.q(), scene.overlook_joints(arm))) < 1.0, "not back at food_detect_start"
        print("REFUSED AS EXPECTED (exit %d), retracted" % rc)
        return 0
    assert rc == 0, "present: %d" % rc
    tip, mouth = arm.tip(), np.array(sc.mouth)
    stop = float(scene.cfg.delivery.stop_m)
    gap = float(np.linalg.norm(tip - (mouth + np.array([0, 0, float(scene.cfg.delivery.aim_dz_m)]))))
    assert abs(gap - stop) < 0.01, "the tip %.0f mm from the mouth, not %.0f" % (1000 * gap, 1000 * stop)
    assert robot.moves[-1] == "impedance", "the approach did not run compliant"
    rc = deliver(robot, arm, FakeFace(robot), record.stage("deliver"), start_at="retract")
    assert rc == 0
    home(robot, arm)
    print("THE BITE GOES THROUGH (%d moves, %.1f s; OVER %.1f mm off the piece; the tip %.0f mm short of the mouth)"
          % (len(robot.moves), time.time() - t0, 1000 * off, 1000 * gap))
    return 0


if __name__ == "__main__":
    sys.exit(main())
