"""The fork picked up from its stand, one whole bite, and the fork hung back, offline, through the module's own code
against tests/fake_robot.py (the level test site: its stand beside the bowl): rammp pick-fork, a bite (to
food_detect_start, perceive, acquire, the presentation, retract, home: tests/test_bite.py's), the stand nudged,
rammp put-fork (the stand found again by its strip).

    python3 tests/test_fork.py [--fakes] [--top]
        --fakes: the stand, the food and the mouth the module's own stand-ins (rammp_feeding/fakes.py), where the fake
        robot has them
        --top: a holder the fork leaves straight up (utensil.pick.holder top): put back where it was picked from, the
        stand not nudged

What it checks: the order of the gripper's commands and where the fingers are at each, the moves' refusals and
geometry, the fork's state across the three. What it cannot: the opening's fit, contact (the block is rendered).
"""

from __future__ import annotations

import json
import sys
import time

import fixture  # noqa: F401  (the test site: before rammp_feeding is imported)
import numpy as np  # noqa: E402

from fake_robot import FakeFace, FakeRobot, Scene  # noqa: E402
from rammp_feeding import fakes, fork, record, scene, scene_camera, sensing  # noqa: E402
from rammp_feeding.acquire import acquire  # noqa: E402
from rammp_feeding.arm import Arm  # noqa: E402
from rammp_feeding.cli import _target, detect, home, perceive  # noqa: E402
from rammp_feeding.config import cfg, state_dir  # noqa: E402
from rammp_feeding.deliver import deliver  # noqa: E402


def main() -> int:
    if not fork.used():
        print("NO STAND ON THIS TEST SITE (utensil.pick.use false)")
        return 1
    sc = Scene()
    robot = FakeRobot(sc, scene.captured("home"))
    fork.save_state(dict(holding=False))
    faked, top = "--fakes" in sys.argv, "--top" in sys.argv
    if top:
        cfg["utensil"]["pick"].update(holder="top", open_side=None)
    if faked:                                        # the module's stand-ins where the fake robot has the things
        cfg["fakes"].update(holder={"centre_m": list(sc.hole_at), "long_deg": sc.hole_yaw_deg},
                            food={"pieces_m": [list(sc.piece_xy)]}, mouth={"at_m": list(sc.mouth)}, open_after_s=0.0)
        fakes.refill()
    arm = Arm(robot)
    record.setup_logging()
    record.session("test_fork")
    t0 = time.time()
    P = cfg.utensil.pick
    g = fork.Gripper(robot.gripper, robot.gripper_position)
    out = fork.pick_up(robot, arm, g, out=fork.record_dir("pick_fork"))
    in_hole = np.asarray(sc.hole_at) - [0.0, 0.0, float(P.insert_m)]
    assert [p for p, _ in robot.grips] == [float(P.gripper_closed), float(P.gripper_open), float(P.gripper_open)], \
        robot.grips                                  # (in closed, open in the opening, open again firm once out)
    off = float(np.linalg.norm(robot.grips[1][1] - in_hole))
    assert off < 0.004, "the fingers opened %.1f mm from the hole's depth" % (1000 * off)
    assert fork.holding() and arm.held.fork
    picked = len(robot.moves)
    arm = Arm(robot)                                 # (the meal's arm: the fork's usual model)
    face = sensing.now().face(arm) if faked else FakeFace(robot)

    if scene_camera.in_use():                        # the scene camera's reading: the fake scene's
        (state_dir() / "scene_state.json").write_text(json.dumps(dict(
            bowl_xy=list(sc.bowl_xy), table_z=sc.table_z, bowl_top_z=sc.table_z + sc.rim_height,
            rim_radius_m=sc.bowl_radius, t=time.time())))
    else:
        scene_camera.locate(robot, None)
    detect(robot, arm)                               # the bite, as rammp feed runs it (the bite's force aside)
    assert perceive(robot, arm) == 0, "perceive"
    acquire(robot, arm, _target(), record.stage("acquire"))
    assert deliver(robot, arm, face, record.stage("deliver"), stop_after="present") == 0, "present"
    assert deliver(robot, arm, face, record.stage("deliver"), start_at="retract") == 0, "retract"
    home(robot, arm)
    assert g_at(arm, scene.captured("home")), "the bite did not end home"
    fed = len(robot.moves) - picked

    if not top:
        sc.move_stand(0.012, 0.009, -6.0)            # (bumped during the meal: the put-back finds it again)
    if faked:
        cfg["fakes"]["holder"] = {"centre_m": list(sc.hole_at), "long_deg": sc.hole_yaw_deg}
    in_hole = np.asarray(sc.hole_at) - [0.0, 0.0, float(P.insert_m)]
    robot.grips.clear()
    fork.put_down(robot, arm, g)
    assert [p for p, _ in robot.grips] == [float(P.gripper_open), float(P.gripper_closed)], robot.grips
    off = float(np.linalg.norm(robot.grips[-1][1] - in_hole))
    assert off < 0.004, "the fingers let go %.1f mm from the hole's depth" % (1000 * off)
    assert not fork.holding() and "hole" not in fork.state()
    assert any(o["name"] == "fork_stand" for o in scene.world()["obstacles"]), "the stand not kept in the world"
    assert g_at(arm, scene.captured("home")), "the put-back did not end home"
    print("THE FORK GOES THROUGH (picked up at %s, %d moves; a bite, %d moves; hung back, %d moves; %.1f s%s%s)"
          % (out["opening_m"], picked, fed, len(robot.moves) - picked - fed, time.time() - t0,
             "; fakes" if faked else "", "; a top holder" if top else ""))
    return 0


def g_at(arm, q) -> bool:
    from rammp_feeding.geometry import max_step
    return max_step(arm.q(), q) < 0.5


if __name__ == "__main__":
    sys.exit(main())
