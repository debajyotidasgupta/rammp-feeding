"""The feeding engine end to end on the real arm stack (tests/test_e2e.sh): the node as sheppy runs it, standalone (no
orchestrator), driven through its own /feeding/execute_action as the orchestrator would, with the real 2.0 driver (each
goal accepted, then cancelled: tests/sim_arm.py), its description node and the 2.0.1 planner. The cameras, SAM3, the
face and the bite are tests/fake_robot.py's; the scene camera's reading is the fake scene's.

  start_meal                    the bowl found, the arm over it
  feed                          one bite taken
  feed, cancelled mid-plunge    (the moves at their own pace) ends as cancelled within a few seconds, the move stopped
  feed                          recovers (straight up out of the bowl), then a whole bite
  feed, cancelled at the mouth  (no bite) the food stays on the fork
  feed                          backs away to the delivery look and presents it again
  feed, cancelled at the mouth
  finish_meal                   recovers (back from the user), then home

Each action: its result, /feeding/status, the planner's own world back (/reset) as it ends, and none of its
subscriptions left once it has ended."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import fixture  # (the test site: before rammp_feeding is imported)

os.environ["RAMMP_SET"] += (",planner.world_file=/share/planner_world.yaml,"
                            "planner.world_file_in_planner=/feeding/planner_world.yaml")

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rammp_orchestrator_interfaces.action import ExecuteAction  # noqa: E402
from rammp_orchestrator_interfaces.msg import Result  # noqa: E402
from std_msgs.msg import Empty, String  # noqa: E402

from fake_robot import FakeFace, Scene  # noqa: E402
from sim_arm import SimArmRobot  # noqa: E402

SC = Scene()


def fake_locate(robot, out):
    """The scene camera's reading of the fake scene (tests/fake_robot.py has no scene camera)."""
    from rammp_feeding.config import state_dir
    st = dict(bowl_xy=list(SC.bowl_xy), table_z=SC.table_z, bowl_top_z=SC.table_z + 0.06, rim_radius_m=SC.bowl_radius,
              t=time.time())
    (state_dir() / "scene_state.json").write_text(json.dumps(st))
    return st


def main() -> int:
    rclpy.init(args=["--ros-args", "-p", "use_orchestrator:=false", "-p", "state_dir:=%s" % fixture.STATE,
                     "-p", "runs_dir:=%s" % (fixture.WORK / "runs"), "-p", "record:=true"])
    from rammp_feeding import deliver as deliver_mod, scene, scene_camera
    from rammp_feeding.engine import Feeding
    from rammp_feeding.ros import Robot
    scene_camera.locate = fake_locate
    bite_at = dict(on=False, started=0.0)              # the bite: 3 N at the tool, 1.5 s into the wait (when on)
    real_bite = deliver_mod.bite

    def bite(robot, arm, timeout_s, settle_s=None):
        bite_at["started"] = time.time()
        if bite_at["on"]:
            J, t0 = arm.kin.jacobian(arm.q())[:3], time.time() + 1.5
            robot.effort_fn = lambda t: J.T @ np.array([0, 0, 3.0]) if t0 < t < t0 + 0.5 else np.zeros(7)
        try:
            return real_bite(robot, arm, timeout_s, settle_s)
        finally:
            robot.effort_fn = None
    deliver_mod.bite = bite

    class TestFeeding(Feeding):
        sim = None

        def _robot(self, roles, check):
            ros = Robot(node=self, group=self._group, roles=(), check=check, static_tf=self._static.sendTransform,
                        reset=self._reset, token=self._token)
            if TestFeeding.sim is None:                 # one fake arm across actions: it stays where it stopped
                TestFeeding.sim = SimArmRobot(SC, ros, scene.captured("home"))
                self._face = FakeFace(TestFeeding.sim)
            TestFeeding.sim.ros = ros
            return TestFeeding.sim

    engine = TestFeeding()
    client = rclpy.create_node("e2e_orchestrator")
    status, resets = [], []
    client.create_subscription(String, "/feeding/status", lambda m: status.append(json.loads(m.data)), 10)
    client.create_subscription(Empty, "/reset", lambda m: resets.append(time.time()), 10)
    ex = MultiThreadedExecutor()
    ex.add_node(engine)
    ex.add_node(client)
    threading.Thread(target=ex.spin, daemon=True).start()
    act = ActionClient(client, ExecuteAction, "/feeding/execute_action")
    t_end = time.time() + 600                           # the planner compiles its kernels on a first run
    while not act.wait_for_server(timeout_sec=5.0):
        assert time.time() < t_end, "the engine's action server did not come up"
    time.sleep(1.0)
    idle_subs = client.count_subscribers("/joint_states")

    def run(action_id, cancel_when=None):
        """One action as the orchestrator sends it, cancelled once cancel_when(t0) says so: (Result, the /reset
        messages it sent, seconds from the cancel to its result)."""
        while engine._current is not None:
            time.sleep(0.1)
        n_reset = len(resets)
        sent = act.send_goal_async(ExecuteAction.Goal(engine_id="feeding", action_id=action_id, params_json="{}"))
        while not sent.done():
            time.sleep(0.05)
        h = sent.result()
        assert h.accepted, "%s rejected" % action_id
        t0, res, t_cancel = time.time(), h.get_result_async(), None
        if cancel_when is not None:
            while not cancel_when(t0):
                assert not res.done(), "%s ended before its cancel" % action_id
                time.sleep(0.05)
            assert client.count_subscribers("/joint_states") > idle_subs, "no subscription while the action runs"
            t_cancel = time.time()
            h.cancel_goal_async()
        while not res.done():
            time.sleep(0.05)
            assert time.time() - t0 < 120, "%s ran past the orchestrator's 120 s" % action_id
        t_done = time.time()
        r = res.result().result.result
        time.sleep(0.5)                                 # (the action's scope closes as it reports)
        assert client.count_subscribers("/joint_states") == idle_subs, "%s left a subscription behind" % action_id
        print("  %-12s status %d %-10s %s in %.1f s%s" % (action_id, r.status, r.error_code, r.payload_json, t_done - t0,
              "" if t_cancel is None else ", %.1f s after the cancel" % (t_done - t_cancel)), flush=True)
        return r, len(resets) - n_reset, None if t_cancel is None else t_done - t_cancel

    from rammp_common_interfaces.srv import AcquireControl, ReleaseControl
    token = list(range(1, 17))                          # the orchestrator hands its token over before an action
    acq = client.create_client(AcquireControl, "/feeding/acquire_control")
    rel = client.create_client(ReleaseControl, "/feeding/release_control")
    assert acq.wait_for_service(timeout_sec=10.0) and rel.wait_for_service(timeout_sec=10.0), "no ~/acquire_control"
    f = acq.call_async(AcquireControl.Request(owner_id="feeding", token=token))
    while not f.done():
        time.sleep(0.05)
    assert f.result().accepted and list(f.result().token) == token, f.result()
    S = Result
    r, n_reset, _ = run("start_meal")
    assert engine.sim.ros.token == token, "the action's arm goals do not carry the orchestrator's token"
    f = rel.call_async(ReleaseControl.Request(token=token))
    while not f.done():
        time.sleep(0.05)
    assert f.result().released and engine._token == [0] * 16, f.result()
    assert r.status == S.STATUS_SUCCESS, r
    assert n_reset == 1, "start_meal did not put the planner's own world back"
    bite_at["on"] = True
    r, n_reset, _ = run("feed")
    assert r.status == S.STATUS_SUCCESS and json.loads(r.payload_json) == {"bite_taken": True}, r
    assert n_reset == 1 and not engine._holding
    from rammp_feeding import record
    meal = record._dir["session"]                       # (the meal's record: finish_meal closes it)

    def moves_since(k: int) -> list[str]:
        """The labels of the moves logged since the k-th (the run record's events)."""
        lines = (meal / "events.jsonl").read_text().splitlines()
        return [e["label"] for e in map(json.loads, lines) if e["kind"] == "move"][k:]
    from rammp_feeding.arm import Arm
    from rammp_feeding.geometry import fold
    arm = Arm(engine.sim, kin=engine.sim.kin)          # (where the fake's tip is, read without stepping it)

    def in_the_bowl() -> bool:
        tip = arm.tip(list(fold(np.degrees(engine.sim.raw))))
        return tip[2] < scene.tip_floor(float(tip[0]), float(tip[1])) - 0.01
    engine.sim.realtime = True                          # the plunge takes its 8 s: cancelled in the middle of it, the
    n_moves = len(engine.sim.moves)                     # tip down in the bowl
    r, _, after_cancel = run("feed", cancel_when=lambda t0: len(engine.sim.moves) >= n_moves + 2 and in_the_bowl())
    engine.sim.realtime = False
    assert r.status == S.STATUS_CANCELLED, r
    assert after_cancel < 5.0, "the cancel took %.1f s" % after_cancel
    assert engine.sim.traj.cancelled, "no move was in flight at the cancel (or it was not stopped)"
    assert moves_since(0)[-1] == "plunge", moves_since(0)[-3:]
    low = arm.tip(list(fold(np.degrees(engine.sim.raw))))[2]
    assert not engine._holding
    k = len(moves_since(0))
    r, _, _ = run("feed")
    assert r.status == S.STATUS_SUCCESS and json.loads(r.payload_json)["bite_taken"], r
    assert moves_since(k)[:2] == ["out of the bowl", "to food_detect_start"], moves_since(k)[:3]
    print("    (the cancelled plunge left the tip at z %.3f, in the bowl; the next feed rose out of it first)" % low,
          flush=True)
    bite_at["on"] = False                               # no bite this time: cancelled while it waits at the mouth
    at_mouth = lambda t0: bite_at["started"] > t0 and time.time() > bite_at["started"] + 2  # noqa: E731
    r, _, after_cancel = run("feed", cancel_when=at_mouth)
    assert r.status == S.STATUS_CANCELLED, r
    assert after_cancel < 5.0, "the cancel took %.1f s" % after_cancel
    assert engine._holding, "the food left on the fork was forgotten"
    k = len(moves_since(0))
    r, _, _ = run("feed", cancel_when=at_mouth)          # the same food, presented again
    assert r.status == S.STATUS_CANCELLED, r
    assert moves_since(k)[0] == "back to the delivery look" and "plunge" not in moves_since(k), moves_since(k)
    k = len(moves_since(0))
    r, n_reset, _ = run("finish_meal")
    assert r.status == S.STATUS_SUCCESS, r
    assert n_reset == 1
    assert moves_since(k)[:2] == ["back to the delivery look", "back to food_detect_start"], moves_since(k)
    assert max(abs(a - b) for a, b in zip(engine.sim.joints(), scene.captured("home"))) < 1.0, "not home"
    assert {s["state"] for s in status} >= {"idle", "executing"} and status[-1]["state"] == "idle", status[-3:]
    assert all(s["mode"] == "standalone" and not s["mock"] for s in status)
    assert engine.sim.accepted == len(engine.sim.moves), (engine.sim.accepted, len(engine.sim.moves))
    print("E2E ENGINE THROUGH THE REAL ARM STACK (%d moves, each goal accepted by the driver; %d status messages; "
          "every action's subscriptions gone as it ended)" % (len(engine.sim.moves), len(status)), flush=True)
    ex.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
