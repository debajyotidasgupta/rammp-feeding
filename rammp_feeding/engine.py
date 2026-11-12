"""rammp_feeding as a RAMMP capability engine (rammp-kitchen-capability-engines' CapabilityEngine): registered with the
orchestrator, it runs an action only when the orchestrator starts one -- nothing moves otherwise.

  start_meal   the scene camera finds the bowl; the arm goes to look at it and centres over it (food_detect_start)
  feed         one bite: SAM3 finds the pieces, the fork plunges and scoops one, swings to the user, waits for an open
               mouth, stops 80 mm short, feels the bite, and goes back over the bowl. A piece still on the fork (the
               user did not take it) is presented again by the next feed, not scooped anew; an empty bowl ends it
  finish_meal  the arm back to feeding_start

Each action builds its own ros.Robot on this node: its subscriptions, clients and publishers live only while it runs.
Our collision world goes into the shared planner while the action plans and the planner's own comes back as it ends. A
cancel stops the move in flight, waits for the arm to settle, and ends the action as cancelled. The waits at the mouth
are bounded (node.*) so that a bite ends within the orchestrator's 120 s.

Its ROS parameters (docs/interface.md) select the site's profiles and state; the node loads config/ with them before
importing anything that reads it.

    ros2 launch rammp_feeding feeding.launch.py                          # with the orchestrator
    ros2 launch rammp_feeding feeding.launch.py use_orchestrator:=false  # standalone
"""

from __future__ import annotations

import logging

from capability_engine import Action, Cancelled, CapabilityEngine, EngineError, run
from rammp_orchestrator_interfaces.msg import ActionAvailability

from rammp_feeding import config
from rammp_feeding.config import PARAMS, apply_parameters

log = logging.getLogger("rammp")


class Feeding(CapabilityEngine):
    display_name = "Eating"
    perception_classes = ["food container", "food"]
    required_resources = ["arm"]                             # the arm only: nothing here commands the gripper
    actions = [
        Action("start_meal", "Find the bowl", sticky_followup=True),
        Action("feed", "Feed a bite", sticky_followup=True),
        Action("finish_meal", "Done eating"),
    ]

    def __init__(self, engine_id: str = "feeding"):
        super().__init__(engine_id)
        values = {name: self.declare_parameter(name, default).value for name, (default, _) in PARAMS.items()}
        apply_parameters(values)
        from rammp_feeding import record
        record.setup_logging()
        from std_msgs.msg import Empty
        from tf2_ros import StaticTransformBroadcaster
        self._static = StaticTransformBroadcaster(self)      # (publishers for the node's life: what it publishes)
        self._reset = self.create_publisher(Empty, str(config.cfg.ros.planner_reset), 1)
        from rammp_common_interfaces.srv import AcquireControl, ReleaseControl
        self._token = [0] * 16                               # the orchestrator's execution token: on every arm goal
        self.create_service(AcquireControl, "~/acquire_control", self._on_acquire_control, callback_group=self._group)
        self.create_service(ReleaseControl, "~/release_control", self._on_release_control, callback_group=self._group)
        self._meal = False                                   # start_meal has found the bowl, this selection
        self._holding = False                                # a piece on the fork, not yet taken
        self._face = None                                    # MediaPipe, loaded by the first feed and kept
        if not self.mock:
            self._publish_scene_pose()

    # ------------------------------------------------------------------ the engine contract
    def availability(self, instance_id, world_state):
        """Fast, no side effects: feed only once the bowl was found this selection."""
        return [ActionAvailability(action_id="start_meal", available=True),
                ActionAvailability(action_id="feed", available=self._meal,
                                   reason="" if self._meal else "find the bowl first (start_meal)"),
                ActionAvailability(action_id="finish_meal", available=True)]

    def hibernate(self):
        self._meal = False

    def _on_acquire_control(self, request, response):
        """The orchestrator hands the execution token over before it starts an action: it goes on every arm goal of
        that action, with this engine's id as the sender."""
        self._token = [int(b) for b in request.token]
        response.accepted, response.token, response.message = True, list(self._token), "ok"
        return response

    def _on_release_control(self, request, response):
        """...and takes it back once the action has ended."""
        if [int(b) for b in request.token] == self._token:
            self._token = [0] * 16
            response.released, response.message = True, "released"
        else:
            response.released, response.message = False, "not the token this engine holds"
        return response

    def execute(self, action_id, params, ctx):
        if self.mock:                                        # (no hardware: the actions' results, nothing moves)
            return self._mock(action_id, ctx)
        from rammp_feeding import record, scene
        from rammp_feeding.arm import Arm, MoveError
        from rammp_feeding.ros import RobotError

        def check():                                         # polled by every wait and every move in flight
            if ctx.cancelled():
                raise Cancelled()
        stage = ["starting"]

        def step(name, pct, message):
            stage[0] = name
            ctx.checkpoint()                                 # pause and cancel between steps
            ctx.progress(name, pct, message, pausable=False)
            log.info("  [%s] %s", action_id, message)
        roles = ("camera", "scene_camera") if action_id == "start_meal" else ("camera",)
        robot = None
        try:
            robot = self._robot(roles, check)
            step("setup", 0.0, "Getting the arm ready")
            arm = Arm(robot)
            if action_id == "start_meal":
                return self._start_meal(robot, arm, step)
            if action_id == "feed":
                return self._feed(robot, arm, step)
            step("home", 0.1, "Moving the arm back to its start")
            from rammp_feeding.cli import home
            self._recover(robot, arm)
            home(robot, arm)
            self._meal = False
            record.end_session()
            return {"home": True}
        except Cancelled:
            raise
        except (MoveError, RobotError, RuntimeError, OSError, ValueError) as e:
            code, recoverable = {"locate": ("bowl_not_found", True), "detect": ("bowl_not_found", True),
                                 "perceive": ("perception_failed", True), "acquire": ("pickup_failed", False),
                                 "deliver": ("delivery_failed", False)}.get(stage[0], ("move_failed", False))
            if isinstance(e, FileNotFoundError) or "not measured" in str(e) or "not calibrated" in str(e):
                code, recoverable = "not_set_up", False
            elif isinstance(e, RobotError):
                code, recoverable = "robot_unavailable", True
            raise EngineError(code, str(e), recoverable=recoverable)
        finally:
            if robot is not None:
                if getattr(robot, "world_pushed", False):     # the shared planner's own world back, for its other users
                    try:
                        scene.restore_world(robot)
                    except Exception as e:  # noqa: BLE001
                        log.error("  the planner's own world was not restored (%s): it still holds ours", e)
                robot.close()

    # ------------------------------------------------------------------ the actions
    def _start_meal(self, robot, arm, step) -> dict:
        from rammp_feeding import record
        from rammp_feeding.cli import detect
        from rammp_feeding.scene_camera import locate
        record.end_session()
        step("locate", 0.1, "Finding the bowl")
        st = locate(robot, record.session("meal"))
        step("detect", 0.4, "Looking at the bowl")
        detect(robot, arm)
        self._meal = True
        return {"bowl_xy_m": [float(v) for v in st["bowl_xy"]]}

    def _feed(self, robot, arm, step) -> dict:
        from rammp_feeding import scene
        from rammp_feeding.acquire import acquire
        from rammp_feeding.cli import perceive_target
        from rammp_feeding.config import cfg
        from rammp_feeding.deliver import deliver
        from rammp_feeding import record
        from rammp_feeding.geometry import max_step
        N = cfg.node
        if self._holding:                                   # the food still on the fork: present it again
            self._recover(robot, arm, to_bowl=False)
        else:
            if max_step(arm.q(), scene.overlook_joints(arm)) > float(cfg.motion.near_pose_deg):
                step("detect", 0.05, "Moving over the bowl")
                self._recover(robot, arm)
                arm.travel(scene.overlook_joints(arm), "to food_detect_start")
            step("perceive", 0.1, "Finding the food")
            rc, target = perceive_target(robot, arm)
            if rc == 3:
                return {"bite_taken": False, "bowl_empty": True}
            if rc != 0:
                raise EngineError("no_pickable_piece", "food is left in the bowl but no piece can be picked up")
            step("acquire", 0.25, "Picking up the food")
            acquire(robot, arm, target, record.stage("acquire"))
            self._holding = True
        if self._face is None:
            from rammp_feeding.face import Face
            self._face = Face()
        stages = {"swing": (0.5, "Bringing the food over"), "present": (0.6, "Waiting for an open mouth"),
                  "bite": (0.8, "Waiting for the bite"), "retract": (0.9, "Going back to the bowl")}
        stage_dir = record.stage("deliver")
        rc = deliver(robot, arm, self._face, stage_dir, open_timeout_s=float(N.open_mouth_timeout_s),
                     bite_timeout_s=float(N.bite_timeout_s), settle_s=float(N.bite_settle_s),
                     step=lambda name: step("deliver", *stages[name]))
        if rc == 0:
            self._holding = False
            return {"bite_taken": True}
        if rc == 7:
            return {"bite_taken": False, "declined": True, "food_on_fork": True}
        if rc == 4:
            raise EngineError("no_bite", "no bite was felt; the food is still on the fork", recoverable=True)
        raise EngineError("presentation_refused", "the arm could not present the food at the mouth; the food is "
                          "still on the fork", recoverable=True)

    def _mock(self, action_id, ctx) -> dict:
        ctx.checkpoint()
        ctx.progress(action_id, 1.0, "Done (mock)", pausable=False)
        if action_id == "start_meal":
            self._meal = True
            return {"bowl_xy_m": [0.45, 0.0]}
        if action_id == "feed":
            return {"bite_taken": True}
        self._meal = False
        return {"home": True}

    def _recover(self, robot, arm, to_bowl: bool = True) -> None:
        """Clear of the user and the bowl before a planned move, after an action was cancelled mid-way (the planner
        refuses a start inside its world): at the user, back the way the approach came (and on over the bowl, to_bowl);
        with the tip under the free-space floor (in the bowl), straight up to food_detect_start's height."""
        from rammp_feeding import scene
        from rammp_feeding.acquire import rise
        from rammp_feeding.config import cfg
        from rammp_feeding.deliver import retract
        from rammp_feeding.geometry import axis
        T, R = arm.tool()
        Te, Re = arm.tool(scene.captured("deliver"))
        if abs(axis(R)[2]) <= float(cfg.delivery.level_tol) and float((T - Te) @ axis(Re)) > float(cfg.motion.arrived_m):
            if to_bowl:
                retract(robot, arm)
            else:
                arm.straight(Te, Re, float(cfg.delivery.retract.back_off_mps), "back to the delivery look",
                             torque_stop=False)
            return
        tip = arm.tip()
        try:
            low = tip[2] < scene.tip_floor(float(tip[0]), float(tip[1]))
        except (OSError, KeyError, ValueError):              # (no bowl read this meal: nothing of ours to clear)
            low = False
        if low:
            rise(arm, "out of the bowl")

    # ------------------------------------------------------------------ the robot, per action
    def _robot(self, roles, check):
        from rammp_feeding.ros import Robot
        return Robot(node=self, group=self._group, roles=roles, check=check, static_tf=self._static.sendTransform,
                     reset=self._reset, token=self._token)

    def _publish_scene_pose(self) -> None:
        """Our measured pose of the scene camera, latched on /tf_static for the node's life (not when the robot's own
        /tf_static places it)."""
        from rammp_feeding.config import cfg, state_dir
        from rammp_feeding.ros import transform_msg
        from rammp_feeding.scene_camera import SC, extrinsic, from_platform
        if not from_platform() and (state_dir() / "scene_extrinsic.json").exists():
            self._static.sendTransform(transform_msg(self.get_clock().now().to_msg(), str(cfg.arm.base_frame),
                                                     str(SC.link_frame), extrinsic()[0]))


def main():
    run(Feeding)


if __name__ == "__main__":
    main()
