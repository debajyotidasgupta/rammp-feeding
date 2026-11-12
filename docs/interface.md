# rammp_feeding: interface

**Rung:** ad-hoc · **Tier:** experimental

## What it does

One node, a capability engine (`rammp-kitchen-capability-engines`' `CapabilityEngine`, engine id `feeding`). It
registers with the orchestrator and moves the arm only in an action the orchestrator starts, at the user's request:
find the bowl, feed one bite, and go home. Nothing moves when it starts, and nothing moves between actions.

## Publishes

| Topic | Type | Rate | Meaning |
| --- | --- | --- | --- |
| `/feeding/status` | `std_msgs/String` | 1 Hz and on change | Health: JSON `state` (idle, executing, paused), `action`, `instance_id`, `mode`, `registered`, `mock` |
| `/tf_static` | `tf2_msgs/TFMessage` | latched | `arm_mount -> <scene camera link frame>`, our measured pose, only on a robot whose own `/tf_static` does not place the scene camera |
| `/reset` | `std_msgs/Empty` | once per action | The planner's own world back, as each action ends |

## Subscribes

Only while an action runs (each action's subscriptions are destroyed as it ends); the scene camera only in
`start_meal`.

| Topic | Type | Required | Meaning |
| --- | --- | --- | --- |
| `/arm/robot_description` | `std_msgs/String` | yes | The arm's model, rooted at `arm_mount` |
| `/joint_states` | `sensor_msgs/JointState` | yes | `arm_joint_1..7`: joints, and torques for the contact stop and the bite |
| `/control_status` | `rammp_common_interfaces/ControlStatus` | yes | The e-stop |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | yes | The cameras' frames; on the demo robot, the scene camera's mount |
| `/wrist_camera/color/image_raw`, `.../depth/image_rect_raw`, `.../color/camera_info` | `sensor_msgs/Image`, `CameraInfo` | yes | The wrist camera (topics: the selected camera profile) |
| `/scene_camera/color/image_raw`, `.../depth/image_rect_raw`, `.../color/camera_info` | `sensor_msgs/Image`, `CameraInfo` | `start_meal` | The scene camera |

## Calls

| Name | Type | Served by | Use |
| --- | --- | --- | --- |
| `/execute_joint_trajectory` | `rammp_arm_interfaces/action/ExecuteJointTrajectory` | `arm` | Every move, in position control; cancelled to stop |
| `/rammp_curobo/plan_to_joints` | `rammp_curobo_interfaces/action/PlanToJoints` | `planner` | Free-space paths |
| `/rammp_curobo/set_world` | `rammp_curobo_interfaces/srv/SetWorld` | `planner` | Our world while an action plans: `/config/planner_world.yaml`, read by the planner at `/feeding/planner_world.yaml` |
| `/perception/sam3/detect` | `rammp_perception_interfaces/srv/Detect` | `sam3` | Boxes and masks: the bowl and the food |

## Orchestrator interface

Served under `/feeding/`, types from `rammp_orchestrator_interfaces`: `activate`, `execute_action`, `control`,
`hibernate`; and `acquire_control`, `release_control` (`rammp_common_interfaces/srv/AcquireControl`,
`ReleaseControl`). It registers through `/rammp_orchestrator/register_engine`. Activated by the perception classes
`food container` and `food`. Locks the `arm`.

| Action | Label | Params | Result payload |
| --- | --- | --- | --- |
| `start_meal` | Find the bowl | — | `bowl_xy_m` |
| `feed` | Feed a bite | — | `bite_taken`; `bowl_empty` when nothing is left; `declined` and `food_on_fork` when the user did not open their mouth |
| `finish_meal` | Done eating | — | `home` |

The bowl is measured once a meal, by `start_meal` or, when it has not run, by the meal's first `feed`; every bite after it uses that reading (its rim, when SAM3 does not find the bowl in a bite's image). A meal ends with `finish_meal` or the object's deselection. `feed` takes about 50 s and at most about 105 s (the waits at the mouth are
bounded: `node.*` in `config/rammp.yaml`). A piece the user did not take stays on the fork, and the next `feed`
presents it again rather than scooping another. Error codes: `bowl_not_found`, `perception_failed`,
`no_pickable_piece`, `pickup_failed`, `delivery_failed`, `no_bite`, `presentation_refused`, `move_failed`,
`robot_unavailable`, `not_set_up`. A cancel stops the move in flight and waits for the arm to settle.

## Parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `use_orchestrator` | `true` | `false`: standalone (drive it with `ros2 action send_goal /feeding/execute_action ...`) |
| `mock` | `false` | No hardware: every action answers at once, nothing moves |
| `state_dir` | `/config` | The site's state folder (below) |
| `runs_dir` | `/runs` | The run record's folder |
| `record` | `false` | The run record (every stage's images, depth, telemetry, log): a debug setting |
| `camera_profile` | `oak_d_pro` in the fragment | `config/camera/<name>.yaml`: the wrist camera |
| `scene_camera_profile` | `oak_d_pro_w` in the fragment | `config/camera/<name>.yaml`: the scene camera |
| `config` | `""` | Any other setting from `config/rammp.yaml`, as `"a.b=1,c=[1, 2]"` |

## Arbitration

The orchestrator hands its execution token to `/feeding/acquire_control` before it starts an action and takes it
back through `/feeding/release_control` once the action ends. Every arm goal of the action carries that token and the
engine's id (`feeding`) as its sender. On an arm that enforces arbitration, an action started without a token sends no
goals.

## Files (in the container)

| Path | Contents |
| --- | --- |
| `/config/local.yaml` | The site's settings |
| `/config/{home,overlook,deliver}.json` | Captured poses |
| `/config/wrist_mount.json`, `/config/scene_extrinsic.json` | The calibrations (the scene camera's only where the robot does not place it) |
| `/config/scene_state.json` | This meal's bowl and table |
| `/runs/` | The run record, with `record` on |

Host networking and IPC; run as the owner of `/config` and `/runs`; CPU only.
