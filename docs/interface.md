# rammp_feeding: interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Tier:** experimental

## What it does

Feeds a seated user from a bowl (see the [introduction](index.mdx)). It is one node, `rammp_feeding`, running one
command (`rammp feed` under sheppy). It offers no topics or services of its own; it publishes one static transform.

## Subscribes

| Topic | Type | From | Use |
| --- | --- | --- | --- |
| `/robot_description` | `std_msgs/String` (latched) | `arm_state_publisher` | The arm's model for FK, the Jacobian and IK. Without it, the bundled copy of the driver's URDF (`urdf/gen3_7dof_2f85.urdf`). |
| `/joint_states` | `sensor_msgs/JointState` | `arm` | Joint positions, and joint torques for the contact guard and the bite. |
| `/control_status` | `rammp_common_interfaces/ControlStatus` (latched) | `arm` | The e-stop, and whether arbitration is enforced. |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | the camera drivers | Each camera's depth -> colour transform and link -> optical frames. |
| `/wrist_camera/color/image_raw`, `/wrist_camera/depth/image_rect_raw`, and their `camera_info` | `sensor_msgs/Image`, `sensor_msgs/CameraInfo` | `wrist_camera` | The bowl and the food from food_detect_start, the depth under the tip, the mouth, `calibrate-wrist`. |
| `/scene_camera/color/image_raw`, `/scene_camera/depth/<image_raw \| image_rect_raw>`, and their `camera_info` | `sensor_msgs/Image`, `sensor_msgs/CameraInfo` | `scene_camera` | The coarse bowl and table (`locate-scene`), `calibrate-scene`. The depth topic is the camera profile's (`image_raw` on the Gemini 336L, `image_rect_raw` on the D435i). |

## Calls

| Name | Type | On | Use |
| --- | --- | --- | --- |
| `/execute_joint_trajectory` | `rammp_arm_interfaces/action/ExecuteJointTrajectory` | `arm` | Every move. Position control, except the approach to the mouth, which runs in impedance. Cancelled to stop. |
| `/rammp_curobo/plan_to_joints` | `rammp_curobo_interfaces/action/PlanToJoints` | `planner` | Free-space paths from the shared planner, as configured for every client. Each path is checked as it will run before it runs: against the utensil tip's floor over the table and the bowl, and against the utensil's clearance from the bowl and the user. |
| `/rammp_curobo/set_world` | `rammp_curobo_interfaces/srv/SetWorld` | `planner` | The module's world (table, hollow bowl, user, pedestal) by file, `/config/planner_world.yaml`, before each planned move. The planner's own world is put back when the command ends. |
| `/rammp_curobo/get_parameters` | `rcl_interfaces/srv/GetParameters` | `planner` | Its `world` parameter, the world to put back (`planner.restore_world` where it has none). |
| `/perception/sam3/detect` | `rammp_perception_interfaces/srv/Detect` | `sam3` | Boxes and masks. In the wrist camera: the bowl and the food in one call. In the scene camera: the bowl (`perception.scene.bowl_prompts`). Detections without a mask are not used. |

## Publishes

| Topic | Type | Meaning |
| --- | --- | --- |
| `/tf_static` | `tf2_msgs/TFMessage` | `base_link -> <scene camera link_frame>`: the scene camera's pose as measured by `calibrate-scene` (`<state_dir>/scene_extrinsic.json`). |

## Arbitration

Goals carry the token in `RAMMP_ARM_TOKEN` (hex, 16 bytes; zeros by default). The module never calls
`/acquire_control`. Under `arbitration_mode:=enforced` the orchestrator acquires control and hands the module the
token; without one, the module refuses to send goals.

## Files

| Path (in the container) | Contents |
| --- | --- |
| `/config/local.yaml` | The site's settings (`config/machines/<site>.yaml`) |
| `/config/{home,overlook,scoop_end,deliver}.json` | The captured poses (`rammp capture`) |
| `/config/wrist_mount.json` | The wrist camera's mount and the utensil's pixels (`rammp calibrate-wrist`, or entered by hand: [porting, 6b](porting.md#6b-entering-an-existing-calibration-no-board)) |
| `/config/scene_extrinsic.json` | The scene camera's pose (`rammp calibrate-scene`, or entered by hand: [porting, 6b](porting.md#6b-entering-an-existing-calibration-no-board)) |
| `/config/scene_state.json` | The bowl and the table for this session (`rammp locate-scene`, refined by `rammp detect`) |
| `/config/planner_world.yaml` | The world handed to the planner |
| `/runs/` | The run record |

## Requirements

Host networking and IPC. Run it as the user that owns the host's `/config` and `/runs` folders. CPU only: SAM3 and
cuRobo are their own nodes, and MediaPipe's face landmarker runs on the CPU.
