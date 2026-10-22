# rammp_feeding -- Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Tier:** experimental

## What it does

Feeds a seated user from a bowl (see the [introduction](index.mdx)). One node, `rammp_feeding`, running one command
(`rammp feed` under sheppy). It publishes no topics of its own beyond a static transform.

## Subscribes

| Topic | Type | From | Use |
| --- | --- | --- | --- |
| `/robot_description` | `std_msgs/String` | `arm_state_publisher` | The arm's model: our kinematics (IK for straight-line and scoop paths). Without it, a bundled copy of the driver's. |
| `/joint_states` | `sensor_msgs/JointState` | `arm` | Joints, and joint torques for the contact guard and the bite. |
| `/control_status` | `rammp_common_interfaces/ControlStatus` | `arm` | The e-stop; whether arbitration is enforced. |
| `/tf_static` | `tf2_msgs/TFMessage` | the camera drivers | Each camera's depth -> colour and link -> optical frames. |
| `/wrist_camera/color/image_raw`, `/wrist_camera/depth/image_rect_raw` (+ `camera_info`) | `sensor_msgs/Image` | `wrist_camera` | The aim, the floor under the tip, the mouth. |
| `/scene_camera/color/image_raw`, `/scene_camera/depth/image_raw` (+ `camera_info`) | `sensor_msgs/Image` | `scene_camera` | The coarse bowl and table; its own calibration. |
| `/perception/object_detections` | `rammp_perception_interfaces/ObjectDetections` | `scene_segmenter` | The bowl ("food container"). |

## Calls

| Name | Type | On | Use |
| --- | --- | --- | --- |
| `/execute_joint_trajectory` | `rammp_arm_interfaces/action/ExecuteJointTrajectory` | `arm` | Every move. Position control; the approach to the mouth in impedance. Cancelled to stop. |
| `/rammp_curobo/plan_to_joints` | `rammp_curobo_interfaces/action/PlanToJoints` | `planner` | Free-space paths from the shared planner, as it is configured for everyone; each checked as it will run against our utensil-tip floor and, for the utensil's spheres, against our boxes before it runs. |
| `/rammp_curobo/set_world` | `rammp_curobo_interfaces/srv/SetWorld` | `planner` | Our table, bowl, user and pedestal by file (`/config/planner_world.yaml`, mounted read-only into the planner) before each planned move; the planner's own world back when a command ends. |
| `/rammp_curobo/get_parameters` | `rcl_interfaces/srv/GetParameters` | `planner` | Its `world` parameter: the world to put back (`planner.restore_world` where it has none). |
| `/perception/sam3/detect` | `rammp_perception_interfaces/srv/Detect` | `sam3` | The bowl and the food in the wrist camera (boxes and masks); the bowl in the scene camera when the segmenter has none. |

## Publishes

| Topic | Type | Meaning |
| --- | --- | --- |
| `/tf_static` | `tf2_msgs/TFMessage` | `base_link -> scene_camera_link`: the scene camera's pose as measured from the arm (`calibrate-scene`). |

## Arbitration

Goals carry the token in `RAMMP_ARM_TOKEN` (hex, 16 bytes; zeros by default). The module never calls
`/acquire_control`: under `arbitration_mode:=enforced` the orchestrator acquires and hands it the token.

## Requirements

Host networking and IPC, the user's uid (files in `/config` and `/runs` stay host-owned). CPU only: SAM3 and cuRobo
are their own nodes; MediaPipe's face landmarker runs on the CPU.
