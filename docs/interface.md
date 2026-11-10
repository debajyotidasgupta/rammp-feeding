# rammp_feeding: interface

**Rung:** ad-hoc · **Tier:** experimental

One node, `rammp_feeding`, which runs `rammp feed` under sheppy. It offers no topics or services; it publishes one
static transform.

## Uses

| Name | Type | From | Use |
| --- | --- | --- | --- |
| `/robot_description` | `std_msgs/String` | `arm_state_publisher` | The arm's model; the bundled driver URDF if absent |
| `/joint_states` | `sensor_msgs/JointState` | `arm` | Joints, and torques for the contact stop and the bite |
| `/control_status` | `rammp_common_interfaces/ControlStatus` | `arm` | E-stop, arbitration |
| `/tf_static` | `tf2_msgs/TFMessage` | camera drivers | Each camera's frames |
| `/wrist_camera/...`, `/scene_camera/...` | `sensor_msgs/Image`, `CameraInfo` | the cameras | Colour, depth, intrinsics (topics in the camera profiles) |
| `/execute_joint_trajectory` | `rammp_arm_interfaces/action/ExecuteJointTrajectory` | `arm` | Every move, in position control; cancelled to stop |
| `/rammp_curobo/plan_to_joints` | `rammp_curobo_interfaces/action/PlanToJoints` | `planner` | Free-space paths |
| `/rammp_curobo/set_world` | `rammp_curobo_interfaces/srv/SetWorld` | `planner` | Our world (`/config/planner_world.yaml`) while we plan; the planner's own is put back after |
| `/perception/sam3/detect` | `rammp_perception_interfaces/srv/Detect` | `sam3` | Boxes and masks: the bowl and the food |

## Publishes
`/tf_static`: `base_link -> <scene camera link_frame>`, from `/config/scene_extrinsic.json`.

## Arbitration
Goals carry the token in `RAMMP_ARM_TOKEN`. When arbitration is enforced and no token is set, no goals are sent.

## Files (in the container)

| Path | Contents |
| --- | --- |
| `/config/local.yaml` | The site's settings |
| `/config/{home,overlook,deliver}.json` | Captured poses |
| `/config/wrist_mount.json`, `/config/scene_extrinsic.json` | The calibrations |
| `/config/scene_state.json` | This session's bowl and table |
| `/runs/` | The run record |

Host networking and IPC; run as the owner of `/config` and `/runs`; CPU only.
