# The port to the RAMMP stack (2026-10-16)

From a Kortex-API stack of our own (kinova/, feeding/, shell stages, our servers) to one RAMMP module that reuses the
platform's nodes. What changed, what was measured, what is left.

## Replaced by the platform

| Ours (removed) | Now |
| --- | --- |
| Kortex API sessions, `safe_move`, `run_path` over `ExecuteWaypointTrajectory`, `Base.Stop`, the twist watchdog | kinova-gen3-ros2's `/execute_joint_trajectory` (position; impedance for the approach); a cancel halts the arm |
| Kortex `ComputeInverseKinematics` / `ComputeForwardKinematics` | our IK on the arm's own model (`/robot_description`) |
| `make limits`, `make tool` (the arm's soft limits and tool config) | the driver's own limits (URDF velocity caps); the tool frame is the model's flange + `arm.tool_from_flange_m` |
| our cuRobo planner server (HTTP, two warm planners with the utensil's spheres) | RAMMP-CuRobo's `plan_to_joints`; the world by `set_layer` (else `set_world`), the utensil by `attach_spheres` (below) |
| camera servers, the Kinova vision module's depth, `colour_stream` | the camera drivers' topics, `camera_info` and `/tf_static` |
| LocateAnything (+ its fallbacks), Depth Anything 3 depth fill | SAM3's detect (boxes and masks); the scene segmenter for the coarse bowl |
| MediaPipe's HTTP face server and its tracker | MediaPipe in-process (the tracked approach was dropped: open loop, the user's choice) |
| the run-record shell layer (`lib.sh`), `sync.sh`, `setup_jetson.sh` | `record.py`; the module image; sheppy |
| sim/ (the MuJoCo twin, the Isaac strategy search) | removed with the code they drove (in git: abra-site-stack) |

## Measured

- **The URDF and Kortex's own kinematics disagree by ~6.5 mm and ~1 deg** at the tool (1753 telemetry samples from
  the 2026-09-25 runs, at rest and moving alike; no single base or tool offset explains it: 2 mm / 0.6 deg remain after
  fitting both). The driver and cuRobo run the URDF, so it is the model now; the fine aim, the plunge and the approach
  are relative moves from the wrist camera, and `tip_px` is re-measured on the new model at a 10 mm hover anyway.
- **The aim in 3D.** The old image formula took the image of true vertical as the principal point; at the overlook's
  ~0.4 deg tilt that put OVER 2.6 mm off the piece (the bite test). Placing the piece in the base frame from its pixel
  and depth through the camera's pose, and reading "under the tip" along the true vertical, brings it under 0.2 mm.
  `vertical_vp` is gone from the camera profile.
- **The utensil's image region** was 300..480 in y: with the tip over the bowl's centre a piece right under it shows
  at y ~300 and was dropped as "the utensil". Now 365..480 (the tines start at ~372).
- The scoop end is placed by FK of its captured joints (its stored position was Kortex's, ~7 mm off our model).

## Old vs new, on one model (2026-10-16)

The old stack's own offline bite (tests/site_check.py on abra-site-stack) with its FK and IK swapped for the URDF chain,
against the new stack on the same inputs (the 2026-09-30 site, the tip over the bowl's centre, the depth's floor 7 mm
over the table, a 25 mm piece, the glass floor bound, the old run's planned paths fed to both):

| move | end (tip) | tip path | seconds old / new |
| --- | --- | --- | --- |
| detect, swing, retract's return, home (planned) | 0.1 mm | (the planner's) | 8.73/8.73, 3.96/3.94, 9.82/9.80, 8.90/8.65 |
| plunge | 0.0 mm | 1.2 mm | 6.91 / 6.95 |
| scoop | 8.4 mm | 8.4 mm | 4.42 / 4.48 |
| approach | 0.4 mm | 7.9 mm (from the swing's end) | 4.44 / 4.36 |
| retract: back off + in | 0.7 mm | 0.7 mm | 1.67 / 1.68 |

Every difference is one of the model's: food_detect_start's orientation now FK of its captured joints (the old aimed
at Kortex's reading of it: 0.96 deg, 2.3 mm), the scoop end FK of its joints (Kortex's recorded position: 7.8 mm),
and the swing goal following both. On the arm both reproduce the captured physical poses. Found and restored on the
way: the old plan acceleration (72 / 1.6 = 45 deg/s^2) and speed cap (0.98 x 63 deg/s, every joint), the 2 mm "already
there" of a straight move, the bite's wait for the force to settle, the planner world refreshed with the measured
mouth, punch from a hover, the direct home.

The bite: Kortex's own force estimate against ours (J^T f = tau, at the tool frame where Kortex reported it), on the 13
bites of 2026-09-25: correlation 0.97-1.00, peak ratio 1.15 (1.04-1.33); with bite_n 2 N for 0.1 s both decide the
same on all 13 (the same sample on 10, up to 0.4 s earlier on 3). At the utensil tip instead, 2 of the 13 are missed.

## The planner's world, as the old stack's (2026-10-16)

The old stack ran its own cuRobo server: the utensil (5 spheres from the gripper's pads to the tip) and, on the
swing, the piece (one more) as spheres on `bracelet_link`; `world_padding` 0.01 with the bowl's pieces and the pedestal
exact; the standoff (`collision_activation_distance`) 0.01. The port first ran RAMMP-CuRobo as it ships: no utensil,
every box but the pedestal padded 0.02, standoff 0.03. Three generic changes to RAMMP-CuRobo were made to bring it
back (PR #30 for #23, PR #33 for #32, stacked on #28's config and scene layers, and PR #31 for #29). They were closed
on 2026-10-22, unmerged, for the shared planner as it is (below): kept here for what they measured.

- `~/attach_spheres` on an opt-in held-object model (`gen3_real_held.yaml`). Opt-in: even disabled, its extra spheres
  moved pose plans by up to 4.9 mm. The default model stays byte-identical, and so does every pose plan (0.0000 rad).
- `~/set_layer`: runtime layers beside #28's, a padding per layer, boxes as a world file writes them (`Obstacle`:
  name, position, rpy_deg, dims), over a scene `world_mount.yaml` (the pedestal alone). No world file: the planner's
  `/config` mount goes once this is released. (A first version sent a quaternion pose; #28 builds every orientation
  from rpy_deg, so the PR does too, and the module no longer converts.)
- `collision_activation_distance` as a node parameter / launch argument (negative: the config's).

Checked on the cluster's 3090 (job logs under /scratch/dd752/curobo_eq, node 01), last on a tree merging the three
PRs over #28:
- The world as one layer padded by the planner's own `world_padding` is the file's world exactly: the same 17 boxes
  to the last bit and the same pose plans, each way in a fresh process. (Within one process cuRobo's solvers carry
  state from call to call; its joint plans vary run to run regardless: up to 0.08 rad on dev alone.)
- The old server's planner (its `curobo_world.world` and `robot_config`, on the same inputs) against the new one,
  replaying the bite's four planned calls: the boxes identical, the standoff identical, all 52 (53 carrying) arm
  spheres within 3e-8 m. The old spheres sat on `bracelet_link` and so carried the robot model's 5 mm
  `collision_sphere_buffer`; attached spheres are used as given, so `planner.utensil_spheres` holds the radii the old
  ones had in effect (0.017, 0.025). Plans: old vs new 0.005-0.030 rad max on the path, old vs old up to 0.021,
  new vs new 0 to 0.051; the carrying swing differs by 0.018 rad every time (its inputs identical: cuRobo's order of
  spheres and boxes).
- The bite against the real planner, held-object model, standoff 0.01: through for every piece and user offset
  tests/offline.sh tries (the far user refused, as it must), no floor re-plans.

The old server rebuilt its world from a list after each delivery (`/world`), which made every box a padded prop: from
the first delivery on it padded the bowl and the pedestal too, against its own comment. The new stack keeps them exact.

The module was then written for a planner release carrying them (the held-object config, a 0.01 standoff, no file
fallback), and dropped with the PRs.

## Interfaces in step with the deployment (2026-10-21)

rammp-deployments' dev moved its release nodes on: kinova-gen3-ros2 1.1.0 (FROM rammp-base 1.1.0-jp6, interfaces
v1.1.0, which append `speed_scale` to ExecuteJointTrajectory) and rammp-curobo 1.1.0; the perception images
(sha-8091bbee) were rebuilt on rammp-interfaces-ros2 dev with #16 (`DetectedObject.size`). The image was on
rammp-base 1.0.0-jp6, perception interfaces at c418dad (before #16) and the planner's at v1.0.0.

Checked over ROS (cascadilla, ros:humble-ros-base amd64, Cyclone DDS as on the robot), a stand-in built as the
deployment builds its nodes publishing three detections and serving SAM3's detect with three, our module built each way:

| our interfaces | segmenter / SAM3 | trajectory goal to the 1.1.0 arm |
| --- | --- | --- |
| as now: base 1.1.0, perception 2bf67d4, planner v1.1.0 | as published (labels, boxes, masks) | received (2 points, impedance) |
| as before: base 1.0.0, perception c418dad, planner v1.0.0 | only the first detection; the rest an empty label and a zero box; every mask lost | received |
| base 1.0.0, perception 2bf67d4 | as published | received |
| base 1.1.0, perception c418dad | as the old | received |

The perception pin is the fix: DetectedObject is an array element of ObjectDetections, so one more field shifts every
object after the first and the label image. The arm's appended field is read as its default (under Cyclone; under
Fast DDS the mismatched types do not connect at all). The base moved to 1.1.0 anyway, the arm's contract. The
planner's PlanToJoints and SetWorld are the same in v1.0.0 and v1.1.0 (only PlanToPose grew), and kinova-gen3-ros2
1.1.0 builds the same URDF the module bundles (kinova-gen3-driver v1.2.0's, unchanged since v1.1.1).

## The shared planner, as it is (2026-10-21)

The planner is one node for every demo in a sheppy chain, and it reads its config once: the held-object model and
the standoff (PRs #30, #31) would change everyone's plans, and a world or an attachment left behind would be the next
demo's. So the module uses the planner as deployed (rammp-curobo 1.1.0, gen3_real.yaml: padding 0.02, standoff 0.03,
no held object) and keeps what only it knows on its side:

- our world (table, bowl, user, pedestal) pushed by file and set_world just before each planned move, and the
  planner's own world (its `world` parameter, else `planner.restore_world`) put back when every command ends;
- the utensil and, on the swing, the piece checked against our boxes along each planned path as it will run (sampled
  every 0.5 deg), with the old stack's margins (boxes grown 0.01, the bowl's pieces exact, 0.01 clear) -- what the old
  planner's spheres did inside it, here after planning: a hit is a re-plan, as the tip floor's always were.

The tip floor is now checked along the path as it runs, not only at its waypoints. The cost: cuRobo cannot route the
utensil round a box, only be asked again. #33 (runtime layers) would let any demo send measured obstacles without a
file; nothing here depends on it.

## Left for the planner

- The world's padding and the standoff are still planner-wide, not per layer (the standoff: per node, at launch).
- With the utensil in the planner's model, detect / home could use the arm's `go_to_joint_config` directly.
