---
title: Deployment checklist
---

# Deployment checklist: a new Jetson

Everything a new setup needs, from an unboxed Jetson to `rammp feed`. Work through the sections in order: each one
depends on the ones before it. Nothing here comes from another installation. Every value is measured or set on this
one, and kept in its **state directory**, never in the image, so one image serves every installation.

Conventions:
- the state directory is `/srv/rammp_feeding/config` on the host (`/config` in the container);
- the run record is `/srv/rammp_feeding/runs` (`/runs`);
- the module runs as uid 1000;
- positions are in the arm's base frame (`base_link`: x forward, y left, z up), in metres.

| Step | Result |
| --- | --- |
| 1. Hardware | What to have before starting |
| 2. Jetson and network | The host tuned; the arm reachable |
| 3. Software | sheppy, the deployment repo, the module's image, the host folders, the `rammp` command |
| 4. The deployment manifest | The stack's nodes for this robot, the module's node among them |
| 5. The site file | `local.yaml`: the profiles and every hand measurement |
| 6. Bring the stack up | Every node but `feeding`; `rammp check` |
| 7. Captures | `home.json`, `overlook.json`, `scoop_end.json`, `deliver.json` |
| 8. Calibrations | `wrist_mount.json`, then `scene_extrinsic.json`: measured with the board, or entered from an existing calibration (8b) |
| 9. Verification | One bite, stage by stage |
| 10. Feeding | `rammp feed`, under sheppy or by hand |

Every command that moves the arm needs a person at the e-stop (see [Safety](#safety)).

## 1. Hardware

- [ ] **An NVIDIA Jetson AGX Orin** (arm64, 12 cores) with JetPack, Docker and the NVIDIA container runtime. The arm
      driver runs its control loop on CPU 11.
- [ ] **A Kinova Gen3 7-DoF with a Robotiq 2F-85**, mounted **level**. This release supports only the arm stack's
      release generation (kinova-gen3-ros2 1.1.0, rammp-curobo 1.1.0), which models a level mount.
- [ ] **The utensil** (the fork profile: a four-tine fork) seated in the gripper. The module never opens or closes
      the gripper.
- [ ] **A wrist RGB-D camera** on the arm's wrist, looking along the utensil: a RealSense D405 or D435i
      (`config/camera/realsense_d405.yaml`, `realsense_d435i.yaml`).
- [ ] **A fixed scene RGB-D camera** that sees the table and the bowl: an Orbbec Gemini 336L or a RealSense D435i
      (`config/camera/orbbec_gemini336l.yaml`, `realsense_d435i_scene.yaml`).
- [ ] **A table, a bowl and the food**, laid out with the pieces spaced apart. A cut-cube food (20–30 mm) suits
      the food profile.
- [ ] **A tape measure, and a printer** for the ChArUco board (or the values of an existing calibration, 8b).

A camera of another model needs its own profile in `config/camera/`. For a wrist camera that means:
- its `streams` (colour and depth topics);
- `glass_to_optical_m` (its depth origin behind the front glass, from the datasheet);
- a `mount:` block with `camera_in_tool_m: null`, `rotation_quat_xyzw: null` (both measured) and
  `image_rotated_at_delivery`.

For a scene camera: its `streams`, its driver's root frame `link_frame`, and `error_m` (how far its reading may be off).
Intrinsics are never configured: they come from the driver's `camera_info` and `/tf_static`.

## 2. Jetson and network

- [ ] **The arm's control core is isolated.** Add `isolcpus=11 nohz_full=11 rcu_nocbs=11` to the kernel command line
      in `/boot/extlinux/extlinux.conf`, and reboot. Check with `cat /proc/cmdline`.
- [ ] **The arm has a fixed address**, `192.168.1.10` (the Gen3's default, and the address the arm driver connects
      to). The Jetson's NIC on the arm's link gets a fixed address on the same subnet:

      ```bash
      sudo nmcli con mod "<connection>" ipv4.method manual ipv4.addresses 192.168.1.20/24
      sudo nmcli con up "<connection>"
      ```

      Check with `ping 192.168.1.10`.
- [ ] **The arm's link runs at 1 Gb/s**: check `ethtool <arm_nic>` or the switch port's light. At 100 Mb/s, any stray
      traffic starves the driver's 1 kHz control loop, and the driver exits.
- [ ] **DDS receive buffers:** `net.core.rmem_max` at least 32 MiB. Below it, DDS silently drops camera images, and
      SAM3 loses the ~900 KB colour frames.

      ```bash
      echo 'net.core.rmem_max=33554432' | sudo tee /etc/sysctl.d/60-rammp-dds.conf && sudo sysctl --system
      ```

- [ ] **No DDS multicast on the arm's link.** Otherwise Cyclone DDS multicasts the camera images out of the arm's NIC
      (13.6 MB/s while a command subscribes to the cameras), and the arm driver exits ("timeout detected:
      BaseCyclicClient::Refresh"). Drop it on egress:

      ```bash
      sudo tc qdisc add dev <arm_nic> clsact
      sudo tc filter add dev <arm_nic> egress prio 10 protocol ip flower dst_ip 224.0.0.0/4 action drop
      ```

      The filter does not survive a reboot or the NIC being re-opened. Install it from a NetworkManager dispatcher
      script (`/etc/NetworkManager/dispatcher.d/`) that runs these two lines when `<arm_nic>` comes up. On a Jetson
      with separate arm and camera NICs, rammp-deployments' host tuning (`december_2026/config/hosts/jetson-agx-orin/
      tuning.sh install`, with `ARM_IF` and `CAM_IF` set) installs this, the sysctl and the rest of the platform's
      tuning in one step.
- [ ] **Nothing else commands the arm while its driver runs**: the Kinova Web App, the wrist buttons, a gamepad.
      The driver exits as soon as the arm leaves low-level servoing (`WRONG_SERVOING_MODE`).

## 3. Software

- [ ] **sheppy**, the stack's supervisor:

      ```bash
      uv tool install "sheppy @ git+https://github.com/rammp-org/sheppy@dev"
      ```

- [ ] **The deployment repo:** `git clone https://github.com/rammp-org/rammp-deployments.git`.
- [ ] **The module's image**, pinned to its release:

      ```bash
      docker pull debajyotidasgupta/rammp-feeding:0.2.2
      ```

- [ ] **SAM3's weights.** They are gated on Hugging Face and not in SAM3's image. Accept the licence for
      `facebook/sam3`, download the snapshot to the Jetson, and point the `sam3` node's weights mount (`/models/sam3`)
      at it (section 4).
- [ ] **The host folders**, owned by the module's uid:

      ```bash
      sudo mkdir -p /srv/rammp_feeding/config /srv/rammp_feeding/runs
      sudo chown -R 1000:1000 /srv/rammp_feeding
      ```

- [ ] **The `rammp` command**, for running the module one command at a time. Save it as `/usr/local/bin/rammp` and
      make it executable (`sudo chmod +x`):

      ```bash
      #!/usr/bin/env bash
      # rammp <command>: the feeding module's command in its image, against this installation's state and run record.
      tty=-i; [ -t 0 ] && tty=-it
      exec docker run --rm $tty --network host --ipc host --user 1000:1000 -e ROS_DOMAIN_ID=0 -e HOME=/tmp \
        -v /srv/rammp_feeding/config:/config -v /srv/rammp_feeding/runs:/runs \
        debajyotidasgupta/rammp-feeding:0.2.2 ros2 run rammp_feeding rammp "$@"
      ```

      Never run it while sheppy's `feeding` node is up: two clients would command the arm.

## 4. The deployment manifest

The stack runs from rammp-deployments' `december_2026/sheppy-manifest.yaml`. In it:

- [ ] **The `feeding` node**, transcribed from this repo's `rammp-alternative.yaml` (image, mounts, topics). Add it if
      the manifest does not have it yet.
- [ ] **The release planner (`curobo_planner`) mounts the state directory read-only.** While the module plans, it
      hands the planner its collision world through `set_world`, which takes a file path:

      ```yaml
          volumes:
            - curobo_kernel_cache:/root/.cache
            - /srv/rammp_feeding/config:/config:ro
      ```

- [ ] **The `sam3` node's weights mount** points at the snapshot from section 3: `- <snapshot_dir>:/models/sam3:ro`.
- [ ] **The scene camera** runs as `/scene_camera`, with its frames prefixed `scene_camera_`
      (`camera_name: scene_camera`), as the scene camera profiles expect. The manifest's Gemini 336L alternative does
      this. A RealSense D435i on the scene needs an alternative of its own: the realsense driver with `camera_name
      scene_camera`, under `/scene_camera`.
- [ ] **This robot's config**, `december_2026/config/robots/<robot>/mount.yaml`. Make it the host's active robot:

      ```yaml
      robot: <robot>      # must equal the directory name
      rpy_deg: [0.0, 0.0, 0.0]   # level
      ```

      ```bash
      uv run scripts/set_robot.py december_2026/config/robots/<robot>
      ```

- [ ] **This robot's profile**, `december_2026/profiles/<robot>.yaml`. It selects the release generation of the
      arm, its state publisher and the planner; this robot's cameras with their serial numbers; SAM3; and the module:

      ```yaml
      description: <robot> -- feeding
      selections:
        arm: kinova_gen3_driver                     # one generation for all three: the release
        arm_state_publisher: kinova_gen3_description
        planner: curobo_planner
        wrist_camera: d405_docker                   # or d435i_docker
        scene_camera: gemini336l_docker
        sam3: sam3_docker
        feeding: rammp_feeding_docker               # leave out until section 10
      overrides:
        wrist_camera:
          serial_no: "_<wrist camera serial>"       # rs-enumerate-devices lists them
        scene_camera:
          serial_number: "<scene camera serial>"
      ```

      `uv run scripts/validate_manifest.py` (with sheppy installed) checks the manifest and the profile together.

## 5. The site file

Copy `config/machines/template.yaml` from this repo to `/srv/rammp_feeding/config/local.yaml`. Fill in every value
marked REQUIRED, and keep a copy under version control as `config/machines/<robot>.yaml`.

- [ ] **The profiles**: the wrist camera, the scene camera, the utensil, the bowl and the food (section 1).
- [ ] **The hand measurements:**

| What | How | Key |
| --- | --- | --- |
| Wrist camera glass to the utensil's tips | A ruler from the camera's front glass straight out, along the camera's axis, to the level of the tips. Not along the utensil: a slanted reading comes out long. | `utensil.glass_to_tip_m` |
| The bowl's inner floor above the table | A ruler, inside the bowl. The plunge never aims under it, because a clear bowl's depth reads the table through its floor. | `bowl.floor_above_table_m` |
| The table's edges | From the arm's base, in its frame: `{x: [near, far], y: [right, left]}`. The planner's table and the utensil's tip floor use them. | `workspace.table_extent_m` |
| Where the bowl is laid | Its centre, `[x, y]`. Of the bowls the scene camera finds, the one nearest this point is taken. | `workspace.bowl_centre_m` |
| Where the user sits | Their mouth when seated, `[x, y, z]`. The planner's world puts the user's head and torso there, and a mouth measured further than `delivery.seated_tol_m` (0.10 m) away is refused. Fixed for a session. | `motion.free_space.user_mouth_m` |
| The ChArUco board as printed | After printing it (section 8), measure across all its squares and divide: printers scale. For a board printed elsewhere, also its layout. | `perception.wrist_calibration.square_m`; `squares`, `marker_ratio`, `dictionary`, `legacy_pattern` |

The bowl's centre and rim radius, the table's height and the bowl's top are never entered by hand. The scene camera
reads them each session, and the wrist camera refines the centre and radius.

- [ ] **Optional, the contact guard** (`guards.torque_step_nm`): if the arm's own motion trips the 2.5 N·m default
      (the trajectory's acceleration ripples the joint torques by up to about 2 N·m), raise it for this robot.

## 6. Bring the stack up

- [ ] `sheppy up <robot> --manifest december_2026/sheppy-manifest.yaml`, with `feeding` **not** selected (leave it
      out of the profile, or unselect it in sheppy's TUI). The arm driver starts holding the arm where it is.
- [ ] `rammp check`. Each row is OK or names what is missing:
  - `arm`, `trajectories`, `planner`, `planner world`, `sam3`, `camera`, `scene_camera`: the nodes are up;
  - `measured`: the hand measurements and the wrist camera's calibration;
  - `poses`: the captures;
  - `scene pose`: the scene camera's calibration.

  The `robot model` row may WARN: kinova-gen3-ros2 1.1.0's `arm_state_publisher` serves no `/robot_description`, so
  the module uses its bundled copy of the driver's own URDF, the same model.

`check`, `state`, `capture`, `calibrate-wrist` and `home --direct` work before anything is measured. Every other
command refuses until `rammp check` is clean.

## 7. Captures

Each capture records the arm's current joints: `rammp capture <name>` writes `<state_dir>/<name>.json`. To pose the
arm, stop the arm node, move the arm with the Kinova Web App or by hand, then restart the arm node and capture.
Capture them in this order: later poses are reached from earlier ones.

| Capture | Pose | Used as |
| --- | --- | --- |
| `home` | **feeding_start**: the utensil straight down, the arm parked beside the table and clear of the user. Leave room above it: `calibrate-wrist` rises 0.10 m and moves 0.15 m out from there. | The start and end of every session, and the calibration route's start |
| `overlook` | **food_detect_start's arm shape**: the utensil straight down over the bowl, the wrist camera looking down at it. | The orientation and arm shape of food_detect_start. Its position is computed each time: the tip over the bowl's centre, the camera 0.45 m (`workspace.overlook_camera_height_m`) above the table. |
| `scoop_end` | **Where the scoop ends**: the utensil level, its tips along the base's +y (the scoop turns the utensil 90° about the base's x axis from straight down), just past the bowl's rim, clear of the table, at most 0.30 m (`acquisition.scoop.max_backoff_m`) across from the bowl. | The scoop's end (FK of its joints) and the arm shape its IK keeps. The swing to the user starts there. |
| `deliver` | **The delivery look**: the utensil level toward the seated user's mouth, the wrist camera on their face, the food out of its view of the mouth. | The swing's goal, and where the mouth is measured |

## 8. Calibrations

Both calibrations use one printed ChArUco board, unless the setup is calibrated already: then enter its values
instead (8b). `rammp calibrate-wrist --board` writes the sheet to print (`/srv/rammp_feeding/runs/charuco_board.pdf`,
US Letter, 7 × 9 squares of 25 mm). Print it at 100 % and measure it (section 5). A board printed elsewhere works if
its layout is configured.

Both commands start at feeding_start and end there. Before they have run, the planned `rammp home` cannot run (it
needs the calibrations and a scene reading). Bring the arm to feeding_start with `rammp home --direct`, once someone
has checked that the way is clear.

### calibrate-wrist: the wrist camera's mount and the utensil's pixels

1. `rammp calibrate-wrist --plan` prints the route's tool poses without moving.
2. Run `rammp calibrate-wrist`. The arm rises, moves out from the base and stops at pose 0.
3. Lay the board flat under the utensil, **behind the tines' tips**, so that the board fills the image around and
   beyond them. The command waits up to 60 s (`wait_board_s`) for it.
4. The arm visits 7 poses. They pivot on a point 0.30 m along the tool axis: tilted ±10° about the tool's x and y
   axes, and turned ±20° about its own axis, at 10°/s. Then it returns to feeding_start.

What it measures:

- **The mount**, the colour optical frame in the tool frame, by OpenCV's hand-eye solve (Park). The board's pose
  comes from the camera's own `camera_info`.
- **The utensil's pixels**: `tip_px`, the middle of the tines' far end, and `image_region_px`, the utensil's box grown
  by 10 px. The utensil is what stays still in the image while the board moves behind it, searched near the tool's
  axis as the measured mount projects it.

It is accepted only when all of these hold:

| Check | Threshold |
| --- | --- |
| Corners found per view | ≥ 12 (`min_corners`) |
| Views that saw the board | ≥ 5 of 7 (`min_poses`) |
| The print's scale, at pose 0 | The board's distance by the camera's depth matches its distance from the image within 2 % (`max_scale_error`). A mismeasured `square_m` moves the camera by the same factor. |
| The board's place in the base frame across the views | Within 5 mm and 1.0° (`max_spread_m`, `max_spread_deg`) |
| The utensil | Told apart from the board behind it, with the board visible beyond its tips |

The result goes to `<state_dir>/wrist_mount.json`, keyed to the camera and utensil profiles. The configuration takes
the mount, `tip_px` and `image_region_px` from it.

### calibrate-scene: the scene camera's pose

Run it after `calibrate-wrist`, which supplies the mount it depends on.

1. Lay the board where **both** cameras see it: in the scene camera's view, and under the wrist camera's route
   (`calibrate-wrist --plan` prints where pose 0 is).
2. Run `rammp calibrate-scene`. The scene camera finds the board where it lies. The wrist camera then runs the same
   route over it and places the board in the base frame through the measured mount. Keep the board still throughout.

The scene camera's link frame in the base frame goes to `<state_dir>/scene_extrinsic.json`. Every command then
publishes it on `/tf_static` as `base_link -> <link_frame>`. Neither the arm nor a starting guess needs to be in the
scene camera's view. Coarse is enough: the scene camera only places food_detect_start, and the wrist camera
re-centres over the bowl from there.

## 8b. Entering an existing calibration (no board)

If the setup has already been calibrated, skip `calibrate-wrist` and `calibrate-scene` and write the two files they
would have written. Nothing in the module needs changing. It reads only the keys below, whoever wrote the file. Nothing
checks entered values, so verify them afterwards ([below](#verifying-entered-values)).

### The wrist camera and the utensil: `<state_dir>/wrist_mount.json`

```
{
 "camera": "<profiles.camera, e.g. realsense_d435i>",
 "utensil": "<profiles.utensil, e.g. fork>",
 "camera_in_tool_m": [X, Y, Z],
 "rotation_quat_xyzw": [QX, QY, QZ, QW],
 "tip_px": [U, V],
 "image_region_px": [X1, Y1, X2, Y2]
}
```

| Key | Meaning |
| --- | --- |
| `camera` | The wrist camera profile the mount belongs to: exactly `profiles.camera`. A file naming another profile is ignored, and `rammp check` reports the mount as not measured. |
| `utensil` | The utensil profile the pixels belong to: exactly `profiles.utensil`. Otherwise `tip_px` and `image_region_px` are ignored. |
| `camera_in_tool_m` | The origin of the wrist camera's **colour optical frame**, in the **tool frame**, in metres. |
| `rotation_quat_xyzw` | The rotation tool ← colour optical frame, a unit quaternion in ROS order **(x, y, z, w)**. |
| `tip_px` | The tines' tips in the colour image, (u, v) pixels: the image as the driver publishes it (not rotated), at its `camera_info` resolution. |
| `image_region_px` | The utensil's box in the colour image, (x1, y1, x2, y2) pixels. Perception never takes a detection inside it as food. |

The frames:

- **The tool frame** is the URDF's `end_effector_link` (the flange) moved `arm.tool_from_flange_m` = (0, 0, 0.12) m
  along its own z: the 2F-85's TCP. Its axes are the flange's.
- **The colour optical frame** is the driver's `<camera>_color_optical_frame` (REP 103): its origin at the colour
  camera's optical centre, z out of the lens, x to the image's right, y down.

`utensil.glass_to_tip_m` (the ruler reading) stays in `local.yaml`. The tip model is the point `camera_to_tip_m` deep
along the ray through `tip_px`. `rammp check` prints it in the tool frame.

A calibration in other frames converts as follows, with T_a_b the pose of frame b in frame a (4 × 4):

- **Relative to the flange (`end_effector_link`):** T_tool_opt = T_flange_tool⁻¹ · T_flange_opt. T_flange_tool is a
  translation of 0.12 m along z, so the rotation stays the same and 0.12 m comes off z.
- **For the camera's body frame (`<camera>_link`):** T_tool_opt = T_tool_link · T_link_opt, with T_link_opt from the
  driver:

  ```bash
  ros2 run tf2_ros tf2_echo <camera>_link <camera>_color_optical_frame    # translation, and rotation (x, y, z, w)
  ```

### The scene camera: `<state_dir>/scene_extrinsic.json`

```
{
 "parent_frame": "base_link",
 "child_frame": "<the scene camera profile's link_frame, e.g. scene_camera_link>",
 "xyz": [X, Y, Z],
 "quat_xyzw": [QX, QY, QZ, QW]
}
```

This is the pose of the scene camera's **link frame** (its driver's root frame, the profile's `link_frame`) in the
arm's **`base_link`**: the position in metres, and a unit quaternion (x, y, z, w). Only `xyz` and `quat_xyzw` are
read; the two frame names record what the pose is. Every command publishes it on `/tf_static` as
`base_link -> <link_frame>`.

- **A pose of the colour optical frame instead:** T_base_link = T_base_opt · T_link_opt⁻¹, with T_link_opt from
  `ros2 run tf2_ros tf2_echo <link_frame> <camera>_color_optical_frame`.
- **One publisher per frame.** The module publishes this transform itself. If the deployment's `static_tf` node also
  places the scene camera (`config/robots/<robot>/transforms/`), tf would see two parents for its link frame. Keep
  the camera there, or here, not both.

### Common mistakes

- A quaternion in (w, x, y, z) order, or not of unit length.
- Millimetres instead of metres.
- A profile name that doesn't match `profiles.camera` or `profiles.utensil` exactly.
- An old `wrist_mount.json` left from another camera, or a calibration of the wrong frame: the body frame for the
  optical one, or the flange for the tool.

### Verifying entered values

None of these steps needs the board.

1. `rammp check`. The `measured` row gives the tip in the tool frame. It should be where the tines are: beyond the
   fingers along the tool's +z, near its axis. The `scene pose` row should read "measured".
2. `rammp locate-scene`. In `scene_detection.jpg` the bowl's box is green, and the logged centre is where the bowl
   stands (a tape measure from the arm's base is enough).
3. `rammp detect`. The log line `the wrist camera: the bowl at (...), N mm from the scene camera's` is how far the
   two calibrations disagree about the bowl. `detect` refuses beyond twice the scene camera's `error_m` (100 mm by
   default). Most of it is the scene camera's own depth error along its line of sight, which grows when the camera
   sees the table at a shallow angle.
4. `rammp perceive`, then `rammp acquire --stop-after hover10`. The tines stop about 10 mm above the centre of the
   piece: the wrist mount and `tip_px` together. Then `rammp rise`.

If step 3 or 4 is off, the entered values are wrong. Check them against the frames above, or measure them with the
board (section 8).

## 9. Verification

Run these in order. Each stage writes its images to the run record.

1. `rammp check`: every row OK.
2. `rammp state`: the joints, the tool, and a utensil tilt of about 0° at feeding_start. (`check`'s `measured` row
   gives the tip in the tool frame.)
3. `rammp locate-scene`: open `scene_detection.jpg`. The bowl's box should be green, at its place.
4. `rammp home`.
5. `rammp detect --plan`, then `rammp detect`: the log gives the wrist camera's correction to the scene camera's
   bowl centre.
6. `rammp perceive`: open `detection.jpg`. Check the bowl overlay, the food boxes, and the utensil's region in grey.
7. `rammp acquire --stop-after hover10`: the tips should stop about 10 mm above the centre of the target piece. Look
   from two sides. If they are off, check `glass_to_tip_m` and the wrist calibration.
8. `rammp rise`, then `rammp detect` and `rammp perceive` again: `acquire` starts at food_detect_start.
9. `rammp acquire`: over the piece, the plunge and the scoop, ending at scoop_end.
10. The delivery, one stage at a time:
    - `rammp deliver --stop-after swing`;
    - `rammp deliver --start-at present --stop-after present`;
    - `rammp deliver --start-at bite --stop-after bite`;
    - `rammp retract`.
11. `rammp home`.

## 10. Feeding

For each session:

1. Seat the user, and update `motion.free_space.user_mouth_m` if the seat moved. For a new seat, also capture
   `deliver` and set `delivery.second_look_m`.
2. Lay the bowl near `workspace.bowl_centre_m`.
3. Run `rammp check`.
4. Then either:
   - **under sheppy:** select `feeding` (add it to the profile, or select it in the TUI). The node runs `rammp feed`
     at once, attended, and exits when the session ends. Or
   - **by hand:** run `rammp feed --bites N`.

`rammp feed` reads the scene, goes over the bowl, and feeds bite after bite. It stops when the bowl is empty, the
user declines, or a bite is missed, and then returns home. The commands, options and exit codes are in the
[introduction](index.mdx).

## When to re-measure

| Change | Redo |
| --- | --- |
| Utensil changed or re-seated in the gripper | `glass_to_tip_m` with a ruler, then the wrist calibration |
| Arm power-cycled | The gripper re-initialises and can pivot the utensil: check the aim with `acquire --stop-after hover10`, and if it is off, the wrist calibration |
| Wrist camera or its bracket moved | The wrist calibration, then the scene calibration |
| Scene camera moved | The scene calibration |
| New table, or the arm moved against it | `table_extent_m`, `bowl_centre_m`, and the captures |
| New bowl | `floor_above_table_m`; `bowl_centre_m` and `scoop_end` if it sits elsewhere |
| New seat | `user_mouth_m`, `deliver`, `second_look_m` |

## Safety

- Keep a person at the e-stop for every command that moves the arm, and say what will move before running it.
- The e-stop does not hold the arm still: the Gen3 has moved 100–160 mm after one. Stay clear of the bowl and the
  user when stopping near them.
- After an e-stop, a power-cycle or a stop mid-move, the pose is unknown: run `rammp state`, then `rammp home`.
- The module never commands the gripper. It holds the utensil throughout.
- Stopping the module's node, or Ctrl-C on a command, cancels the move in flight, and the arm halts where it is.
