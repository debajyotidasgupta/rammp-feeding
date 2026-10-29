---
title: Porting to a new site
---

# Porting to a new site

How to bring `rammp_feeding` up on a new robot or at a new site. Work through the sections in order: each one
depends on the ones before it. Everything a site measures is kept in its **state directory** (mounted at `/config`),
never in the source tree, so one image serves every site.

| Step | Result |
| --- | --- |
| 1. Hardware and network | The arm and the Jetson on one subnet; DDS kept off the arm's link |
| 2. Deployment | The module's node in the sheppy manifest, with its mounts |
| 3. The machine file | `<state_dir>/local.yaml`: the site's profiles and hand measurements |
| 4. Hand measurements | Ruler readings and layout values |
| 5. Captures | `home.json`, `overlook.json`, `scoop_end.json`, `deliver.json` |
| 6. Calibrations | `wrist_mount.json`, then `scene_extrinsic.json` |
| 7. Verification | One bite, stage by stage |

Every command that moves the arm needs a person at the e-stop (see [Safety](#safety)).

## 1. Hardware and network

- [ ] **The arm** has a fixed address (the Gen3's default is `192.168.1.10`). The Jetson is on the same subnet with a
      fixed address of its own, for example:

      ```bash
      sudo nmcli con mod "<connection>" ipv4.method manual ipv4.addresses 192.168.1.20/24 ipv4.gateway 192.168.1.1
      sudo nmcli con up "<connection>"
      ```

- [ ] **The arm's link runs at 1 Gb/s.** Check the switch port's link light or `ethtool`. At 100 Mb/s, any stray
      traffic on that port starves the driver's 1 kHz control loop.
- [ ] **DDS receive buffers** are large enough for camera images. With the default 208 KB, SAM3 drops the ~900 KB
      colour frames. In `/etc/sysctl.d/60-ros-dds-buffers.conf`:

      ```
      net.core.rmem_max = 33554432
      net.core.rmem_default = 16777216
      ```

      Then apply them with `sudo sysctl --system`.

- [ ] **Cyclone DDS multicast is kept off the arm's network interface.** Without this, DDS sends the camera images to
      239.255.0.1 out of the arm's interface, about 13.6 MB/s while a command subscribes to the cameras. The switch
      floods them to the arm's port, and the arm driver exits ("timeout detected: BaseCyclicClient::Refresh").
      Drop that traffic on egress, and make the filter persistent, for example from a NetworkManager dispatcher
      script:

      ```bash
      sudo tc qdisc add dev <arm_nic> clsact
      sudo tc filter add dev <arm_nic> egress protocol ip flower dst_ip 239.255.0.0/16 action drop
      ```

      This keeps DDS on the host. Nodes running on other hosts would need DDS on another interface.
- [ ] **Nothing else commands the arm while the arm node runs.** That includes the Kinova Web App, the wrist buttons
      and a gamepad. The arm driver exits as soon as the arm leaves low-level servoing (`WRONG_SERVOING_MODE`).

## 2. Deployment

The module runs as one node in the sheppy manifest, transcribed from this repo's `rammp-alternative.yaml` (keep the
two the same).

- [ ] **The image** is pinned to a release tag: `ghcr.io/rammp-org/rammp-feeding:0.2.0`.
- [ ] **The node's mounts**: the state directory at `/config` and the run record at `/runs`, both host folders owned
      by the node's `user:` uid:

      ```yaml
      volumes:
        - /home/<user>/.config/rammp_feeding:/config
        - /home/<user>/rammp_runs:/runs
      ```

- [ ] **The planner node** mounts the same state directory at `/config`, read-only. `set_world` reads the module's
      collision world from a file there (`/config/planner_world.yaml`).
- [ ] **Selected alongside**: `arm`, `arm_state_publisher`, `planner`, `wrist_camera`, `scene_camera` and `sam3`.
      The module uses no other perception node.
- [ ] **The cameras' frames don't collide.** With two cameras of the same model, run the scene camera's driver with
      `camera_name scene_camera`. Its frames are then `scene_camera_*` and its topics `/scene_camera/...`, as
      `config/camera/*_scene.yaml` expects.

For single commands outside sheppy, a wrapper runs the image with the same mounts (see the [index](index.mdx)).

## 3. The machine file

The site's settings sit over `config/rammp.yaml` and the selected profiles. Keep a versioned copy as
`config/machines/<site>.yaml` and install it as `<state_dir>/local.yaml`.

- [ ] **The profiles**: the wrist camera (`config/camera/`), the scene camera, the utensil, the bowl and the food:

      ```yaml
      profiles:
        camera: realsense_d435i            # the wrist camera
        scene_camera: realsense_d435i_scene
        utensil: fork
        bowl: front
        food: cubes
      ```

      A new camera model needs a profile: its topics, `glass_to_optical_m` (its depth origin behind the front glass,
      from the datasheet), and for a scene camera its `link_frame` and `error_m`. Intrinsics are never configured:
      they come from the driver's `camera_info` and `/tf_static`.
- [ ] **The hand measurements** of section 4.
- [ ] **Site-specific limits**, if any. For example, the lab sets every `guards.torque_step_nm` to 10 N·m (see
      [notes/motion.md](notes/motion.md)).

`rammp check` reads this file. Until every value listed in `config.MEASURED` is set or measured, the commands that
need the utensil's tip refuse, naming the missing keys. `state`, `capture`, `calibrate-wrist` and `home --direct`
still run without them.

## 4. Hand measurements

| What | How | Key |
| --- | --- | --- |
| Wrist camera glass to the utensil's tines | A ruler from the camera's front glass straight out, along the camera's axis, to the level of the tines. Not along the utensil: a slanted reading came out 15 mm long. | `utensil.glass_to_tip_m` |
| The bowl's inner floor above the table | A ruler, inside the bowl. The plunge never aims under it, because a clear bowl's depth reads the table through its floor. | `bowl.floor_above_table_m` (bowl profile) |
| The ChArUco board as printed | Measure across all the squares and divide; printers scale. For a board not generated by `calibrate-wrist --board`, also its layout. | `perception.wrist_calibration.square_m`; `squares` (columns, rows), `marker_ratio`, `dictionary`, `legacy_pattern` |
| The table's footprint | Its edges in the arm's base frame (x forward, y left), in metres. The planner's table and the utensil's tip floor use it. | `workspace.table_extent_m` |
| Where the bowl is laid | Its centre in the base frame. Of the bowls the scene camera finds, the one nearest this point is taken. | `workspace.bowl_centre_m` |
| Where the user sits | Their mouth in the base frame. The planner's world puts the user's head and torso there; a mouth measured more than `delivery.seated_tol_m` away is refused. Fixed for a session. | `motion.free_space.user_mouth_m` |
| The second delivery look | The offset from the delivery look for the second look, in the base frame (to the user's right). | `delivery.second_look_m` |

The bowl's centre and rim radius, the table's height and the bowl's top are never entered by hand. The scene camera
reads them, and the wrist camera refines the centre and radius.

## 5. Captures

Each capture records the arm's current joints: `rammp capture <name>` writes `<state_dir>/<name>.json`. Pose the arm
with the arm node stopped, using the Kinova Web App or by hand, then restart the arm node and capture. Capture them in
this order: later poses are reached from earlier ones.

| Capture | Pose | Used as |
| --- | --- | --- |
| `home` | **feeding_start**: the utensil straight down, the arm parked beside the table and clear of the user. Leave room above it: `calibrate-wrist` rises 0.10 m and moves 0.15 m out from there. | The start and end of every session, and the calibration route's start |
| `overlook` | **food_detect_start's arm shape**: the utensil straight down over the bowl, the wrist camera looking down at it. | The orientation and arm shape of food_detect_start. Its position is computed each time: the tip over the bowl's centre, the camera `workspace.overlook_camera_height_m` (0.45 m) above the table. |
| `scoop_end` | **Where the scoop ends**: the utensil level, tines along the base's +y (the scoop turns the utensil 90° about the base's x axis from straight down), just past the bowl's rim, clear of the table. At most `acquisition.scoop.max_backoff_m` (0.30 m) across from where pieces are picked. | The scoop's end position (FK of its joints) and the arm shape its IK keeps. The swing to the user starts there. |
| `deliver` | **The delivery look**: the utensil level toward the user's mouth, the wrist camera on the user's face, the food out of its view of the mouth. | The swing's goal, and where the mouth is measured |

## 6. Calibrations

Both calibrations use one printed ChArUco board. `rammp calibrate-wrist --board` writes the sheet to print
(`<runs_dir>/charuco_board.pdf`, US Letter, 7 × 9 squares of 25 mm). Print it at 100 % and measure it
(section 4). A third-party board works if its layout is configured. Both commands start at feeding_start
(`rammp home` first) and end there.

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
   (`calibrate-wrist --plan` shows where pose 0 is).
2. Run `rammp calibrate-scene`. The scene camera finds the board where it lies. The wrist camera then runs the same
   route over it and places the board in the base frame through the measured mount. Keep the board still throughout.

The scene camera's link frame in the base frame goes to `<state_dir>/scene_extrinsic.json`. Every command then
publishes it on `/tf_static` as `base_link -> <link_frame>`. Neither the arm nor a starting guess needs to be in the
scene camera's view. Coarse is enough: the scene camera only places food_detect_start, and the wrist camera
re-centres over the bowl from there.

## 7. Verification

Run these in order. Each stage writes its images to the run record.

1. `rammp check`: every row OK, including `measured`, `poses` and `scene pose` (each names what is still missing).
   The robot model may WARN: kinova-gen3-ros2 1.1.0's `arm_state_publisher` cannot start ("package not found"), so
   the bundled copy of the driver's URDF is used. It is the same model.
2. `rammp state`: the joints, the tool, the tip and a utensil tilt of about 0°.
3. `rammp home`.
4. `rammp locate-scene`: open `scene_detection.jpg`. The bowl's box should be green, at its place.
5. `rammp detect --plan`, then `rammp detect`: the log gives the wrist camera's correction to the scene camera's
   bowl centre.
6. `rammp perceive`: open `detection.jpg`. Check the bowl overlay, the food boxes, and the utensil's region in grey.
7. `rammp acquire --stop-after hover10`: the tines should stop about 10 mm above the centre of the target piece. Look
   from two sides. If they are off, check `glass_to_tip_m` and re-run `calibrate-wrist`.
8. `rammp rise`.
9. `rammp acquire`, then `rammp deliver` one stage at a time (`--stop-after swing`, `present`, `bite`), then
   `rammp feed --bites 1`.

## When to re-measure

| Change | Redo |
| --- | --- |
| Utensil changed or re-seated in the gripper | `glass_to_tip_m` with a ruler, then `calibrate-wrist` |
| Arm power-cycled | The gripper re-initialises and can pivot the utensil: check the aim with `acquire --stop-after hover10`, and if it's off, `calibrate-wrist` |
| Wrist camera or its bracket moved | `calibrate-wrist`, then `calibrate-scene` |
| Scene camera moved | `calibrate-scene` |
| New table, or the arm moved against it | `table_extent_m`, and the captures that depend on it |
| New bowl | `floor_above_table_m`; `bowl_centre_m` and `scoop_end` if it sits elsewhere |
| New session | The user's seat (`user_mouth_m`), and `locate-scene` (`rammp feed` does it itself) |

## Safety

- Keep a person at the e-stop for every command that moves the arm, and say what will move before running it.
- The e-stop does not hold the arm still: the Gen3 has moved 100–160 mm after one. Stay clear of the bowl and the
  user when stopping near them.
- After an e-stop, a power-cycle or a stop mid-move, the pose is unknown: run `rammp state`, then `rammp home`.
- The module never commands the gripper. It holds the utensil throughout.
- Stopping the module's node, or Ctrl-C on a command, cancels the move in flight, and the arm halts where it is.
