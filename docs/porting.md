---
title: Deployment checklist
---

# Deployment checklist

From a new Jetson to `rammp feed`, in order.
- Site state lives in `/srv/rammp_feeding/config` (`/config` in the container), and runs in `/srv/rammp_feeding/runs`.
- The module runs as uid 1000.
- Positions are metres in `arm_mount`, the arm model's gravity-aligned root (x forward, y left, z up).
- Keep someone at the e-stop for every command that moves the arm.

## 1. Hardware
- An NVIDIA Jetson AGX Orin (arm64) with Docker and the NVIDIA runtime.
- A Kinova Gen3 7-DoF with a Robotiq 2F-85, on the arm stack's 2.0 generation (kinova-gen3-ros2 2.0.0, rammp-curobo
  2.0.1). rammp-feeding 0.2.x is for the 1.1.0 stack.
- A fork in the gripper. The module never opens or closes it.
- A wrist RGB-D camera (RealSense D405 or D435i), and a fixed scene camera that sees the table (Orbbec Gemini 336L or
  D435i). Another model needs its own `config/camera/` profile. Intrinsics always come from `camera_info`.
- A table and a bowl, with the food spaced apart. A tape measure, and a printer for the ChArUco board.

## 2. Jetson and network
- Isolate the arm driver's core: add `isolcpus=11 nohz_full=11 rcu_nocbs=11` to `/boot/extlinux/extlinux.conf`, then
  reboot.
- The arm is at `192.168.1.10`; give the Jetson's arm NIC a fixed address on that subnet (e.g. `192.168.1.20/24`).
  The link must run at 1 Gb/s.
- DDS receive buffer: `echo 'net.core.rmem_max=33554432' | sudo tee /etc/sysctl.d/60-rammp-dds.conf && sudo sysctl --system`.
- No multicast out of the arm NIC; otherwise the driver exits. Re-apply this from a NetworkManager dispatcher script,
  or use rammp-deployments' `config/hosts/jetson-agx-orin/tuning.sh install`:
  ```bash
  sudo tc qdisc add dev <arm_nic> clsact
  sudo tc filter add dev <arm_nic> egress prio 10 protocol ip flower dst_ip 224.0.0.0/4 action drop
  ```
- Nothing else may command the arm while its driver runs: not the Web App, the wrist buttons, or a gamepad.

## 3. Software
```bash
uv tool install "sheppy @ git+https://github.com/rammp-org/sheppy@dev"
git clone https://github.com/rammp-org/rammp-deployments.git
docker pull debajyotidasgupta/rammp-feeding:0.4.2
sudo mkdir -p /srv/rammp_feeding/config /srv/rammp_feeding/runs && sudo chown -R 1000:1000 /srv/rammp_feeding
```
- SAM3's weights are gated: accept the `facebook/sam3` licence on Hugging Face and download the snapshot to the Jetson.
- Save this as `/usr/local/bin/rammp`, make it executable, and never run it while sheppy's `feeding` node is up:
  ```bash
  #!/usr/bin/env bash
  tty=-i; [ -t 0 ] && tty=-it
  exec docker run --rm $tty --network host --ipc host --user 1000:1000 -e ROS_DOMAIN_ID=0 -e HOME=/tmp \
    -v /srv/rammp_feeding/config:/config -v /srv/rammp_feeding/runs:/runs \
    debajyotidasgupta/rammp-feeding:0.4.2 ros2 run rammp_feeding rammp "$@"
  ```

## 4. The deployment manifest
Edit rammp-deployments' `december_2026/sheppy-manifest.yaml`:
- **feeding:** the `feeding` node, copied from this repo's `rammp-alternative.yaml`.
- **planner:** both `curobo_planner` alternatives mount `/srv/rammp_feeding/config:/feeding:ro`. Our world reaches the
  planner as a file there; the planner's own world comes back through `/reset`.
- **arm:** the driver's own contact stop (`contact_detection`, `enforce` by default) stops and latches the arm until
  `/clear_faults`. Set it to `observe` in the profile's overrides if it should not stop the plunge or the scoop.
- **sam3:** its weights mount points at the snapshot: `<snapshot_dir>:/models/sam3:ro`.
- **scene camera:** runs as `/scene_camera`, with `camera_name: scene_camera`.
- **The robot:** `config/robots/<robot>/` holds `mount.yaml` (`robot: <robot>`, `rpy_deg` of the mount) and the
  planner's `curobo/` (as dojo's). Activate it with `uv run scripts/set_robot.py december_2026/config/robots/<robot>`.
- **The profile:** `profiles/<robot>.yaml` selects `kinova_gen3_driver`, `kinova_gen3_description`, `curobo_planner`,
  the two cameras with their serials, and `sam3_docker`. Leave `feeding` out until section 7. Check it with
  `uv run scripts/validate_manifest.py`.

## 5. The site file
Copy `config/machines/template.yaml` to `/srv/rammp_feeding/config/local.yaml` and fill in every REQUIRED value:

| Key | How |
| --- | --- |
| `profiles.camera`, `profiles.scene_camera` | Your camera models |
| `utensil.glass_to_tip_m` | Ruler from the wrist camera's glass to the fork tips, **along the camera's axis** |
| `workspace.table_extent_m` | Tape from the arm's base: `{x: [near, far], y: [right, left]}` |
| `workspace.bowl_centre_m` | Where the bowl is laid, `[x, y]`. `rammp setup` can write it from the wrist camera. |
| `perception.wrist_calibration.square_m` | The printed board's square, measured across all of them |

`bowl.floor_above_table_m` (5 mm) and `utensil.punch_floor_margin_m` (5 mm) are defaults. Change them only if the
plunge stops short (section 7).

A state file the node cannot read (a YAML typo in `local.yaml`, a damaged JSON) counts as missing: the node still
starts, its log names the file at startup, `rammp check` fails on it, and every action answers `not_set_up` until it
is fixed or redone.

## 6. Calibrate and capture
`sheppy up <robot> --manifest december_2026/sheppy-manifest.yaml`, without `feeding`.

To pose the arm by hand: stop the arm node, move the arm with the Kinova Web App, then restart the node.

1. `rammp capture home`: feeding_start, with the fork straight down and the arm parked beside the table, clear of the
   user.
2. `rammp calibrate-wrist --board` writes the sheet to print. Then run `rammp calibrate-wrist`: lay the board under the
   arm's first pose and keep it still. It measures the wrist camera's mount and the fork's pixels.
3. `rammp setup`. It walks through the rest, each step asking first:
   - **site:** the site values, to confirm;
   - **home:** recapture feeding_start;
   - **scene:** the scene camera's pose, through the board;
   - **overlook:** pose the arm over the bowl, the fork straight down;
   - **bowl:** the scene camera, then the wrist camera, find the bowl, and its centre can be written to `local.yaml`;
   - **deliver:** pose the arm facing the seated user, the fork level and the camera on their face. It checks that a
     face is in view.

   Then it goes home and runs `rammp check`.
4. `rammp check`: every row OK.

If the setup was calibrated elsewhere, write the two files instead of running the calibrations (see below).

## 7. Verify, then feed
1. `rammp detect`, `rammp perceive`, then `rammp acquire --stop-after hover10`: the tips should stop about 10 mm above
   the piece's centre. If not, re-measure `glass_to_tip_m` and run `calibrate-wrist` again. Then `rammp rise`.
2. `rammp detect`, `rammp perceive`, `rammp acquire`. Open `acquire/plunge_end.jpg` (the run record: `record:
   {enabled: true}` in `local.yaml` while you set up) and look from the side: the tines should be well into the piece.
   If they only touch it, lower `punch_floor_margin_m`.
3. `rammp deliver`, seated facing the delivery look.
4. `rammp feed --bites N` by hand. Then add `feeding` to the sheppy profile: the orchestrator starts its actions
   (standalone, `ros2 action send_goal /feeding/execute_action rammp_orchestrator_interfaces/action/ExecuteAction
   '{engine_id: feeding, action_id: start_meal}'`, then `feed`).

## When the setup changes

| Change | Redo |
| --- | --- |
| The arm moved against the table or the user | `table_extent_m`, then `rammp setup` |
| Only the user's seat | `rammp setup deliver` |
| A new bowl position | `rammp setup bowl` |
| The scene camera moved | `rammp setup scene` |
| The fork changed or re-seated, or the arm power-cycled | Check the 10 mm hover. If it is off, `glass_to_tip_m` and `calibrate-wrist` |
| The wrist camera moved | `calibrate-wrist`, then `rammp setup scene` |

## Entering an existing calibration
`<state_dir>/wrist_mount.json`. The pose is of the colour optical frame in the tool frame, which is the flange plus
0.12 m along z. The quaternion is (x, y, z, w). The pixels are in the image as the driver publishes it.
```json
{"camera": "<profiles.camera>", "utensil": "<profiles.utensil>", "camera_in_tool_m": [x, y, z],
 "rotation_quat_xyzw": [qx, qy, qz, qw], "tip_px": [u, v], "image_region_px": [x1, y1, x2, y2]}
```
`<state_dir>/scene_extrinsic.json`. The pose is of the scene camera's link frame in `arm_mount`:
```json
{"parent_frame": "arm_mount", "child_frame": "scene_camera_link", "xyz": [x, y, z], "quat_xyzw": [qx, qy, qz, qw]}
```
Common mistakes:
- a (w, x, y, z) quaternion;
- millimetres instead of metres;
- profile names that don't match `local.yaml`;
- the camera's body frame given for its optical frame.

Verify with the 10 mm hover.

## Safety
- The e-stop does not hold the arm still: it has moved 100–160 mm after one.
- After an e-stop, a power-cycle or a stop mid-move: `rammp state`, then `rammp home`.
- Stopping the node, or Ctrl-C, cancels the move in flight.
