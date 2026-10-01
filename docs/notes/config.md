# config/rammp.yaml: where the numbers came from

- `deployment.arm.soft_limits.joint_speed_dps` [70 x4, 63 x3]: set by the user 2026-09-25 (was 40 everywhere):
  J1-J4 10 deg/s under their 80 hard limit, J5-J7 10% under their 70.
- `deployment.servers.startup_timeout_s` 300: on 2026-09-25 the detector took over 120 s to warm.
- `motion.joint_accel_cap_dps2` 72: was 40, then 60; with the faster speeds of 2026-09-25 short moves never reached cruise.
- `motion.controller_accel_factor` 1.6: the controller measured 1.55-1.56x the time scaling's estimate (115 vs 72,
  84 vs 54 deg/s^2, 2026-09-25) and refused; each retry then halved the move.
- `motion.settle_dps` 10 (was 5); `goto_min_s` 0.8 / `goto_pad_s` 0.3 (were 1.5 / 1.0 s: at 65 deg/s the pad was a
  third of every swing).
- `motion.free_space.dps` 61 (and `delivery.swing_dps`, `delivery.retract.home_dps`): just under 0.98 x the slowest
  joint's soft limit (63). The scripted swings went 20 -> 44 -> 65 -> 61 deg/s (2026-09-24/25).
- `guards.torque_step_nm.approach` 4.0: the user sitting down shook the wheelchair mount past 2.5.
- `guards.torque_step_nm.scoop` 6.0: 2.5 tripped on the arm's own acceleration.
- `guards.torque_step_nm.free_space` 6.0: at 65 deg/s the braking on arrival stepped J2 -6.6 -> -3.4 N.m
  (2026-09-25, nothing touched); likewise the descent into feeding_start (J4 -6.9 -> -4.0 N.m at 0.22 m/s).
- `acquisition.over_speed_mps` 0.15: was 0.05, then 0.11.
- `acquisition.plunge.no_floor.floor_max_above_table_m` 0.025: the round bowl's floor reads 7 mm above the table;
  packed pieces read 40+ mm (2026-09-25). `depth_below_top_m` 0.025: the user, 2026-09-25 (25-30 mm).
- `acquisition.scoop.speed_dps` 55: x2.2 the earlier 25 (2026-09-25, user: faster everywhere but the plunge).
- `perception.scene.bowl_prompts`: from the side a bowl reads as a dish or a plate with a jar; "bowl" alone boxed a
  lidded box 1 m off (2026-09-30). Stopping at the first prompt that finds a reachable bowl: 4.4 s.
- `perception.depth_tries` 3: a single 10 s wait for a dropped stream cost bite 2 of 2026-09-25 16 s; the stream
  drops and recovers in under 1 s.
- `delivery.aim_dz_m` -0.010: the detected mouth felt nearer the nose (user, 2026-09-24).
- `delivery.approach_mps` 0.077 (was 0.035) and `sidestep_mps` 0.044 (was 0.02): x2.2, 2026-09-25.
- `delivery.mouth_check.depth_range_m`: on 2026-09-25 a stale depth frame put the mouth at 921 mm (the face was ~0.52 m).
- `delivery.stop_m` 0.080 (not 0.050): safe for any cube-sized piece (user, 2026-09-24).
- `delivery.lift_m` 0.175: at +60 mm the swing touched the bowl and the head sat at the top of the image.
- `workspace.overlook_camera_height_m` 0.45: the user asked for a lower arm (2026-09-30: the old 0.545 m lift looked
  unnatural; 35 and 40 cm alike for them, "not a hard constraint"). 0.35 sat inside cuRobo's buffer of the rim; 0.40
  planned, but tests/scene_error_study.py (2026-10-01, 25 sampled extrinsic errors per placement, up to 5 cm and
  1.5 deg, 4 placements incl. the site's) showed 1-3 of 25 paths per camera touching the TRUE bowl / table when the
  reading is trusted, and almost no plans at all with the 5 cm margin. At 0.45 with the margin: 25/25 planned for
  every placement, none touching the truth, the tip >= 66 mm clear, the true bowl always in the wrist camera's view
  and within OVER's 10 cm; the tip's peak 22-25 cm above the table (the old fixed lift: ~43 cm above this table).
  Raw rows: docs/notes/scene_error_study_20261001.json.

# Profiles (config/utensil, config/food)

- fork.yaml `tip_px` [690, 436]: 2026-09-25 18:00, after the arm's power-cycle re-seated the fork, the tips moved
  ~35 px up in the overlook image (was (685, 476), set at 16:3x). Before that, (644, 405) put the tips 34.5 mm off
  the camera's vertical where they were ~49 mm off (the white fork's top 12 rows, x 657..724): a cube plunge landed
  ~15 mm -X, ~12 mm -Y of centre. fork_20260924.yaml keeps the old values. On 2026-09-24 a robot restart pivoted
  the fork by (-14, -65) px.
- fork.yaml `camera_to_tip_m` 0.265: measured 260-261 (ruler to the bare table + depth; the camera case + 6.7 mm),
  stored long on purpose, so every stop errs high: a nominal 10 mm hover looks like 15-20 mm by eye.
- fork.yaml `tine_offset_tool_m` -5 mm x: the user at a 10 mm hover on a cube, 2026-09-25: base -5 mm y at the
  overlook = tool -5 mm x (2026-09-24's -6 mm aim offset was the same effect).
- fork.yaml `head_width_m` 0.023: the user's ruler, 2026-09-25; it spans ~110 px (x 640..750) = 22.5 mm at camera_to_tip.
- fork.yaml `by_camera.realsense_d405`: 2026-09-30, a spoon, run as the fork by the user's choice ("nothing changes
  ... it behaves like a fork"); 0.26 m camera-to-tip measured by the user with the D405.
- cubes.yaml `prompt` "piece of food": on two packed-cube frames (2026-09-25) one box per piece in 1.1 s, as good as
  "fruit cube"; "food pieces" / "food cube" / "bite-sized piece" merged pieces or boxed the whole container. The
  fallback prompt exists for a pile read as one box (2026-09-25).
- cubes.yaml `aim_offset_base_m` 0: the strawberry offset put one side of the fork past a cube's corner (2026-09-25).
