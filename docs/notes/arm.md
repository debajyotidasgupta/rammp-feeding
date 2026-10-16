# Arm-side notes

History and rationale moved out of the code comments (2026-10-01). Grouped by file.

## kinova/scripts/acquire.py

- Frozen 2026-09-23 after it ran end to end.
- Measured on the arm 2026-09-23, fork on a strawberry: OVER put the berry 1.2 mm from the fork's vertical line;
  the stepped approach re-measured at a 30 mm hover and stopped 10 mm above (user-confirmed); punch went 35.0 mm
  down and stopped ~11 mm above the floor as planned; scoop raised the berry point 155.7 mm, 0.2 mm off vertical,
  fork horizontal ("perfectly correct", user).
- 2026-09-23: the berry FELL in that scoop because it was too slow (6 deg/s, 25 s): food slides under gravity as
  the utensil tilts. Hence `acquisition.scoop.speed_dps` fast on purpose. The user also placed the aim 5 mm off
  along Y (now applied in OVER from calibration: `cfg.aim_offset()`) and said the punch can go deeper: floor + 5 mm.
- `utensil.punch_floor_margin_m` is nominal; the real stop was ~5-10 mm because `camera_to_tip` was stored long.
- `acquisition.plunge.slow_above_food_m`: the overlook read the food TOP 4-7 mm high; the floor read from the
  overlook matched the later close re-measures to within 1 mm on every run (why the stepped route re-measured,
  and why the slow zone starts above the top).
- plunge as one move: user, 2026-09-24, no stopping on the way down (OVER -> punch in one smooth trajectory,
  easing to ~5 mm/s through the food).
- pick in one process: user, 2026-09-25, no program restarts/reconnection so the arm does not stand still between
  steps. The floor is still re-measured from above the berry after OVER: read from the overlook frame it disagreed
  by up to 11 mm on 2 of 17 runs.
- `under_over` handed on in memory, not via the `--under` file: 2026-09-25, the first feed after a stepped-route
  test stopped because the shell's host-owned file could not be overwritten by the container's root.
- `acquisition.plunge.refuse_if_floor_filled_over`: the Depth Anything 3 fallback gets the floor level right
  (median ~1 mm) but its local shape can be 25-60 mm off.
- `guards.torque_step_nm.scoop`: 2.5 N.m tripped on the arm's own acceleration in open air.
- `acquisition.scoop.rot_exp` < 1: user, 2026-09-24, at 1.6 the food rose steep first and gravity worked on it.

## kinova/scripts/safe_move.py

- UTENSIL DOWN, 2026-09-23: a diagnostic moved J6 alone while J2/J4 stayed put (-4.10 - 57.08 - 40.08 = -101.3)
  and the fork swung 79 deg off vertical. Hence complete 7-joint targets only, and the pitch-sum check.
- CONTACT, same session: the fork tines swung into the arm's own joint 6 housing and broke, with the tool angle
  legal throughout. sim/safety.py (a self-collision model) was deleted in a cleanup and not restored for the new
  utensil, so the torque step is the contact detector.
- JOINT LIMITS: four retries were lost to J6 = -129.6 against a +-120.3 limit before targets were checked; Kortex
  gives no error, no clamp.
- Refused multi-joint moves came back as a bare METHOD_FAILED whenever one joint had a sliver of travel. Several
  wrong theories (angle wrap encoding, the deprecated API, the gamepad) were tested and disproved before
  `GetTrajectoryErrorReport()` was read: INVALID_JOINT_ACCELERATION on J3, 0.7 deg of travel, 8.0 requested,
  7.56 max. Fix: JOINT_CONSTRAINT_DURATION instead of a speed constraint (under a per-joint speed limit a
  short-travel joint finishes early and J2+J4+J6 wanders).
- ExecuteAction is used rather than PlayJointTrajectory/PlayCartesianTrajectory, deprecated in this SDK.
- SERVOING_MANUALLY_CONTROLLED: an earlier version refused to move in that state and blamed the gamepad; the
  gamepad was off and blameless. The state appears after any trajectory, including ours.
- INVALID_DURATION minimum is not stable between tries: a 75 deg swing half was refused at 1.69 s, then 1.93 s for
  the re-sent one (`motion.goto_duration_retries`).
- `_braked_on_arrival`: torque steps from the arm's own braking seen at 65 deg/s and 0.22 m/s, nothing touched.
- `check_tool`, 2026-09-30: another user's 200 mm tool frame (1.025 kg; ours 120 mm / 0.831 kg) was left on the
  arm and every cartesian target came out 80 mm low.
- ContactGuard (unused since the plunge became one joint trajectory watched by TorqueGuard). Measured on this arm:
  - a force LEVEL threshold is unusable: fz drifts +1.0 .. -2.4 N over a 170 mm descent as the arm changes shape;
    2 N tripped after 5 mm of free air;
  - a RATE threshold (N/s) is speed-dependent: 0.8 N/s fired falsely at 20 mm/s on drift of 1.2 N/s;
  - per mm the drift is ~0.014 N/mm typical, 0.06 N/mm worst over a short window; strawberry flesh 0.15-0.27 N/mm,
    a rigid surface far more: 0.10 N/mm sits between;
  - every move starts with a ~4 N/s kick lasting ~0.4 s (acceleration), hence `settle_s`;
  - upward push on the tip reads +fz (finger-press test).

## kinova/motion.py

- `_time_scale` per-joint acceleration: the earlier leading-joint profile let J1 go from ~0 (the scripted lift)
  to full speed (the swing) in one segment, 628 deg/s^2; the controller refused the lift+swing trajectory at every
  speed down to 7 deg/s. (The scripted lift and swing are gone: free-space moves are cuRobo paths now.)
- `motion.accel_retries`: a fixed `controller_accel_factor` (1.6) was not enough for every path: the retract
  measured 93-97 vs 80 deg/s^2, 2026-09-25. Hence rescaling by the controller's reported ratio.
- `homotopy` limit steering: without it the (former scripted) swing only fitted 60 mm of retraction.
- `fk_tip` retries: ComputeForwardKinematics failed once (METHOD_FAILED) right after a trajectory.
- 2026-10-01: the `ramp` parameter was removed from `run_path` / `_waypoints`.

## kinova/scripts/preflight.py, arm_tool.py

- tool check, 2026-09-30: a 200 mm frame / 1.025 kg (ours 120 mm / 0.831 kg) left by another user; make detect's
  (then scripted) lift went 80 mm off.
- other clients, 2026-09-30: another project's driver connected to the arm mid-session.
- kinematics, 2026-09-30: at the same joints the reported tool pose differed by 3.2 deg / 8 mm from the capture:
  the captured poses were stale.

## kinova/scripts/arm_limits.py

- 2026-09-25: the trajectory modes read 40 deg/s (joystick 50), set by whoever configured the arm before this repo;
  raised to 70 deg/s at the user's request (`deployment.arm.soft_limits.joint_speed_dps`). Hard limits: 80 deg/s
  J1-J4, 70 J5-J7; accelerations 5.2 / 10 rad/s^2.
- 2026-10-02 (abra): the other team had the trajectory modes at 25 deg/s (ANGULAR_TRAJECTORY accel 50 deg/s^2,
  which --apply does not set); set to ours with the user's OK. --apply saved nothing then (the run record only exists
  under a stage script's RAMMP_RECORD); it now always writes <runs_dir>/arm_limits_before_<time>.json, and
  --restore puts a saved file back (their values: abra's rammp_runs/arm_limits_before_20261002_110500.json).
- 2026-10-02: the planner container lacked the external calibration files (the wrist mount the tip model reads):
  `make planner` crashed on abra; it now mounts them as the driver does.

## kinova/scripts/capture_home.py

- The start pose is frozen rather than perceived because the face camera, mounted on the wheelchair armrest at
  deployment, carries ~100 mm of error and cannot seed a pose.
- Joints 1/3/5/7 are continuous and legitimately sit near 0/360, hence the fold before averaging.

## docker/Dockerfile.driver

- numpy is pinned as on the 2026-09 deployment; the golden arm trace (tests/arm_trace.py, since removed) was
  float-exact against it.

## docker/Dockerfile.vision

- The depth colorizer (depth_stream.py) used to be baked in so a container restart could not lose it (it had lived
  in /tmp and vanished on recreate); removed with depth_stream.py.
- RTSP reconnect crash: "Could not get frame" -> PAUSED -> started, then ParameterAlreadyDeclaredException.

## docker/Dockerfile.planner_x86

- Was docker/Dockerfile.curobo_eval (cascadilla RTX 3070, 2026-10-01), run with tests/curobo_eval.py (removed).

## calibration/README.md

- Captured from the device 2026-09-22 with kinova/scripts/camera_intrinsics.py (removed),
  `VisionConfig.GetIntrinsicParametersProfile`, vision device 9.
- Device vs package default: depth fx, fy 334.810 vs 360.013 (+7.53% shipped); depth cx 238.536 vs 243.872
  (+5.34 px); cy 139.694 vs 137.922 (-1.77 px); colour identical to 3 dp at all resolutions. The scale error:
  ~2 mm at the plate centre, ~8 mm at the plate edge and at face range (a food cube).
- 16:9 colour profiles: the device reports a principal point that is not centred (cy/h = 0.331 at 720p, 0.090 at
  1080p) and the 1080p value is not consistent with any crop of a 4:3 sensor. (Pointed to
  docs/camera-calibration.md, which does not exist.) The colour cx/cy in config/camera/kinova_wrist.yaml is now
  measured.
- calibration/colour_intrinsics.json was deleted: the intrinsics live in config/camera/*.yaml.

## setup_kortex_sdk.sh

- Everything else (Dockerfiles, scripts) is versioned, so the SDK fetch is the only per-host step.
