# rammp — robot-assisted feeding, Kinova Gen3

The arm is a **Kinova Gen3 7-DoF** on a wheelchair mount at the user's right,
holding a **fixed utensil** in a Robotiq 2F-85 — a fork originally, a spoon since
2026-09-23. "Utensil" means whatever is clamped there; it must be able to pierce
the food. There is **no wrist force/torque sensor**: all force sensing is joint
torque. Compute is a **Jetson AGX Orin** running the driver in a container.

## Commands

```
make status      what is running, is the arm reachable
make state       where is the arm right now

make start       back to feeding_start       (retraces the outbound path)
make start-direct  straight to feeding_start from anywhere, shortest joint path (SPEED=10: deg/s; clear workspace only)
make detect      feeding_start -> food_detect_start
make perceive    at food_detect_start: capture, detect, depth, target
make acquire     over -> plunge -> scoop
                 (STOP_AFTER=over|plunge; the old stepped route for aim checks:
                  STOP_AFTER=hover30|hover10, then START_AT=punch)
make deliver     swing -> present -> bite -> retract   (from the end of the scoop)
                 (STOP_AFTER=swing|present|bite, START_AT=present|bite|retract, VALIDATE=1)
make retract     back from the user to food_detect_start
make feed        the whole loop from feeding_start: bite after bite until the bowl is empty or the
                 user does not open their mouth, then back to feeding_start (BITES=1: one bite)

make review      the latest session: timeline, key numbers, contact sheet per bite (RUN=<session|bite>)
make locate-bowl where the bowl and the table are, from the latest perceive (to set `workspace:`)

make twin        the digital twin: replay a recorded bite in MuJoCo -> sim/out/*.mp4 (on the Mac; BITE=3|all,
                 SESSION=<run dir>, STILL=<s> for one frame, SPEED=1.5); make twin-poses: the calibrated poses

make models      every model loaded and warm (detector + DA3, face + colour stream, depth); make feed runs it
make camera      the selected camera behind the camera server (make models starts it too)
make depth       the Kinova vision nodes: two depth sessions, 31 s apart (the arm ends each every ~62 s)
make limits      the arm's own speed limits (APPLY=1: set them from config deployment.arm.soft_limits)
make preflight   before any motion: tool, faults, kinematics vs the capture, speed limits, servers (FIX=1
                 sets our tool + limits). make feed runs it first; nothing moves
make tool        the arm's tool frame + mass vs ours (APPLY=1: set ours, saving the arm's; RESTORE=<file>).
                 Every move refuses while they differ: another user's tool frame shifts every cartesian target
```

Frozen 2026-09-24 (tag `acquire-v1`): two consecutive full runs, fork on strawberries, both picked up and held ("the most perfect scoop"). Run to run: OVER 82/81 mm, punch 45.3/44.3 mm down, scoop 6.3/6.2 s, rise 164.5/163.8 mm, 1.0/1.0 mm off vertical.

## Configuration (`config/`, read by `settings.py`)

Every number that is tuned or changes with the deployment, and every on/off switch, is in YAML -- not in the code:

```
config/rammp.yaml             deployment (arm IP, containers, venvs, ports, models), features (on/off),
                              motion limits, contact-guard thresholds, acquisition / perception / delivery tunables
config/camera/<name>.yaml     intrinsics, depth->colour extrinsic, depth scale/offset correction, camera-in-tool, streams
config/utensil/<name>.yaml    camera-to-tip, tip pixel, image region, punch margin
config/food/<name>.yaml       the detector prompt, the aim offset
config/bowl/<name>.yaml       the overlook pose over it, where the scoop ends, where the swing may start, and
                              `captured_at`: where the table and bowl were when those poses were captured
config/local.yaml             per-machine overrides (not committed; same keys as rammp.yaml)
```

**The table and the bowl can move** (`rammp.yaml: workspace`, `kinova/scene.py`). `workspace.table_height_m`
and `workspace.bowl_centre_m` say where they are now; every bowl-dependent pose -- food_detect_start, the scoop
end, the retract's descent -- is the captured one shifted by the difference from the bowl profile's
`captured_at`, with the camera's orientation and height above the bowl unchanged (so every orientation-only
calibration still holds). Unshifted, the captured joints are used exactly; shifted, food_detect_start's joints
come from the controller's IK seeded with them, and a shift that would need a different arm shape, come near a
joint limit, or is out of reach is refused **before anything moves** -- re-capture the pose there. Heights that
depend on the table are table-relative: the retract's lowest utensil tip (86 mm above it) and the lift's
clearance over it (250 mm). Measured 2026-09-24 at this overlook (`make locate-bowl`, three runs, +-2 mm): bowl
centre (0.2692, 0.2765), floor z 0.221, table z 0.214 beside the bowl.

```
make perceive && make locate-bowl                     # where they are, and the YAML to paste if they moved
RAMMP_SET="workspace.bowl_centre_m=[0.29, 0.27]" make feed    # or put it in config/local.yaml
```

Later wins: `rammp.yaml` < the selected profiles < `local.yaml` < `RAMMP_SET`. Profiles are chosen in
`rammp.yaml: profiles` or per run:

```
RAMMP_UTENSIL=spoon make feed                         # a new utensil: copy config/utensil/fork.yaml, re-measure
RAMMP_BOWL=side RAMMP_CAMERA=realsense make perceive
RAMMP_SET="delivery.stop_m=0.06,features.wait_for_open_mouth=false" make deliver
python3 -m settings get delivery.bite_n               # any key, from the shell
python3 -m settings dump                              # the merged result
```

Python reads it as `from settings import cfg` (`cfg.delivery.bite_n`, `cfg.get("motion.joint_limits_deg.2")`).
Shared pieces: `feeding/camera.py` (intrinsics, depth correction, colour<->depth pixel mapping),
`kinova/motion.py` (IK, time scaling, homotopy, validated and watched trajectories), `kinova/scene.py` (the
table and bowl) and `runlog.py` (logging and the run record) -- stages import these rather than re-defining them.

## The run record (`feeding/lib.sh`, `runlog.py`, `make review`)

Every run leaves everything needed to debug it afterwards, without re-running anything:

```
~/rammp_runs/<date>_<what>/          a session (make feed; or a stage run on its own)
  session.json  config.yaml          when, host, git commit + dirty files, profiles, workspace; the merged config
  console.log                        everything printed, each line timestamped (ms)
  events.jsonl                       stage start / end, exit codes, durations, bite results
  moves/arm/                         detect / home: telemetry per move
  bite_01/perceive/                  colour, depth (+ .png), the DA3 fill (+ _filled.png), detection, target
  bite_01/acquire/                   under_*.json / .png (where the floor was read), scoop_log.json, and arm/:
                                       plan_*.json      every validated trajectory (waypoints, durations, speed)
                                       *_arm_*.npz      telemetry per move: t, joints, velocities, torques, tool
                                                        pose, tool force (the watch loops' own samples: no extra
                                                        controller calls, no delay)
                                       *_depth_at_over.npz, *_under_over.json, *_scoop_food_path.json
                                       events.jsonl, log.txt (every arm log line, timestamped)
  bite_01/deliver/look_N/            each look at the face: colour, depth, face.json, face_upright.jpg,
                                       open_wait.json (mouth opening over time), open_wait_last.jpg (declined)
  bite_01/deliver/arm/               swing / approach / bite force trace / retract
  bite_01/summary.json, contact_sheet.jpg     written by make feed after each bite (in the background)
~/rammp_runs/latest -> the current bite (make acquire / deliver after make perceive attach to it)
```

`make review` (or `python -m feeding.review <session|bite> [--brief] [--write]`) prints each bite's stage
times, detections, target, floor reading, plunge, scoop, approach, bite and retract numbers, every refusal with
its reason, and writes a contact sheet: every image of the bite plus a torque plot per move and the bite force
against its threshold. Recording never raises and never slows a move; without RAMMP_RECORD (a module run by
hand, the tests) everything still logs to the console and nothing is written.

## The colour stream (`feeding/colour_stream.py`, in the face server)

The face server (`make models`) keeps the wrist camera's colour stream open: the open-mouth wait reads frames as
they arrive (no ~1.6 s stream set-up per look, no stale frames) and `grab_frame` gets a fresh JPEG from it in
~0.05 s (`GET :8766/frame`; `GET :8766/stream` = its status). Every dropout -- a gap over 0.5 s, a lost stream, a
reconnect -- goes to `~/rammp_runs/colour_stream.jsonl` (on 2026-09-25 the camera stopped delivering colour for
over a minute with its RTSP server still answering). Each wait also records its frames' ages and a snapshot
every 0.5 s (`look_N/wait_frames/`).

## Tests (no arm needed)

```
python tests/arm_trace.py /tmp/t.json --compare tests/golden/arm_trace.json.gz   # the arm stages, on a fake controller
bash tests/regress.sh check          # (jetson) perception / measurement / face outputs on saved runs
python3 tests/arm_values.py          # (driver container) every arm-side constant and CLI default
bash tests/record_smoke.sh           # (jetson) the run record end to end, into a scratch directory
.venv/bin/python tests/twin_check.py # (mac) the twin's arm vs the controller's tool pose on a recorded session
```

`tests/fake_kortex.py` stands in for the controller (IK / FK / validation / execution on a simulated clock), so
every stage -- including refusals, the controller refusing a speed, a torque stop, cache reuse, a moved bowl --
runs deterministically in ~1 min; the trace records every request the stages send it, their output, exit codes
and state files. A matching trace means a change altered nothing the arm or the operator would see. State files
go to a scratch directory (`RAMMP_STATE_DIR`), never the arm's live `/tmp`.

## Acquisition (`make acquire` -> `feeding/acquire.sh` -> `kinova/scripts/acquire.py`)

| step | what | measured |
|---|---|---|
| OVER | translate at the overlook height until the tip's vertical line meets the food | 82 mm, aim user-confirmed at the hover |
| PLUNGE | **one smooth move** from the overlook to the **measured bowl floor** (nominal; really ~5–10 mm above it): 20 mm/s, easing to 5 mm/s from 20 mm above the food top | 2026-09-24: 173 mm in 20.2 s, same end depth as the stepped punch |
| SCOOP | food rises while the utensil turns vertical -> horizontal (+Y); the turn **starts the moment the food lifts** (90°·s^0.8). With the bowl in front of the user it also **backs off to the old scoop end** (`scoop_end`, 206 mm), mostly after the first 20% |  6.3 s at 20 °/s; rose 164.5 mm, 1.0 mm off vertical; berry held |

**Packed food** (`acquisition.plunge.no_floor`, 2026-09-25). When no bowl floor shows beside the target -- the
floor ring reads the neighbouring pieces, so the food looks under 10 mm tall, or there is no floor reading at
all, or the "floor" is more than 25 mm above the table (`floor_max_above_table_m`: a real bowl floor sits a few mm
above it) -- the plunge goes `depth_below_top_m` (25 mm) below the target's top instead of to the floor, never closer
than 10 mm to the table (`workspace.table_height_m`: shortened to that, refused if that leaves < 10 mm in the
food), and never on a top read from the Depth Anything 3 fallback. The scoop then takes the tip as the piece's
bottom. The torque guard still stops the plunge on contact.

**Why one plunge, not steps.** The stepped route stopped at 30 and 10 mm to re-measure, because the overlook
reads the food TOP 4–7 mm high. But the punch aims at the FLOOR, and the overlook's floor reading matched the
close re-measures to within 1 mm on every run (e.g. 172.3 mm, then 65.3 after 107 mm down: predicted 65.3).
The slow zone starts 20 mm above the top reading, which covers its error. The stepped route stays for aim
checks at a 10 mm hover (STOP_AFTER=hover10).

How it got here (2026-09-23/24), each change the user's call:
- the first full run **dropped the berry**: a 25 s scoop, and the food slid off while the fork was steep ->
  **20 °/s with a smooth speed profile** (sin² ease in/out over 25% each end; giving every waypoint full
  speed was refused at any speed as INVALID_JOINT_ACCELERATION);
- the fork rose near-vertical before turning -> the turn **starts the moment the food lifts** (exponent 1.6 -> 0.8);
- the punch was shallow -> floor + h/3, then + 7, + 2, and now **the measured floor, nominal**;
- the aim -> **+5 mm X** (into the bulk of the flesh past the tines) and **−6 mm Y** (the fork head's curve),
  `aim_offset_base_m`, fork-specific.

**No camera roll after the scoop** (user, 2026-09-24): rolling the fork 90° to put the camera on top turns the food's
curvature from facing up to facing sideways, which makes the bite harder. The fork stays as the scoop leaves it.

**After the scoop** (until the feed-the-user stages exist): take the food off, then lift **80 mm** with the
utensil horizontal and joint-move to food_detect_start. A straight joint move from the scoop end dips the
tip 94 mm over the bowl, and turning the utensil back down about its tip is unreachable (IK fails 40–60% in).

Why each piece is the way it is:
- **Re-measure at every stage, from where the arm is.** At a 30 mm hover the food was
  4.7–6.7 mm closer than the overlook reading predicted, every time.
- **The scoop is a joint-space path, not a Cartesian move.** As one straight Cartesian
  rotation it is unreachable past ~57°: J6 hits its limit. The end pose is reachable in
  hundreds of other arm shapes, so the arm reshapes (J1 ~80°, J2 ~75°) while the food
  holds its vertical line. It is planned live from the actual punch pose, IK-checked,
  validated by the controller, and run as one smooth trajectory.
- **The scoop pivots on the utensil tip in the arm's tool frame,** computed from
  calibration only: Kinova's camera position on the wrist + camera-to-tip + the tip's
  pixel. For the fork, (0, 9, 142) mm.

## The demo (2026-09-24, recorded)

`make feed` from feeding_start, bowl in front of the user: **3 bites, 126–130 s each, no stops, no retries**;
the loop saw the bowl empty and went home. Per bite: perceive ~12 s (resident detector), plunge + scoop
flowing together, one-trajectory swing, open-mouth wait, approach to 80 mm, bite at 1.0–1.7 s (2.5–3.2 N),
fast retract.

## The demo (2026-09-25, recorded): 6 of 6

Packed fruit cubes (melon, apple, pineapple) in a clear container, fork: `make feed BITES=6` --
**6 of 6 eaten, no stops, 73-85 s a bite** (perceive 6-9 s, acquire 21-25 s, deliver 46-57 s), then one more
in an unlimited run. What changed since 2026-09-24, each in its section below:

- **Aim**: the fork's tip pixel re-measured (the power-cycle re-seats it: check at a 10 mm hover after one),
  the fork's tine offset in the tip model, the aim on the piece's **top surface** (depth), not the box centre.
- **Food**: food profiles (`config/food/`), the generic prompt `piece of food`, fallback prompts, grapes and
  the background never targeted; packed food plunges 25 mm below its top; "empty" confirmed by depth.
- **Speed**: the arm's soft limits 70 (J1-J4) / 63 (J5-J7) deg/s, every move planned under them and under the
  controller's own acceleration measure -- a bite 126-130 s -> ~78 s.
- **Robustness**: depth from two staggered RTSP sessions (no dropouts), a resident colour stream, a stale
  depth server killed with -9, braking on arrival not taken for contact, a mouth depth that is not a face
  never approached.

## Delivery (`make deliver` -> `feeding/deliver.sh` -> `kinova/scripts/deliver.py`, `feeding/detect_mouth.py`)

Built 2026-09-24, leg by leg, with the user at the arm; ran end to end once (swing, present, one bite, retract)
as scratch scripts before being moved here. **Not yet run from these files.**

| step | what | why |
|---|---|---|
| SWING | **one trajectory**: lift **175 mm**, J1 −90° toward the user **while** retracting the utensil along its axis (120 mm, as far as the joints allow) | +60 mm touched the bowl; retraction in parallel, not after, keeps the tip from leading into the user |
| PRESENT | capture → face/nose/mouth → straight approach to the tip **80 mm short** of the mouth, **10 mm below** it, 20 mm/s | 80 mm suits any cube-sized piece; the stop felt nearer the nose |
| BITE | read-only: tool-force estimate > **1 N** for 0.1 s, against a baseline at rest | one bite peaked 5.7 N; J4/J5/J6 carry it, J7 barely (0.11 N·m) |
| RETRACT | back off along the utensil → reverse swing → down to scoop end + 80 mm → joint move to food_detect_start; **fast** (60 mm/s, 25/20/15 °/s) | no food, moving away |

- **No camera roll.** Rolling the utensil to put the camera on top turns the food sideways for the bite. The
  image is rolled 90° instead and rotated upright in software.
- **Detection: MediaPipe face landmarks first** (`feeding/detect_mouth_mp.py`, ~25 ms a frame on the CPU, in
  `~/venvs/mp`); LocateAnything (30–40 s) only if MediaPipe finds no face. On 8 saved frames MediaPipe found the
  face every time — including two where LocateAnything's "mouth" was the whole frame — and its depths agreed
  with LocateAnything's to 1–6 mm. Depth comes from many face points with the food (~250 mm nearer) dropped,
  so food covering the mouth or nose no longer blinds it.
- **Detected from 50 mm to the user's right** (the swing ends there) so the food is not in the line of sight.
  If the mouth is still unusable: sidestep further (twice), then the nose estimate.
- **Waits for an open mouth by default** (`WAIT_OPEN=` to skip): approach only once the user opens their mouth (lip gap / mouth width > 0.20 for 0.3 s;
  closed reads ≤ 0.02), up to 60 s — a "ready" signal.
- **Only a fresh detection moves the arm.** A stale second file once let an approach run to an old reading
  (it happened to stop further away, because the user had moved back). The approach now reads only the
  detection made at its pose, newer than its capture, under 2 min old.
- **Retract does not stop on torque while backing off** — stopping could leave the food in the mouth — and
  steers the arm back into the scoop's own shape before the joint move home: without that, the joint move
  dipped the utensil tip to the bowl floor (222 mm); with it, never below ~340 mm.
- The 1 N bite threshold sits at the level of non-bite bumps (1.0–1.1 N: leaning in, lips on the food); the
  0.1 s hold rejects spikes, not a sustained touch. One recorded bite so far.

### The open-mouth wait and the face (2026-09-25)

The wait reads the resident colour stream (frames arrive with ~0 s age; the camera itself lags reality by
**0.4 s**, `feeding/camera_latency.py`), writes a live readout to `runs_dir/open_wait_live.txt` every 0.25 s,
and keeps a snapshot every 0.5 s (`look_N/wait_frames/`). A detection is registered within ~0.3 s of the mouth
opening. The mouth's depth must be a face at the fork -- 0.25-0.75 m and within 80 mm of the face box's depth
(`delivery.mouth_check`) -- else the approach exits 5 without moving and one fresh look is taken: after a
depth stall the stream once delivered a stale frame (mouth "921 mm"; the face was 0.52 m).

## Per-utensil calibration — measure every time the utensil changes

`config/utensil/<name>.yaml` holds, for each utensil (select with `profiles.utensil` or `RAMMP_UTENSIL`):

| field | fork (current) | how to measure |
|---|---|---|
| `camera_to_tip_m` | **0.265** | at food_detect_start: ruler from the tip to **bare table** (state the unit!) + the sensor's bare-table depth; cross-check with camera-case-to-tip |
| `tip_px` | (644, 405) | where the utensil meets the food, **not** its image box centre (a fork's head is wide) — a 10 mm hover and your eye settle it |
| `image_region_px` | fork's image region | so detection never calls the utensil food |
| `punch_floor_margin_m` | 0.0 | how far above the bowl floor the plunge stops |

OVER aims at the detected centre + the utensil's **tine offset** (`tine_offset_tool_m`, tool frame, also part of the tip
model for the plunge, the scoop and the approach to the mouth: the fork head curves, so
its tines meet the food off the tool axis; fork: (0, −5) mm, 2026-09-25) + the food's **aim offset**. The aim
offset is per food (`config/food/<name>.yaml`, `profiles.food` or `RAMMP_FOOD`): `strawberry`
(+5, −6) mm, set by eye at a 10 mm hover -- into the berry's bulk past the tines, and the fork head's curve;
`cubes` (0, 0): a cube is about as wide as the fork head, and the strawberry offset put one side of the fork past
a cube's corner (2026-09-25). Check a new food at a 10 mm hover (`make acquire STOP_AFTER=hover10`).

A different bowl: `config/bowl/<name>.yaml` (overlook pose file, `scoop_end`, swing-start region). A different
camera: `config/camera/<name>.yaml`; if its depth reads consistently off, set `depth.scale` / `depth.offset_m`
(corrected = raw * scale + offset) from a ruler check rather than touching the code.

Both breakages on 2026-09-23 came from one wrong number: camera-to-tip taken as
105/115 mm (inches read as centimetres) when it is ~260 mm.

**The e-stop does not hold the arm.** Three times today the arm moved 100–160 mm after an
e-stop. Near the bowl, an e-stop can itself push things around.

## The motions (`kinova/scripts/sequence.py`)

```
feeding_start ──[one smooth vertical lift, 337 mm]──> top of lift
              ──[joint swing, via its midpoint]─────> food_detect_start
```

`make start` is exactly that backwards, then a joint snap onto the pose.

- **Up first.** feeding_start sits *below table height*, parked beside it. Going
  straight up first puts the whole swing ~300 mm above the bowl.
- **Smooth**, not a staircase: one straight-line Cartesian move each way.
- **The swing goes via its midpoint**, which makes the direction of travel
  explicit. The short way for J1 goes round the *back*, away from the table.
- **home is a retrace.** It re-uses the recorded waypoints, and refuses to start
  from any pose it doesn't recognise rather than improvising a path.

Measured: both directions run cleanly, utensil never more than 1.2° off
vertical; two consecutive round trips land the camera view within **1 px**
(≈0.3 mm at berry depth) of each other.

### Speed and the arm's limits (2026-09-25)

The arm's own **soft limits** (Kinova ControlConfig, `make limits`) were 40 deg/s; raised at the user's request to
**70 deg/s J1-J4, 63 deg/s J5-J7** (hard limits 80 / 70; `deployment.arm.soft_limits`, the old values saved to
the run record). Every speed is kept under them at runtime (`motion.capped`): asking more is refused, not
clamped, and the retries went slower than asked. Transit moves run at 61 deg/s, the lift at 0.22 m/s, the
descent into feeding_start 0.13 (at 0.22 its braking read as contact); the plunge is unchanged.

Trajectories are planned for the **controller's acceleration measure** (it sees ~1.6x the time scaling's):
`controller_accel_factor`, and a plan still refused on acceleration is rescaled by the controller's own
measured/limit ratio and validated again at the same speed (`accel_retries`) -- never above a limit, never
halved. A joint move the controller refuses as too short is re-sent at its stated minimum. A torque-step stop
with the arm already at its target is its own **braking**, not contact (J2 steps 2.7-3.2 N.m at 61 deg/s);
the swings and the retract's move home use 6 N.m guards.

### Every move goes through `safe_move.py`

Each check exists because its absence cost something on this arm:

| check | why |
|---|---|
| **utensil stays down** | Tool pitch is set by the sum J2+J4+J6 ≈ −180°. Moving *one* of them alone swung the fork 79° off vertical (2026-09-23). Only complete 7-joint targets are accepted. |
| **joint-torque contact abort** | The fork was then broken by its own wrist — tines dragged across the joint-6 housing — with the tool *angle* legal throughout. There is no self-collision model yet, so a torque step is the only thing that notices contact. |
| **time-synchronised joint moves** | A per-joint *speed* constraint makes a joint with a sliver of travel accelerate and stop within it; the controller refuses with a bare `METHOD_FAILED`. The real reason (`INVALID_JOINT_ACCELERATION`) is only in `GetTrajectoryErrorReport()`, now read on every abort. A *duration* constraint fixes it — and is also what actually keeps J2+J4+J6 interpolating linearly. |
| **joint limits pre-checked** | Kortex silently ignores a target past a limit. |
| **fail fast** | If nothing moves within 3 s, stop and report the controller's reason. |

Don't infer a refused move's cause from the arm state: it reads
`SERVOING_MANUALLY_CONTROLLED` for a while after *any* trajectory, ours included.
Several wrong theories (the gamepad, angle-wrap encoding, the deprecated API)
were tested and disproved before the error report was read. **Read the report
first.**

## Perception (`make perceive` → `feeding/stage1.sh`)

Runs on the jetson host; results go to the run record (`~/rammp_runs/<session>/bite_NN/perceive/`), **outside**
the synced tree. Refuses unless the arm is at food_detect_start, because the
image→motion mapping was measured in exactly that orientation.

1. **Detect** (`perceive.py`, LocateAnything-3B) — **one query per category.** A
   combined prompt gets stuck repeating one category and silently drops the
   rest, and *which* one repeats isn't stable. Food must lie inside the bowl
   box; a bowl box that merged with the utensil is squared. **The utensil is
   not detected** — it's fixed to the arm, as is the camera, so it occupies the
   same pixels every time (`utensil_px` in `food_detect_pose.json`). A `spoon`
   query once returned the whole bowl and suppressed every berry.
2. **Depth** (sensor, 20-frame median) — read *at each detection*, not by
   segmenting the bowl: the floor is concave, so no single floor level separates
   berries from the wall. Measured berries 23–31 mm tall. A berry hidden under
   the utensil has no return and is skipped, not guessed.
3. **Registration check** against the frozen colour principal point
   (`calibration/colour_intrinsics.json`). It's a camera constant, so it's
   calibrated once and each run only checks agreement (typically a few px, where
   1 px ≈ 0.34 mm).
4. **Target** (`locate_target.py`) — the most central visible berry. With the
   camera vertical, the utensil tip's vertical line lands at
   `p* = pp + (p_tip − pp)·Z_tip/Z_berry`, and the tool translates until the
   berry sits there. No hand–eye calibration needed.

### Food detection and the aim (2026-09-25)

- **Prompt per food profile** (`config/food/<name>.yaml`): cubes ask for `piece of food` -- on packed cubes one box
  per piece, as good as `fruit cube` and generic for steak / chicken; `food pieces`, `food cube`, `bite-sized
  piece` merged pieces or boxed the whole container.
- **Fallback prompts** (`fallback_prompts`, cubes: `fruit cube`), asked only when no box is usable.
- **Never targeted** (`exclude_grapes`): a box whose centre is red with little green (a grape -- it rolls off
  the fork) or blue (the mat, the bowl's wall).
- **The aim** is the centre of the uppermost piece's top inside the chosen box (depth pixels within 6 mm of its
  nearest point, the largest patch; `perception.aim_top_surface`) -- a merged box or a seam still aims at one
  piece. The box centre once sent the fork into the gap between a piece and a grape.
- **Packed food** (`acquisition.plunge.no_floor`): when the "floor" beside the piece reads food (under 10 mm
  "tall", no reading, or more than 25 mm above the table) the plunge goes 25 mm below the piece's top, never
  closer than 10 mm to the table.
- **"Empty" is confirmed by depth**: with no usable box, a quadratic fitted to the bowl's own surface (flat
  container or round bowl) -- under 5% of its inner area standing 15 mm clear of it is empty (exit 3), else
  "food left, none pickable" (exit 5): `make feed` stops saying so instead of finishing.

### Speed of perception (2026-09-25, offline on saved frames; the robot was off)

| step | before | now | how |
|---|---|---|---|
| colour frame | 22 s | 1.5 s | `grab_frame.py` reads the stream; gst recorded until a timeout |
| depth frame | ~5 s | ~0.5 s (expected) | `depth_server.py` keeps the stream open in kinova_vision; median of fresh frames |
| berry query | 5.6 s | **~1.0 s** | on the bowl crop only (0.5x image-reading), the model's `fast` mode, stop at the first repeated / sliver box -- same berries on 8/8 frames after the filters |
| bowl query | 1.8 s / bite | once per session | the bowl does not move; `make feed` clears the cache |
| OVER -> plunge | ~15 s standing still | ~3 s | `acquire pick`: OVER, fresh depth, plan, plunge, scoop in one arm program |

Tried and NOT used: re-using the overlook depth for the plunge (off by 11 and 5 mm on 2 of 17 runs); int8 weight-only
quantization (1.79 s vs 1.17 s, different boxes -- no fused kernels); int4 (torchao's kernels unsupported with this
PyTorch). The PyTorch build warns it is not built for the Orin (sm 8.7) but runs bf16 GEMM at 34 TFLOPS, near peak.

### Depth fallback: Depth Anything 3 fills what the sensor misses

`feeding/depth_fill.py` runs after every depth capture (perceive, the plunge's floor reading, the face). Sensor
pixels are kept as measured; holes -- or a whole missing frame (an RGB-only camera) -- are filled from DA3,
projected into the sensor's frame, fitted to the sensor (global scale/offset, then the fit's own error around
each hole spread into it), with `depth_filled_mask.npy` marking what was filled.

Measured on 4 saved frames with two 40x40 px holes cut out and compared with the sensor's own values: median
error -0.6 to -1.9 mm, but 90% within 23-29 mm and up to ~60 mm -- the LEVEL is right, the local SHAPE is not.
So: fine for finding and ranking food; the plunge REFUSES if more than half its floor reading is filled, or if
the depth is raw DA3 (no sensor at all: ~300 mm off); the approach stands 40 mm further off (120 mm) if the
mouth's depth is filled.

**Models stay resident** (`make models`, run by `make feed` first; `make models-stop`): LocateAnything-3B and
Depth Anything 3 in one GPU server (8.5 GB of 61 GB), the MediaPipe face landmarker in its own; each runs a
warm-up inference before it answers -- all up and warm in ~21 s.

| model | size | resident memory | warm inference |
|---|---|---|---|
| LocateAnything-3B | 7.2 GB | ~7.2 GB | berries 4.5-6 s, bowl 1.8 s |
| Depth Anything 3 (DA3METRIC-LARGE) | 0.33 B params, 1.3 GB | 1.25 GB | ~0.1 s |
| MediaPipe face landmarker | 3.7 MB | CPU | ~25 ms |

### Which depth source

**The depth stream itself** (2026-09-25): the arm ends each RTSP depth session every ~62 s, whatever the client
sends (an explicit keep-alive and a longer tcp-timeout changed nothing), and recovery took 10-20 s -- 20-26 s
lost per capture that landed in a gap. `make depth` runs two depth nodes 31 s apart (`/camera` and `/camera2`)
and the depth server merges both (`camera.streams.depth_topics`): 30 of 30 requests answered over 150 s while
the sessions paused 6 and 8 times. Captures take frames since the arm stopped (5 s buffered), so a brief gap
costs nothing. Stop the depth server with `pkill -9` (`make models` does): rclpy swallows SIGTERM and leaves a
stale server holding the port.

Scored against checks independent of all of them (a flat table; a strawberry is
~25–35 mm tall):

| | table noise | berry heights |
|---|---|---|
| **sensor** | 5.1 mm | **32, 31, 31 mm** |
| Depth Anything 3 raw | 12.4 mm, invents a 39 mm tilt | 88, 58, 74 mm |
| DA3 fused to the sensor | 2.1 mm | 13, 9, 14 mm — smooth and wrong |

**The sensor is the truth.** DA3 is available as an optional cross-check
(`STAGE1_DA3=1`) and is only worth anything where the sensor returns nothing.

## Poses (`calibration/`)

| file | pose |
|---|---|
| `home_pose.json` | **feeding_start** — J1 −90° (arm swung clear of the user), utensil straight down, tool (0.0236, −0.1766, 0.2082) |
| `lift_pose.json` | top of the lift — feeding_start's xy, z 0.545, J6 3.6° off its limit |
| `food_detect_pose.json` | **food_detect_start** — over the bowl, **in front of the user** (2026-09-24), camera straight down, tool (0.2205, 0.2826, 0.5344); named by `config/bowl/front.yaml`. The old pose (bowl to the user's right) is kept as `previous_pose` |
| `colour_intrinsics.json` | colour principal point, measured -- the record; the values used are in `config/camera/kinova_wrist.yaml` |
| `depth_480x270.ini` | this depth sensor's own intrinsics, not the package default |

**food_detect_start leans J2 ~+8° toward the table**, not −9.7° back. The camera
orientation is identical either way (J4 compensates; J2+J4 doesn't change), but
the camera sits ~120 mm nearer the base: no berry is hidden behind the utensil,
J2 holding torque roughly halves, and J6 has 5.4° to its limit (was 1.1°). The
rim's top sits at the image edge — at this bowl size (~560 px) the full rim and
a clear view past the utensil (~460 px) can't both fit.

**Orientation is the same at every stage** — utensil straight down (0.5°) — so
stages differ only in position, and nothing ever needs a wrist rotation. The
0.5° residual is in a direction J2/J4/J6 can't reach (they all pitch in one
plane).

**J1 and J3 are stored as intended exact values** (−90.000, 0.000), not as
measured: commanding −90.000000 lands at −89.998138 every time. If you ever
`make capture`, restore those two by hand.

## Known open

- **Mixed plates.** Packed cubes detect one box per piece; a pile of grapes, slices and small pieces often comes
  back as one box on the wrist camera (the loop then stops "food left, none pickable", exit 5). It is the
  image: the same container photographed sharper gives one box per piece with the same model and prompt
  (`captures/camera_quality_vs_detection.jpg`, sharpness 254 vs 828). Stock YOLO11 / YOLO-World were worse
  and a Real-ESRGAN upscale changed nothing (`captures/yolo_vs_locateanything.jpg`). The options: even light
  over the bowl; the camera's 1080p mode (needs a colour recalibration); splitting a pile box by depth.
- **A new food: test its prompt first** on a saved frame (`config/food/<name>.yaml`); only fruit was tested.
- **The bite reads 0.0-0.3 s** on most bites: the force is often already over 1 N when the wait starts (the
  user leaning in); harmless (the retract follows the bite) but it is not a measured bite time.
- **A spoon** needs its own utensil profile (camera-to-tip, tip pixel, tine offset) before use.
- **No self-collision model.** Torque is the only contact detector.
- **Disk is at 93%** on the jetson (3.8 GB free, 2026-09-24, after removing the Depth Anything V2 weights and
  the pip/apt caches). The rest is the OS/JetPack (~9 GB), LocateAnything-3B (7.2 GB), `~/venvs/la` (5.7 GB)
  and the Docker images (3.3 GB), all in use. Depth Anything 3 (1.3 GB) is the depth fallback (see Depth fallback) -- keep it.
- The tool frame is the **configured** one (120 mm, 0.831 kg), not the flange.
- The user-facing camera can **abort** motion but never drive it: ~100 mm error,
  and on the armrest its extrinsic to the arm isn't fixed.

## Layout

```
settings.py                  loads config/ (python: `from settings import cfg`; shell: `python3 -m settings get`)
runlog.py                    logging and the run record (events, plans, telemetry)
config/                      every tunable, switch and deployment value (above)
tests/                       fake_kortex + arm_trace (the arm stages), regress.sh, arm_values.py, record_smoke.sh,
                             twin_check.py
kinova/
  connection.py              session, routers, teardown
  motion.py                  shared: IK, time scaling, homotopy, guarded trajectories
  scene.py                   the table and bowl: shifted poses, table-relative heights
  scripts/
    sequence.py              the frozen motions: detect / home
    acquire.py               over / descend / punch / scoop
    deliver.py               swing / approach / sidestep / bite / retract
    safe_move.py             every move goes through here
    feeding_start.py         `feeding start` == sequence home
    arm_state.py             where is the arm (--watch)
    arm_limits.py            the arm's kinematic limits: show, or apply config (make limits)
    capture_home.py          re-measure a pose
    check_connection.py      identity, firmware, live joints
feeding/
  lib.sh                     sourced by every stage script: config, containers, the run record
  camera.py                  shared: intrinsics, depth correction, colour<->depth mapping
  viz.py                     debug images (depth PNGs with marks)
  review.py                  make review: a run record read back
  locate_bowl.py             make locate-bowl
  colour_stream.py           the resident colour stream (in the face server): live frames, dropout log
  camera_latency.py          how far the colour picture lags the arm (a J7 +-5 deg test; 0.4 s measured)
  camera_server.py           the camera, kept open: depth medians + colour frames, from any camera (make camera)
  detect_mouth_mp.py         face / mouth (MediaPipe): the open-mouth wait, live readout, snapshots
  stage1.sh                  make perceive
  perceive.py                detection
  locate_anything.py         LocateAnything-3B wrapper, label-aware parsing
  depth_probe.py             median depth frame (runs in the vision container)
  locate_target.py           depth per detection, registration check, target
  deliver.sh                 make deliver
  detect_mouth.py            face / nose / mouth at the delivery pose
  run_depth_anything3.py     optional DA3 cross-check
sim/                         the digital twin (above): scene.py, record.py, twin.py, procedural.py, pose_human.py
calibration/                 measured poses and intrinsics (above)
docker/                      driver and vision images
third_party/                 kinova-gen3-ros2 (submodule)
```

The gripper is **never commanded** anywhere in this tree.

## The camera: any camera, behind the camera server (`make camera`)

Nothing in the stack reads a camera directly: colour frames and depth medians come from
`feeding/camera_server.py` (`/frame`, `/mjpeg`, `/median`), and the selected camera profile
(`config/camera/<name>.yaml`, `profiles.camera`, a machine may pick its own in `local.yaml`) says where it gets them:

| `streams.kind` | the camera | how |
|---|---|---|
| `kinova_vision` | the arm's vision module (`kinova_wrist.yaml`) | depth from our kinova_vision nodes (`make depth`), colour over RTSP from the arm; the server runs in the vision container |
| `ros2` | any ROS 2 RGB-D camera (`realsense_d405.yaml`: the site D405) | its colour + aligned-depth topics, published by whatever drives it; the server runs in `{name}_camera` on that camera's ROS domain / DDS config (`camera.streams.ros`, per machine) |

A new camera: copy a profile, set its topics (or RTSP) and intrinsics, and the utensil's measurements against it
go under the utensil profile's `by_camera.<camera>` (camera-to-tip, tip pixel, image region), which apply only with
that camera selected. Then calibrate at food_detect_start as for any new setup.

## The digital twin (`sim/`, `make twin`)

A replay, not a simulation: the arm moves exactly as a recorded bite's telemetry says, in a scene built from the
same calibration the stack uses, with the camera frames the stack saw beside it.

```
rsync -a jetson:rammp_runs/<date>_feed ~/rammp_runs/     # a feed session, to the Mac
.venv/bin/python sim/pose_human.py                       # once: the seated person (needs torch + smplx and the
                                                         # licensed SMPL-X model in sim/assets/smplx/)
make twin BITE=3                                         # -> sim/out/twin_<session>_bite03.mp4 (real time, 1080p)
make twin BITE=all SPEED=1.5                             # every bite, one film
make twin BITE=3 STILL=52.9                              # one frame
make twin-poses                                          # feeding_start, food_detect_start, lift, scoop_end
```

| | from |
|---|---|
| the arm | Kinova's Gen3 model (MuJoCo Menagerie) + the 2F-85, closed on the fork; joints from each move's `arm/*_arm_*.npz`. FK agrees with the controller's own tool pose to 6 mm / 1.2 deg over a bite |
| the fork | `kinova/motion.tip_in_tool` (tip_px, camera_to_tip, the tine offset) and `utensil.head_width_m` |
| the wrist camera | `config/camera`: the mount and the measured intrinsics, principal point included, so the TWIN tile lines up with the REAL one (the tine tips land at v 425 against the measured 436) |
| the table, the food | `workspace.table_height_m`; each piece where `perceive/detection.json` puts it, mapped exactly as `feeding/locate_target.py` maps its target; the picked one rides the tines from the plunge's end until the bite |
| the container | the detection's `bowl` box: the clear clamshell on its blue mat (it sat at (0.240, 0.236) m in all six bites of 2026-09-25 -- ~30-40 mm from `workspace.bowl_centre_m`, which `make locate-bowl` measured on 2026-09-24 with the round bowl) |
| the person | SMPL-X, seated, placed so the mouth lands on the approach's measured `mouth_base`; the jaw opens with `open_wait.json`, the upper body leans in for the bite (the arm stops 80 mm short) |
| set dressing | the room, the wheelchair and the arm's mount, the laptop |

The HUD: the phase (from `events.jsonl`), joint speeds against the soft limits, the tip's speed and height, the
mouth-opening signal during the wait, the force at the fork during the bite, and the REAL | TWIN wrist-camera
tiles. `sim/record.py` reads a bite; `sim/scene.py` builds the scene; `sim/twin.py` directs and renders;
`sim/procedural.py` makes the textures and meshes (into `sim/out/cache/`, git-ignored).

## Machines

| host | role |
|---|---|
| abra@192.168.1.11 | **the site Jetson from 2026-09-30** (AGX Orin, JetPack 6.2 / R36.4.7, Python 3.10), wired to the arm at **192.168.1.10**. A SHARED team machine: everything of ours is under `~/cornell/` (below) |
| jetson | the lab Jetson until 2026-09-25 (JetPack R39, Python 3.12); arm 192.168.2.67; root `~`; its containers predate the naming (`tcp_rec`, `kinova_vision`: in its local.yaml) |
| triphammer | x86-64, RTX 4070 Ti; no PREEMPT_RT, on wifi |
| the Mac | source of truth; `./sync.sh [host]` pushes with `--delete` (never `config/local.yaml`) |

Edit on the Mac, never on a replica. A machine's own facts live in its `config/local.yaml` (never synced):
`deployment.root` (where it keeps venvs, models, the model cache, sources, runs), `deployment.name` (the prefix of
its containers and images), the arm's `ip`, and a port only if one clashes. Everything else in
`config/rammp.yaml` is derived from them (`{root}/venvs/la`, `{name}_driver`, ...); `RAMMP_ROOT=<dir>` overrides the
root for one command.

### The shared site Jetson (`abra@192.168.1.11`, `~/cornell/`)

Other people's `sheppy-*` containers run on it (foxglove holds :8765, a USB RealSense and an Orbbec are theirs):
leave them alone. Everything of ours is in one folder, so it can go in one step:

```
~/cornell/rammp        this tree (SYNC_USER=abra SYNC_DEST=cornell/rammp ./sync.sh 192.168.1.11)
~/cornell/venvs/la     detector: torch 2.10 (jetson-ai-lab jp6/cu126), transformers 4.57.1, decord2, DA3 (-e)
~/cornell/venvs/mp     face: mediapipe 1.0.1
~/cornell/da3src       Depth Anything 3 @ 3d835ec
~/cornell/.cache/huggingface   the model cache (LocateAnything-3B, DA3METRIC-LARGE)
~/cornell/models       face_landmarker.task
~/cornell/rammp_runs   the run record
~/cornell/logs, cache  setup logs, the pip cache
docker                 images cornell/driver:py310 (docker/Dockerfile.driver: Python 3.10, numpy, kortex_api),
                       cornell/kortex-vision:humble (Dockerfile.vision: ROS 2 Humble + kinova_vision);
                       containers cornell_driver, cornell_vision; our ROS 2 on domain 42 (theirs is 0)
```

Its `config/local.yaml` holds only its facts: `root: ~/cornell`, `name: cornell`, the arm at 192.168.1.10, the
detector on :8775, ROS domain 42. Set up from nothing: `./setup_jetson.sh` (the venvs, DA3, the models, all under
the root; `decord2` because decord has no aarch64 wheel), then `make images driver depth models`. Clean up:

```
docker rm -f cornell_driver cornell_vision; docker rmi cornell/driver:py310 cornell/kortex-vision:humble
pkill -f "cornell/venvs"; rm -rf ~/cornell
```

## History

Everything before the reset — MuJoCo testbed, earlier perception and skill work,
architecture and poster documents — is in `a85881d` and nothing else.
