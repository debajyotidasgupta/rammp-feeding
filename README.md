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
```

Frozen 2026-09-24 (tag `acquire-v1`): two consecutive full runs, fork on strawberries, both picked up and held ("the most perfect scoop"). Run to run: OVER 82/81 mm, punch 45.3/44.3 mm down, scoop 6.3/6.2 s, rise 164.5/163.8 mm, 1.0/1.0 mm off vertical.

## Acquisition (`make acquire` -> `feeding/acquire.sh` -> `kinova/scripts/acquire.py`)

| step | what | measured |
|---|---|---|
| OVER | translate at the overlook height until the tip's vertical line meets the food | 82 mm, aim user-confirmed at the hover |
| PLUNGE | **one smooth move** from the overlook to the **measured bowl floor** (nominal; really ~5–10 mm above it): 20 mm/s, easing to 5 mm/s from 20 mm above the food top | 2026-09-24: 173 mm in 20.2 s, same end depth as the stepped punch |
| SCOOP | food rises while the utensil turns vertical -> horizontal (+Y); the turn **starts the moment the food lifts** (90°·s^0.8). With the bowl in front of the user it also **backs off to the old scoop end** (`scoop_end`, 206 mm), mostly after the first 20% |  6.3 s at 20 °/s; rose 164.5 mm, 1.0 mm off vertical; berry held |

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

## Per-utensil calibration — measure every time the utensil changes

`calibration/food_detect_pose.json` holds, for the current utensil:

| field | fork (current) | how to measure |
|---|---|---|
| `camera_to_tip_m` | **0.265** | at food_detect_start: ruler from the tip to **bare table** (state the unit!) + the sensor's bare-table depth; cross-check with camera-case-to-tip |
| `utensil_tip_px` | (658, 470) | where the utensil meets the food, **not** its image box centre (a fork's head is wide) — a 10 mm hover and your eye settle it |
| `utensil_px` | fork's image region | so detection never calls the utensil food |
| `aim_offset_base_m` | (+5, −6) mm | at a 10 mm hover, by eye: where the tines actually meet the food vs the tool axis |

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

Runs on the jetson host; results go to `~/rammp_runs/<timestamp>/`, **outside**
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

### Which depth source

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
| `food_detect_pose.json` | **food_detect_start** — over the bowl, **in front of the user** (2026-09-24), camera straight down, tool (0.2205, 0.2826, 0.5344); also the fixed utensil image region. The old pose (bowl to the user's right) is kept as `previous_pose` |
| `colour_intrinsics.json` | colour principal point, measured |
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

- **The spoon's camera-to-tip distance is unmeasured.** 105 mm was the fork's.
  It dominates the target's ±4.7 mm; calipers would bring that to ~±1 mm. So
  does its lateral offset from the tool axis — the quantity that swept the fork
  into joint 6.
- **No self-collision model.** Torque is the only contact detector.
- **Disk is at 95%** on the jetson. The unused Depth Anything V2 weights
  (~390 MB) can go.
- The tool frame is the **configured** one (120 mm, 0.831 kg), not the flange.
- The user-facing camera can **abort** motion but never drive it: ~100 mm error,
  and on the armrest its extrinsic to the arm isn't fixed.

## Layout

```
kinova/
  connection.py              session, routers, teardown
  scripts/
    sequence.py              the frozen motions: detect / home
    acquire.py               over / descend / punch / scoop
    deliver.py               swing / approach / sidestep / bite / retract
    safe_move.py             every move goes through here
    feeding_start.py         `feeding start` == sequence home
    arm_state.py             where is the arm (--watch)
    capture_home.py          re-measure a pose
    check_connection.py      identity, firmware, live joints
feeding/
  stage1.sh                  make perceive
  perceive.py                detection
  locate_anything.py         LocateAnything-3B wrapper, label-aware parsing
  depth_probe.py             median depth frame (runs in the vision container)
  locate_target.py           depth per detection, registration check, target
  deliver.sh                 make deliver
  detect_mouth.py            face / nose / mouth at the delivery pose
  run_depth_anything3.py     optional DA3 cross-check
calibration/                 measured poses and intrinsics (above)
docker/                      driver and vision images
third_party/                 kinova-gen3-ros2 (submodule)
```

The gripper is **never commanded** anywhere in this tree.

## Machines

| host | role |
|---|---|
| jetson | AGX Orin, arm64; driver container `tcp_rec`, vision `kinova_vision`, GPU venv `~/venvs/la` |
| triphammer | x86-64, RTX 4070 Ti; no PREEMPT_RT, on wifi |
| the Mac | source of truth; `./sync.sh [host]` pushes with `--delete` |

Arm at **192.168.2.67**. Edit on the Mac, never on a replica.

## History

Everything before the reset — MuJoCo testbed, earlier perception and skill work,
architecture and poster documents — is in `a85881d` and nothing else.
