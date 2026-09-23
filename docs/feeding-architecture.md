# Robot-assisted feeding — architecture

Kinova Gen3 7-DoF on a wheelchair, fork in a Robotiq 2F-85, feeding a seated
user. One bite cycle, looped until the plate is empty or the user stops it.

Everything below is ROS 2 and goes through the `kinova-gen3-ros2` interface
(`kinova_gen3_interfaces`) — no side channels to the arm.

---

## 0. The constraint that shapes everything

**A fork with pointed tines is moved toward a person's face, repeatedly, by a
7-DoF arm that can reach ~0.9 m.** Every other decision is downstream of that.
Concretely this rules out three otherwise-reasonable choices:

- **No position control in the approach phase.** A position controller meeting
  an unexpected obstacle (the user leaning in) will drive through it at full
  authority. The approach runs in `ee_pose_impedance` so contact is compliant.
- **No planner-owned safety.** The layer that decides where to go cannot also be
  the layer that vetoes it. The safety monitor is a separate node with its own
  view of the world and its own authority to `EStop`. A planner that checks once,
  at plan time, is not enough on its own: in simulation a fully clearance-checked
  path collided with the user at several hundred newtons simply because the user
  leaned forward after the plan was made.
- **Clearance is a whole-arm property, not a tool-frame one.** A feeding arm
  reaches *around* a seated person, so most of its length is closer to them than
  the fork is. A tip-only monitor in simulation reported a comfortable 84–90 mm
  while the wrist and forearm were being driven into the user's chest; at first
  contact the fork tip was 381 mm from anything. The monitor must check every
  link against every no-go volume.
- **No perception in the control loop.** A dropped frame or a 200 ms model
  inference must never stall or jerk a trajectory that is 10 cm from an eye.
  Perception produces *goals*; the controller executes them independently.

---

## 1. Verified hardware budget

Measured, not assumed:

| | |
|---|---|
| GPU | **1×** RTX 4070 Ti, **12282 MiB** (~11 GB free), compute cap **8.9** (Ada: FP8 + INT4 via TensorRT) |
| Depth camera | Gen3 L53 2D/3D, **480×270 `16UC1` millimetres @ 30 Hz**, 87–91% valid pixels |
| Colour camera | **1280×720 H.264 @ 30 Hz** |
| Arm | 7-DoF, spherical wrist, `BRAKE_TYPE_NOT_INSTALLED`, 2F-85 gripper |
| Joint feedback | position + **torque** per actuator, 1 kHz cyclic |

Note it is **one** GPU, not two. Everything GPU-side shares those 11 GB.

---

## 2. Perception is episodic, not continuous

This is the single biggest lever on the GPU budget and it is easy to miss.

A bite cycle takes ~10–20 s. Within it, food detection needs to run **once**
(plate is static, food does not move on its own). Table and plate geometry need
to run **once per meal**. Only face tracking needs to be continuous, and only
during the transport/present phase.

So the 3B detector runs on the order of **once every 15 seconds**, not at 30 Hz.
That changes the quantization question completely — see §7.

---

## 3. Depth: use the sensor, not a monocular model

`Depth-Anything-V2` default checkpoints are **relative** depth — no metric scale,
unusable for placing a fork in space. The `Metric-Indoor-*` variants are metric
but are still an *estimate*, while the L53 gives you **true millimetres** at
30 Hz.

**Primary depth = the arm's depth sensor.** Depth Anything earns a place only as
**hole-filling**: we measured 9–13% invalid pixels, concentrated on dark and
specular surfaces — which is exactly what a dark plate rim or a wet food cube
looks like. Run it only on the plate ROI, only when the invalid fraction inside
that ROI exceeds a threshold, and fit its relative output to the valid metric
pixels around the hole to recover scale.

Do not put it in the default path.

---

## 4. Node graph

```
                    ┌───────────────────────────────────────────┐
                    │ safety_monitor        (own subscriptions)  │
                    │  head no-go volume · speed caps · torque   │
                    │  residual · deadman · /estop  ── AUTHORITY │
                    └───────────────────────────────────────────┘
                                     │ /estop
  ── perception (episodic, GPU) ──   ▼   ── skills ──      ── control ──
  scene_geometry   ─┐                                   ┌ GoToEEPose
   plane fit → table├─► food_world ──► feeding_skill ───┤ ExecuteJointTrajectory
   plate ROI        │   (3D cubes)     (state machine)  ├ /setpoint/pose  (stream)
  food_detector    ─┘                        ▲          ├ /setpoint/gripper
   LocateAnything-3B                         │          └ JointImpedanceGains
                                             │
  face_tracker (continuous, CPU) ────────────┘
   mouth + nose keypoints → 3D via depth
```

### 4.1 `scene_geometry` — table and plate, once per meal
- Point cloud from `/camera/depth/image_raw` + `camera_info`.
- **RANSAC plane fit** → table plane. Gives the support surface and, critically,
  a *floor* for the skewer descent: never command below plane − ε.
- Plate: `LocateAnything-3B` prompted `"plate"` → 2D box → mask ∩ points above
  plane → plate centre and rim height.
- Publishes a latched `TableGeometry` (plane coefficients, plate pose, rim z).

### 4.2 `food_detector` — where the food is, once per cycle
- `LocateAnything-3B`, running on the Jetson. The prompt IS the interface, and
  **the prompt matters more than expected.** Measured on a real wrist-camera
  frame of a bowl holding two strawberries and an apple:

  | prompt | boxes | time | result |
  |---|---|---|---|
  | **`food pieces</c>bowl`** | **3** | **2.2 s** | **both strawberries + the apple. Use this.** |
  | `strawberry` | 3 | 9.8 s | same three, but you must know the fruit |
  | `food pieces` | 8 | 13.4 s | the three, plus junk in the corners |
  | `fruit</c>bowl</c>table` | 5 | 11.7 s | the three, plus two spurious |
  | `bowl` | 1 | 2.0 s | correct |
  | `fruit` | 63 | 9.9 s | **degenerate** — rows of thin slivers, unusable |
  | `table` | 1 | 11.3 s | boxes the ENTIRE frame; true and useless |

  Two results worth carrying forward:

  **Naming the container with the food sharpens both accuracy and speed.**
  `food pieces` alone scatters boxes into the corners and takes 13 s;
  `food pieces</c>bowl` returns exactly the three items in 2.2 s. The container
  constrains where food can be, and the model uses that. This is the prompt to
  use, and it needs no prior knowledge of which fruit is in the bowl.

  **A bare generic noun can collapse entirely.** `fruit` produced 63 sliver
  boxes. Prompt choice is not cosmetic here; validate any new prompt on real
  food before trusting it.
- **Do not ask the detector for the table.** "Box the table" is answered
  correctly by boxing everything. Table geometry comes from the depth plane fit
  in §4.1; the detector is for objects ON it.
- Generation must be bounded. Unconstrained, a single query emitted **341
  duplicate boxes** and ran to the 2048-token cap, which is where 43-88 s
  queries came from. With greedy NMS and a 384-token budget the same queries
  take **2-11 s**. Latency here was runaway decoding, not model speed.
- 2D boxes → centroid → **metric depth lookup** → 3D position; cube top face
  from the depth patch, not from the box.
- Publishes `FoodPiece[]`: pose, size, confidence, graspability score.
- **Reject rather than guess**: a box whose depth patch is >30% invalid is
  dropped, not interpolated. A missed cube costs one retry; a phantom cube
  drives the fork into the plate.

### 4.3 `face_tracker` — demoted to a monitor, and here is why

**The wrist camera cannot do this job during approach.** It is mounted on the
moving arm; as the fork advances, the face leaves the frame and the fork and food
occlude the lower half of it. So face tracking has to come from the second
camera looking at the user.

**But that camera cannot close a position loop either.** Two separate defects,
and the second is the fatal one:

1. *It is imprecise.* ~100 mm of error on the reported mouth position, with an
   effort underway to bring it to ~80 mm. On its own this might be survivable —
   error that is mostly bias can be filtered or calibrated out.
2. *Its extrinsic is not fixed.* At deployment it mounts on the **wheelchair
   armrest**, not on a rigid frame. The transform from that camera to the arm
   base changes whenever the armrest is adjusted, the chair is reclined, or the
   user shifts their weight. There is no calibration that survives this, because
   there is no constant to calibrate.

Defect 2 means the face camera can never be a source of setpoints, at any
precision. A 100 mm error against a mouth is not a tracking inaccuracy; it is
the distance between a bite and an eye.

**What it can still do,** and these are real:

- **Differential motion.** The *change* in apparent mouth position between
  frames is far more trustworthy than the absolute position, because the error
  is dominated by a slowly-varying bias (extrinsic + depth scale) rather than
  per-frame noise. "The mouth just moved 60 mm toward the fork" is usable even
  when "the mouth is at (x, y, z)" is not. This is the lean-in detector.
- **Presence gating.** Is a person there; is the head roughly upright and
  facing forward. A coarse yes/no that must pass before PRESENT is entered.
- **Coarse proximity.** Fork-to-mouth distance as a monotone quantity. Good
  enough to **veto** an approach, never to aim one.

Summary: the face camera holds an **abort authority, not a setpoint authority**.
It can stop the arm. It cannot steer it. §5.4 is what steers it instead.

---

## 5. The bite cycle

A state machine (not a behaviour tree — the transitions are mostly linear and
the failure handling is explicit).

```
IDLE
 └─► SCAN            scene_geometry once; food_detector → FoodPiece[]
      └─► SELECT     choose next cube (isolation, distance from rim, confidence)
           └─► APPROACH_ABOVE   GoToEEPose 5 cm above cube, fork vertical
                └─► SKEWER      tilt + descend under force monitor
                     └─► VERIFY  did we get it? (torque + gripper + re-detect)
                          ├─ no ─► SELECT (retry, max 2, then next cube)
                          └─► REORIENT   tines toward user, food up
                               └─► TRANSPORT   impedance mode, speed-capped
                                    └─► PRESENT   hold at mouth − offset
                                         └─► BITE_DETECT
                                              └─► RETRACT ─► SELECT / DONE
```

### SKEWER — and then SCOOP, not lift

Corrected from a teleop demo on the real arm, 2026-09-22. Two things the
straight-down-then-up plan gets wrong, both observed:

**Depth must stop in the flesh, not at the container.** In the demo the tines
went far enough to touch the BOTTOM of the bowl. That is a failure twice over:
it loads the fork against a rigid surface, and it means depth was governed by
the commanded position rather than by contact. Target is roughly **5 mm short**
of where that demo stopped. Position cannot deliver that — fruit sits at
whatever height it happens to sit — so the descent must terminate on force.
The bowl bottom is the abort floor, exactly as the table plane is for a plate.

**Lifting straight up loses the food.** Pulling back along the approach axis
lets the piece slide off the tines; it was visibly slipping in the demo. The
motion after insertion is a **scoop**: rotate the fork so the tines sweep under
the piece and come up level, carrying it on the shafts rather than balanced on
the points. This is the same requirement the RETAIN phase encodes in simulation
(`sim/skewer.py`, `retention_axis()`), but it belongs *inside* the skewer, not
as a separate later step -- the piece is lost in the first centimetre of lift.

So the sequence is: stop above, tilt, descend under force monitoring, stop when
the flesh gives, **scoop laterally at depth**, then lift.

### Measured on the arm (docs/demo-analysis.md)

Terminating the descent is **not** a force threshold. Two demos, both 136 mm in
3.6 s, one stopping correctly in the flesh and one loading against the bowl:

| depth band (mm) | hit the bowl | stopped in flesh |
|---|---|---|
| 91-113 | 0.208 N/mm | **0.270** <- peak |
| 113-136 | **0.369** <- still rising | **0.155** <- fallen back |
| final \|F\| | 14.07 N | 5.66 N |

**Stiffness that rises then FALLS means the skin broke and the tines are in
flesh — stop. Stiffness that keeps rising means a rigid surface — abort.** A
level threshold cannot separate them: the bowl run passed through 5.7 N on its
way to 14 N, so anything low enough to catch it would stop the good run short
of the fruit.

The scoop is **~93 mm of lateral travel at constant depth**, taken immediately
after the descent bottoms out and *before* any lift — not a rotation during the
lift, which is what the simulator had.

The window between "in the flesh" and "on the bowl base" measured **4 mm**. That
is the error budget for depth sensing, the fork TCP and controller lag combined
(the last of which is ~7 mm under impedance).

**And the depth is bounded from above too.** The demo never lifted: z varied by
1.9 mm across the entire scoop-transport-present sequence, because the fork
**slides out across the bowl rim** rather than lifting over it. So the pierce
must be deep enough to reach flesh AND shallow enough to keep the fork above the
rim. That makes the bowl rim a third height to measure, makes the scoop distance
a function of bowl geometry rather than a constant, and means a deep bowl with
low-sitting food may admit no valid depth at all -- check before committing to a
piece. A flat plate removes the constraint entirely, so this skill is not
container-agnostic.

### Descent, concretely
- Descend along the fork axis under `ee_pose_impedance` with **low stiffness in
  the approach axis, high laterally**, so the tines self-align into the cube
  rather than shoving it across the plate.
- Terminate on **force threshold**, not on position. The table plane from §4.1 is
  the hard floor; if the threshold has not tripped by plane − ε, abort.
- Grip is fixed — the 2F-85 holds the *fork*, it never opens during a meal.
  (Worth an explicit interlock: gripper commands are rejected while a meal is
  active, or a dropped fork becomes a falling-object problem.)

### PRESENT / BITE_DETECT
- Approach stops at a **standoff** and the arm **holds** — it does not push
  toward the mouth. The user closes the last distance. This was always both
  safer and how assistive feeding is normally done; with §4.3's demotion it is
  now also the *only* thing making the mouth-side approach tolerable, so the
  standoff is sized generously rather than tightly.
- The standoff is **not** computed from the nose keypoint any more. It is baked
  into the taught PRESENT pose (§5.4).
- Hold pose in impedance mode with low stiffness so a bump is absorbed.

---

### 5.4 Frozen mouth-side motion — the consequence of §4.3

The bite cycle splits in two, along the line of *which sensor can be trusted to
locate the goal*.

| | bowl side | mouth side |
|---|---|---|
| states | SCAN, SELECT, ALIGN, DESCEND, DESC_FINAL, SCOOP, SETTLE | TRANSPORT, PRESENT, RETRACT |
| goal comes from | **wrist camera + joint torque** | **a taught pose** |
| why that works | the wrist camera is bolted to the arm, so its extrinsic is fixed and known | nothing on the mouth side has a fixed extrinsic to the base |
| closed loop? | yes, unchanged | no — replayed |

**The bowl side does not change at all.** Its sensing is rigidly mounted to the
thing it is steering. The table, the bowl and the food are located in the wrist
camera's frame and the descent is terminated on torque (§5.3). None of that
touches the face camera, so none of it is affected.

**The mouth side is taught once and replayed.** Before a meal, the user or an
attendant drives the fork with the gamepad to a comfortable bite position, and
that joint vector is captured — the same `capture_home` path that recorded the
start pose. Every subsequent cycle replays it.

**Why this is not a downgrade.** It looks like one — vision replaced by a
recording — but the geometry favours it. *The arm and the user are bolted to the
same rigid body.* The arm is mounted on the wheelchair; the user is sitting in
the wheelchair. So the mouth-to-base transform is approximately constant over a
session, and varies only with the user's posture. The camera is the one thing in
the system that is **not** rigidly attached to that body. A taught pose captures
the mouth-to-base transform exactly, with no calibration step and no extrinsic
to estimate — it measures the very quantity the camera is bad at.

**Replay in joint space, not Cartesian.** The taught pose is a joint vector, and
it is replayed as one. Going through Cartesian space would re-introduce IK,
which means re-introducing the redundancy choice (`sim/README.md`: the same goal
solved twice differed by 187 mm of clearance) for no benefit, since the goal is
already known to be reachable — it was reached, by hand, during teaching.

**What this costs, stated plainly:**

- **Posture drift is uncorrected.** If the user slumps over a twenty-minute
  meal, the taught pose goes stale and nothing notices. Mitigations: re-teach
  between courses; keep the standoff generous; and use §4.3's *differential*
  signal, which is exactly the residual capability the camera retains.
- **The mouth-side safety monitor loses its geometry.** `sim/safety.py` checks
  14 arm geoms against no-go volumes — but those volumes came from a known
  human mesh. On hardware we will not know where the person is. At the mouth,
  safety therefore degrades to **impedance compliance + joint-torque limits +
  the face camera's veto**, with no clearance computation behind it. This is
  the weakest point in the deployed system and should be named as such rather
  than papered over.

---

## 6. Bite detection — the hard one

The user's instinct (joint torque in the last joints) is correct, but raw torque
will not work: the arm's own gravity and inertia dominate a bite by an order of
magnitude.

**Use the gravity-compensated residual.** The driver already carries a Pinocchio
model (`Dynamics::gravity(q)`, `mass_matrix`) — that is exactly the predictor we
need:

```
τ_residual = τ_measured − τ_gravity(q) − τ_inertial(q, q̇, q̈)
```

While holding at the mouth the arm is static, so `τ_inertial ≈ 0` and the
residual is just external wrench mapped through `Jᵀ`. A bite is a transient tug
along the fork axis.

Detection = **a threshold on the residual projected onto the fork axis, plus a
minimum duration**, with the baseline re-zeroed at the start of every PRESENT
(the food's own mass changes the static residual).

Practical notes:
- Calibrate the threshold per user; bite force varies enormously.
- **Always pair it with a timeout and the gamepad.** Force-only bite detection
  will have false negatives, and a fork parked at someone's mouth indefinitely is
  a bad failure. The user already holds a controller in the trial video — that is
  the natural confirm/abort channel and should not be designed out.

### Measured in simulation

`sim/validate_bite.py` runs this scheme against a plant whose dynamic model the
estimator gets deliberately wrong. Threshold 1.0 N over a fresh baseline, 60 ms
minimum duration, four trials per cell:

| applied bite | detected | latency |
|---|---|---|
| 0.5 N | 0/4 | — |
| 1.0 N | 2/4 | 251 ms |
| 2.0 N | 4/4 | 141 ms |
| 5.0 N | 4/4 | 108 ms |

Zero false positives across every no-bite hold. **Reliable from ≈2 N, in ≈140 ms.**

The bias suppression is worth being precise about, because the two mitigations
are not equally important. Projecting onto the fork axis does most of the work:
at 3% link-mass error the residual carries 0.27 N of apparent force, but only
0.05 N of it lies along the fork, because model error is dominated by *vertical*
mass error while the fork is horizontal at PRESENT. Re-zeroing then takes it to
0.01 N. Re-zeroing remains worth doing — it also absorbs the food's own weight,
and it is the only one of the two that still works when the fork is tilted — but
the axial projection is the load-bearing choice.

These numbers are only as good as the noise model behind them. `TORQUE_NOISE` is
a placeholder: log `/joint_states` at rest for 60 s on the real arm and take the
per-joint standard deviation before trusting any threshold derived here.

---

## 7. GPU budget and quantization

**Superseded by measurement.** This section was written against triphammer's
12 GB RTX 4070 Ti. Perception now runs on the **Jetson AGX Orin**, which is on
the arm's own network and has **61 GB of unified memory**. That changes the
question completely: memory was the constraint, and it is gone. Latency is the
constraint, because face tracking is continuous (§4.3).

Measured on the Orin — full detail and method in `docs/perception-quantization.md`:

| | Orin AGX (sm_87) | 4070 Ti (sm_89) |
|---|---|---|
| FP8 | **unavailable** (needs CC ≥ 8.9) | available |
| INT8 | **native** (`_int_mm`) | native |
| INT4 | tinygemm path only | available |

Single FFN projection, batch 1, 2048 → 11008:

| | latency | vs bf16 | rel err |
|---|---|---|---|
| bf16 | 0.271 ms | 1.00× | — |
| **int8 weight-only + `torch.compile`** | **0.155 ms** | **1.75×** | **0.0046** |
| int8 weight-only, eager | 1.255 ms | 0.22× | 0.0049 |
| int4 tile-packed + compile | 0.191 ms | 1.40× | 0.0652 |

**Decision: INT8 weight-only, compiled.** Three things drive it:

1. **`torch.compile` is mandatory, not an optimisation.** Eager quantisation is a
   4.5× *pessimisation* here — the dequantise step never fuses.
2. **INT4 loses on this hardware**, on both axes: slower than INT8 *and* 14×
   the relative error. Ampere implements INT8 in hardware; the int4 path goes
   through PTX JIT on an officially-unsupported arch.
3. **FP8 is not available at all** on sm_87.

Quantization error in a detector is **box displacement**, so every variant is
validated on box-centre shift against the BF16 reference, not on latency alone
(`feeding/bench_locate_anything.py`). And re-validate on real food images:
degradation shows first on small low-contrast objects, which is exactly what a
beige cube on a white plate is. Do not accept a published benchmark for this.

Face tracking may still need to leave the GPU. If the quantised model cannot
hold ~10 Hz on the face prompt, food detection keeps this model (episodic, §2)
and face tracking moves to something small — the architecture already keeps
perception out of the control loop, so a slow detector lowers the goal-refresh
rate rather than stalling a trajectory.

---

## 8. Deployment

Three facts that constrain placement, all verified:

- `triphammer` is at **192.168.2.136/24** — same subnet as the arm (192.168.2.67).
  It can reach it directly. `cascadilla` and `buttermilk` cannot (campus net).
- `triphammer` has the **only working GPU** (RTX 4070 Ti) but runs
  `5.15.0-139-generic` — **no PREEMPT_RT** — and is on **wifi**.
- The 1 kHz cyclic loop wants PREEMPT_RT and wired ethernet.

```
Jetson (PREEMPT_RT, wired) ── kinova_gen3_node ── 1 kHz ──► arm
                                    │ ROS 2 / Cyclone DDS
triphammer (GPU, 12 GB)   ── perception nodes, safety_monitor
```

Running the driver on triphammer works for bring-up and for everything except
trusting the low-level loop — acceptable while developing skills, not for
sustained impedance control near a face. Wire it before believing any timing
number.

---

## 9. Open questions — measure before building

1. **Depth minimum range.** Unknown, and it decides whether the wrist camera can
   see the cube at skewer depth at all. If min range > standoff, the last few cm
   are blind and SKEWER must run open-loop off a pose captured higher up.
   *Measure first — this changes the skewer design.*
2. **Second camera for the face.** §4.3. Not currently on the chair.
3. **Depth stream stability.** The RTSP session drops every ~62 s (documented in
   the README; server advertises `timeout=30`, `rtspsrc` refreshes too slowly).
   Tolerable when perception is episodic; **not** tolerable for continuous face
   tracking — another reason that camera should be separate and not go through
   `kinova_vision`.
4. **Bite force range** for this user population, to set the residual threshold.
   Simulation says ≈2 N is reliably detectable in ≈140 ms (§6); what is still
   unmeasured is whether real users *produce* 2 N, and what the per-joint torque
   noise actually is on this arm.
5. **Fork geometry in the URDF.** `gen3_7dof_2f85.urdf` models the gripper, not a
   fork held in it. Collision checking and the tool frame are both wrong until
   the fork is added — the TCP must be the **tine tips**, not the gripper face.
   *Done in simulation* (`sim/assets/kinova_gen3/gen3_fork.xml`, TCP at the tine
   tips); the URDF the real driver loads still needs the same treatment.

   *Coarse number now in hand:* **camera to tine tip is ~105 mm** (measured by
   hand). The arm already carries a configured 120 mm tool transform along z
   (`GetToolConfiguration`: 0, 0, 0.12, no rotation, 0.831 kg), so the
   `tool_pose` the arm reports lands within a few cm of the real tine tip.
   That is enough to report a standoff and to reason about clearance. It is
   **not** enough for the 4 mm flesh-to-bowl window, and it does not need to
   be: that window is resolved on torque (§5.3), not on geometry. Pivot
   calibration of the true TCP remains open, but it is no longer on the
   critical path.
6. **Camera intrinsics.** *Partly answered.* The sim now uses the intrinsics
   `kinova_vision` actually publishes (colour fy=1298.63 -> fovy 30.99 deg,
   depth fy=360.01 -> fovy 41.11 deg); the menagerie's 41.84 deg colour camera
   was 10.8 deg too wide. But those are Kinova's GENERIC defaults, not this
   unit's, and `cy` in the 16:9 colour profiles is not centred and does not
   scale with resolution. See `docs/camera-calibration.md`.
7. **Camera-to-base extrinsics, and the fork TCP.** Neither exists. The static
   face camera reports the mouth in camera coordinates and nothing knows where
   that camera is relative to the arm; the fork's tip is in no URDF. These block
   running the loop on hardware and are the real calibration work --
   `docs/camera-calibration.md`.
8. **Retention pose reachability.** Two of four plate positions had *no*
   reachable retention pose when the fork's heading was pinned to the approach
   direction. Freeing the heading fixed it in simulation, but this is a real
   workspace limit of a knee-height mount and is worth checking against the
   physical setup before the plate position is fixed.

---

## 10. Simulation and learning — sequencing

### Do not build a new simulator

`rammp-org/ramms-sim` (UE 5.7, updated 2026-09-12) **already ships a Kinova Gen 3
manipulator**, plus sensor simulation, a camera capture system, URDF
interoperability and Python integration. `ramms-mujoco-physics` /
`ramms-mujoco-support` (updated 2026-09-11) bring MuJoCo, and
`ramms-human-physics` does human body simulation — which is the one asset a
feeding scenario needs and that a generic simulator would not have.

Extending that is a fraction of the work of standing up something new, and it
keeps one simulation story for the platform rather than two.

What it does **not** appear to cover yet, and what a feeding scenario needs:
a seated user with a trackable face, deformable food, and a fork on the tool
flange.

### The highest-value use of sim here is not policy training

It is **testing the failure paths that cannot be safely tested on a person.**
That is available today and needs no ML at all: the driver's `SimTransport`
already runs the full ROS 2 interface headless — verified, `/list_controllers`
in sim returns exactly what it returns against the real arm, all 8 controllers.

So the state machine (§5) and, especially, `safety_monitor` (§0) can be built and
fault-injected right now:

- detection returns nothing / returns a phantom cube
- face track lost mid-transport
- the depth stream stalls (it does, every ~62 s — §9.3)
- the bite never comes; does PRESENT time out cleanly?
- user leans into the fork during approach — does impedance + the monitor stop it?

None of that requires food physics or a learned policy, and all of it is work
that must exist before hardware trials with a real person.

**Status: the harness for all five now exists** (`sim/`, see its README). Not
`SimTransport` — a MuJoCo cell with the seated user, the fork, depth-image
detections, a torque-residual bite channel and a whole-arm clearance monitor, so
that geometry and contact are real rather than stubbed. Every fault above is a
flag in `sim/faults.py`. The state machine and `safety_monitor` themselves are
still to be written; this is what they get built against. Running it has already
returned five defects and two design corrections that would otherwise have been
found on hardware, or on a person — they are listed in `sim/README.md`.

### On diffusion policy: right tool, wrong stage, and scope it narrowly

Three reasons not to lead with it:

1. **Most of this loop is geometry, not learning.** Plane fitting, pose
   estimation, motion planning — classical methods here are more reliable, far
   more debuggable, and critically more *boundable*. The safety architecture in
   §0 depends on explicit goals that an independent monitor can veto. An
   end-to-end learned policy driving a fork toward a face is very hard to put
   hard geometric guarantees around.
2. **Demos are the bottleneck, and sim demos are the wrong ones.** Skewering is
   contact-rich on *deformable* food — variable friction, moisture, yield. That
   sim-to-real gap is itself a research problem. Real teleop demos are worth far
   more per sample.
3. **You already have the demo-collection rig.** The trial video shows the user
   driving the arm with a gamepad. Instrumenting that path to log
   (observation, action) pairs makes every trial you already run into data, at
   nearly zero marginal cost.

**Where learning genuinely earns its place: SKEWER.** It is the one stage that is
contact-rich, food-dependent and hard to model analytically — which is exactly
where the assistive-feeding literature concentrates. Even there, keep it
bounded: the policy *proposes* an approach pose and descent, while the table
plane, the force threshold and the safety monitor retain veto.

### Suggested order

1. **Now** — state machine + `safety_monitor` against `SimTransport`; fault
   injection. No GPU, no food physics, no person.
2. **Next** — log teleop demos on hardware through the existing gamepad path.
   Free data from trials you are running anyway.
3. **Then** — extend RAMMS with the seated user, fork and deformable food, for
   augmentation and for rehearsing approach geometry safely.
4. **Only then** — a learned skewering policy on real demos, sim-augmented,
   proposing into the geometric layer rather than replacing it.
