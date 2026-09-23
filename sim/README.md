# Feeding simulation (MuJoCo)

A testbed for the parts of the feeding stack that must not be debugged on a
person. Built instead of extending `ramms-sim`, per the call that it is not yet
stable enough to build on, and deliberately narrow: enough to develop and
fault-inject the skill state machine and the safety monitor
(`docs/feeding-architecture.md` §10 step 1), not a photoreal data generator.

```
sim/
  world.py         the model, and named handles onto it
  robot.py         the arm as the stack sees it: q, qd, measured torque, residual
  human.py         the person: lean, and a bite that is a real force
  sensors.py       colour / depth / face cameras, with the depth sensor's warts
  perception.py    detections that come through the depth image, not from ground truth
  safety.py        whole-arm clearance, and a clearance-checked leg planner
  faults.py        the five failure paths from architecture §10
  ik.py            6-DoF IK with null-space elbow control and multi-restart
  skewer.py        where to put the tines, and how to keep the piece on them
  selftest.py      25 checks; run after any change
  demo_meal.py     the whole loop, end to end
  validate_bite.py how small a bite the torque residual can actually see
  pose_human.py    one-time: SMPL-X -> seated mesh + landmarks (needs torch/smplx)
  render.py        offscreen renders from named cameras
  reach_check.py   IK reachability of the layout
```

**Where it runs.** The Mac only, for now. triphammer is on Ubuntu 20.04 with
Python 3.8 and no `python3-venv`, which cannot run numpy 2.x or MuJoCo 3.13, and
upgrading it needs the admin — the same blocker as the missing
`nvidia-container-toolkit`. Worth bundling both into one request, since the GPU
box is also where the perception nodes will eventually have to run.

**First run on a new machine:** the posed human meshes are derivatives of the
licensed SMPL-X body model and are not kept in the repo, so generate them once
with `python sim/pose_human.py` (needs `torch`, `smplx` and
`sim/assets/smplx/SMPLX_NEUTRAL.npz`). Nothing else in `sim/` needs torch.

Then run `python sim/selftest.py`. It prints measured numbers, not just
verdicts, because most of these are quantities the design depends on.

## Frame

Z-up. The user faces **+X**, their **left is +Y**, so the arm — mounted on their
right, as in the trial video — sits at **−Y**. SMPL-X ships Y-up/+Z-forward; the
`(x,y,z) -> (z,x,y)` permutation in `pose_human.py` converts it (det=+1, so
handedness is preserved) and drops the model so the floor is z=0.

| | |
|---|---|
| mouth / nose tip | (0.101, −0.001, 1.207) / (0.115, 0.000, 1.236) |
| head no-go sphere | r = 0.123 at (−0.004, 0.011, 1.225) |
| lumbar lean hinge | (−0.016, 0.000, 0.715) |
| seat height (pelvis) | 0.605 m |
| arm mount | (0.34, −0.32, 0.55), knee height, forward of the thigh |
| table top | z = 0.73, near edge x = 0.49 |

## What it models, and why each piece is there

**The user can lean.** The body is split at the lumbar spine by SMPL-X skinning
weight into a seated lower half and an upper half on a pitch hinge. This is not
cosmetic: *"the user leans into the fork during approach"* is the scenario the
whole safety architecture exists for, and a rigid mesh cannot express it. At 20°
the mouth travels **161 mm forward** — further than the 90 mm PRESENT standoff,
so the user can close on the fork faster than the arm can withdraw.

**The arm reports measured torque, not ground truth.** `Arm.residual()` is
`measured − predicted`, where both the gravity feed-forward and the prediction
come from `World.est`, a deliberately imperfect copy of the plant (3% link-mass
error by default). With a perfect model the residual is identically zero and any
detection threshold "works"; the achievable threshold is set by model error, and
this is where that shows up.

**Bites are forces, not events.** `Human.start_bite` applies a real tug to the
food body. It reaches the arm through the weld and the joint torques. The
detector is never told a bite happened.

**Detections come through the depth image.** `perception.py` projects each piece
into the depth frame, samples the patch, and **rejects rather than interpolates**
when more than 30% of it is invalid — so architecture §4.2's rule is exercised
rather than asserted. It reports the measured top-face point *and* the inferred
centre separately, because they differ by a half-extent (7.5 mm on a 15 mm cube)
and skewering aims at the centre.

**Clearance is whole-arm.** See below.

## Things the simulator found

Each of these was a real defect or a real design gap, not a modelling artifact.

**A tip-only safety check is not a safety check.** The first meal demo measured
fork-tip distance to the user, reported a comfortable 84–90 mm at PRESENT, and
was driving the wrist and forearm into the person's chest at **411 N** the whole
time. At the instant of first contact the fork tip was **381 mm** from anything.
A feeding arm reaches *around* a seated person; most of its length is closer to
them than the tool is. `safety.py` now checks all 14 arm collision geoms against
all four no-go volumes with `mj_geomDistance`.

**Redundancy resolution decides whether the elbow is inside the user.** The Gen3
is 7-DoF, so a position+orientation goal leaves exactly one free dimension, and
that dimension is where the elbow goes. Two solves of the *same* goal, both with
sub-millimetre residuals, differ by +45 mm and −142 mm of clearance. IK now
carries a null-space elbow bias, and the planner re-seeds when a leg is unsafe.

**A straight line between two safe poses is not safe.** Both endpoints of the
LIFT→REORIENT leg cleared by 74 mm; the joint-space interpolation between them
put the upper arm **161 mm inside** the torso, at 15% along. Every leg is now
swept before it is driven.

**Damped least squares winds continuous joints.** Gen3 joints 1/3/5/7 are
unlimited, and a solve came back with joint 3 at 502° — a 380° journey to a pose
130° away. Kinematically identical, catastrophic as a path. Solutions are now
unwound onto the branch nearest the seed.

**IK has silent local minima.** It does not fail loudly; it returns a pose tens
of millimetres and several degrees off, which downstream looks like a slightly
sloppy move. `solve()` retries from several seeds and keeps the best.

**IK was corrupting the live state.** The previous version iterated on
`d.qpos` and left the solution there, so every caller teleported the arm to the
goal before servoing toward it — the physics then ran from a corrupted initial
condition at each phase boundary. It now uses scratch state.

**Index arithmetic does not survive scene edits.** Adding the lean hinge moved
the arm's joints from index 0..6 to 1..7 and every `d.qpos[:7]` in the codebase
silently addressed the wrong DoF. Everything goes through `world.py` by name now.

## Measured numbers

Bite detection (`python sim/validate_bite.py`), 1.0 N threshold over a fresh
baseline, 60 ms minimum duration:

| applied bite | detected | latency |
|---|---|---|
| 0.5 N | 0/4 | — |
| 1.0 N | 2/4 | 251 ms |
| 2.0 N | 4/4 | 141 ms |
| 5.0 N | 4/4 | 108 ms |

No false positives in any no-bite hold. Reliable detection from **≈2 N**, in
about **140 ms**. Two things suppress the model-error bias, and the first does
most of the work: projecting onto the fork axis cuts a 3%-model-error bias from
0.27 N to 0.05 N (model error is dominated by *vertical* mass error, and the fork
is horizontal at PRESENT), and re-zeroing takes it to 0.01 N.

Nominal meal vs. the lean fault (`python sim/demo_meal.py`):

| | no fault | user leans 23° mid-transport |
|---|---|---|
| peak force into the user | none — no contact | **568 N** |
| minimum whole-arm clearance | +44 mm | −5 mm |
| bites detected | 2/2 | 1/1 |

The fault case is the architecture's §0 claim made concrete: the path *was*
clearance-checked, against the geometry as it was when the plan was made. The
person then moved. That is the argument for impedance control in the approach
and for a runtime monitor with its own view of the world, rather than a planner
that checks once. **Treat 568 N as a direction, not a prediction** — its
magnitude is set by two stiff position controllers fighting, and the real number
depends on gains this model does not claim to match.

## Not modelled

Deformable food — rigid contact cannot represent piercing (with contact on, the
tines punt an 8 g cube 1.4 m across the table), so fork-vs-food contact is
excluded and the grasp is a weld. "Retained" therefore means the constraint
held, not that physics proved the piece would not slip. There is no
force-terminated skewer; insertion depth is commanded, not sensed. Label errors
in detection (a shadow read as food, two touching cubes read as one) need the
real model to characterise. The depth camera's minimum range is a guess
(architecture §9) and the camera intrinsics are derived from the MuJoCo frusta,
not from the published `camera_info` — anything calibrated against them is
calibrated against the simulator.
