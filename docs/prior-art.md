# Ideas taken from `empriselab/feeding-deployment`

A deployed feeding system already exists on `emprise`
(`~/deployment_ws/src/feeding-deployment`). Same task, **different hardware**:

| | feeding-deployment | this setup |
|---|---|---|
| fork | **perpendicular**, on an actuated wrist (`set_wrist_state(pitch, roll)`) | **inline** with the tool axis, fixed in the gripper |
| force sensing | **custom wrist force/torque sensor** | joint torques only |
| container | flat **plate** | **bowl** with a rim |
| stack | ROS 1 Noetic, Ubuntu 20.04, separate NUC | ROS 2 Humble, Jetson, `kinova-gen3-ros2` |

So: take the ideas, not the code. Some of it is actively wrong for us, and the
hardware difference is why.

## The geometry difference has a sharp consequence

Their fork is perpendicular, so a bite pulls **laterally** on it. Ours is inline,
so a bite pulls **axially**. Measured on our arm: for an axial pull the three
wrist joints carry **0%** of the signal -- the Gen3's spherical wrist has its
three axes intersecting, so an axial force's line of action passes through the
point and generates no moment about any of them. For a lateral pull the wrist
sees ~30%.

Their geometry puts the load exactly where a wrist sensor reads it. Ours puts it
where the wrist is blind and only joints 1-4 see it. **Their sensing approach
does not transfer**; ours has to use the full Jacobian across the shoulder and
elbow (`docs/demo-analysis.md`).

## Worth stealing

**1. A geometric depth fallback that can only move you sideways.**
`skewering_skill` computes an expected depth from `camera_height - plate_height -
food_thickness` and compares the measured depth patch against it. If the
measurement is implausible it falls back to the geometric value -- but the z is
overwritten with the plate height either way, so *a wrong fallback can never
drive the fork deeper than a good reading would.* Failure is confined to the
lateral axis, which is recoverable; depth is not.

**2. Validate that a depth reading is plausibly ON THE SURFACE, not merely
present.** Their comment names the real failure modes: a match latching onto the
fork tine, a hand crossing the frame, the background behind the plate. Any of
those returns a confident, valid-looking number. Our `perception.py` rejects on
invalid-pixel fraction, which does not catch these at all.

**3. Skewer perpendicular to the food's MAJOR AXIS**, and offset ~5 mm along
that axis toward the end further from the robot. This answers the entry-angle
question `docs/skill-sequence.md` left open. `SkeweringOrientation` is exposed as
a user preference (horizontal | vertical).

**4. Parameters with ranges, defaults and user-editability**, not constants:
`SkeweringDepth` 0.01-0.05 m default **0.015**, `FoodDippingDepth` 0.01-0.03,
`Speed` low/medium/high. Their tuned 15 mm is within a millimetre of the 14.3 mm
our simulator derived independently for a 15 mm piece -- mutual confirmation from
two unrelated routes.

**5. Timed auto-continue with user override.** `BiteSelectionAutocontinueSeconds`
and `PickupConfirm` (-1 skip / 0 wait / >0 count down). This is the concrete form
of the "always pair force detection with a timeout and the gamepad" requirement
in architecture Sec. 6.

**6. Behaviour trees as declarative, parameterised YAML** with human-readable
descriptions, rather than parameters buried in code.

## Do NOT copy

**Driving the utensil down to the container surface.** Their code sets
`food_base[2,3] = plate_height` -- the fork descends to the plate. A commented-out
line shows `max(food_top - skewering_depth, plate_height)` was tried and
abandoned. That is reasonable for a perpendicular fork on a flat plate.

For us it is the exact failure we measured: inline tines driven to a **bowl**
base loaded to **16.3 N** at **2.36 N/mm** and bent the fork. And the bowl adds a
constraint their plate does not have -- the fork must stay **above the rim** to
slide out, since it never lifts (`docs/skill-sequence.md`).

So we keep the stiffness-shape terminator and the rim bound. Their depth rule is
sound for their hardware and dangerous on ours.

## Their joint-torque bite detector, and how ours differs

`.claude/worktrees/joint-torque-bite-detector/` holds a mature detector fit on
~300 bites over 20 deployment days, with a measured **0.07 false fires per
minute** on a real negative set. `feeding/bite_detector.py` is now rebuilt on its
structure. Their findings, and what each means here:

**Only joint 7 -- and that INVERTS for us.** Their fork is perpendicular, tip
~0.255 m off joint 7's axis: largest moment arm, quietest baseline (0.021 Nm sd),
SNR ~19, and it beats all-seven on every axis. Our fork is inline, and we
measured joints 5/6/7 carrying **0.0%** of an axial pull -- the Gen3's spherical
wrist axes intersect, so the force line passes through them. We watch joints 1-4
(33/29/23/15% of the signal). Same arm, opposite answer, from fork orientation
alone.

**A settling gate is mandatory.** While the arm settles into the transfer pose
its torque sd is 0.3-0.85 Nm, the same size as a bite -- no threshold separates
them. The detector must refuse to arm until the arm is measurably still and
report that (`ready`). Ours had no such gate: it re-zeroed for 200 ms and armed
regardless.

**Baseline needs a dead time.** The baseline window must end `gap` before now or
the bite's own leading edge lands in its reference. 150 ms upstream; our demo
showed a ~120 ms rise.

**Window length is the binding constraint, not the threshold.** Shortening
span/gap from 0.50/0.20 to 0.25/0.15 took the fraction of real bites the detector
could even ATTEMPT from **31% to 75%** at equal accuracy. Users bite sooner than
a long baseline allows. This is invisible without deployment data.

**An absolute floor beats k*sigma.** A bite peaks ~12x above pre-bite
disturbances (0.341 vs 0.029 Nm), so an absolute threshold separates two
populations while k*sigma scales with local noise and shrinks both. Detection
*rises* with the floor, because suppressing an early fire leaves the detector
armed rather than burning the window.

**Freeze the baseline.** A sliding baseline treats sustained load as normal, so a
user who bites and HOLDS stops being detected after ~0.4 s.

**Refresh off.** Wander during a still hold is episodic, not drift: worst
excursion 2.3x the net change (p90 7.3x). There is no trend to chase and
refreshing occasionally re-baselines onto the bite's leading edge.

Their `abs_min_nm = 0.25` is tuned for joint 7 with a 0.255 m moment arm. Ours is
a different channel and the number is carried over as a STARTING POINT only.

## The compliant controller

`control/robot_controller/compliant_controller.py`, after
`empriselab/gen3_compliant_controllers` (which ships a formulation PDF). Joint-
space, 1 kHz, with a disturbance-observer structure and Ruckig for online
trajectory generation. Tuned gains, 7-DoF case:

```
K_p  [100 x4, 50 x3]     K_d  [3 x4, 2 x3]
K_r  [0.3 x4, 0.18 x3]   K_l  [75 x4, 40 x3]   K_lp [5 x4, 4 x3]
DT 0.001   policy period 0.1 s   torque LPF alpha 0.01   damping 0.05
```

The 2:1 proximal-to-wrist split matches the arm's own torque limits (105 vs
52 Nm) and the `max_ref_speed` the driver prints (1.40 vs 1.22 rad/s), so it is
a property of the hardware rather than a tuning quirk -- useful as a sanity
reference for our impedance gains.

**They resolve the 7-DoF redundancy by LOCKING A JOINT** ("fix one of the joints
of 7DoF to become 6DoF"), with a separate gain set for the 6-DoF case. We use a
null-space elbow bias instead, after measuring that an unconstrained solve puts
the elbow inside the user. Locking is simpler and gives up reach; the bias keeps
reach and needs the clearance check. Worth knowing both work.

## The transfer protocol is richer than ours

`actions/transfer_tool.py` + `actions/feel_the_bite/`:

- **inside-mouth AND outside-mouth transfer**, selectable, with
  `outside_mouth_distance` a parameter. Our architecture Sec. 5 commits to
  outside-mouth only ("the arm holds; the user closes the last distance"). Theirs
  treats it as a user preference.
- the standoff is **calibrated**, not fixed: `calibrated_outside_mouth_distance`.
- a real interaction state machine: `relay_ready_to_initiate_transfer` ->
  `detect_initiate_transfer` -> `execute_transfer` -> `detect_transfer_complete`.
  Ours collapses this to PRESENT -> BITE.
- the mouth frame is explicitly oriented "facing away from the wheelchair".

## Head perception: DECA + Kabsch-Umeyama, at 10 Hz

`perception/head_perception/deca_perception.py` reconstructs a 3D face with DECA
and registers it to depth-derived landmarks via Kabsch-Umeyama, giving a head
pose and hence a mouth pose. Polled at **10 Hz** -- which answers our open
question about whether continuous face tracking can hold ~10 Hz: theirs does.

Two operational lessons in the comments, both hard-won:

- `wait_for_head_perception_data` **blocks until a valid reading arrives**
  rather than dereferencing a None frame. A DECA frame can be None for three
  distinct reasons (no face, too few depth landmarks, reading rejected as noisy).
- the noisy-readings filter is a **sticky flag on the perception node**, so it
  must be cleared on EVERY exit including a mid-skill teleop takeover. Left
  latched, every later transfer blocks on rejected readings -- which presents as
  head perception being broken rather than as a leaked flag.

## Practice worth copying outright

**Log the exact input that produced a safety-critical pose.** They save the
single RGB frame used to compute each bite-transfer pose, with the resulting pose
as metadata. When a transfer goes somewhere wrong, the frame that caused it is
already on disk.

## What to do about it here

1. Adopt the **depth fallback that can only move laterally** in
   `sim/perception.py` -- currently we reject on invalid-pixel fraction only.
2. Add the **surface-plausibility gate** (a reading can latch onto the fork, a
   hand, or the background and still look valid).
3. Add **per-pose frame logging** before any hardware transfer trial.
4. Re-sweep `abs_min_nm` for OUR joint set against real bites; 0.25 Nm is
   inherited from a channel with a 0.255 m moment arm and is a placeholder here.
5. Compare their compliant gains against the ~7 mm impedance tracking lag we
   measured, before tightening anything.
