# Teleop demo analysis — 2026-09-22

Two operator-driven demonstrations on the real arm, logged at 40 Hz through the
high-level API while the operator drove with the gamepad. Both are on
`triphammer:~/rammp_data/` and `jetson:/tmp/demo{1,2}`.

| | demo 1 | demo 2 |
|---|---|---|
| duration | 180 s (motion in the first 20 s) | 345 s, two full cycles |
| samples | 7 151 | 13 719 |
| outcome | **went too deep, tines loaded against the bowl** | **stopped in the flesh, then scooped and fed** |

The state log is the usable record. See "Video alignment is broken" below.

## The full motion, from demo 2

```
t= 85      arrive over the bowl
t=143-149  descend            z 609 -> 336 mm
t=149.4    deepest            z 336 mm, |F| 6.4 N
t=149-151  SCOOP              93 mm laterally at constant z (dz = -1 mm)
t=151-154  hold
t=155-162  transport          y +248 -> -236 mm, 484 mm, constant z
t=162-204  at the person      42 s
t=249      second cycle back to the bowl
t=310      back to the person
```

The scoop is not a wrist flick. It is **~93 mm of lateral travel at constant
depth**, immediately after the descent bottoms out and before any lift. The
simulator had this as a *rotation applied during the lift*, which is wrong twice
over: wrong motion, and too late.

## Terminating the descent: stiffness, not force

Both demos descended 136 mm in 3.6 s, so they are directly comparable.

| depth band (mm) | demo 1 stiffness | demo 2 stiffness |
|---|---|---|
| 0-23 | 0.108 | 0.165 |
| 23-45 | 0.140 | 0.157 |
| 45-68 | 0.087 | 0.086 |
| 68-91 | 0.115 | 0.097 |
| 91-113 | 0.208 | **0.270**  <- peak |
| 113-136 | **0.369**  <- still rising | **0.155**  <- fallen back |
| final \|F\| | **14.07 N** | **5.66 N** |
| peak \|F\| | 16.29 N | 8.20 N |

**The discriminator is the SHAPE, not a level.**

- Stiffness rises and then **falls** -> the skin has broken and the tines are in
  soft flesh. This is the stop condition.
- Stiffness **keeps rising** -> the tines are loading a rigid surface (the bowl
  base, or the plate). This is the abort condition.

A single force threshold cannot separate these: demo 1 passed through 5.7 N on
its way to 14 N, so any threshold low enough to stop demo 1 in time would also
have stopped demo 2 before it reached the fruit.

Finer binning of demo 1's last millimetres gives **2.36 N/mm** over the final
1.2 mm, against 0.08-0.17 N/mm in free descent — an 18x jump. The bowl is
unmistakable once touched. The point is to stop *before* that.

The window is narrow: in demo 1 the gap between "in the flesh" and "on the bowl
base" was **4 mm**. That is the error budget for everything downstream --
depth sensing, the fork TCP, and the controller's own tracking lag (~7 mm under
impedance, measured during the table scan).

## The kinematic model is validated

FK from the driver's joint angles reproduces its own reported `/ee_state` to
**0.0 mm** at `pinch_site` in `sim/assets/kinova_gen3/gen3_fork.xml`. So the
menagerie model and the real arm agree exactly, and the frames reconcile:

```
pinch_site        = bracelet_link - 61.5 mm along local z   ( = /ee_state )
Kortex tool frame = bracelet_link - 182   mm                ( = pinch_site - 120.5 mm )
arm's configured tool transform, read from the device:        120 mm
```

That 0.5 mm agreement between the measured 120.5 mm and the configured 120 mm is
the check that the whole chain -- Kortex degrees, 0..360 wrapping, radians, model
joint order -- is correct end to end.

Consequences:

- The **full arm configuration** can be reconstructed from the logged joints, not
  just the tool pose. `feeding/trace3d.py` renders it; the whole-arm clearance
  monitor can therefore be replayed against recorded demos.
- The simulator can be driven from real joint trajectories.
- `sim` tcp (the fork tine tips as modelled) sits at bracelet - 206.5 mm, i.e.
  **24.5 mm beyond** the arm's configured tool frame. That is a *prior* for the
  fork TCP from the model's fork geometry, NOT a measurement -- the fork in the
  model is a guess and the real one has never been measured.

## What the arm gives you for free

`BaseCyclic.RefreshFeedback()` over the high-level channel returns, per
actuator, `position / velocity / torque / current`, plus the base's own
`tool_external_wrench_force_{x,y,z}`. That last one is Kinova's gravity-
compensated external force estimate -- we do not have to derive it. It reads
3-8 N during free motion (not zero), so **relative** change is trustworthy and
absolute level is not.

Per-joint torque noise at rest, measured: **0.010-0.045 Nm**. The simulator had
been assuming 0.15 Nm, so the real arm is 3-15x quieter and bite/pierce
detection is correspondingly more sensitive than `sim/validate_bite.py`
predicted at the time.

## Video alignment is broken

The wrist RTSP feed dies and restarts roughly every **65 s** -- six times in a
six-minute recording. Segmented capture (`kinova/scripts/record_demo.sh`) keeps
the coverage, but the deeper problem is timestamps: segment 3 holds **36.5 s of
video for 65 s of wall time**, so frames are dropped AND the container clock
does not preserve real time. Video time cannot be mapped to state time, and the
error drifts within each segment.

For qualitative review this is fine. For `(observation, action)` training pairs
it is not: the action would be attributed to the wrong frame. Fixing it needs
either RTP timestamps carried through to the container, or a frame-level
timestamp sidecar written at capture.

## What this changes

1. **SKEWER gains a scoop phase** -- lateral, at depth, before any lift.
2. **Descent terminates on stiffness shape**, not a force level.
3. **The bowl base is the abort floor**, detected by monotonically rising
   stiffness, the same role the table plane plays for a plate.
4. The 4 mm window makes the **fork TCP** the dominant unknown. Manual pivot
   calibration failed (32-47 mm residual, see below); it needs either calipers
   or an automated contact probe.

## Calibration attempt: why it failed

Pivot calibration needs the tine tips on one physical point across varied wrist
orientations. The operator's gamepad supplies translation only, so the
orientations had to come from the script -- but the procedure required a 40 mm
lift between poses to avoid bending the fork, and re-finding contact by eye
after each lift is worth +-10-20 mm. The corrections the geometry actually
needed were **~5 mm**. The measurement noise exceeded the signal by 3-4x, and
the solve returned a 150 mm offset with a 32-47 mm residual.

The fix is to take the human judgement out: descend automatically and stop on
the wrench, so "touching" is measured rather than estimated. The stiffness
signature above is exactly that detector.
