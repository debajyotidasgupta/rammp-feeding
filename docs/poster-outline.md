# Poster outline — robot-assisted feeding on the Kinova Gen3

Everything below is measured on this setup unless marked otherwise. Numbers with
no source are not in the poster.

---

## Title block

**Force-Terminated Skewering for Robot-Assisted Feeding with an Inline Fork**

Subtitle candidates, pick one to match the emphasis you want:
- *What a wheelchair-mounted Gen3 can and cannot feel through its own joints*
- *Deriving a bite-acquisition controller from teleoperated demonstration*

---

## 1. The problem (top-left, small)

A 7-DoF arm moves a pointed fork toward a person's face, repeatedly, and must
stop **inside the food** and **not** in the container or the person.

One sentence on why it is hard, with the number that makes it concrete:

> The gap between "tines in the flesh" and "tines on the bowl base" measured
> **4 mm**.

**Figure:** the wheelchair setup photo — user, Gen3 at knee height, bowl on the
table, static D435i on the chair frame. Annotate the two cameras and the fork.

---

## 2. The setup (top-centre) — THE ANCHOR PANEL

| | |
|---|---|
| arm | Kinova Gen3 7-DoF, wheelchair-mounted forward of the knee |
| end effector | Robotiq 2F-85 gripping a **fixed inline fork** (no actuated wrist) |
| force sensing | **joint torque only** — no wrist F/T sensor |
| wrist camera | colour 1280x720, depth 480x270 16UC1 @ 30 Hz |
| static camera | Intel RealSense D435i on the chair frame, for the face |
| compute | Jetson AGX Orin, **PREEMPT_RT**, RT loop pinned to isolated core 11 |
| perception GPU | RTX 4070 Ti / 4090, LocateAnything-3B in BF16 |
| middleware | ROS 2 Humble, `kinova-gen3-ros2`, native arm64 |

**Say the inline fork out loud.** It is the single fact that makes this setup
different, and section 5 is the consequence.

**Figure:** the system flow diagram (`docs/poster-flow.png`, 1900x1010).
Regenerate with `python -m feeding.poster_flow` after any measurement changes --
every number in it is read off `docs/`, so a stale figure is a wrong figure.

---

## 3. Perception, and what it costs (left column)

- **Food:** LocateAnything-3B, open vocabulary. Prompt matters more than
  expected:

  | prompt | boxes | time |
  |---|---|---|
  | **`food pieces</c>bowl`** | **3, all correct** | **2.2 s** |
  | `food pieces` | 8, junk in the corners | 13.4 s |
  | `fruit` | 63, degenerate | 9.9 s |

  Naming the container with the food sharpens accuracy *and* is 6x faster.

- **Table:** RANSAC plane fit on the wrist depth stream.
  **774 mm above the floor, 1.09 mm residual, 31 717 inliers.**
  No monocular depth model needed — and a monocular model could not give this,
  because the quantity is a *separation between two parallel planes*, which is
  exactly what scale-ambiguous depth cannot resolve.

- **Calibration defect found:** the shipped `camera_info` for this arm's depth
  stream is **7.53% wrong** (fx 360.013 published vs **334.810** on the device).
  That is a scale error, so it grows off-axis: **8 mm at the plate edge**.

**Figure:** annotated wrist frame with `food pieces</c>bowl` boxes; beside it the
depth plane fit with the two parallel surfaces (table / floor) called out.

---

## 4. Learning the skill from teleoperation (centre column)

An operator drove the full motion on the real arm; state logged at 40 Hz
(13 719 samples) with per-joint torque.

**The descent is a staircase, not a plunge:**

```
DESCEND 89 mm/s  ->  ALIGN 87 mm/s  ->  DESCEND 86 mm/s  ->  ALIGN
   ->  DESCEND_FINAL 51 mm/s (40% slower)  ->  SCOOP  ->  hold 3.5 s
```

**Two things the demonstration corrected in our design:**

1. **The scoop is lateral at depth, before any lift** — 93 mm at constant z
   (dz = **1 mm**). We had modelled it as a rotation *during* the lift, which
   loses the piece in the first centimetre.
2. **There is no lift at all.** Height varied **1.9 mm** across the entire
   scoop -> transport -> present sequence (483 mm of travel, 40 s). The fork
   **slides out across the bowl rim**.

**Consequence — a constraint neither we nor the prior system had:** the pierce
depth is bounded from *both* sides. Deep enough to reach flesh, shallow enough
that the fork clears the rim. A deep bowl with low-sitting food may admit **no
valid depth**, which is a reachability check to run *before* committing to a
piece.

**Figure:** the 3D trace, phase-coloured, reconstructed from joint angles by FK
(`feeding/trace3d.py`). The front view shows the staircase and the flat
scoop-transport run in one picture.

---

## 5. THE RESULT — terminating on stiffness, not force (centre-right, LARGEST)

Two demonstrations, both descending 136 mm in 3.6 s. One stopped in the flesh;
one went too far and loaded the bowl base.

| depth band (mm) | hit the bowl | stopped in flesh |
|---|---|---|
| 91-113 | 0.208 N/mm | **0.270** <- peak |
| 113-136 | **0.369** still rising | **0.155** fallen back |
| final \|F\| | 14.07 N | 5.66 N |

```
stiffness rises then FALLS  ->  skin broke, tines in flesh  ->  STOP
stiffness keeps RISING      ->  loading a rigid surface     ->  ABORT
```

**A force threshold provably cannot separate these.** The bowl run passed
*through* 5.7 N on its way to 14 N, so any level low enough to catch it would
stop the good run short of the fruit. Only the **shape** works.

Free descent for reference: 0.08–0.17 N/mm. Bowl contact, finely binned:
**2.36 N/mm** over the final 1.2 mm — an 18x jump.

**Figure:** the two stiffness curves overlaid, with the STOP and ABORT regions
shaded. This is the poster's money figure.

---

## 6. Where the bite signal lives (right column)

For an inline fork the answer is counterintuitive, and it is the opposite of the
perpendicular-fork case.

Measured: 5 N axial pull at the tines, share of the torque signal per joint.

| joint | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| axial pull | 29% | **33%** | 23% | 15% | **0%** | **0%** | **0%** |
| lateral pull | — | — | — | — | \[wrist sees ~30% total] | | |

**The three wrist joints are blind to an axial bite.** The Gen3 has a spherical
wrist — joints 5/6/7 axes intersect at a point, and an axial force's line of
action passes through it, producing no moment about any of them.

A perpendicular fork puts the load exactly where a wrist sensor reads it. **Ours
puts it where only the shoulder and elbow can see it.** Same arm, opposite
answer, from fork orientation alone.

Supporting measurement: per-joint torque noise at rest is **0.010–0.045 N·m**,
3–15x quieter than we had assumed in simulation.

**Figure:** arm schematic with joints coloured by signal share; inset of the
spherical wrist showing the force line through the axis intersection.

---

## 7. Safety: clearance is a whole-arm property (bottom-right)

Found in simulation before it could happen on hardware:

> A tip-only monitor reported **84–90 mm** of clearance while the wrist and
> forearm were being driven into the user's chest at **411 N**. At first contact
> the fork tip was **381 mm** from anything.

A feeding arm reaches *around* a person; most of its length is nearer to them
than the tool is.

Two more, same source:
- **Redundancy decides whether the elbow is inside the user.** Two IK solutions
  of the same goal, both sub-millimetre, differ by **+45 mm** and **−142 mm** of
  clearance.
- **A straight line between two safe poses is not safe** — both endpoints
  cleared by 74 mm; the interpolation passed **161 mm inside** the torso.

**Figure:** the sim render at the moment of the 411 N contact, tip-clearance and
whole-arm clearance both labelled.

---

## 8. What is not solved (bottom strip, honest and short)

- **Fork TCP is unmeasured.** Manual pivot calibration failed: the corrections
  needed were ~5 mm and re-finding contact by eye contributed ±10–20 mm, giving
  a 32–47 mm residual. It dominates the 4 mm budget. Fix: automated contact
  probing using the stiffness detector above.
- **Camera→base extrinsics do not exist**, so the face track cannot yet drive
  motion.
- **Video/state alignment is broken** — the wrist RTSP session drops every
  ~65 s and container timestamps do not preserve wall time, so demonstrations
  cannot yet be used as (observation, action) training pairs.

---

## Layout

```
+----------------+------------------------+------------------+
| 1 problem      | 2 SETUP + flow diagram | 6 bite signal    |
| (photo)        |                        | (joint shares)   |
+----------------+------------------------+------------------+
| 3 perception   | 5 THE RESULT           | 7 safety         |
| (boxes, plane) |   stiffness curves     | (sim collision)  |
|                |   LARGEST PANEL        |                  |
+----------------+------------------------+------------------+
| 4 skill from teleop (3D trace, staircase)| 8 not solved     |
+------------------------------------------+------------------+
```

Reading order 1 → 2 → 5 → 6, with 3/4/7 as support. If a viewer reads only one
panel, it should be 5.
