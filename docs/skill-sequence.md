# The bite cycle, concretely

Every number here is measured from the teleop demo of 2026-09-22
(`docs/demo-analysis.md`), not chosen. Cycle 1 of demo 2, t=139-204 s, is the
reference: it is the one complete run where the operator pierced the fruit,
scooped it, carried it and fed it without hitting the bowl.

Where a number is a design choice rather than a measurement, it says so.

---

## The sequence

| # | phase | measured | terminator |
|---|---|---|---|
| 1 | `ALIGN_ABOVE` | hold over bowl, then lateral at **87 mm/s** | food detected, tines above target |
| 2 | `DESCEND_COARSE` | **86-89 mm/s**, in 100-135 mm steps | reach standoff above the piece |
| 3 | `ALIGN` | lateral **87 mm/s** at constant z, 60-85 mm | re-centred on the piece |
| 4 | `DESCEND_FINAL` | **51 mm/s** (40% slower), ~37 mm | **stiffness rise-then-fall** |
| 5 | `SCOOP` | **93 mm lateral at constant z**, **61 mm/s**, 1.5 s, dz = 0 | travel complete |
| 6 | `SETTLE` | **3.5 s** hold at depth | timer |
| 7 | `TRANSPORT` | **483 mm at 62 mm/s**, 7.9 s, **constant z** | reach mouth standoff |
| 8 | `PRESENT` | **25.6 s** hold | bite detected, or timeout |
| 9 | `RETRACT` | — | back to bowl |

Steps 2-4 alternate vertical and horizontal rather than plunging: the operator
descends, re-centres, descends, re-centres, and only then makes the final slow
approach. That is not hesitation, it is closing the loop on a target whose
position is only known to a few millimetres.

## Phase 4 is the one that matters

**Do not terminate on a force level.** Two demos, both descending 136 mm in
3.6 s:

| depth band (mm) | hit the bowl | stopped in flesh |
|---|---|---|
| 91-113 | 0.208 N/mm | **0.270** <- peak |
| 113-136 | **0.369** still rising | **0.155** fallen back |
| final \|F\| | 14.07 N | 5.66 N |

```
stiffness rises then FALLS  ->  skin broke, tines in flesh  ->  STOP
stiffness keeps RISING      ->  loading a rigid surface     ->  ABORT
```

The bowl run passed through 5.7 N on its way to 14 N, so any level low enough to
catch it would have stopped the good run short of the fruit. Only the shape
separates them.

Abort is not optional: the gap between "in the flesh" and "on the bowl base"
measured **4 mm**.

Free-descent reference: **0.08-0.17 N/mm**, \|F\| 3-8 N. Bowl contact, finely
binned: **2.36 N/mm** over the final 1.2 mm — an 18x jump, unmistakable once
touched. The job is to stop before it.

## There is no LIFT, and that is a constraint not an omission

Measured across the whole scoop -> transport -> present sequence: z varied by
**1.9 mm** over 483 mm of travel and 40 s. The fork was never raised.

Confirmed with the operator: **the fork slides out ACROSS THE BOWL RIM** at
constant height. It does not lift over it.

That makes the pierce depth bounded from BOTH sides:

```
deep enough    -> tines into the flesh          (stiffness rise-then-fall)
shallow enough -> fork body stays ABOVE THE RIM  (or it cannot slide out)
```

Neither the architecture nor `sim/` had the upper bound. It has consequences:

- The **bowl rim height** is a needed measurement, alongside the table plane and
  the food surface. Three heights, not two.
- The **scoop distance is set by geometry**, not chosen: it must carry the piece
  from the pierce point clear of the rim. 93 mm for this bowl.
- A **flat plate behaves completely differently** -- no rim to clear, so the
  constraint vanishes and a lift may be preferable. The skill is not
  container-agnostic.
- If the food sits too low in a deep bowl, there may be NO valid pierce depth:
  the tines cannot reach the flesh while the fork stays above the rim. That is a
  reachability check to run before committing to a piece, not a failure to
  discover halfway down.

## Phase 5 is what I had wrong

The scoop is **lateral travel at constant depth**, taken immediately after the
descent bottoms out and BEFORE any lift. Measured dz = -1 mm over 93 mm of
travel.

The simulator had it as a *rotation applied during the lift*. Wrong motion and
too late: the piece is lost in the first centimetre of lifting, which is
exactly what the operator reported seeing.

## Budgets

| quantity | value | source |
|---|---|---|
| flesh-to-bowl window | **4 mm** | demo 1 |
| impedance tracking lag | **~7 mm** | table scan |
| depth plane-fit residual | **1.09 mm** | table plane |
| depth systematic (pre-fix) | 8 mm | wrong intrinsics, now corrected |
| fork TCP | **UNKNOWN** | calibration failed |
| joint torque noise | 0.010-0.045 Nm | demo, at rest |

The TCP is the dominant unmeasured term and it alone could exceed the 4 mm
window. Until it exists, phase 4 cannot be run open-loop to a computed depth --
the stiffness terminator is doing all the work.

## What is NOT specified here

- **Approach tilt.** Architecture Sec. 5 calls for tilting before the descent so
  the tines self-align. The demo descended vertically throughout, so there is no
  measurement to quote. Worth a dedicated demo.
- **Where in the piece to enter.** The operator aimed by eye. `sim/skewer.py`
  targets the thickest cross-section; that has never been validated on real
  fruit.
- **Bite detection.** The 25.6 s PRESENT ended by operator judgement, not by a
  detected bite. The torque channel exists and is 3-15x quieter than simulated,
  but no real bite has been recorded yet.
