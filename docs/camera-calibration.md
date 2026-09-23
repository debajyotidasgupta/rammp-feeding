# Camera calibration — what actually needs doing

Short answer: **the intrinsics are probably fine and are worth ten minutes to
verify; the extrinsics do not exist yet and are mandatory.** Those are different
jobs, and the second one is the one that blocks the feeding loop.

There are four transforms in the chain between "a camera saw the mouth" and "the
arm moved the fork there". Two are factory-known, two are not known by anyone.

```
  mouth pixel ──(1 intrinsics)──► ray in camera frame
      │
      └──(2 camera → robot base)──► point in base frame      ← NOT KNOWN
                    │
                    └──(3 base → flange)── from joint encoders, known
                              │
                              └──(4 flange → fork tine tips)──► the TCP  ← NOT KNOWN
```

(3) is the arm's own forward kinematics and is as good as the encoders. (1) is
factory. **(2) and (4) are the work.**

---

## 1. Intrinsics — verify, probably don't recalibrate

### The Kinova vision module

Each module is factory-calibrated and stores its own intrinsics, readable over
the Kortex API. **`kinova_vision` does not use them by default.** It publishes
`camera_info` from generic files shipped inside the package:

```
ros2_kortex_vision/launch/calibration/default_color_calib_1280x720.ini
ros2_kortex_vision/launch/calibration/default_depth_calib_480x270.ini
```

and only reads per-unit values if you pass `color_camera_info_url` /
`depth_camera_info_url`. So out of the box you are consuming *a* calibration,
not *your* calibration, and nothing warns you.

What those defaults contain:

| file | fx | fy | cx | cy | cx/w | cy/h |
|---|---|---|---|---|---|---|
| color 320×240 | 331.289 | 330.585 | 156.143 | 117.923 | 0.488 | 0.491 |
| color 640×480 | 653.682 | 651.856 | 311.753 | 232.401 | 0.487 | 0.484 |
| **color 1280×720** | 1297.673 | 1298.631 | 620.914 | **238.280** | 0.485 | **0.331** |
| **color 1920×1080** | 2612.346 | 2609.466 | 946.434 | **96.968** | 0.493 | **0.090** |
| depth 424×240 | 314.625 | 314.625 | 212.257 | 120.856 | 0.501 | 0.504 |
| depth 480×270 | 360.013 | 360.013 | 243.872 | 137.922 | 0.508 | 0.511 |

`cx/w` sits at ~0.49 everywhere. `cy/h` is centred at 4:3 and **is not centred at
either 16:9 resolution**, and does not scale between them. Distortion is all
zeros throughout.

### What the device actually says — measured 2026-09-22

Run against the arm (`kinova/scripts/camera_intrinsics.py`, vision device 9):

**Colour: the device reports exactly the shipped numbers, to three decimals, at
all four resolutions** — including `cy` = 238.280 at 720p and 96.968 at 1080p.
So the odd principal point is not a packaging mistake; it is what the module
claims about itself.

That does not make it right, and one of the two is provably not:

- At 1280×720 an off-centre crop is *possible*. A 4:3 frame 1280 wide is 960
  tall with `cy` ≈ 0.484 × 960 ≈ 465; a 720-tall window with `cy` = 238.28 sits
  at rows 227–947, which fits inside 960.
- At 1920×1080 it is **impossible**. The 4:3 frame is 1440 tall with `cy` ≈ 697;
  a 1080-tall window with `cy` = 96.97 would start at row 600 and end at row
  1680, past the bottom of a 1440-row sensor.

So the 1080p profile is wrong, and 720p is unverified rather than vindicated.
Treat the colour principal point as **unknown until you calibrate it yourself**
with a target. Nothing that aims at a face should rest on it.

**Depth: the device does NOT match the shipped file.**

| | device | shipped default | |
|---|---|---|---|
| fx, fy @480×270 | **334.810** | 360.013 | shipped **+7.53%** |
| cx | **238.536** | 243.872 | +5.34 px |
| cy | **139.694** | 137.922 | −1.77 px |
| fx, fy @424×240 | **297.609** | 314.625 | shipped +5.72% |

A focal-length error is a *scale* error, so it grows off-axis:

| scene | true X | using the shipped file | error |
|---|---|---|---|
| food on plate, Z = 0.35 m, 100 px off-axis | 104.5 mm | 102.4 mm | −2.1 mm |
| plate edge, Z = 0.35 m, 180 px off-axis | 188.2 mm | 180.2 mm | **−8.0 mm** |
| face at mouth, Z = 0.50 m, 150 px off-axis | 224.0 mm | 215.7 mm | **−8.3 mm** |

8 mm is half a food cube. This was live in every depth deprojection until now.
`calibration/depth_480x270.ini` holds the device's values and `make depth` passes
them with `depth_camera_info_url`.

**Depth → colour extrinsics (factory, from the device):**

```
t = [-0.02706  -0.00997  -0.00471] m      |t| = 29.2 mm
R = identity
```

A 29 mm baseline, no rotation. Any operation that finds a pixel in the colour
image and samples the depth image at it must account for this.

### The RealSense D435i

Factory intrinsics live in the camera's EEPROM and `librealsense` applies them
automatically, including the depth↔colour extrinsics. Generally good to ~1–2%.
Recalibrate only if the depth looks visibly warped.

One hardware note first: the D435i currently enumerates at **480 Mbps (USB 2.0)**
on the USB 2 hub, while the USB 3 hub sits empty. On USB 2 librealsense clamps
the available profiles hard. Move the cable before characterising anything.

---

## 2. Camera → robot base. Mandatory, and nobody has it.

The static D435i is what makes face tracking possible at all (architecture
§4.3: the wrist camera cannot do this job — it is on the moving arm and the fork
occludes the mouth exactly when it matters). It reports the mouth in **camera**
coordinates. The arm needs it in **base** coordinates. Nothing at the factory
knows where the camera was bolted to the chair.

This is eye-to-hand calibration (camera fixed, target moving with the arm):

1. Mount a fiducial rigidly on the flange — not on the fork, which flexes.
2. Drive the arm to 15–25 poses spread over the region where feeding happens,
   with **varied orientation**, not just varied position. Pure translation
   leaves the rotation unobservable and the solve will look converged and be
   wrong.
3. Record (base→flange from the encoders, camera→target from the fiducial).
4. Solve `AX = XB` for camera→base.
5. Validate on **held-out poses**, reporting reprojection error in millimetres
   at the working distance. Do not report the fit residual as the accuracy.

Accuracy target: the fork is presented ~90 mm from the mouth and the head no-go
sphere has a 123 mm radius, so a 10 mm error is tolerable and a 30 mm error is
not. Aim to demonstrate **< 10 mm** on held-out poses.

Re-do it whenever the camera is bumped, and treat "was the camera moved?" as a
standing question before each session. A mount that can drift silently is worth
replacing with one that cannot.

## 3. Flange → fork tine tips. Also mandatory.

The fork is a custom part held in a 2F-85. Its tip pose is in **no** URDF. The
driver's `/ee_state` reports the flange/tool frame from
`models/gen3_7dof_2f85.urdf`, which models the gripper and knows nothing about a
fork.

Everything the skewer does is expressed at the tine tips: insertion depth
(14.3 mm through a 15 mm cube), the plate floor, the mouth standoff. An
unmeasured tool offset is a systematic error in every one of them.

Measure it by pivot calibration: touch one fixed physical point from ≥4 widely
different orientations and solve for the offset that keeps the tip stationary.
The gripper is already closed on the fork, so this can be done now, and must be
redone if the fork is ever re-seated in the jaws.

## 4. Wrist camera → flange

Kinova publishes a nominal transform. Probably adequate; verify it once against
the same fiducial used in §2 and move on.

---

## Order of work

1. Move the D435i to a USB 3 port. (Minutes, and it changes what is measurable.)
2. ~~Run `camera_intrinsics.py`~~ **done** — the depth stream was 7.53% off and
   is now corrected via `calibration/depth_480x270.ini`. What remains is the
   colour principal point, which the device reports but cannot be right at
   1080p; calibrate colour against a target.
3. Pivot-calibrate the fork TCP.
4. Eye-to-hand for the D435i → base, validated on held-out poses.
5. Only then trust any 3D number the perception stack produces.

Until 3 and 4 exist, the feeding loop can be developed in simulation — where all
four transforms are exact by construction — but cannot be run on hardware
against a person.
