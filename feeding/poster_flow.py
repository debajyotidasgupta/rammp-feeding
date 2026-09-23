#!/usr/bin/env python3
"""Render the system flow diagram for the poster.

Deliberately drawn rather than hand-made in a vector tool so it stays in sync
with the repo: every box names the module that implements it and every measured
number is the one in docs/. Redraw after any measurement changes.
"""
from __future__ import annotations

import argparse
from PIL import Image, ImageDraw, ImageFont


def font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


C = {
    "sense":  (219, 234, 254),   # blue
    "percept":(220, 252, 231),   # green
    "skill":  (254, 243, 199),   # amber
    "control":(252, 231, 243),   # pink
    "safety": (254, 226, 226),   # red
    "hw":     (229, 231, 235),   # grey
}
EDGE = (55, 65, 81)


def box(d, xy, title, lines, fill, w_title=None):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle([x0, y0, x1, y1], 10, fill=fill, outline=EDGE, width=2)
    d.text((x0 + 12, y0 + 9), title, fill=(17, 24, 39), font=w_title or font(17, True))
    y = y0 + 34
    for ln, col in lines:
        d.text((x0 + 12, y), ln, fill=col, font=font(13))
        y += 17


def arrow(d, p0, p1, label=None, col=EDGE, dashed=False):
    x0, y0 = p0; x1, y1 = p1
    if dashed:
        n = 14
        for i in range(0, n, 2):
            a = (x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
            b = (x0 + (x1 - x0) * (i + 1) / n, y0 + (y1 - y0) * (i + 1) / n)
            d.line([a, b], fill=col, width=3)
    else:
        d.line([p0, p1], fill=col, width=3)
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    for s in (2.6, -2.6):
        d.line([(x1, y1), (x1 + 15 * math.cos(ang + s), y1 + 15 * math.sin(ang + s))],
               fill=col, width=3)
    if label:
        # Centre the label in the gap between the boxes and knock out the line
        # behind it -- the gaps are only ~60 px, so an offset label lands on a
        # box border and becomes unreadable.
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        f = font(12, True)
        bx = d.textbbox((mx, my - 13), label, font=f, anchor="mm")
        d.rectangle([bx[0] - 3, bx[1] - 2, bx[2] + 3, bx[3] + 2], fill="white")
        d.text((mx, my - 13), label, fill=col, font=f, anchor="mm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/poster-flow.png")
    a = ap.parse_args()
    W, H = 1900, 1010
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    G = (107, 114, 128); K = (17, 24, 39)

    d.text((40, 24), "Robot-assisted feeding — system flow", fill=K, font=font(30, True))
    d.text((40, 62), "Kinova Gen3 · inline fork · joint-torque sensing only · "
                     "measured values in grey", fill=G, font=font(15))

    # ---- sensing column -----------------------------------------------------
    box(d, (40, 110, 400, 250), "SENSING", [
        ("wrist colour  1280x720 @30 Hz", G),
        ("wrist depth   480x270 16UC1", G),
        ("   fx 334.810 (device)", G),
        ("   NOT 360.013 (shipped, 7.5% off)", G),
        ("static D435i  640x480 @30 Hz", G),
        ("joint torque  0.010-0.045 Nm noise", G),
    ], C["sense"])

    box(d, (40, 280, 400, 420), "PERCEPTION", [
        ("LocateAnything-3B  BF16", K),
        ("  prompt: food pieces</c>bowl", K),
        ("  3 boxes, 2.2 s", G),
        ("RANSAC plane fit  (wrist depth)", K),
        ("  table 774 mm, 1.09 mm resid", G),
        ("face track  (static cam)  NOT YET", (185, 28, 28)),
    ], C["percept"])

    # ---- skill column -------------------------------------------------------
    box(d, (460, 110, 900, 470), "BITE CYCLE  (state machine)", [
        ("SCAN        detect food + plane", K),
        ("SELECT      reachable? rim clearance?", K),
        ("ALIGN       lateral 87 mm/s", K),
        ("DESCEND     86-89 mm/s, staircase", K),
        ("DESC_FINAL  51 mm/s  <- 40% slower", K),
        ("   terminate on STIFFNESS SHAPE", (185, 28, 28)),
        ("SCOOP       93 mm lateral, dz=1 mm", K),
        ("SETTLE      3.5 s", K),
        ("TRANSPORT   483 mm @ 62 mm/s", K),
        ("   constant z — NO LIFT", (185, 28, 28)),
        ("PRESENT     hold; bite or timeout", K),
        ("RETRACT", K),
    ], C["skill"])

    # ---- control column -----------------------------------------------------
    box(d, (960, 110, 1380, 300), "CONTROL", [
        ("kinova-gen3-ros2  (arm64 native)", K),
        ("ee_pose_impedance  during approach", K),
        ("  tracking lag ~7 mm (measured)", G),
        ("Jetson AGX Orin, PREEMPT_RT", K),
        ("  RT loop SCHED_FIFO 80, core 11", G),
        ("  1 kHz cyclic", G),
    ], C["control"])

    box(d, (960, 330, 1380, 470), "ARM", [
        ("Gen3 7-DoF, knee-height mount", K),
        ("2F-85 + FIXED INLINE FORK", K),
        ("no wrist F/T sensor", (185, 28, 28)),
        ("tool frame = flange + 120 mm", G),
        ("fork TCP  UNMEASURED", (185, 28, 28)),
    ], C["hw"])

    # ---- safety, spanning ---------------------------------------------------
    box(d, (1440, 110, 1860, 470), "SAFETY MONITOR", [
        ("independent view, own authority", K),
        ("", G),
        ("WHOLE-ARM clearance", K),
        ("  14 arm geoms x 4 no-go volumes", G),
        ("  tip-only missed a 411 N collision", (185, 28, 28)),
        ("  (tip read 84-90 mm clear)", G),
        ("", G),
        ("path checked, not just endpoints", K),
        ("  endpoints +74 mm, midpoint -161 mm", G),
        ("", G),
        ("IK redundancy guarded", K),
        ("  same goal: +45 vs -142 mm", G),
        ("", G),
        ("/estop  ·  timeout  ·  gamepad", K),
    ], C["safety"])

    # ---- the terminator callout --------------------------------------------
    box(d, (460, 505, 900, 745), "DESCENT TERMINATOR  (the result)", [
        ("stiffness RISES then FALLS", K),
        ("   -> skin broke, tines in flesh -> STOP", (21, 128, 61)),
        ("stiffness KEEPS RISING", K),
        ("   -> rigid surface -> ABORT", (185, 28, 28)),
        ("", G),
        ("flesh   0.270 -> 0.155 N/mm", G),
        ("bowl    0.208 -> 0.369 N/mm", G),
        ("a force LEVEL cannot separate them:", K),
        ("the bowl run passed through 5.7 N", G),
        ("on its way to 14 N", G),
    ], C["skill"])

    box(d, (960, 505, 1380, 745), "BITE SIGNAL", [
        ("axial pull at the tines, 5 N:", K),
        ("  joint 1  29%     joint 5   0%", G),
        ("  joint 2  33%     joint 6   0%", G),
        ("  joint 3  23%     joint 7   0%", G),
        ("  joint 4  15%", G),
        ("", G),
        ("spherical wrist: axes intersect, so", K),
        ("an axial force makes no moment", K),
        ("-> watch joints 1-4, never the wrist", (185, 28, 28)),
        ("(a perpendicular fork inverts this)", G),
    ], C["control"])

    box(d, (1440, 505, 1860, 745), "SIMULATION  (MuJoCo)", [
        ("fault injection before hardware:", K),
        ("  detection blackout / phantom food", G),
        ("  face track lost mid-transport", G),
        ("  depth stall (real: every ~65 s)", G),
        ("  bite never comes", G),
        ("  user leans into the fork", G),
        ("", G),
        ("seated SMPL-X user, lumbar hinge", K),
        ("  20 deg lean -> mouth +161 mm", G),
        ("FK validated vs real arm: 0.0 mm", (21, 128, 61)),
    ], C["percept"])

    # ---- arrows -------------------------------------------------------------
    arrow(d, (400, 170), (460, 170), "images")
    arrow(d, (400, 330), (460, 330), "3D goals")
    arrow(d, (220, 250), (220, 280))
    arrow(d, (900, 200), (960, 200), "setpoints")
    arrow(d, (1170, 300), (1170, 330))
    arrow(d, (1380, 380), (1440, 380), "state", col=(185, 28, 28))
    arrow(d, (1440, 160), (1380, 160), "/estop", col=(185, 28, 28))
    arrow(d, (1160, 470), (1160, 505), None, col=(185, 28, 28))
    arrow(d, (960, 620), (900, 620), "tau", col=(185, 28, 28))
    arrow(d, (680, 470), (680, 505))
    # feedback: arm state back to perception/skill
    d.line([(1060, 470), (1060, 492), (425, 492), (425, 335)], fill=(107, 114, 128), width=2)
    d.text((706, 474), "joint state  40 Hz / 1 kHz", fill=G, font=font(12, True))

    # ---- footer -------------------------------------------------------------
    d.text((40, 775), "NOT YET SOLVED", fill=(185, 28, 28), font=font(19, True))
    for i, t in enumerate([
        "fork TCP unmeasured — pivot calibration failed (32–47 mm residual); dominates the 4 mm flesh-to-bowl window",
        "static camera → robot base extrinsics do not exist, so the face track cannot drive motion",
        "wrist video drops every ~65 s and its timestamps do not preserve wall time — demos are not yet (observation, action) pairs",
    ]):
        d.text((40, 807 + i * 22), "•  " + t, fill=(75, 85, 99), font=font(14))

    d.text((40, 895), "MEASURED ON THIS SETUP", fill=(21, 128, 61), font=font(19, True))
    cols = [
        ["table height        774 mm  (1.09 mm residual)",
         "depth intrinsics    7.53% error found + corrected",
         "food detection      2.2 s, 3/3 correct"],
        ["flesh↔bowl window   4 mm",
         "scoop               93 mm lateral, dz = 1 mm",
         "lift                none — slides across the rim"],
        ["joint torque noise  0.010–0.045 N·m",
         "impedance lag       ~7 mm",
         "FK vs real arm      0.0 mm"],
    ]
    for c, col in enumerate(cols):
        for i, t in enumerate(col):
            d.text((40 + c * 620, 927 + i * 22), t, fill=(55, 65, 81), font=font(14))

    im.save(a.out)
    print("wrote %s  (%dx%d)" % (a.out, W, H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
