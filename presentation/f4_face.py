import json
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.patches import Circle, FancyBboxPatch
from style import *

up = Image.open("data/face_colour.jpg").convert("RGB").transpose(Image.ROTATE_270)
up = ImageEnhance.Brightness(ImageEnhance.Color(up).enhance(0.7)).enhance(0.6)
lm = json.load(open("data/face_landmarks.json")); L = np.array(lm["landmarks"])
f = json.load(open("data/face_face.json"))
OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
        150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
mouth = L[[13, 14]].mean(0); nose = L[1]
opening = np.linalg.norm(L[13] - L[14]) / np.linalg.norm(L[61] - L[291])

fig = frame("Delivery · the face",
            "Finding the mouth, and waiting until it opens",
            "MediaPipe's 478-point face mesh on the wrist camera, rotated upright: the lips, their depth, and an open mouth as the signal to come in",
            "MediaPipe face landmarker (resident, CPU, ~25 ms) · depth sensor at each landmark, food pixels dropped · delivery 2026-09-24 19:48")
ax = fig.add_axes([0.03, 0.05, 0.40, 0.76]); ax.imshow(up); ax.axis("off")
x0, x1 = L[:, 0].min() - 150, L[:, 0].max() + 150; y0, y1 = L[:, 1].min() - 170, L[:, 1].max() + 230
ax.set_xlim(max(0, x0), min(up.size[0], x1)); ax.set_ylim(min(up.size[1], y1), max(0, y0))
ax.scatter(L[:468, 0], L[:468, 1], s=2.2, color=CYAN, alpha=0.75, linewidths=0)
ax.plot(L[OVAL, 0], L[OVAL, 1], color=CYAN, lw=1.0, alpha=0.6)
ax.plot(L[OUTER, 0], L[OUTER, 1], color=BERRY, lw=2.0, path_effects=glow(BERRY, 7, 0.3))
ax.plot(L[INNER, 0], L[INNER, 1], color=BERRY, lw=1.2, alpha=0.9)
for rad, a_ in ((22, 1.0), (36, 0.35)):
    ax.add_patch(Circle(mouth, rad, fill=False, ec=AMBER, lw=1.6, alpha=a_))
ax.scatter(*nose, s=40, color=LEAF, edgecolor=BG, zorder=5)
tag(ax, nose[0] + 60, nose[1] - 10, "NOSE  %.0f mm" % (1000 * f["nose"]["depth_m"]), LEAF, size=10)
tag(ax, mouth[0] + 60, mouth[1] + 8, "MOUTH  %.0f mm\nfrom the lips' own depth" % (1000 * f["mouth"]["depth_m"]), AMBER, size=10)

# right column: the mouth-open gauge and what the arm does with it
axg = fig.add_axes([0.50, 0.60, 0.44, 0.10])
axg.barh([0], [1.2], color=PANEL, height=0.5); axg.barh([0], [opening], color=BERRY, height=0.5)
axg.axvline(0.20, color=LEAF, lw=1.6); axg.axvline(0.02, color=DIM, lw=1, ls=":")
axg.set_xlim(0, 1.2); axg.set_yticks([]); axg.grid(False)
for sp in axg.spines.values(): sp.set_visible(False)
tag(axg, 0.215, 0.30, "open: above 0.20 for 0.3 s", LEAF, size=8.5, ha="left", va="bottom")
tag(axg, opening + 0.02, 0, "%.2f" % opening, TEXT, size=13, va="center")
fig.text(0.50, 0.745, "Mouth opening  = inner-lip gap ÷ mouth width", family=DISPLAY, size=19, color=TEXT)
fig.text(0.50, 0.555, "a closed mouth reads 0.00-0.02 on every saved frame", family=BODY, size=8.5, color=MUTED)

steps = [("1  Wait", "The arm holds 50 mm to the user's right, food on the fork. Open mouth = ready. "
          "No open mouth in 60 s = declined: the arm takes the food back."),
         ("2  Locate", "Lips from the mesh; depth read at the lip landmarks (the food on the fork, ~250 mm nearer, is dropped). "
          "Fallbacks: \"lips\" query, sidestep, nose estimate, Depth Anything 3 fill."),
         ("3  Approach", "Straight line to 80 mm short of the mouth and 10 mm below it, at 35 mm/s, eased in and out. "
          "Planned from this detection only, under 2 minutes old."),
         ("4  Bite", "The arm holds still. A bite is > 1 N on the tool-force estimate for 0.1 s; then it retracts, fast.")]
for k, (h, t) in enumerate(steps):
    y = 0.475 - k * 0.108
    fig.text(0.50, y, h, family=DISPLAY, size=18, color=AMBER if k == 0 else TEXT)
    import textwrap
    fig.text(0.60, y + 0.022, "\n".join(textwrap.wrap(t, 66)), family=BODY, size=9.6, color=MUTED, va="top", linespacing=1.35)
fig.savefig("out/04_face_mouth.png")
print("ok opening %.2f nose %.3f mouth %.3f" % (opening, f["nose"]["depth_m"], f["mouth"]["depth_m"]))
