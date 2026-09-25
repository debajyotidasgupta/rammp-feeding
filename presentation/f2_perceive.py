import json, re
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.patches import Ellipse, Rectangle, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from style import *

img = Image.open("data/perceive_colour.jpg").convert("RGB")
img = ImageEnhance.Brightness(ImageEnhance.Color(img).enhance(0.75)).enhance(0.62)
det = json.load(open("data/perceive_detection.json")); tg = json.load(open("data/perceive_target.json"))
log = open("data/perceive_locate_log.txt").read()
rows = [tuple(map(float, m)) for m in re.findall(r"berry at \(\s*(\d+),\s*(\d+)\)\s+top (\d+) mm\s+floor beside it (\d+) mm\s+height (\d+) mm", log)]
VP = np.array([657.6, 236.6]); TIP = np.array([644, 405]); U = [640, 405, 750, 719]
pstar = np.array(tg["p_star"]); c = np.array(tg["berry_bbox_centre"]); dx, dy = tg["delta_xy_m"]

fig = frame("Acquisition · perception",
            "What the arm sees from above the bowl",
            "One frame at food_detect_start: every berry measured in depth, the most central one chosen, and the slide that puts the fork over it",
            "LocateAnything-3B (resident, bowl crop, fast mode) · depth sensor 480x270, 15-frame median · run 2026-09-24 20:22")
ax = fig.add_axes([0.035, 0.07, 0.62, 0.72]); ax.imshow(img); ax.set_xlim(250, 1110); ax.set_ylim(719, 0)
ax.axis("off")

# bowl: the inscribed ellipse of its box, as a ring
b = det["bowl"][0]
for w_, a_ in ((10, 0.10), (2, 0.9)):
    ax.add_patch(Ellipse(((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2), b["x2"] - b["x1"], b["y2"] - b["y1"] + 60,
                         fill=False, ec=CYAN, lw=w_, alpha=a_ * 0.6))
tag(ax, b["x1"] - 12, 600, "BOWL\ndetected once\nper session", CYAN, size=9, ha="right")

# fork: stored region, never detected
ax.add_patch(Rectangle((U[0], U[1]), U[2] - U[0], U[3] - U[1], fc=VIOLET, alpha=0.13, ec=VIOLET, lw=1.2, ls="--"))
tag(ax, U[2] + 10, 690, "FORK\nstored region\nnever detected", VIOLET, size=8.5)
ax.scatter(*TIP, s=38, color=VIOLET, edgecolor=BG, zorder=6)

def brackets(x1, y1, x2, y2, col, L=16, lw=2.2):
    for (x, y, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        ax.plot([x, x + sx * L], [y, y], color=col, lw=lw, solid_capstyle="round", path_effects=glow(col, 6, 0.2))
        ax.plot([x, x], [y, y + sy * L], color=col, lw=lw, solid_capstyle="round", path_effects=glow(col, 6, 0.2))

tgt = min(range(len(det["food"])), key=lambda i: np.hypot(det["food"][i]["cx"] - c[0], det["food"][i]["cy"] - c[1]))
for i, f in enumerate(det["food"]):
    col = BERRY if i == tgt else "#F59AA4"
    brackets(f["x1"], f["y1"], f["x2"], f["y2"], col, lw=2.6 if i == tgt else 1.8)
    r = min(rows, key=lambda r: np.hypot(r[0] - f["cx"], r[1] - f["cy"]))
    lab = "BERRY %d · %d mm tall\ntop %d mm from the camera" % (i + 1, r[4], r[2])
    tag(ax, f["x1"], f["y1"] - 8, lab, col, size=8.5, va="bottom")

# target reticle
for rad, a_ in ((34, 0.9), (48, 0.35)):
    ax.add_patch(Circle(c, rad, fill=False, ec=BERRY, lw=1.6, alpha=a_))
ax.plot([c[0] - 60, c[0] - 38], [c[1], c[1]], color=BERRY, lw=1.6); ax.plot([c[0] + 38, c[0] + 60], [c[1], c[1]], color=BERRY, lw=1.6)
ax.plot([c[0], c[0]], [c[1] - 60, c[1] - 38], color=BERRY, lw=1.6); ax.plot([c[0], c[0]], [c[1] + 38, c[1] + 60], color=BERRY, lw=1.6)

# the tip's vertical line: through the image of true vertical and the tip pixel, to berry depth
ax.plot([TIP[0], pstar[0]], [TIP[1], pstar[1]], color=AMBER, lw=1.6, ls=(0, (4, 3)))
ax.scatter(*pstar, s=70, marker="x", color=AMBER, lw=2.2, zorder=7)
tag(ax, pstar[0] - 12, pstar[1] + 22, "p*  where the tip's vertical\nline meets berry depth", AMBER, size=8.5, ha="right")
ax.add_patch(FancyArrowPatch(pstar, c, arrowstyle="-|>,head_length=9,head_width=5", color=AMBER, lw=2.0,
                             connectionstyle="arc3,rad=-0.25", path_effects=glow(AMBER, 7, 0.2)))
tag(ax, (pstar[0] + c[0]) / 2 + 26, (pstar[1] + c[1]) / 2 + 58,
    "OVER  ΔX %+.1f mm  ΔY %+.1f mm\n+ fork aim offset (+5, −6) mm" % (1000 * dx, 1000 * dy), AMBER, size=9)
tag(ax, c[0] + 66, c[1] + 4, "TARGET\nmost central =\nmost rim clearance", BERRY, size=9, va="center")

# right column: the depth frame + the numbers
axd = fig.add_axes([0.675, 0.43, 0.29, 0.33])
d = np.load("data/perceive_depth.npy").astype(float); d[d <= 0] = np.nan
cm = LinearSegmentedColormap.from_list("depth", [BERRY, AMBER, LEAF, CYAN, "#1E4A5E"]); cm.set_bad(BG)
im = axd.imshow(d, cmap=cm, vmin=395, vmax=445, interpolation="nearest"); axd.axis("off")
axd.set_title("Depth sensor, same moment", loc="left", color=TEXT, family=DISPLAY, size=17, pad=8)
cb = fig.add_axes([0.675, 0.405, 0.29, 0.012]); cbar = fig.colorbar(im, cax=cb, orientation="horizontal")
cbar.outline.set_visible(False); cb.tick_params(colors=MUTED, labelsize=8); cb.set_xlabel("mm from the camera · black = no return", color=MUTED, size=8.5)
facts = [("berries found", "%d, in ~1 s" % len(det["food"])), ("heights", " · ".join("%d mm" % r[4] for r in rows)),
         ("floor under the target", "%d mm" % rows[[np.hypot(r[0]-c[0], r[1]-c[1]) for r in rows].index(min(np.hypot(r[0]-c[0], r[1]-c[1]) for r in rows))][3]),
         ("registration check", "±4 px"),
         ("1 pixel at berry depth", "≈ 0.34 mm")]
for k, (a_, v) in enumerate(facts):
    y = 0.285 - k * 0.046
    fig.text(0.675, y, a_.upper(), family=BODY, size=8, color=MUTED, weight="demibold")
    fig.text(0.965, y, v, family=MONO, size=9.5, color=TEXT, ha="right")
fig.savefig("out/02_perception_overlook.png")
print("ok", len(det["food"]), rows)
