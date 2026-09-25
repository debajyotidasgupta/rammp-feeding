from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import *

fig = frame("Architecture",
            "What runs where, and what stays loaded",
            "One Jetson AGX Orin beside the wheelchair: sensors, resident models, planning, and the arm's own controller",
            "Jetson AGX Orin 64 GB (MAXN) · Kinova Gen3 7-DoF + Robotiq 2F-85 · wrist camera colour 1280x720 + depth 480x270")
ax = fig.add_axes([0.03, 0.06, 0.94, 0.74]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

def box(x, y, w, h, title, lines, col, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=col, lw=1.6))
    ax.text(x + 1.2, y + h - 1.6, title, family=DISPLAY, size=15, color=col, va="top")
    for k, l in enumerate(lines):
        ax.text(x + 1.2, y + h - 5.2 - k * 2.6, l, family=BODY, size=8.6, color=TEXT if not l.startswith("·") else MUTED, va="top")

def arrow(a, b, col=DIM, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=6,head_width=3.5", color=col, lw=1.4, connectionstyle="arc3,rad=%.2f" % rad))

box(1, 36, 18, 21, "Wrist camera", ["colour 1280x720, RTSP", "depth 480x270, ROS", "· one frame in 1.5 s", "· fresh-depth median", "  in ~0.5 s (resident)"], CYAN)
box(1, 7, 18, 23, "Arm state", ["joints + torques, 40 Hz", "tool-force estimate", "· the bite trigger", "· contact guards", "· where every plan", "  starts from"], LEAF)

ax.text(23, 58.5, "RESIDENT, WARM BEFORE THE FIRST BITE  ·  make models", family=BODY, size=9, color=AMBER, weight="demibold")
box(23, 42, 23, 14.5, "LocateAnything-3B", ["berries, bowl · GPU 7.2 GB", "· bowl crop, fast mode,", "  stop on repeats: ~1 s"], BERRY)
box(23, 24.5, 23, 14.5, "Depth Anything 3", ["fills sensor holes · GPU 1.25 GB", "· fitted to the sensor", "· ~0.1 s warm"], VIOLET)
box(23, 7, 23, 14.5, "MediaPipe face mesh", ["478 landmarks · CPU", "· lips, nose, mouth opening", "· ~25 ms"], AMBER)

box(51, 32, 22, 25, "Perception", ["berry centre + height", "floor under the fork", "the fork tip's vertical line", "mouth position + depth", "· fallbacks: lips query,", "  sidestep, nose, DA3 fill"], CYAN)
box(51, 7, 22, 21, "Planning", ["IK homotopy on the 7th DoF", "per-joint time scaling", "  (speed + 40 °/s² caps)", "· plans cached when the", "  start pose repeats"], VIOLET)

box(78, 7, 21, 50, "Kinova Gen3", ["Kortex controller", "· validates every", "  trajectory before", "  it moves", "", "Guards, every move:", "· utensil tilt", "· joint limits (3-5° margin)", "· torque step, per stage", "  (2.5 / 4 / 6 N·m)", "· food off its path", "  > 15 mm stops the scoop", "· stale detection refused"], LEAF)

# bus lines instead of crossing arrows
def seg(xs, ys, col, lw=1.4):
    ax.plot(xs, ys, color=col, lw=lw, solid_capstyle="round")
seg([19.6, 21.2], [46.5, 46.5], CYAN); seg([21.2, 21.2], [14.2, 49.2], CYAN)
for y in (49.2, 31.7, 14.2): arrow((21.2, y), (22.5, y), CYAN)
seg([46.6, 48.6], [49.2, 49.2], MUTED); seg([46.6, 48.6], [31.7, 31.7], MUTED); seg([46.6, 48.6], [14.2, 14.2], MUTED)
seg([48.6, 48.6], [14.2, 49.2], MUTED); arrow((48.6, 44.5), (50.5, 44.5), MUTED)
arrow((62, 31.6), (62, 28.7), MUTED)
arrow((73.6, 17), (77.5, 17), LEAF)
seg([88.5, 88.5], [6.6, 4.6], LEAF); seg([88.5, 10], [4.6, 4.6], LEAF); arrow((10, 4.6), (10, 6.4), LEAF)
ax.text(49, 2.2, "joint states + torques back to every stage", family=BODY, size=8.5, color=LEAF, ha="center")
fig.savefig("out/09_system.png")
print("ok")
