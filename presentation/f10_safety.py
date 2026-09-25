import textwrap
from matplotlib.patches import FancyBboxPatch
from style import *

rows = [
    ("09-23", "A single-joint wrist move swung the fork into joint 6 and broke it",
     "Only complete 7-joint targets; the utensil's tilt is checked on every move; a torque-step guard"),
    ("09-23", "A spoon broke: camera-to-tip read from a ruler in inches, not cm",
     "Per-utensil calibration: ruler to bare table + the depth sensor, two methods that must agree"),
    ("09-23", "After an e-stop the arm sagged 100-160 mm",
     "Documented; at the face the arm stops level with the mouth, so a sag carries the fork down, away"),
    ("09-24", "An approach used a stale detection: the user had moved back",
     "Only a detection made at this pose, newer than its capture, under 2 minutes old"),
    ("09-24", "An empty bowl would have re-used an old target and plunged at nothing",
     "perceive says 'the bowl is empty'; acquire refuses unless the newest run has a target"),
    ("09-24", "From the bowl in front, the swing would have swept the fork through the user's face",
     "The swing refuses scoop ends outside its validated region; the scoop backs off to one first"),
    ("09-24", "Re-running an interrupted retract would have 'backed off' toward the user",
     "The retract recognises which legs are done; from any pose it does not know, it refuses"),
    ("09-24", "The contact guard stopped the arm on its own acceleration, and when the user sat down",
     "Guard matched to each move: 2.5 N·m in the plunge, 4 near the face, 6 in open air"),
    ("09-25", "The depth fallback is right on average but up to ~60 mm off locally",
     "The plunge refuses a mostly-filled floor; the approach stands 40 mm further off"),
]
fig = frame("Design · safety",
            "Every guard has a story",
            "Each check in the loop exists because something went wrong, or nearly did, during development",
            "from the development log, 2026-09-23 to 09-25 · every trajectory is also validated by the Kinova controller before it moves")
ax = fig.add_axes([0.045, 0.05, 0.91, 0.71]); ax.set_xlim(0, 100); ax.set_ylim(0, len(rows)); ax.axis("off")
ax.text(8, len(rows) + 0.05, "WHAT HAPPENED", family=BODY, size=9, color=BERRY, weight="demibold", va="bottom")
ax.text(55, len(rows) + 0.05, "THE GUARD IT PRODUCED", family=BODY, size=9, color=LEAF, weight="demibold", va="bottom")
ax.plot([4.5, 4.5], [0.25, len(rows) - 0.25], color=GRID, lw=2)
for k, (d, what, guard) in enumerate(rows):
    y = len(rows) - 0.5 - k
    ax.scatter(4.5, y, s=46, color=BERRY if "nearly" not in what else AMBER, zorder=3, edgecolor=BG, lw=1.5)
    ax.text(0, y, d, family=MONO, size=8.5, color=MUTED, va="center")
    ax.add_patch(FancyBboxPatch((7.5, y - 0.40), 43.5, 0.80, boxstyle="round,pad=0,rounding_size=0.25", fc=PANEL, ec="none"))
    ax.text(8.6, y, "\n".join(textwrap.wrap(what, 58)), family=BODY, size=9.2, color=TEXT, va="center", linespacing=1.25)
    ax.annotate("", xy=(54, y), xytext=(51.6, y), arrowprops=dict(arrowstyle="-|>", color=DIM, lw=1.2))
    ax.text(55, y, "\n".join(textwrap.wrap(guard, 62)), family=BODY, size=9.2, color=LEAF, va="center", linespacing=1.25)
fig.savefig("out/10_safety.png")
print("ok")
