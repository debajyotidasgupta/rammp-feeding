import numpy as np
from matplotlib.patches import Rectangle
from style import *

fig = frame("The whole loop · time",
            "Where one bite's time goes",
            "Moving, standing still, or waiting for the user: the first dry run, the timed test bite, and what today's changes leave",
            "dry run 211 s · timed test bite 126 s (stage timestamps + move durations), 2026-09-24 · third bar: projected")

# stage timeline of the timed bite: (name, start s, dur s, moving s, colour, note for the still part)
st = [("perceive", 0, 12, 0.0, CYAN, "see the bowl"), ("over", 12, 2, 0.8, CYAN, ""),
      ("plunge + scoop", 14, 47, 17.9, BERRY, "re-measure + plan: 29 s still"), ("swing", 61, 12, 10.2, VIOLET, ""),
      ("present", 73, 22, 9.8, AMBER, "wait 2.6 s · detect"), ("bite", 95, 5, 0.0, BERRY, "the user"),
      ("retract", 100, 26, 23.8, LEAF, "")]
axg = fig.add_axes([0.06, 0.13, 0.88, 0.28])
for i, (n, s0, dur, mov, c, note) in enumerate(st):
    user = 2.6 if n == "present" else (dur if n == "bite" else 0)
    still = dur - mov - user
    x = s0
    if still > 0:
        axg.add_patch(Rectangle((x, 0.15), still, 0.7, fc=PANEL, ec=GRID, hatch="////", lw=0.6)); x += still
    if user > 0:
        axg.add_patch(Rectangle((x, 0.15), user, 0.7, fc=DIM, ec=BG, lw=0.6)); x += user
    if mov > 0:
        axg.add_patch(Rectangle((x, 0.15), mov, 0.7, fc=c, ec=BG, lw=0.6))
    tag(axg, s0 + dur / 2, 1.02, n.upper(), c if n != "bite" else TEXT, size=9, ha="center", va="bottom")
    if note:
        tag(axg, s0 + dur / 2, -0.06, note, MUTED, size=8, ha="center", va="top")
axg.set_xlim(0, 126); axg.set_ylim(-0.45, 1.35); axg.set_yticks([]); axg.grid(False)
axg.set_xlabel("seconds into the timed test bite (126 s)")
for sp in ("left",): axg.spines[sp].set_visible(False)

# comparison bars: moving / still / user
bars = [("first dry run", 84.0, 125.0, 2.0, ""), ("timed test bite", 62.5, 58.5, 5.0, ""),
        ("with today's changes", 62.5, 16.0, 5.0, "projected")]
axb = fig.add_axes([0.20, 0.50, 0.74, 0.26])
for k, (n, mov, still, user, flag) in enumerate(bars):
    y = 2 - k
    axb.barh(y, mov, color=CYAN, height=0.56, alpha=0.95 if not flag else 0.6)
    axb.barh(y, still, left=mov, color=PANEL, height=0.56, hatch="////", edgecolor=GRID)
    axb.barh(y, user, left=mov + still, color=DIM, height=0.56)
    tot = mov + still + user
    tag(axb, tot + 2, y, ("~%.0f s" if flag else "%.0f s") % tot, TEXT if not flag else AMBER, size=15, va="center")
    fig.text(0.19, 0.50 + (y + 0.5) / 3.0 * 0.27, n + ("\n(%s)" % flag if flag else ""), family=BODY, size=10.5,
             color=TEXT if not flag else AMBER, ha="right", va="center", weight="demibold")
axb.set_axisbelow(True); axb.set_xlim(0, 240); axb.set_ylim(-0.5, 2.5); axb.set_yticks([]); axb.grid(axis="y", visible=False)
axb.spines["left"].set_visible(False)
axb.set_xlabel("seconds per bite")
axl = fig.add_axes([0.20, 0.785, 0.74, 0.03]); axl.set_xlim(0, 100); axl.set_ylim(0, 1); axl.axis("off")
for k, (lab, fc, h) in enumerate((("arm moving", CYAN, ""), ("arm still: computing, re-measuring, program start-up", PANEL, "////"),
                                  ("waiting for the user", DIM, ""))):
    x0 = (0, 17, 64)[k]
    axl.add_patch(Rectangle((x0, 0.15), 2.2, 0.7, fc=fc, ec=GRID, hatch=h, lw=0.6))
    tag(axl, x0 + 3, 0.5, lab, MUTED, size=8.5, va="center")
fig.savefig("out/06_time_per_bite.png")
print("ok")
