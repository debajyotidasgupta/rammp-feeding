import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from style import *

r = np.array(json.load(open("data/bite_torque_1.json"))); t = r[:, 0]; tq = r[:, 1:8]; wr = r[:, 15:18]
rest = np.array(json.load(open("data/rest_torque.json"))); sig = rest[:, 1:8].std(0)
base = np.median(tq[t < 8], 0); z = np.abs(tq - base) / sig
F = np.linalg.norm(wr - np.median(wr[t < 8], 0), axis=1)
Fs = np.convolve(F, np.ones(3) / 3, mode="same")

fig = frame("Delivery · bite detection",
            "The bite has a signature",
            "One recorded bite, 60 s at the mouth: the controller's tool-force estimate and all seven joints, against the noise at rest",
            "recorded 2026-09-24 · joint torques + tool-wrench estimate @ 40 Hz · arm still, 80 mm from the mouth")
ax = fig.add_axes([0.06, 0.39, 0.66, 0.39])
ax2 = fig.add_axes([0.06, 0.14, 0.66, 0.19], sharex=ax)
ax3 = fig.add_axes([0.78, 0.14, 0.18, 0.60])

# --- force trace
bite = (t > 26.4) & (t < 28.7)
ax.axvspan(26.4, 28.7, color=BERRY, alpha=0.10, lw=0)
ax.fill_between(t, 0, Fs, color=CYAN, alpha=0.10, lw=0)
ax.plot(t, Fs, color=CYAN, lw=1.4, path_effects=glow(CYAN))
ax.plot(t[bite], Fs[bite], color=BERRY, lw=1.8, path_effects=glow(BERRY, 8, 0.25))
for thr, c, lab, ls in ((1.0, LEAF, "1.0 N  trigger (in use)", "-"), (0.5, DIM, "0.5 N  first idea: fires on leaning in", ":"),
                        (2.0, DIM, "2.0 N  considered: too stiff", ":")):
    ax.axhline(thr, color=c, lw=1.1 if ls == "-" else 0.9, ls=ls)
    tag(ax, 59.4, thr + 0.12, lab, c, va="bottom", ha="right", size=8.5)
ip = np.argmax(F)
ax.scatter([t[ip]], [F[ip]], s=60, color=BERRY, zorder=5, edgecolor=BG, lw=1.5)
tag(ax, t[ip] + 0.8, F[ip] - 0.1, "BITE  %.1f N peak\n2.3 s, then released" % F[ip], BERRY, size=11, va="top")
# non-bite bumps
for tc in (12.2, 17.1, 39.1):
    m = (t > tc - 1.2) & (t < tc + 1.2); k = np.argmax(np.where(m, F, 0))
    ax.annotate("", xy=(t[k], F[k] + 0.08), xytext=(t[k], F[k] + 1.1),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.0))
    tag(ax, t[k], F[k] + 1.17, "%.1f N" % F[k], AMBER, ha="center", size=8.5)
tag(ax, 1.0, 3.35, "leaning in, lips on the food:\nbumps up to %.1f N, no bite" % max(F[(t > 11) & (t < 13.5)].max(), F[(t > 38) & (t < 40.5)].max()), AMBER, size=9)
after = np.median(F[t > 32]); tag(ax, 41.5, 2.55, "resting level after the bite: +%.2f N\nthe berry's weight has gone" % after, LEAF, size=9)
ax.set_ylim(0, 6.3); ax.set_xlim(0, 60)
ax.set_ylabel("tool force estimate, N (vs rest)")
ax.tick_params(labelbottom=False)

# --- joint heat strip
cm = LinearSegmentedColormap.from_list("heat", [BG, "#1E4A5E", CYAN, AMBER, BERRY])
ax2.imshow(np.log10(1 + z.T), aspect="auto", cmap=cm, extent=[t[0], t[-1], 7.5, 0.5], vmin=0, vmax=2.2, interpolation="nearest")
ax2.set_yticks(range(1, 8)); ax2.set_yticklabels(["J%d" % j for j in range(1, 8)], family=MONO, size=8)
ax2.grid(False); ax2.set_xlabel("seconds at the mouth")
tag(ax2, 0.6, 1.3, "each joint's torque change, in multiples of its own noise at rest", MUTED, size=8.5)
tag(ax2, 41, 4.6, "J5 settles at a new level: the food is gone", LEAF, size=8.5)

# --- per-joint SNR bars
peak = z[bite].max(0)
order = np.arange(7)
ax3.barh(order, peak, color=[BERRY if p > 80 else CYAN for p in peak], height=0.62)
for j, p in enumerate(peak):
    tag(ax3, p + 3, j, "%.0f×" % p, TEXT, va="center", size=9)
ax3.set_yticks(order); ax3.set_yticklabels(["J%d" % (j + 1) for j in order], family=MONO)
ax3.invert_yaxis(); ax3.set_xlim(0, 165); ax3.grid(axis="y", visible=False)
ax3.set_title("Which joints feel it", loc="left", color=TEXT, family=DISPLAY, size=17, pad=10)
ax3.set_xlabel("bite peak ÷ noise at rest")
tag(ax3, 40, 6.0, "J7 rolls about the fork's\nown axis: a bite\nbarely twists it", MUTED, size=8, va="center")
fig.savefig("out/01_bite_signature.png")
print("ok", peak.round(0), "after %.2f" % after)
