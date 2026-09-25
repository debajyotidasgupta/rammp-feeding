import json
import numpy as np
from style import *

FXD = FYD = 334.810; CXD = 238.536; CYD = 139.694; T = np.array([-0.02706, -0.00997, -0.00471])
CAM = np.array([0.2205 + 0.05639, 0.2826, 0.5344 + 0.12305])     # camera in the base frame at food_detect_start
log = json.load(open("data/perceive_scoop_log.json")); Pv = np.array([r["piv"] for r in log]); tt = np.array([r["t"] for r in log])
d = np.load("data/perceive_depth.npy").astype(float) / 1000.0
v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]; m = d > 0.3
z = d[m]; xc = (u[m] - CXD) * z / FXD + T[0]; yc = (v[m] - CYD) * z / FYD + T[1]; zc = z + T[2]
W = np.stack([CAM[0] - yc, CAM[1] - xc, CAM[2] - zc], 1)          # camera -> base frame at this orientation

fig = frame("Acquisition · the scoop",
            "Scoop and carry, in one motion",
            "The berry's recorded path with the fork drawn along it: turning from the moment it lifts, backing out of the bowl as it turns",
            "scoop log (tool pose + fork calibration), 287 samples, 7.2 s at 25 °/s · bowl slice from the overlook depth · base frame")
ax = fig.add_axes([0.06, 0.12, 0.60, 0.66])
X0 = Pv[0, 0]
sl = np.abs(W[:, 0] - X0) < 0.006
ys, zs = W[sl, 1], W[sl, 2]; o = np.argsort(ys)
ax.fill_between(1000 * ys[o], 1000 * zs[o], 1000 * (zs.min() - 0.02), color=CYAN, alpha=0.12, lw=0)
ax.plot(1000 * ys[o], 1000 * zs[o], ".", color=CYAN, ms=1.6, alpha=0.8)
tag(ax, 1000 * ys[o][len(o) // 4], 1000 * zs.min() - 12, "bowl and table, sliced through the berry (depth sensor)", CYAN, size=8.5)

rise = (Pv[:, 2] - Pv[0, 2]) / (Pv[-1, 2] - Pv[0, 2]); s = np.clip(rise, 0, 1)
ax.plot(1000 * Pv[:, 1], 1000 * Pv[:, 2], color=BERRY, lw=2.2, path_effects=glow(BERRY, 8, 0.25), zorder=4)
for k, sk in enumerate(np.linspace(0, 1, 9)):
    i = np.argmin(np.abs(s - sk)); th = np.radians(90.0 * sk ** 0.8)
    a = np.array([np.sin(th), -np.cos(th)])             # fork axis in the (Y, Z) plane: down -> +Y
    F = Pv[i, 1:3]; tip = F + 0.015 * a; back = F - 0.140 * a
    al = 0.25 + 0.75 * sk
    ax.plot(1000 * np.array([back[0], tip[0]]), 1000 * np.array([back[1], tip[1]]), color=TEXT, lw=2.2, alpha=al, solid_capstyle="round", zorder=5)
    ax.scatter(1000 * F[0], 1000 * F[1], s=70, color=BERRY, alpha=al, edgecolor=BG, zorder=6)
    if k in (0, 3, 8):
        tag(ax, 1000 * back[0] + (8 if k else -8), 1000 * back[1] + 6, "%.0f°" % (90 * sk ** 0.8), AMBER, size=9.5, ha="left" if k else "right")
tag(ax, 1000 * Pv[0, 1] + 10, 1000 * Pv[0, 2] - 14, "pierced at the floor", BERRY, size=9)
tag(ax, 1000 * Pv[-1, 1] + 8, 1000 * Pv[-1, 2] + 14, "level, food on top:\nthe old scoop-end pose", BERRY, size=9)
ax.annotate("", xy=(1000 * Pv[-1, 1], 1000 * Pv[-1, 2] - 55), xytext=(1000 * Pv[0, 1], 1000 * Pv[-1, 2] - 55),
            arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.4))
tag(ax, 1000 * (Pv[0, 1] + Pv[-1, 1]) / 2, 1000 * Pv[-1, 2] - 48, "backs off %.0f mm" % (1000 * np.hypot(*(Pv[-1, :2] - Pv[0, :2]))), AMBER, size=9.5, ha="center", va="bottom")
ax.set_xlim(1000 * (Pv[-1, 1] - 0.13), 1000 * (Pv[0, 1] + 0.14)); ax.invert_xaxis()
ax.set_ylim(1000 * (zs.min() - 0.03), 1000 * (Pv[-1, 2] + 0.12)); ax.set_aspect("equal")
ax.set_xlabel("Y, mm (base frame; the user sits toward +Y)"); ax.set_ylabel("height Z, mm")

# right: the rotation profile and the food's vertical offset from its path
axr = fig.add_axes([0.72, 0.47, 0.25, 0.30])
sp = np.linspace(0, 1, 200)
axr.plot(sp, 90 * sp ** 0.8, color=AMBER, lw=2, label="now: 90°·s^0.8")
axr.plot(sp, 90 * sp ** 1.6, color=DIM, lw=1.4, ls="--", label="first try: 90°·s^1.6")
axr.set_xlabel("scoop progress s"); axr.set_ylabel("fork angle from vertical, °"); axr.set_xlim(0, 1); axr.set_ylim(0, 92)
axr.set_title("Turn early, while the berry is steep", loc="left", color=TEXT, family=DISPLAY, size=15, pad=6)
axr.legend(loc="lower right", fontsize=8.5, labelcolor=MUTED)
axo = fig.add_axes([0.72, 0.12, 0.25, 0.22])
off = np.array([r["off"] for r in log]) * 1000
axo.fill_between(tt, 0, off, color=LEAF, alpha=0.25, lw=0); axo.plot(tt, off, color=LEAF, lw=1.4)
axo.axhline(15, color=BERRY, lw=1, ls=":"); tag(axo, 0.2, 13.5, "stop if the food drifts 15 mm off its path", BERRY, size=8, va="top")
axo.set_ylim(0, 17); axo.set_xlim(0, tt[-1]); axo.set_xlabel("seconds"); axo.set_ylabel("off path, mm")
tag(axo, tt[-1] * 0.45, 5, "worst %.1f mm" % off.max(), LEAF, size=10)
fig.savefig("out/07_scoop.png")
print("ok", Pv[0].round(3), Pv[-1].round(3), off.max())
