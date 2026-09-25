import json, re
import numpy as np
from PIL import Image
from style import *

FXD = FYD = 334.810; CXD = 238.536; CYD = 139.694; T = np.array([-0.02706, -0.00997, -0.00471])
FXC, FYC, CXC, CYC = 1297.673, 1298.631, 657.4, 225.7
d = np.load("data/perceive_depth.npy").astype(float) / 1000.0
img = np.asarray(Image.open("data/perceive_colour.jpg").convert("RGB")).astype(float) / 255.0
tg = json.load(open("data/perceive_target.json"))
rows = [tuple(map(float, m)) for m in re.findall(r"berry at \(\s*(\d+),\s*(\d+)\)\s+top (\d+) mm\s+floor beside it (\d+) mm\s+height (\d+) mm", open("data/perceive_locate_log.txt").read())]

v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
m = (d > 0.30) & (d < 0.47)
z = d[m]; X = (u[m] - CXD) * z / FXD + T[0]; Y = (v[m] - CYD) * z / FYD + T[1]; Zc = z + T[2]
uc = np.clip(np.round(FXC * X / Zc + CXC).astype(int), 0, img.shape[1] - 1)
vc = np.clip(np.round(FYC * Y / Zc + CYC).astype(int), 0, img.shape[0] - 1)
inside = (FXC * X / Zc + CXC > 0) & (FXC * X / Zc + CXC < img.shape[1]) & (FYC * Y / Zc + CYC > 0) & (FYC * Y / Zc + CYC < img.shape[0])
col = img[vc, uc]; col[~inside] = [0.35, 0.45, 0.5]
# keep the bowl and its surroundings
ctr = np.array([np.median(X[z < 0.44]), np.median(Y[z < 0.44])])
keep = np.hypot(X - ctr[0], Y - ctr[1]) < 0.16
X, Y, z, col = X[keep], Y[keep], z[keep], col[keep]

fig = frame("Acquisition · depth",
            "The bowl in depth, and the plunge through it",
            "The same overlook frame as a point cloud: each berry a bump on the concave floor, and one smooth plunge that slows before the food",
            "depth sensor 480x270, 15-frame median, coloured from the registered colour frame · camera frame, metres · run 2026-09-24 20:22")
ax = fig.add_axes([0.02, 0.04, 0.64, 0.80], projection="3d")
ax.set_facecolor(BG)
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
    a_.set_pane_color((0, 0, 0, 0)); a_._axinfo["grid"]["color"] = GRID
ax.scatter(X, Y, -z, c=col, s=1.6, linewidths=0, depthshade=False)

# target berry and the plunge along the fork tip's vertical (the camera looks straight down)
c = tg["berry_bbox_centre"]; zb = tg["berry_depth_m"]
tx = (c[0] - CXC) * zb / FXC; ty = (c[1] - CYC) * zb / FYC
r3 = min(rows, key=lambda r: np.hypot(r[0] - c[0], r[1] - c[1]))
top, floor = r3[2] / 1000.0, r3[3] / 1000.0; tip0 = 0.265
slow_from = top - 0.020
ax.set_xlim(ctr[0] - 0.16, ctr[0] + 0.16); ax.set_ylim(ctr[1] - 0.16, ctr[1] + 0.16); ax.set_zlim(-0.47, -0.25)
ax.set_box_aspect((1, 1, 0.75)); ax.view_init(elev=26, azim=-62)
ax.set_xticklabels([]); ax.set_yticklabels([])
ax.set_zticks([-0.45, -0.40, -0.35, -0.30, -0.25]); ax.set_zticklabels(["450", "400", "350", "300", "250 mm"], family=MONO, size=8)

# the plunge, drawn ON TOP: project the 3D points to the figure (mplot3d does not layer lines over dense clouds)
from mpl_toolkits.mplot3d import proj3d
fig.canvas.draw()
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")
def P(x, y, zz):
    X2, Y2, _ = proj3d.proj_transform(x, y, zz, ax.get_proj())
    return fig.transFigure.inverted().transform(ax.transData.transform((X2, Y2)))
p0, p1, p2 = P(tx, ty, -tip0), P(tx, ty, -slow_from), P(tx, ty, -floor)
ov.plot([p0[0], p1[0]], [p0[1], p1[1]], color=AMBER, lw=2.6, path_effects=glow(AMBER, 8, 0.25), solid_capstyle="round")
ov.plot([p1[0], p2[0]], [p1[1], p2[1]], color=BERRY, lw=3.4, path_effects=glow(BERRY, 9, 0.3), solid_capstyle="round")
ov.scatter(*p0, s=60, color=VIOLET, zorder=5, edgecolor=BG); ov.scatter(*p2, s=80, color=BERRY, marker="v", zorder=5, edgecolor=BG)
tag(ov, p0[0] - 0.012, p0[1], "fork tip at the overlook\n265 mm below the camera", VIOLET, size=9, ha="right", va="center")
tag(ov, p0[0] - 0.012, (p0[1] + p1[1]) / 2, "50 mm/s", AMBER, size=10, ha="right", va="center")
tag(ov, p1[0] - 0.012, (p1[1] + p2[1]) / 2 + 0.01, "8 mm/s through the food", BERRY, size=10, ha="right", va="center")
tag(ov, p2[0] - 0.012, p2[1] - 0.025, "stop: the measured bowl floor", BERRY, size=9, ha="right", va="center")

# right: the profile under the target, the numbers behind the plunge
axp = fig.add_axes([0.70, 0.18, 0.26, 0.56])
line = np.abs(Y - ty) < 0.004
xs, zs = X[line], z[line]; o = np.argsort(xs)
axp.fill_between(1000 * (xs[o] - tx), 1000 * zs[o], 470, color=CYAN, alpha=0.18, lw=0)
axp.plot(1000 * (xs[o] - tx), 1000 * zs[o], color=CYAN, lw=1.4)
axp.axvline(0, color=AMBER, lw=1.2, ls=(0, (4, 3)))
for yv, lab, cc in ((top * 1000, "berry top %d mm" % (top * 1000), BERRY), (floor * 1000, "floor %d mm" % (floor * 1000), LEAF),
                    (slow_from * 1000, "slow from here", AMBER)):
    axp.axhline(yv, color=cc, lw=0.9, ls=":")
    tag(axp, 46, yv - 1, lab, cc, size=8.5, ha="right", va="bottom")
axp.set_ylim(460, 375); axp.set_xlim(-60, 60)
axp.set_xlabel("mm across, through the target"); axp.set_ylabel("mm below the camera")
axp.set_title("Profile under the fork", loc="left", color=TEXT, family=DISPLAY, size=17, pad=8)
fig.savefig("out/03_depth_cloud.png")
print("ok points", len(X), "top %.3f floor %.3f" % (top, floor))
