import json
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import proj3d
from style import *

FXD = FYD = 334.810; CXD = 238.536; CYD = 139.694; T = np.array([-0.02706, -0.00997, -0.00471])
FXC, FYC, CXC, CYC = 1297.673, 1298.631, 657.4, 225.7
d = np.load("data/face_depth.npy").astype(float) / 1000.0
img = np.asarray(Image.open("data/face_colour.jpg").convert("RGB")).astype(float) / 255.0
Hc = img.shape[0]
f = json.load(open("data/face_face.json")); L = np.array(json.load(open("data/face_landmarks.json"))["landmarks"])

v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
m = (d > 0.20) & (d < 0.72)
z = d[m]; X = (u[m] - CXD) * z / FXD + T[0]; Y = (v[m] - CYD) * z / FYD + T[1]; Zc = z + T[2]
uc = FXC * X / Zc + CXC; vc = FYC * Y / Zc + CYC
ok = (uc >= 0) & (uc < img.shape[1]) & (vc >= 0) & (vc < Hc)
X, Y, Zc, uc, vc = X[ok], Y[ok], Zc[ok], uc[ok], vc[ok]
col = img[vc.astype(int), uc.astype(int)]

def disp(x, y, zc):          # camera frame (x = image right = world down, y = image down) -> upright display
    return -np.asarray(y), np.asarray(zc), -np.asarray(x)

# landmarks lifted to 3D: nearest cloud point in colour-pixel space (upright (x, y) -> original (u, v) = (y, H-1-x))
lu, lv = L[:, 1], Hc - 1 - L[:, 0]
P = []
for a_, b_ in zip(lu, lv):
    k = np.argmin((uc - a_) ** 2 + (vc - b_) ** 2)
    P.append((X[k], Y[k], Zc[k]) if (uc[k] - a_) ** 2 + (vc[k] - b_) ** 2 < 64 else (np.nan,) * 3)
P = np.array(P)
face_z = np.nanmedian(P[:468, 2]); near = np.abs(P[:, 2] - face_z) < 0.06; P[~near] = np.nan

# the fork tip at the detection pose and the approach to 80 mm short, 10 mm below (camera frame)
zt = 0.265; tip = np.array([(644 - 657.6) * zt / FXC, (405 - 236.6) * zt / FYC, zt])
mc = np.array(f["mouth"]["cam_xyz"])
stop = mc - np.array([0, 0, 0.080]) + np.array([0.010, 0, 0])

fig = frame("Delivery · 3D",
            "The face, reconstructed from the wrist camera",
            "Depth and colour at the delivery pose: the face, the berry on the fork, the lips in 3D, and the approach to 80 mm",
            "depth 480x270 median + registered colour · MediaPipe landmarks lifted onto the surface · 2026-09-24 19:48")
ax = fig.add_axes([0.0, 0.02, 0.70, 0.82], projection="3d"); ax.set_facecolor(BG)
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
    a_.set_pane_color((0, 0, 0, 0)); a_._axinfo["grid"]["color"] = (0, 0, 0, 0); a_.line.set_color((0, 0, 0, 0))
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
sub = np.random.default_rng(0).random(len(X)) < 0.9
ax.scatter(*disp(X[sub], Y[sub], Zc[sub]), c=col[sub], s=1.3, linewidths=0, depthshade=False, alpha=0.95)
ax.set_xlim(-0.22, 0.18); ax.set_ylim(0.20, 0.72); ax.set_zlim(-0.16, 0.16)
ax.set_box_aspect((0.40, 0.52, 0.32)); ax.view_init(elev=16, azim=-42)

fig.canvas.draw()
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")
def Pf(p):
    a_, b_, c_ = disp(*p)
    x2, y2, _ = proj3d.proj_transform(a_, b_, c_, ax.get_proj())
    return fig.transFigure.inverted().transform(ax.transData.transform((x2, y2)))
good = ~np.isnan(P[:, 0])
pts = np.array([Pf(p) for p in P[good]])
ov.scatter(pts[:, 0], pts[:, 1], s=1.4, color=CYAN, alpha=0.8, linewidths=0)
OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]
lip = np.array([Pf(P[i]) for i in OUTER if good[i]])
if len(lip) > 3:
    ov.plot(lip[:, 0], lip[:, 1], color=BERRY, lw=1.8, path_effects=glow(BERRY, 6, 0.3))
t2, s2, m2 = Pf(tip), Pf(stop), Pf(mc)
ov.plot([t2[0], s2[0]], [t2[1], s2[1]], color=AMBER, lw=2.4, path_effects=glow(AMBER, 8, 0.25), solid_capstyle="round")
ov.plot([s2[0], m2[0]], [s2[1], m2[1]], color=LEAF, lw=1.4, ls=(0, (3, 3)))
ov.scatter(*t2, s=60, color=VIOLET, edgecolor=BG, zorder=5); ov.scatter(*s2, s=90, color=AMBER, edgecolor=BG, zorder=5)
ov.scatter(*m2, s=50, color=BERRY, edgecolor=BG, zorder=5)
tag(ov, t2[0] - 0.012, t2[1] - 0.01, "fork tip at the\ndetection pose", VIOLET, size=9, ha="right", va="top")
tag(ov, s2[0] + 0.012, s2[1] - 0.045, "STOP  80 mm short, 10 mm low", AMBER, size=9.5, ha="left")
tag(ov, m2[0] + 0.012, m2[1] + 0.01, "mouth %.0f mm" % (1000 * mc[2]), BERRY, size=9.5)
tag(ov, (t2[0] + s2[0]) / 2 + 0.012, (t2[1] + s2[1]) / 2 - 0.035, "approach %.0f mm at 35 mm/s" % (1000 * np.linalg.norm(stop - tip)), AMBER, size=9)

facts = [("points in the cloud", "%d" % len(X)), ("landmarks on the surface", "%d of 478" % good.sum()),
         ("nose / mouth depth", "%.0f / %.0f mm" % (1000 * f["nose"]["depth_m"], 1000 * mc[2])),
         ("mouth opening", "%.2f" % f.get("mouth_open", np.nan)), ("detection", "~25 ms, CPU")]
fig.text(0.72, 0.70, "What the arm knows\nat this moment", family=DISPLAY, size=22, color=TEXT, va="top", linespacing=1.0)
for k, (a_, v_) in enumerate(facts):
    y = 0.56 - k * 0.058
    fig.text(0.72, y, a_.upper(), family=BODY, size=8, color=MUTED, weight="demibold")
    fig.text(0.955, y, v_, family=MONO, size=10, color=TEXT, ha="right")
fig.text(0.72, 0.21, "The food on the fork shows up ~250 mm\nnearer than the face: those points are\ndropped before the mouth's depth is read.",
         family=BODY, size=9.5, color=MUTED, va="top", linespacing=1.4)
fig.savefig("out/05_face_3d.png")
print("ok", len(X), good.sum(), "stop", stop.round(3), "tip", tip.round(3))
