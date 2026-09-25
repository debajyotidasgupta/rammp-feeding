import json
import numpy as np
from mpl_toolkits.mplot3d import proj3d
from gen3fk import frames
from style import *

HOME = [-90.0, 8.071, 0.0, -139.199, 0.446, -49.528, -174.304]
LIFT = [-90.501, 14.243, 0.441, -78.165, -0.354, -116.713, -173.798]
DET = [154.812, -9.686, -31.181, -59.005, 4.715, -113.549, 35.326]
SCOOP_END = [-102.4, -42.0, -37.2, -97.5, -41.1, -111.2, 131.8]
sw = json.load(open("data/swing_plan.json"))["path"]; rt = json.load(open("data/retract_plan.json"))
scoop = np.array([r["piv"] for r in json.load(open("data/perceive_scoop_log.json"))])
TIP = np.array([0.0, 0.0087, 0.142 + 0.0])                 # fork tip in the tool frame (calibration, approx)

def fold(a): return (a + 180) % 360 - 180
def interp(a, b, n):
    return [[x + fold(y - x) * k / n for x, y in zip(a, b)] for k in range(n + 1)]
def mid(a, b): return [x + fold(y - x) / 2 for x, y in zip(a, b)]
def tip_of(q):
    T = frames(q)[-1]; return (T @ np.r_[TIP, 1])[:3]
def arm(q):
    return np.array([F[:3, 3] for F in frames(q)])

legs = [("to food_detect_start", interp(HOME, LIFT, 20) + interp(LIFT, mid(LIFT, DET), 20) + interp(mid(LIFT, DET), DET, 20), CYAN),
        ("swing to the user", sw, VIOLET),
        ("retract", rt["path2"] + rt["path3"] + interp(rt["path3"][-1], DET, 20), LEAF)]

fig = frame("The whole loop · motion",
            "One bite, in the arm's own frame",
            "Drawn from the joint paths the arm actually ran: out to the bowl, plunge and scoop, the swing, the approach, and home",
            "Gen3 kinematics (Kinova DH; 2.5-7.7 mm from the controller on 5 poses) · swing and retract: cached demo joint paths")
ax = fig.add_axes([0.0, 0.03, 0.74, 0.80], projection="3d"); ax.set_facecolor(BG)
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
    a_.set_pane_color((0, 0, 0, 0)); a_._axinfo["grid"]["color"] = (0.14, 0.22, 0.25, 0.6)
# table, bowl, user (stylised)
xx, yy = np.meshgrid(np.linspace(0.05, 0.45, 2), np.linspace(-0.05, 0.50, 2))
ax.plot_surface(xx, yy, np.full_like(xx, 0.205), color="#1C2E36", alpha=0.55, shade=False)
th = np.linspace(0, 2 * np.pi, 80); bc = np.array([0.276, 0.270])
for zz, r_ in ((0.205, 0.075), (0.265, 0.095)):
    ax.plot(bc[0] + r_ * np.cos(th), bc[1] + r_ * np.sin(th), zz, color=CYAN, lw=1.2, alpha=0.7)
mouth = np.array([-0.26, 0.31, 0.615])
u_, v_ = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 16))
ax.plot_wireframe(mouth[0] - 0.06 + 0.085 * np.cos(u_) * np.sin(v_), mouth[1] + 0.075 * np.sin(u_) * np.sin(v_),
                  mouth[2] + 0.06 + 0.11 * np.cos(v_), color=DIM, lw=0.4, alpha=0.5)
ax.scatter([0], [0], [0], s=160, marker="s", color=MUTED, alpha=0.7)                      # the base

for name, path, col in legs:
    P = np.array([tip_of(q) for q in path]); ax.plot(*P.T, color=col, lw=2.2, alpha=0.95)
for q in sw[::13]:
    A = arm(q); ax.plot(*A.T, color=VIOLET, lw=2.0, alpha=0.18)
for q, c_, a_ in ((HOME, MUTED, 0.35), (SCOOP_END, MUTED, 0.35), (DET, CYAN, 0.95), (sw[-1], VIOLET, 1.0)):
    A = arm(q); ax.plot(*A.T, color=c_, lw=3.4, alpha=a_, solid_capstyle="round"); ax.scatter(*A.T, s=12, color=c_, alpha=a_)
ax.plot(*scoop.T, color=BERRY, lw=2.6)                                        # the food during the scoop
dt = tip_of(DET); pl = np.array([[dt[0] + 0.02, dt[1] - 0.02, dt[2]], [scoop[0, 0], scoop[0, 1], scoop[0, 2] - 0.017]])
ax.plot(*pl.T, color=AMBER, lw=2.4)                                           # over + plunge (tip)
se = tip_of(sw[-1]); stop = mouth + np.array([0.08, 0, -0.01])
ax.plot(*np.array([se, stop]).T, color=AMBER, lw=2.4, ls=(0, (3, 2)))         # the approach
ax.set_xlim(-0.35, 0.42); ax.set_ylim(-0.25, 0.45); ax.set_zlim(0.0, 0.78)
ax.set_box_aspect((0.77, 0.70, 0.78)); ax.view_init(elev=20, azim=-148)
ax.set_xlabel("X, m", color=MUTED, labelpad=-6); ax.set_ylabel("Y, m", color=MUTED, labelpad=-6); ax.set_zlabel("Z, m", color=MUTED, labelpad=-6)
ax.tick_params(labelsize=7, pad=-2)

fig.canvas.draw()
ov = fig.add_axes([0, 0, 1, 1], facecolor="none"); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")
def P2(p):
    x2, y2, _ = proj3d.proj_transform(*p, ax.get_proj()); return fig.transFigure.inverted().transform(ax.transData.transform((x2, y2)))
for p, lab, c in ((tip_of(HOME), "feeding_start", TEXT), (tip_of(DET), "food_detect_start\n(over the bowl)", CYAN),
                  (scoop[-1], "scoop end", BERRY), (se, "swing end,\n50 mm to the user's right", VIOLET), (stop, "stop, 80 mm\nfrom the mouth", AMBER),
                  (mouth + np.array([-0.06, 0, 0.19]), "user", MUTED)):
    q2 = P2(p); dx_, dy_, ha_ = (-0.012, 0.03, "right") if lab.startswith("stop") else (0.008, 0.008, "left")
    tag(ov, q2[0] + dx_, q2[1] + dy_, lab, c, size=9, ha=ha_)
keys = [("out to the bowl", CYAN), ("over + plunge", AMBER), ("scoop + back-off", BERRY), ("swing to the user", VIOLET),
        ("approach (dashed)", AMBER), ("retract home", LEAF)]
for k, (lab, c) in enumerate(keys):
    y = 0.70 - k * 0.05
    fig.add_artist(plt.Line2D([0.765, 0.80], [y, y], color=c, lw=3, ls="-" if "dashed" not in lab else (0, (3, 2))))
    fig.text(0.81, y, lab, family=BODY, size=10.5, color=TEXT, va="center", weight="demibold")
fig.text(0.765, 0.33, "Ghosted: the arm through the swing,\nevery 13th waypoint of 104. It lifts\n175 mm, turns J1 90° toward the user\nand retracts the fork 120 mm as it turns,\nso the tip never leads toward anyone.",
         family=BODY, size=9.5, color=MUTED, va="top", linespacing=1.45)
fig.savefig("out/08_motion_3d.png")
print("ok", tip_of(sw[-1]).round(3), tip_of(DET).round(3))
