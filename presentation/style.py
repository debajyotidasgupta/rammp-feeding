"""Shared look for the RAMMP presentation figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

BG = "#0F1B21"; PANEL = "#15252D"; GRID = "#243841"; TEXT = "#E6EEF0"; MUTED = "#8CA2AB"; DIM = "#5B707A"
BERRY = "#FF4F64"; LEAF = "#6BD6A0"; AMBER = "#FFB547"; CYAN = "#5CC8FF"; VIOLET = "#B38CFF"
DISPLAY = "DIN Condensed"; BODY = "Avenir Next"; MONO = "Menlo"
W, H, DPI = 12.0, 6.75, 160          # 1920 x 1080

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "font.family": BODY, "font.size": 10.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
})


def glow(color, w=6, a=0.18):
    return [pe.Stroke(linewidth=w, foreground=color, alpha=a), pe.Normal()]


def frame(eyebrow, title, subtitle, source):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.text(0.045, 0.935, eyebrow.upper(), family=BODY, size=10.5, color=AMBER, weight="demibold")
    fig.text(0.045, 0.868, title, family=DISPLAY, size=34, color=TEXT)
    fig.text(0.045, 0.828, subtitle, family=BODY, size=12, color=MUTED)
    fig.text(0.045, 0.022, source, family=MONO, size=7.6, color=DIM)
    fig.text(0.955, 0.022, "RAMMP · robot-assisted feeding", family=MONO, size=7.6, color=DIM, ha="right")
    return fig


def tag(ax, x, y, text, color, **kw):
    kw.setdefault("size", 9.5)
    return ax.text(x, y, text, color=color, family=BODY, weight="demibold",
                   path_effects=[pe.withStroke(linewidth=3, foreground=BG)], **kw)
