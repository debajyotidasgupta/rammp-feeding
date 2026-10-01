#!/usr/bin/env python3
"""Two recorded bites, side by side in the twin: e.g. the scripted moves against cuRobo's, on a site's geometry.

    .venv/bin/python -m sim.compare_planners scripted.json curobo.json [--out sim/out/compare.mp4]
        (the dumps: tests/site_check.py --dump X.json [--planner curobo])

Each panel replays its bite's executed moves (joint waypoints and their durations) in real time; the clocks run
until each bite ends. The scene is the site's: the table at its height, the bowl where the scene camera put it,
the person with the mouth where the site file says.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sim.scene import Food, Layout, arm_qpos_adr, build, new_data
from sim.twin import CYAN, F, ball, seg

W, H, FPS = 960, 720, 24
GAP_S = 0.4                                         # between moves (the stack's own pauses are not recorded)
PHASES = [("sequence detect", "to food_detect_start"), ("acquire pick", "over + plunge + scoop"),
          ("deliver swing", "swing to the user (with food)"), ("deliver approach", "approach"),
          ("deliver bite", "bite"), ("deliver retract", "retract"), ("sequence home", "home")]


def fold(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


class Timeline:
    """A bite's executed moves as joints (deg) against time, with the step each instant belongs to."""

    def __init__(self, dump: dict):
        self.t, self.q, self.step = [0.0], [fold(dump["motions"][0]["start"])], [dump["motions"][0]["step"]]
        t = 0.0
        for m in dump["motions"]:
            prev = self.q[-1]
            start = prev + fold(fold(m["start"]) - prev)
            self.t.append(t); self.q.append(start); self.step.append(m["step"])
            for q, dt in m["points"]:
                target = self.q[-1] + fold(fold(q) - self.q[-1])
                t += float(dt)
                self.t.append(t); self.q.append(target); self.step.append(m["step"])
            t += GAP_S
            self.t.append(t); self.q.append(self.q[-1]); self.step.append(m["step"])
        self.t, self.q = np.array(self.t), np.array(self.q)
        self.duration = float(self.t[-1])
        self.moves = len(dump["motions"])
        self.motion_s = sum(sum(p[1] for p in m["points"]) for m in dump["motions"])

    def at(self, t: float) -> tuple[np.ndarray, str]:
        t = min(max(t, 0.0), self.duration)
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        i = min(max(i, 0), len(self.t) - 2)
        a = 0.0 if self.t[i + 1] <= self.t[i] else (t - self.t[i]) / (self.t[i + 1] - self.t[i])
        return self.q[i] + (self.q[i + 1] - self.q[i]) * a, self.step[i + 1]

    def phase_start(self, key: str) -> float | None:
        for t, s in zip(self.t, self.step):
            if s.startswith(key):
                return float(t)
        return None


def panel_scene(dump: dict):
    ws, mouth = dump["workspace"], tuple(dump["user"]["mouth_m"])
    lay = Layout.from_config(mouth=mouth, bowl_xy=tuple(ws["bowl_xy"]),
                             foods=[Food(tuple(ws["bowl_xy"]), 0.024, (0.95, 0.62, 0.30))], width=W, height=H)
    lay.table_z = float(ws["table_z"])
    m = build(lay)
    return m, new_data(m)


def render_panel(r, m, d, tl: Timeline, t: float, title: str, trail: list, carry: dict) -> Image.Image:
    q, step = tl.at(t)
    d.qpos[arm_qpos_adr(m)] = np.radians(q)
    # the food: in the bowl until the pick ends, on the tip until the bite, then gone; the jaw open around the bite
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm_tip")
    fb = m.body("food0").mocapid[0]
    t_swing, t_bite = tl.phase_start("deliver swing"), tl.phase_start("deliver retract")
    t_app = tl.phase_start("deliver approach")
    mujoco.mj_kinematics(m, d)
    if t_swing is not None and t_bite is not None and t_swing <= t < t_bite:
        d.mocap_pos[fb] = d.site_xpos[tip_id] - d.xmat[m.body("arm_flange").id].reshape(3, 3)[:, 2] * 0.012
    elif t_bite is not None and t >= t_bite:
        d.mocap_pos[fb] = [0, 0, -5]
    else:
        d.mocap_pos[fb] = carry["home"]
    open_ = t_app is not None and t_bite is not None and t_app <= t < t_bite + 0.6
    for i in range(m.ngeom):
        n = m.geom(i).name or ""
        if n.endswith("_closed") and m.geom_bodyid[i] == m.body("user_upper").id:
            m.geom_group[i] = 5 if open_ else 1
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_camlight(m, d)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = opt.geomgroup[2] = 1
    opt.geomgroup[4] = 1 if open_ else 0
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (-0.02, 0.16, 0.22); cam.distance, cam.azimuth, cam.elevation = 1.55, 146 + 4 * math.sin(t / 9), -18
    r.update_scene(d, camera=cam, scene_option=opt)
    tip = d.site_xpos[tip_id].copy()
    if not trail or np.linalg.norm(trail[-1][1] - tip) > 0.003:
        trail.append((t, tip))
    trail[:] = [(tt, p) for tt, p in trail if t - tt < 6.0]
    for (t1, p1), (t2, p2) in zip(trail, trail[1:]):
        seg(r.scene, p1, p2, CYAN, r=0.0018, alpha=0.2 + 0.7 * max(0.0, 1 - (t - t2) / 6.0))
    ball(r.scene, tip, 0.004, CYAN, alpha=0.9, glow=1.0)
    im = Image.fromarray(r.render())
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, W, 112], fill=(4, 6, 10, 190))
    dr.text((24, 16), title, font=F["big"], fill=(255, 255, 255))
    done = t >= tl.duration
    clock = min(t, tl.duration)
    dr.text((24, 64), "%5.1f s%s   %d moves, %.1f s of motion" % (clock, "  DONE" if done else "", tl.moves, tl.motion_s),
            font=F["mid"], fill=(52, 211, 153) if done else (190, 205, 225))
    name = next((lab for key, lab in PHASES if step.startswith(key)), step)
    dr.rectangle([0, H - 56, W, H], fill=(4, 6, 10, 190))
    dr.text((24, H - 44), "done" if done else name, font=F["midb"], fill=(251, 191, 36))
    x0, x1 = 24, W - 24
    dr.rounded_rectangle([x0, H - 12, x1, H - 6], 3, fill=(40, 48, 62))
    dr.rounded_rectangle([x0, H - 12, x0 + (x1 - x0) * clock / tl.duration, H - 6], 3, fill=(56, 189, 248))
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left"); ap.add_argument("right")
    ap.add_argument("--titles", nargs=2, default=["our scripted moves", "cuRobo's free-space moves"])
    ap.add_argument("--out", default="sim/out/compare_planners.mp4")
    a = ap.parse_args()
    dumps = [json.loads(Path(p).read_text()) for p in (a.left, a.right)]
    panels = []
    for dump in dumps:
        m, d = panel_scene(dump)
        r = mujoco.Renderer(m, H, W, max_geom=8000)
        home = d.mocap_pos[m.body("food0").mocapid[0]].copy()
        panels.append(dict(m=m, d=d, r=r, tl=Timeline(dump), trail=[], carry=dict(home=home)))
    total = max(p["tl"].duration for p in panels) + 2.0
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (2 * W, H + 60),
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    ws = dumps[0]["workspace"]
    for k in range(int(total * FPS)):
        t = k / FPS
        frame = Image.new("RGB", (2 * W, H + 60), (5, 7, 11))
        for i, (p, title) in enumerate(zip(panels, a.titles)):
            frame.paste(render_panel(p["r"], p["m"], p["d"], p["tl"], t, title, p["trail"], p["carry"]), (i * W, 60))
        dr = ImageDraw.Draw(frame)
        dr.text((24, 14), "RAMMP  ·  the same bite on the 2026-09-30 site geometry (table z %.3f m, bowl at (%.2f, %.2f) m), "
                "real kinematics, real time" % (ws["table_z"], ws["bowl_xy"][0], ws["bowl_xy"][1]), font=F["mid"],
                fill=(200, 212, 230))
        dr.line([(W, 60), (W, H + 60)], fill=(30, 36, 48), width=3)
        ff.stdin.write(np.asarray(frame).tobytes())
    ff.stdin.close(); ff.wait()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
