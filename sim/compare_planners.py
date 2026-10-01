#!/usr/bin/env python3
"""Two recorded bites, side by side in the twin: e.g. the scripted moves against cuRobo's, on a site's geometry.

    .venv/bin/python -m sim.compare_planners scripted.json curobo.json [--out sim/out/compare.mp4]
        (the dumps: tests/site_check.py --dump X.json [--planner curobo]) -- MuJoCo's own renderer, a quick look

    the film (path traced in Blender, sim/blender_render.py):
    .venv/bin/python -m sim.compare_planners scripted.json curobo.json --export sim/out/cmp   # scene + poses
    blender -b -P sim/blender_render.py -- sim/out/cmp/left  (and .../right)                 # frames, on a GPU
    .venv/bin/python -m sim.compare_planners scripted.json curobo.json --compose sim/out/cmp # + the HUD -> mp4

Each panel replays its bite's executed moves (joint waypoints and their durations) in real time; the clocks run
until each bite ends. The scene is the site's: the table at its height, the bowl where the scene camera put it,
the person with the mouth where the site file says. MuJoCo poses every frame; Blender only draws them.
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
TOP = 60                                            # the title band over both panels
GAP_S = 0.4                                         # between moves (the stack's own pauses are not recorded)
TAIL_S = 2.0
PHASES = [("sequence detect", "to food_detect_start"), ("acquire pick", "over + plunge + scoop"),
          ("deliver swing", "swing to the user (with food)"), ("deliver approach", "approach"),
          ("deliver bite", "bite"), ("deliver retract", "retract"), ("sequence home", "home")]
VISIBLE_GROUPS = (1, 2, 4)                          # the scene's drawn groups (3 = collision, 5 = hidden)


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
                             foods=[Food(tuple(ws["bowl_xy"]), 0.024, (0.95, 0.62, 0.30))], width=W, height=H,
                             laptop=False, hair=False)
    lay.table_z = float(ws["table_z"])
    m = build(lay)
    return m, new_data(m)


class Panel:
    """One bite in one scene: poses the scene at t (arm, food, the user's mouth) and frames it."""

    def __init__(self, dump: dict, title: str):
        self.dump, self.title = dump, title
        self.m, self.d = panel_scene(dump)
        self.tl = Timeline(dump)
        self.tip_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "arm_tip")
        self.food = self.m.body("food0").mocapid[0]
        self.home = self.d.mocap_pos[self.food].copy()
        self.closed = [i for i in range(self.m.ngeom) if (self.m.geom(i).name or "").endswith("_closed")
                       and self.m.geom_bodyid[i] == self.m.body("user_upper").id]
        self.t_swing, self.t_app = self.tl.phase_start("deliver swing"), self.tl.phase_start("deliver approach")
        self.t_bite = self.tl.phase_start("deliver retract")
        ws = dump["workspace"]
        self.bowl = np.array([*ws["bowl_xy"], float(ws["table_z"]) + 0.05])
        self.mouth = np.array(dump["user"]["mouth_m"], float)

    def pose(self, t: float) -> tuple[str, bool]:
        """The scene at t; returns (step, mouth open)."""
        m, d, tl = self.m, self.d, self.tl
        q, step = tl.at(t)
        d.qpos[arm_qpos_adr(m)] = np.radians(q)
        mujoco.mj_kinematics(m, d)
        # the food: in the bowl until the pick ends, on the tip until the bite, then gone
        if self.t_swing is not None and self.t_bite is not None and self.t_swing <= t < self.t_bite:
            d.mocap_pos[self.food] = d.site_xpos[self.tip_id] - d.xmat[m.body("arm_flange").id].reshape(3, 3)[:, 2] * 0.012
        elif self.t_bite is not None and t >= self.t_bite:
            d.mocap_pos[self.food] = [0, 0, -5]
        else:
            d.mocap_pos[self.food] = self.home
        open_ = self.t_app is not None and self.t_bite is not None and self.t_app <= t < self.t_bite + 0.6
        for i in self.closed:
            m.geom_group[i] = 5 if open_ else 1
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); mujoco.mj_camlight(m, d)
        return step, open_

    def tip(self) -> np.ndarray:
        return self.d.site_xpos[self.tip_id].copy()

    def camera(self, t: float) -> tuple[np.ndarray, float, float, float]:
        """(lookat, distance, azimuth, elevation): wide, in on the bowl for the pick, the face for the bite, wide again;
        eased between, a slow drift throughout."""
        tz = float(self.dump["workspace"]["table_z"])
        wide = (np.array([0.02, 0.20, tz + 0.27]), 1.75, 178.0, -19.0)           # across the table, facing the user
        bowl = (self.bowl + [0.0, 0.0, 0.05], 0.90, 172.0, -30.0)
        face = (self.mouth + [0.10, -0.02, -0.03], 0.85, 225.0, -8.0)             # the bite, from the user's left
        keys = [(0.0, wide)]
        tp = self.tl.phase_start("acquire pick")
        for t0, shot in ((tp, bowl), (self.t_swing, wide), (self.t_app, face), (self.t_bite, wide)):
            if t0 is not None:
                keys.append((t0 - 0.6, shot))
        i = max(k for k, (t0, _) in enumerate(keys) if t0 <= t) if t >= keys[0][0] else 0
        cur = keys[i][1]
        if i > 0:
            a = min(1.0, max(0.0, (t - keys[i][0]) / 2.2))
            a = a * a * (3 - 2 * a)
            prev = keys[i - 1][1]
            cur = tuple(p + (c - p) * a for p, c in zip(prev, cur))
        look, dist, az, el = cur
        return np.asarray(look, float), float(dist), float(az + 3.0 * math.sin(t / 7.0)), float(el)

    def hud(self, t: float, step: str) -> dict:
        tl = self.tl
        done = t >= tl.duration
        name = next((lab for key, lab in PHASES if step.startswith(key)), step)
        return dict(clock=min(t, tl.duration), done=done, phase="done" if done else name,
                    frac=min(t, tl.duration) / tl.duration)


def draw_hud(im: Image.Image, p: Panel, h: dict) -> Image.Image:
    dr = ImageDraw.Draw(im, "RGBA")
    w, hh = im.size
    dr.rectangle([0, 0, w, 112], fill=(4, 6, 10, 150))
    dr.text((24, 16), p.title, font=F["big"], fill=(255, 255, 255))
    dr.text((24, 64), "%5.1f s%s   %d moves, %.1f s of motion" % (h["clock"], "  DONE" if h["done"] else "",
                                                              p.tl.moves, p.tl.motion_s),
            font=F["mid"], fill=(52, 211, 153) if h["done"] else (220, 228, 240))
    dr.rectangle([0, hh - 56, w, hh], fill=(4, 6, 10, 150))
    dr.text((24, hh - 44), h["phase"], font=F["midb"], fill=(251, 191, 36))
    x0, x1 = 24, w - 24
    dr.rounded_rectangle([x0, hh - 12, x1, hh - 6], 3, fill=(40, 48, 62))
    dr.rounded_rectangle([x0, hh - 12, x0 + (x1 - x0) * h["frac"], hh - 6], 3, fill=(56, 189, 248))
    return im


def render_panel(r, p: Panel, t: float, trail: list) -> Image.Image:
    step, open_ = p.pose(t)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = opt.geomgroup[2] = 1
    opt.geomgroup[4] = 1 if open_ else 0
    cam = mujoco.MjvCamera()
    look, cam.distance, cam.azimuth, cam.elevation = p.camera(t)
    cam.lookat[:] = look
    r.update_scene(p.d, camera=cam, scene_option=opt)
    tip = p.tip()
    if not trail or np.linalg.norm(trail[-1][1] - tip) > 0.003:
        trail.append((t, tip))
    trail[:] = [(tt, q) for tt, q in trail if t - tt < 6.0]
    for (t1, p1), (t2, p2) in zip(trail, trail[1:]):
        seg(r.scene, p1, p2, CYAN, r=0.0018, alpha=0.2 + 0.7 * max(0.0, 1 - (t - t2) / 6.0))
    ball(r.scene, tip, 0.004, CYAN, alpha=0.9, glow=1.0)
    return draw_hud(Image.fromarray(r.render()), p, p.hud(t, step))


def banner(ws: dict) -> str:
    return ("RAMMP  ·  the same bite on the 2026-09-30 site geometry (table z %.3f m, bowl at (%.2f, %.2f) m), "
            "real kinematics, real time" % (ws["table_z"], ws["bowl_xy"][0], ws["bowl_xy"][1]))


def ffmpeg(out: Path, w: int, h: int) -> subprocess.Popen:
    out.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)


# ------------------------------------------------------------------------------------------------ for Blender
def _material(m: mujoco.MjModel, i: int) -> dict:
    tex = int(m.mat_texid[i][mujoco.mjtTextureRole.mjTEXROLE_RGB])
    return dict(name=m.material(i).name, rgba=[float(v) for v in m.mat_rgba[i]], specular=float(m.mat_specular[i]),
                shininess=float(m.mat_shininess[i]), reflectance=float(m.mat_reflectance[i]),
                emission=float(m.mat_emission[i]), texrepeat=[float(v) for v in m.mat_texrepeat[i]],
                texture=m.texture(tex).name if tex >= 0 else None)


def export(p: Panel, out: Path, frames: int) -> None:
    """The scene (meshes, primitives, materials) and every frame's geom poses, tip, camera -> out/ for Blender."""
    from sim import procedural as P
    m, d = p.m, p.d
    out.mkdir(parents=True, exist_ok=True)
    geoms = [i for i in range(m.ngeom) if m.geom_group[i] in VISIBLE_GROUPS or i in p.closed]
    meshes = {}
    for i in geoms:
        k = int(m.geom_dataid[i])
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and k not in meshes:
            v0, nv = m.mesh_vertadr[k], m.mesh_vertnum[k]
            f0, nf = m.mesh_faceadr[k], m.mesh_facenum[k]
            meshes[k] = (m.mesh_vert[v0:v0 + nv].astype(np.float32), m.mesh_face[f0:f0 + nf].astype(np.int32))
    np.savez_compressed(out / "meshes.npz", **{"v%d" % k: v for k, (v, f) in meshes.items()},
                        **{"f%d" % k: f for k, (v, f) in meshes.items()})
    gdesc = []
    for i in geoms:
        mat = int(m.geom_matid[i])
        gdesc.append(dict(id=i, name=m.geom(i).name or "g%d" % i, type=int(m.geom_type[i]),
                          size=[float(v) for v in m.geom_size[i]], mesh=int(m.geom_dataid[i]),
                          rgba=[float(v) for v in m.geom_rgba[i]], material=m.material(mat).name if mat >= 0 else None,
                          body=m.body(m.geom_bodyid[i]).name, group=int(m.geom_group[i])))
    textures = {"wood": str(P.wood_png()), "concrete": str(P.floor_png()), "fabric": str(P.fabric_png()),
                "screen": str(P.screen_png())}
    floor_z = float(next(m.geom_pos[i][2] for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE))
    n = frames
    pos = np.zeros((n, len(geoms), 3), np.float32); quat = np.zeros((n, len(geoms), 4), np.float32)
    vis = np.zeros((n, len(geoms)), bool); tips = np.zeros((n, 3), np.float32); cams = np.zeros((n, 6), np.float32)
    huds = []
    q4 = np.zeros(4)
    for k in range(n):
        t = k / FPS
        step, open_ = p.pose(t)
        for j, i in enumerate(geoms):
            pos[k, j] = d.geom_xpos[i]
            mujoco.mju_mat2Quat(q4, d.geom_xmat[i]); quat[k, j] = q4
            g = int(m.geom_group[i])
            vis[k, j] = g in (1, 2) or (g == 4 and open_)
        tips[k] = p.tip()
        look, dist, az, el = p.camera(t)
        cams[k] = [*look, dist, az, el]
        huds.append(p.hud(t, step))
    np.savez_compressed(out / "frames.npz", pos=pos, quat=quat, vis=vis, tip=tips, cam=cams)
    (out / "scene.json").write_text(json.dumps(dict(
        geoms=gdesc, materials=[_material(m, i) for i in range(m.nmat)], textures=textures, floor_z=floor_z,
        fovy=float(m.vis.global_.fovy), fps=FPS, width=W, height=H, frames=n,
        table_z=float(p.dump["workspace"]["table_z"]), bowl=p.bowl.tolist(), mouth=p.mouth.tolist()), indent=1))
    (out / "hud.json").write_text(json.dumps(huds))
    print("%s: %d geoms (%d meshes), %d frames" % (out, len(geoms), len(meshes), n))


def compose(panels: list[Panel], root: Path, out: Path) -> None:
    """Blender's frames (root/left, root/right) + the HUD, side by side -> out."""
    dirs = [root / "left", root / "right"]
    huds = [json.loads((dd / "hud.json").read_text()) for dd in dirs]
    n = min(len(h) for h in huds)
    ff = ffmpeg(out, 2 * W, H + TOP)
    ws = panels[0].dump["workspace"]
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = (1 - 0.32 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.4)[..., None]
    for k in range(n):
        frame = Image.new("RGB", (2 * W, H + TOP), (5, 7, 11))
        for i, (p, dd) in enumerate(zip(panels, dirs)):
            im = np.asarray(Image.open(dd / "frames" / ("%04d.png" % (k + 1))).convert("RGB"), float)
            im = Image.fromarray(np.clip(im * vignette, 0, 255).astype(np.uint8))
            frame.paste(draw_hud(im, p, huds[i][k]), (i * W, TOP))
        dr = ImageDraw.Draw(frame)
        dr.text((24, 16), banner(ws), font=F["mid"], fill=(200, 212, 230))
        dr.line([(W, TOP), (W, H + TOP)], fill=(5, 7, 11), width=4)
        ff.stdin.write(np.asarray(frame).tobytes())
    ff.stdin.close(); ff.wait()
    print(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left"); ap.add_argument("right")
    ap.add_argument("--titles", nargs=2, default=["our scripted moves", "cuRobo's free-space moves"])
    ap.add_argument("--out", default="sim/out/compare_planners.mp4")
    ap.add_argument("--export", help="write the scene + poses for sim/blender_render.py into this folder")
    ap.add_argument("--compose", help="Blender's frames in this folder (left/, right/) + the HUD -> --out")
    a = ap.parse_args()
    dumps = [json.loads(Path(p).read_text()) for p in (a.left, a.right)]
    panels = [Panel(dump, title) for dump, title in zip(dumps, a.titles)]
    frames = int((max(p.tl.duration for p in panels) + TAIL_S) * FPS)
    if a.export:
        for p, side in zip(panels, ("left", "right")):
            export(p, Path(a.export) / side, frames)
        return 0
    if a.compose:
        compose(panels, Path(a.compose), Path(a.out))
        return 0
    rs = [mujoco.Renderer(p.m, H, W, max_geom=8000) for p in panels]
    trails = [[], []]
    ff = ffmpeg(Path(a.out), 2 * W, H + TOP)
    for k in range(frames):
        t = k / FPS
        frame = Image.new("RGB", (2 * W, H + TOP), (5, 7, 11))
        for i, (p, r) in enumerate(zip(panels, rs)):
            frame.paste(render_panel(r, p, t, trails[i]), (i * W, TOP))
        dr = ImageDraw.Draw(frame)
        dr.text((24, 14), banner(dumps[0]["workspace"]), font=F["mid"], fill=(200, 212, 230))
        dr.line([(W, TOP), (W, H + TOP)], fill=(30, 36, 48), width=3)
        ff.stdin.write(np.asarray(frame).tobytes())
    ff.stdin.close(); ff.wait()
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
