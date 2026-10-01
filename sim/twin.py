#!/usr/bin/env python3
"""The feeding digital twin: replay a recorded bite in MuJoCo, with the real camera frames beside it.

    .venv/bin/python -m sim.twin render --bite 3                    # -> sim/out/twin_<session>_bite03.mp4
    .venv/bin/python -m sim.twin render --bite 3 --still 24         # one frame at t = 24 s (a PNG)
    .venv/bin/python -m sim.twin render --bite all --speed 1.5      # every bite of the session, one film
    .venv/bin/python -m sim.twin poses                              # the calibrated poses, one sheet
    make twin BITE=3 [SESSION=~/rammp_runs/<run>]

The arm moves exactly as the record says (each move's joint telemetry); the scene is rebuilt per bite from the
calibration and that bite's perception (sim/scene.py, sim/record.py). The HUD's REAL tile is the camera frame the
stack saw at that moment (detection, target, the open-mouth wait); the TWIN tile is the sim's wrist camera
rendered with the real intrinsics, so the two line up.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from settings import cfg
from sim.record import Bite, load_bite
from sim.scene import Layout, arm_qpos_adr, build, new_data

HERE = Path(__file__).parent
OUT = HERE / "out"
W, H = 1920, 1080
FPS = 30
STAGE_RGB = {"perceive": (56, 189, 248), "acquire": (251, 191, 36), "deliver": (52, 211, 153)}
CYAN = (0.22, 0.85, 1.0)
AMBER = (1.0, 0.72, 0.18)
GREEN = (0.30, 0.95, 0.55)
SOFT_LIMITS = [70.0] * 4 + [63.0] * 3
ROTATE_AT_DELIVERY = bool(cfg.camera.mount.get("image_rotated_at_delivery", False))


# ------------------------------------------------------------------------------------------------ fonts
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for n in names:
        if os.path.exists(n):
            return ImageFont.truetype(n, size)
    return ImageFont.load_default()


F = {k: _font(*v) for k, v in {"title": (40, True), "big": (34, True), "mid": (22, False), "midb": (22, True),
                                "small": (17, False), "smallb": (17, True), "tiny": (14, False)}.items()}


# ------------------------------------------------------------------------------------------------ camera director
@dataclass
class Shot:
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float
    drift: float = 0.0              # deg/s of slow orbit
    track: float = 0.0              # 0..1: how much the lookat follows the fork tip


def shots(b: Bite) -> dict[str, Shot]:
    bx, by = b.bowl_xy
    tz = float(cfg.workspace.table_height_m)
    m = np.array(b.mouth)
    return {
        "establish": Shot((0.02, 0.22, 0.40), 2.0, 206, -24, drift=-1.2),              # front left: face + arm
        "bowl": Shot((bx, by, tz + 0.08), 0.72, 196, -24, drift=-1.6, track=0.25),     # from across the table
        "plunge": Shot((bx, by, tz + 0.05), 0.50, 188, -16, drift=-1.2, track=0.3),
        "swing": Shot((0.02, 0.22, 0.50), 1.30, 158, -14, drift=2.0, track=0.4),
        "face": Shot((m[0] + 0.07, m[1], m[2] - 0.02), 0.92, -84, -5, drift=0.8),        # profile, the user's left
        "retract": Shot((0.02, 0.20, 0.34), 2.25, 140, -26, drift=1.5),
    }


SHOT_OF = {"Perceive": "establish", "Aim": "bowl", "Over the piece": "bowl", "Plunge": "plunge", "Scoop": "plunge",
           "Swing to the user": "swing", "Find the face": "face", "Wait for an open mouth": "face",
           "Approach": "face", "Bite": "face", "Retract": "retract"}


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Director:
    """Which shot at t, with a 1.8 s eased blend on every change of shot."""

    def __init__(self, b: Bite):
        self.S = shots(b)
        self.cuts: list[tuple[float, str]] = []
        for p in b.phases:
            s = SHOT_OF.get(p.title, "establish")
            if not self.cuts or self.cuts[-1][1] != s:
                self.cuts.append((p.t0, s))

    def camera(self, t: float, tip: np.ndarray) -> mujoco.MjvCamera:
        i = max(k for k, (t0, _) in enumerate(self.cuts) if t0 <= t) if t >= self.cuts[0][0] else 0
        t0, name = self.cuts[i]
        cur = self._params(self.S[name], t - t0, tip)
        if i > 0:
            prev_t0, prev = self.cuts[i - 1]
            a = _smooth((t - t0) / 1.8)
            if a < 1:
                old = self._params(self.S[prev], t - prev_t0, tip)
                az_old = old[2] + 360 * round((cur[2] - old[2]) / 360)
                cur = tuple(o + (c - o) * a for o, c in zip(old[:2], cur[:2])) + \
                    (az_old + (cur[2] - az_old) * a, old[3] + (cur[3] - old[3]) * a)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = cur[0], cur[1], cur[2], cur[3]
        return cam

    @staticmethod
    def _params(s: Shot, dt: float, tip: np.ndarray) -> tuple:
        look = np.array(s.lookat) * (1 - s.track) + tip * s.track
        return (look, s.distance * (1 - 0.012 * min(dt, 12)), s.azimuth + s.drift * dt, s.elevation)


# ------------------------------------------------------------------------------------------------ 3D overlays
def _geom(scn: mujoco.MjvScene) -> mujoco.MjvGeom | None:
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def seg(scn, a, b, rgb, r=0.0012, alpha=0.9, glow=0.8) -> None:
    g = _geom(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.array([*rgb, alpha], np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, np.asarray(a, float), np.asarray(b, float))
    g.emission = glow
    g.category = mujoco.mjtCatBit.mjCAT_DECOR


def ball(scn, p, r, rgb, alpha=0.5, glow=0.6) -> None:
    g = _geom(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), np.asarray(p, float),
                        np.eye(3).flatten(), np.array([*rgb, alpha], np.float32))
    g.emission = glow
    g.category = mujoco.mjtCatBit.mjCAT_DECOR


def ring(scn, c, n, radius, rgb, r=0.0012, alpha=0.9, k=40, glow=0.9) -> None:
    """A circle of `radius` about centre c, normal n."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    a = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    pts = [np.asarray(c) + radius * (math.cos(t) * a + math.sin(t) * b) for t in np.linspace(0, 2 * math.pi, k + 1)]
    for p, q in zip(pts, pts[1:]):
        seg(scn, p, q, rgb, r, alpha, glow)


def square(scn, c, half, rgb, yaw=0.0, r=0.0011, alpha=0.95) -> None:
    ca, sa = math.cos(yaw), math.sin(yaw)
    pts = [np.asarray(c) + np.array([ca * x - sa * y, sa * x + ca * y, 0]) for x, y in
           ((-half, -half), (half, -half), (half, half), (-half, half), (-half, -half))]
    for p, q in zip(pts, pts[1:]):
        seg(scn, p, q, rgb, r, alpha)


# ------------------------------------------------------------------------------------------------ projection
def project(scn: mujoco.MjvScene, p: np.ndarray, w: int, h: int) -> tuple[float, float] | None:
    """World point -> pixel in the last rendered view (mjvGLCamera frustum)."""
    c = scn.camera[0]
    pos, fwd, up = np.array(c.pos), np.array(c.forward), np.array(c.up)
    right = np.cross(fwd, up)
    d = np.asarray(p) - pos
    z = d @ fwd
    if z <= 1e-4:
        return None
    x, y = (d @ right) * c.frustum_near / z, (d @ up) * c.frustum_near / z
    half_h = (c.frustum_top - c.frustum_bottom) / 2
    xn = (x - c.frustum_center) / (half_h * w / h)
    yn = (y - (c.frustum_top + c.frustum_bottom) / 2) / half_h
    return (xn + 1) / 2 * w, (1 - yn) / 2 * h


# ------------------------------------------------------------------------------------------------ the replay
class Twin:
    def __init__(self, b: Bite):
        self.b = b
        lay = Layout.from_config(mouth=b.mouth, foods=b.foods, bowl_xy=b.bowl_xy, bowl_half=b.bowl_half,
                                 width=W, height=H)
        self.m = build(lay)
        self.d = new_data(self.m)
        self.adr = arm_qpos_adr(self.m)
        self.lean_adr = self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "lean")]
        self.tip_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "arm_tip")
        self.mouth_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "mouth")
        self.flange_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arm_flange")
        self.cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "arm_wrist")
        self.food_mocap = [self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "food%d" % i)]
                           for i in range(len(b.foods))]
        self.food_home = [self.d.mocap_pos[k].copy() for k in self.food_mocap]
        self.food_quat = [self.d.mocap_quat[k].copy() for k in self.food_mocap]
        self.main = mujoco.Renderer(self.m, H, W, max_geom=12000)
        self.wrist = mujoco.Renderer(self.m, 270, 480, max_geom=4000)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[1] = self.opt.geomgroup[2] = 1
        self.director = Director(b)
        self.trail: list[tuple[float, np.ndarray]] = []
        self.grip: tuple[np.ndarray, np.ndarray] | None = None    # the picked piece in the flange frame
        self.frames = {k: Image.open(p).convert("RGB") for k, p in b.images.items()}
        self.wait = [(t, Image.open(p).convert("RGB")) for t, p in b.wait_frames]
        self.table_z = float(cfg.workspace.table_height_m)

    # ---------------------------------------------------------------- state at t
    def pose(self, t: float) -> None:
        b, d, m = self.b, self.d, self.m
        d.qpos[self.adr] = np.radians(b.q_at(t))
        ev = b.events
        open_ = ("mouth_open" in ev and ev["mouth_open"] <= t <= ev.get("bite", 1e9) + 0.45)
        after = t - ev.get("bite", 1e9)
        if 0.9 < after < 3.4:                                    # chewing
            open_ = math.sin(after * 2 * math.pi * 1.6) > 0.35
        self.opt.geomgroup[1] = 1
        self.opt.geomgroup[4] = 1 if open_ else 0
        self._closed_visible(not open_)
        d.qpos[self.lean_adr] = self._lean(t)
        mujoco.mj_kinematics(m, d)
        # the picked piece: in the bowl, then on the tines from the plunge's end, gone at the bite
        k = b.picked
        if k >= 0:
            mc = self.food_mocap[k]
            fl_p, fl_R = d.xpos[self.flange_id], d.xmat[self.flange_id].reshape(3, 3)
            if t >= ev.get("plunged", 1e9) and after < 0.35:
                if self.grip is None:
                    tip = d.site_xpos[self.tip_id]
                    centre = tip - fl_R[:, 2] * (b.foods[k].size * 0.45)
                    q = np.zeros(4)
                    mujoco.mju_mat2Quat(q, fl_R.flatten())
                    qi = np.zeros(4)
                    mujoco.mju_negQuat(qi, q)
                    rel = np.zeros(4)
                    mujoco.mju_mulQuat(rel, qi, self.food_quat[k])
                    self.grip = (fl_R.T @ (centre - fl_p), rel)
                off, rel = self.grip
                d.mocap_pos[mc] = fl_p + fl_R @ off
                q = np.zeros(4)
                mujoco.mju_mat2Quat(q, fl_R.flatten())
                mujoco.mju_mulQuat(d.mocap_quat[mc], q, rel)
            elif after >= 0.35:
                d.mocap_pos[mc] = [0, 0, -5]
            else:
                self.grip = None
                d.mocap_pos[mc], d.mocap_quat[mc] = self.food_home[k], self.food_quat[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)                                 # cameras and lights follow (kinematics alone does not)

    def _closed_visible(self, on: bool) -> None:
        """The closed-jaw head parts are group 1 with the lower body; hide just them while the mouth is open."""
        if not hasattr(self, "_closed_ids"):
            self._closed_ids = [i for i in range(self.m.ngeom) if (self.m.geom(i).name or "").endswith("_closed")
                                and self.m.geom_bodyid[i] == self.m.body("user_upper").id]
        for i in self._closed_ids:
            self.m.geom_group[i] = 1 if on else 5

    def _lean(self, t: float) -> float:
        """The user leans in to take the food off the fork (the arm stops 80 mm short), then sits back."""
        ev = self.b.events
        if "presented" not in ev or "bite" not in ev:
            return 0.0
        reach = 0.12
        t_in, t_bite = ev["presented"] + 0.2, ev["bite"]
        if t < t_in:
            return 0.0
        if t < t_bite:
            return reach * _smooth((t - t_in) / max(0.3, t_bite - t_in))
        return reach * (1 - _smooth((t - t_bite - 0.3) / 1.4))

    # ---------------------------------------------------------------- overlays
    def overlays(self, scn: mujoco.MjvScene, t: float, ph_title: str) -> None:
        d, b = self.d, self.b
        tip = d.site_xpos[self.tip_id].copy()
        self.trail = [(tt, p) for tt, p in self.trail if t - tt < 7.0 and tt <= t]
        if not self.trail or np.linalg.norm(self.trail[-1][1] - tip) > 0.002:
            self.trail.append((t, tip))
        for (t1, p1), (t2, p2) in zip(self.trail, self.trail[1:]):
            a = max(0.0, 1 - (t - t2) / 7.0)
            seg(scn, p1, p2, CYAN, r=0.0016, alpha=0.15 + 0.75 * a, glow=0.9)
        ball(scn, tip, 0.0035, CYAN, alpha=0.9, glow=1.0)
        if ph_title in ("Perceive", "Aim"):
            self._frustum(scn, t)
            for i, f in enumerate(b.foods):
                p = d.mocap_pos[self.food_mocap[i]]
                col = AMBER if (i == b.picked and ph_title == "Aim") else GREEN
                square(scn, [p[0], p[1], p[2] + f.size * 0.55], f.size * 0.75, col, yaw=f.yaw)
        if ph_title in ("Aim", "Over the piece", "Plunge") and b.picked >= 0:
            p = self.food_home[b.picked]
            ring(scn, [p[0], p[1], p[2] + b.foods[b.picked].size * 0.52], [0, 0, 1], 0.017, AMBER, r=0.001)
            if ph_title != "Plunge":
                for z0 in np.arange(p[2] + 0.02, tip[2] - 0.004, 0.012):
                    seg(scn, [p[0], p[1], z0], [p[0], p[1], min(tip[2], z0 + 0.006)], AMBER, r=0.0007, alpha=0.8)
        if ph_title in ("Find the face", "Wait for an open mouth", "Approach", "Bite"):
            mo = d.site_xpos[self.mouth_id].copy()
            pulse = 0.5 + 0.5 * math.sin(t * 5)
            ring(scn, mo, [1, 0, 0], 0.028 + 0.004 * pulse, AMBER if ph_title == "Wait for an open mouth" else GREEN,
                 r=0.0010)
            ring(scn, mo, [1, 0, 0], 0.045, CYAN, r=0.0006, alpha=0.5)
            stop = np.array(b.mouth) + np.array([0.08, 0, 0])
            if ph_title in ("Approach", "Bite"):
                ball(scn, stop, 0.006, GREEN, alpha=0.35, glow=0.8)
                for z0 in np.linspace(0, 1, 14)[:-1:2]:
                    seg(scn, tip + (mo - tip) * z0, tip + (mo - tip) * (z0 + 1 / 13), GREEN, r=0.0007, alpha=0.7)
        if "bite" in b.events and 0 <= t - b.events["bite"] < 0.9:
            k = (t - b.events["bite"]) / 0.9
            ring(scn, d.site_xpos[self.mouth_id], [1, 0, 0], 0.03 + 0.08 * k, GREEN, r=0.0016, alpha=1 - k)

    def _frustum(self, scn, t: float) -> None:
        d = self.d
        p = d.cam_xpos[self.cam_id]
        R = d.cam_xmat[self.cam_id].reshape(3, 3)
        c = cfg.camera.colour
        z = p[2] - self.table_z
        corners = []
        for u, v in ((0, 0), (c.width, 0), (c.width, c.height), (0, c.height)):
            ray = R @ np.array([(u - c.cx) / c.fx, -(v - c.cy) / c.fy, -1.0])
            corners.append(p + ray * (z / max(1e-6, -ray[2])))
        a = 0.35 + 0.25 * math.sin(t * 3)
        for q in corners:
            seg(scn, p, q, CYAN, r=0.0008, alpha=a)
        for q1, q2 in zip(corners, corners[1:] + corners[:1]):
            seg(scn, q1 + [0, 0, 0.001], q2 + [0, 0, 0.001], CYAN, r=0.0012, alpha=0.8)

    # ---------------------------------------------------------------- one frame
    def frame(self, t: float, total: tuple[float, float] | None = None) -> Image.Image:
        b = self.b
        self.pose(t)
        ph = b.phase_at(t)
        tip = self.d.site_xpos[self.tip_id].copy()
        cam = self.director.camera(t, tip)
        self.main.update_scene(self.d, camera=cam, scene_option=self.opt)
        self.overlays(self.main.scene, t, ph.title)
        img = Image.fromarray(self.main.render())
        self.wrist.update_scene(self.d, camera="arm_wrist", scene_option=self.opt)
        wimg = Image.fromarray(self.wrist.render())
        if ph.stage == "deliver" and ph.title not in ("Swing to the user", "Retract") and ROTATE_AT_DELIVERY:
            wimg = wimg.rotate(-90, expand=True)                 # as the stack sees it: world-up is up
        labels = []
        if ph.title in ("Perceive", "Aim"):
            for i, f in enumerate(b.foods):
                p = self.d.mocap_pos[self.food_mocap[i]] + [0, 0, f.size * 0.9]
                px = project(self.main.scene, p, W, H)
                if px:
                    labels.append((px, "food %d" % (i + 1), i == b.picked))
        return self.hud(img, wimg, t, ph, labels, total)

    # ---------------------------------------------------------------- HUD
    def _real_frame(self, t: float, title: str) -> tuple[Image.Image | None, str]:
        fr = self.frames
        if title == "Perceive" and "detection" in fr:
            return fr["detection"], "LocateAnything · detections"
        if title in ("Aim", "Over the piece") and "target" in fr:
            return fr["target"], "target · the tip's vertical line"
        if title in ("Plunge",) and "over_depth" in fr:
            return fr["over_depth"], "depth at OVER · plunge depth"
        if title in ("Find the face", "Wait for an open mouth", "Approach") and self.wait:
            best = min(self.wait, key=lambda w: abs(w[0] - t) if w[0] <= t + 0.25 else 1e9)
            return best[1], "resident colour stream · MediaPipe"
        if title == "Bite" and self.wait:
            return self.wait[-1][1], "the open mouth that started the approach"
        return None, ""

    def hud(self, img: Image.Image, wimg: Image.Image, t: float, ph, labels, total) -> Image.Image:
        b = self.b
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # vignette-ish top and bottom bands
        for y in range(0, 150):
            d.line([(0, y), (W, y)], fill=(4, 6, 10, int(170 * (1 - y / 150) ** 1.6)))
        for y in range(H - 230, H):
            d.line([(0, y), (W, y)], fill=(4, 6, 10, int(215 * ((y - (H - 230)) / 230) ** 1.4)))
        # title
        d.text((48, 34), "RAMMP", font=F["title"], fill=(255, 255, 255, 255))
        d.text((200, 42), "feeding digital twin", font=F["mid"], fill=(186, 200, 220, 255))
        run = b.session.name.split("_")[0]
        sub = "replay of the recorded run %s-%s-%s  ·  bite %d of %s  ·  MuJoCo · Kinova Gen3 + 2F-85 + fork" % (
            run[:4], run[4:6], run[6:8], b.info["n"], b.info.get("n_bites") or "?")
        d.text((50, 88), sub, font=F["small"], fill=(150, 165, 185, 255))
        # stage chip + phase
        col = STAGE_RGB[ph.stage]
        x0, y0 = 48, H - 212
        tw = d.textlength(ph.stage.upper(), font=F["smallb"])
        d.rounded_rectangle([x0, y0, x0 + tw + 28, y0 + 30], 15, fill=(*col, 60), outline=(*col, 255), width=2)
        d.text((x0 + 14, y0 + 5), ph.stage.upper(), font=F["smallb"], fill=(*col, 255))
        d.text((x0, y0 + 42), ph.title, font=F["big"], fill=(255, 255, 255, 255))
        d.text((x0, y0 + 86), ph.detail, font=F["mid"], fill=(190, 205, 225, 255))
        # live readouts
        self._readouts(d, t, ph)
        # timeline
        self._timeline(d, t, total)
        # the REAL | TWIN tiles
        real, cap = self._real_frame(t, ph.title)
        tx, ty, tw_, th_ = W - 48 - 480, 34, 480, 270
        tiles = [(wimg, "TWIN · wrist camera (real intrinsics)")]
        if real is not None:
            tiles.insert(0, (real, "REAL · " + cap))
        for k, (im, caption) in enumerate(tiles):
            ytile = ty + k * (th_ + 44)
            fit = im.copy()
            fit.thumbnail((tw_, th_))
            bg = Image.new("RGB", (tw_, th_), (6, 8, 12))
            bg.paste(fit, ((tw_ - fit.width) // 2, (th_ - fit.height) // 2))
            img.paste(bg, (tx, ytile))
            c = (56, 189, 248) if caption.startswith("TWIN") else (251, 191, 36)
            d.rectangle([tx - 2, ytile - 2, tx + tw_ + 1, ytile + th_ + 1], outline=(*c, 230), width=2)
            d.rectangle([tx - 2, ytile + th_ + 2, tx + tw_ + 1, ytile + th_ + 30], fill=(6, 8, 12, 200))
            d.text((tx + 10, ytile + th_ + 7), caption, font=F["small"], fill=(*c, 255))
        placed: list[tuple[float, float]] = []
        for (u, v), text, picked in sorted(labels, key=lambda x: x[0][0]):
            while any(abs(u - pu) < 95 and abs(v - pv) < 28 for pu, pv in placed):
                v -= 30
            placed.append((u, v))
            c = (251, 191, 36) if picked else (74, 222, 128)
            wl = d.textlength(text, font=F["smallb"])
            d.rounded_rectangle([u - wl / 2 - 8, v - 30, u + wl / 2 + 8, v - 6], 6, fill=(6, 8, 12, 190), outline=(*c, 255))
            d.text((u - wl / 2, v - 27), text, font=F["smallb"], fill=(*c, 255))
            d.line([(u, v - 6), (u, v + 4)], fill=(*c, 255), width=2)
        out = img.convert("RGBA")
        out.alpha_composite(ov)
        return out.convert("RGB")

    def _readouts(self, d: ImageDraw.ImageDraw, t: float, ph) -> None:
        b = self.b
        dt = 1 / FPS
        q1, q2 = b.q_at(t - dt), b.q_at(t + dt)
        dq = np.abs(q2 - q1) / (2 * dt)
        # joint speeds against the soft limits
        x0, y0 = 760, H - 206
        d.text((x0, y0), "JOINT SPEED  (soft limit 70 / 63 °/s)", font=F["tiny"], fill=(150, 165, 185, 255))
        for k in range(7):
            yy = y0 + 24 + k * 16
            frac = min(1.0, dq[k] / SOFT_LIMITS[k])
            d.text((x0, yy - 3), "J%d" % (k + 1), font=F["tiny"], fill=(170, 185, 205, 255))
            d.rounded_rectangle([x0 + 30, yy, x0 + 230, yy + 9], 4, fill=(40, 48, 62, 200))
            c = (52, 211, 153) if frac < 0.7 else (251, 191, 36) if frac < 0.95 else (248, 113, 113)
            if frac > 0.005:
                d.rounded_rectangle([x0 + 30, yy, x0 + 30 + max(9, 200 * frac), yy + 9], 4, fill=(*c, 255))
            d.text((x0 + 240, yy - 3), "%4.0f" % dq[k], font=F["tiny"], fill=(170, 185, 205, 255))
        # tip speed and height
        tip_now = self.d.site_xpos[self.tip_id].copy()
        x1 = x0 + 330
        speed = getattr(self, "_last_tip", None)
        v = 0.0 if speed is None else np.linalg.norm(tip_now - speed[1]) / max(1e-3, t - speed[0])
        self._last_tip = (t, tip_now)
        d.text((x1, y0), "FORK TIP", font=F["tiny"], fill=(150, 165, 185, 255))
        d.text((x1, y0 + 20), "%3.0f mm/s" % (1000 * v), font=F["midb"], fill=(255, 255, 255, 255))
        d.text((x1, y0 + 52), "%3.0f mm above the table" % (1000 * (tip_now[2] - self.table_z)), font=F["small"],
               fill=(190, 205, 225, 255))
        # the phase's own signal
        x2 = x1 + 290
        if ph.title == "Wait for an open mouth" or (ph.title == "Approach" and len(b.open_t)):
            o = b.opening_at(t)
            d.text((x2, y0), "MOUTH OPENING  (threshold %.2f)" % 0.2, font=F["tiny"], fill=(150, 165, 185, 255))
            d.rounded_rectangle([x2, y0 + 24, x2 + 300, y0 + 40], 8, fill=(40, 48, 62, 220))
            frac = min(1.0, o / 0.8)
            c = (52, 211, 153) if o > 0.2 else (251, 191, 36)
            d.rounded_rectangle([x2, y0 + 24, x2 + max(16, 300 * frac), y0 + 40], 8, fill=(*c, 255))
            thx = x2 + 300 * 0.2 / 0.8
            d.line([(thx, y0 + 20), (thx, y0 + 44)], fill=(255, 255, 255, 255), width=2)
            d.text((x2, y0 + 50), "%.2f %s" % (o, "OPEN" if o > 0.2 else ""), font=F["midb"], fill=(*c, 255))
        bs = next((s for s in b.segments if s.label == "bite"), None)
        if ph.title == "Bite" and bs is not None:
            f = np.linalg.norm(bs.force - bs.force[:5].mean(axis=0), axis=1)
            k = int(np.searchsorted(bs.t, t))
            d.text((x2, y0), "FORCE AT THE FORK (from joint torques)", font=F["tiny"], fill=(150, 165, 185, 255))
            gx, gy, gw, gh = x2, y0 + 22, 330, 70
            d.rectangle([gx, gy, gx + gw, gy + gh], fill=(20, 26, 36, 200))
            top = max(3.0, float(f.max()) * 1.15)
            pts = [(gx + gw * (tt - bs.t[0]) / (bs.t[-1] - bs.t[0]), gy + gh - gh * v_ / top)
                   for tt, v_ in zip(bs.t[:k + 1], f[:k + 1])]
            if len(pts) > 1:
                d.line(pts, fill=(52, 211, 153, 255), width=3)
            be = b.events.get("bite")
            if be is not None and t >= be:
                bx = gx + gw * (be - bs.t[0]) / (bs.t[-1] - bs.t[0])
                d.line([(bx, gy), (bx, gy + gh)], fill=(251, 191, 36, 255), width=2)
                d.text((bx + 6, gy + 2), "bite!", font=F["smallb"], fill=(251, 191, 36, 255))
            d.text((gx, gy + gh + 6), "%.1f N" % (f[min(k, len(f) - 1)]), font=F["midb"], fill=(255, 255, 255, 255))

    def _timeline(self, d: ImageDraw.ImageDraw, t: float, total) -> None:
        b = self.b
        x0, x1, y = 48, W - 48, H - 52
        dur = b.duration
        d.rounded_rectangle([x0, y, x1, y + 8], 4, fill=(40, 48, 62, 230))
        for p in b.phases:
            a = x0 + (x1 - x0) * p.t0 / dur
            e = x0 + (x1 - x0) * min(p.t1, dur) / dur
            c = STAGE_RGB[p.stage]
            done = min(e, x0 + (x1 - x0) * t / dur)
            d.rectangle([a, y, max(a, e - 2), y + 8], fill=(*c, 70))
            if done > a:
                d.rectangle([a, y, max(a, done - 2), y + 8], fill=(*c, 255))
        px = x0 + (x1 - x0) * min(1.0, t / dur)
        d.ellipse([px - 8, y - 4, px + 8, y + 12], fill=(255, 255, 255, 255))
        d.text((x0, y + 16), "t = %5.1f s" % t, font=F["small"], fill=(190, 205, 225, 255))
        right = "%.1f s bite" % dur
        if total:
            right = "film %.0f / %.0f s  ·  " % total + right
        d.text((x1 - d.textlength(right, font=F["small"]), y + 16), right, font=F["small"], fill=(150, 165, 185, 255))
        for label, key in (("mouth open", "mouth_open"), ("bite", "bite"), ("on the fork", "plunged")):
            if key in b.events:
                ex = x0 + (x1 - x0) * b.events[key] / dur
                d.line([(ex, y - 10), (ex, y - 2)], fill=(255, 255, 255, 200), width=2)
                d.text((ex - d.textlength(label, font=F["tiny"]) / 2, y - 28), label, font=F["tiny"],
                       fill=(210, 220, 235, 255))


# ------------------------------------------------------------------------------------------------ commands
def latest_session() -> Path:
    runs = Path(os.path.expanduser(str(cfg.deployment.runs_dir)))
    cands = sorted(p for p in runs.glob("*_feed") if (p / "events.jsonl").exists())
    if not cands:
        raise SystemExit("no feed sessions under %s -- pass --session (rsync one from the Jetson)" % runs)
    return cands[-1]


def bites_in(session: Path) -> list[int]:
    return sorted(int(p.name[5:]) for p in session.glob("bite_*") if (p / "perceive" / "target.json").exists()
                  and (p / "summary.json").exists())


def _ffmpeg(path: Path) -> subprocess.Popen:
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                            stdin=subprocess.PIPE)


def title_card(text: str, sub: str) -> Image.Image:
    im = Image.new("RGB", (W, H), (5, 7, 11))
    d = ImageDraw.Draw(im)
    for r in range(600, 0, -6):
        c = int(22 * (1 - r / 600))
        d.ellipse([W / 2 - r * 1.6, H / 2 - r, W / 2 + r * 1.6, H / 2 + r], fill=(5 + c // 2, 8 + c, 14 + c * 2))
    tw = d.textlength(text, font=F["title"])
    d.text(((W - tw) / 2, H / 2 - 50), text, font=F["title"], fill=(255, 255, 255))
    sw = d.textlength(sub, font=F["mid"])
    d.text(((W - sw) / 2, H / 2 + 10), sub, font=F["mid"], fill=(150, 170, 200))
    return im


def render(a: argparse.Namespace) -> int:
    session = Path(os.path.expanduser(a.session)) if a.session else latest_session()
    nums = bites_in(session) if a.bite == "all" else [int(a.bite)]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.still is not None:
        tw = Twin(load_bite(session, nums[0]))
        tw.frame(max(0.0, a.still - 0.1))
        out = OUT / ("twin_%s_bite%02d_t%05.1f.png" % (session.name, nums[0], a.still))
        tw.frame(a.still).save(out)
        print(out)
        return 0
    tag = "all" if a.bite == "all" else "bite%02d" % nums[0]
    path = Path(a.out) if a.out else OUT / ("twin_%s_%s.mp4" % (session.name, tag))
    ff = _ffmpeg(path)
    bites = [load_bite(session, n) for n in nums]
    total_s = sum(b.duration for b in bites) / a.speed
    done_s = 0.0
    t_wall = time.time()
    run = session.name.split("_")[0]
    card = title_card("RAMMP  ·  robot-assisted feeding", "digital twin of the recorded run %s-%s-%s  ·  %d bite%s"
                      % (run[:4], run[4:6], run[6:8], len(bites), "s" if len(bites) > 1 else ""))
    for k in range(int(FPS * 1.6)):
        fade = min(1.0, k / (FPS * 0.5), (FPS * 1.6 - k) / (FPS * 0.4))
        ff.stdin.write(np.asarray(Image.blend(Image.new("RGB", (W, H)), card, fade)).tobytes())
    for b in bites:
        tw = Twin(b)
        n = int(b.duration / a.speed * FPS)
        for i in range(n):
            t = i * a.speed / FPS
            im = tw.frame(t, (done_s + i / FPS, total_s))
            if i < FPS // 2:
                im = Image.blend(Image.new("RGB", (W, H)), im, i / (FPS // 2))
            ff.stdin.write(np.asarray(im).tobytes())
            if i % (FPS * 5) == 0:
                el = time.time() - t_wall
                print("  %s  t=%5.1f / %.1f s   (%.1f frames/s)" % (b.name, t, b.duration,
                                                                  (i + 1) / max(1e-3, el)), flush=True)
        done_s += n / FPS
        t_wall = time.time()
    ff.stdin.close()
    ff.wait()
    print("wrote %s (%.0f s)" % (path, done_s + 1.6))
    return 0


def poses(a: argparse.Namespace) -> int:
    """The calibrated poses, rendered side by side (the arm in each, the fork where it is)."""
    cal = Path("calibration")
    items = [("feeding_start", json.loads((cal / "home_pose.json").read_text())["joints_deg"]),
             ("food_detect_start", json.loads((cal / "food_detect_pose.json").read_text())["joints_deg"]),
             ("scoop_end", list(cfg.bowl.scoop_end.joints_deg))]
    from sim.scene import Food
    lay = Layout.from_config(mouth=(-0.277, 0.306, 0.635), foods=[Food((0.245, 0.245)), Food((0.225, 0.215),
                             rgb=(0.95, 0.85, 0.35)), Food((0.26, 0.21), rgb=(0.95, 0.55, 0.3))],
                             bowl_xy=(0.240, 0.236), width=W, height=H)
    m = build(lay)
    dd = new_data(m)
    r = mujoco.Renderer(m, 540, 960)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = opt.geomgroup[2] = 1
    sheet = Image.new("RGB", (1920, 1080), (5, 7, 11))
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm_tip")
    for k, (name, q) in enumerate(items):
        dd.qpos[arm_qpos_adr(m)] = np.radians(q)
        mujoco.mj_forward(m, dd)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0.0, 0.2, 0.38)
        cam.distance, cam.azimuth, cam.elevation = 1.7, 150, -18
        r.update_scene(dd, camera=cam, scene_option=opt)
        ball(r.scene, dd.site_xpos[tip_id], 0.006, CYAN, alpha=0.9, glow=1.0)
        im = Image.fromarray(r.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 960, 70], fill=(5, 7, 11))
        dr.text((24, 12), name, font=F["big"], fill=(255, 255, 255))
        tp = dd.site_xpos[tip_id]
        dr.text((26, 52 - 4), "joints %s   tip (%.3f, %.3f, %.3f) m" % (" ".join("%.0f" % x for x in q), *tp),
                font=F["tiny"], fill=(150, 170, 200))
        sheet.paste(im, ((k % 2) * 960, (k // 2) * 540))
    out = OUT / "twin_poses.png"
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="replay a recorded bite to an MP4 (or one frame with --still)")
    r.add_argument("--session", help="a feed session dir (default: the newest under runs_dir)")
    r.add_argument("--bite", default="1", help="the bite number, or 'all'")
    r.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    r.add_argument("--still", type=float, help="render one frame at this time (s) to a PNG")
    r.add_argument("--out", help="the output path")
    sub.add_parser("poses", help="the calibrated poses, one sheet")
    a = ap.parse_args()
    return {"render": render, "poses": poses}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
