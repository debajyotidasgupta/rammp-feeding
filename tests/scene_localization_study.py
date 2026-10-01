#!/usr/bin/env python3
"""Can the scene camera find the bowl from where it is put? feeding/locate_scene's own geometry on rendered depth.

    .venv/bin/python tests/scene_localization_study.py [--samples 8] [--out out/studies/localization.json]

For each scene camera placement and each bowl position on the table, MuJoCo renders the scene camera's depth
image (the twin: the table, an opaque round bowl, the user, the wheelchair, the arm parked at feeding_start; the
Orbbec's depth intrinsics; 0.3 % depth noise). The detector is replaced by the bowl's true outline in the colour
image, grown and shifted by up to 10 % (what a detector's box does). locate_scene.measure() then reports the bowl
centre, the table and the rim height -- with the true extrinsic (the method's own error) and with sampled
extrinsic errors (scene_camera.extrinsic.error_m in position, ROT_DEG in angle: what deployment day's calibration
leaves). The rows (truth and reports) feed tests/scene_error_study.py --reports, which plans on each report and
judges the plan on the truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.update(RAMMP_NO_LOCAL="1", RAMMP_CAMERA="realsense_d405", RAMMP_SCENE_CAMERA="orbbec_gemini336l",
                  RAMMP_SET="deployment.runs_dir=%s" % tempfile.mkdtemp(prefix="scene_loc_"))

import mujoco  # noqa: E402

from feeding import locate_scene as LS  # noqa: E402
from settings import cfg  # noqa: E402
from sim.scene import PRINCIPAL_V_SIGN, Layout, arm_qpos_adr, build, new_data  # noqa: E402

TABLE_Z, RIM_R, BOWL_H = -0.024, 0.054, 0.070            # the 2026-09-30 site's table and bowl
MOUTH = (-0.277, 0.306, 0.397)
ROT_DEG = 1.5
DEPTH_NOISE = 0.003
HOME = json.loads((ROOT / "calibration" / "home_pose.json").read_text())["joints_deg"]
LOOK = np.array([0.33, 0.17, TABLE_Z])                     # the middle of the bowl positions


def looking(c, target=LOOK) -> np.ndarray:
    """base <- camera link frame (x forward, y left, z up) for a camera at c looking at target."""
    x = np.asarray(target, float) - np.asarray(c, float); x /= np.linalg.norm(x)
    up = np.array([0, 0, 1.0]) if abs(x[2]) < 0.95 else np.array([1.0, 0, 0])
    y = np.cross(up, x); y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


PLACEMENTS = {                                             # where the scene camera stands: (centre, base <- link)
    "site (box-opening calibration)": (np.array([-0.065, 0.642, 0.352]),
                                       LS.quat_mat([0.044383, 0.020535, -0.265541, 0.962859])),
    "across the table": (np.array([1.10, 0.25, 0.45]), looking([1.10, 0.25, 0.45])),
    "high, over the table": (np.array([0.40, 0.25, 1.10]), looking([0.40, 0.25, 1.10])),
    "low, from the side": (np.array([0.35, 1.00, 0.12]), looking([0.35, 1.00, 0.12])),
    "behind the arm": (np.array([-0.35, -0.55, 0.65]), looking([-0.35, -0.55, 0.65])),
}
BOWLS = [(x, y) for x in (0.20, 0.30, 0.40, 0.50) for y in (-0.05, 0.10, 0.25, 0.40)]


def rot(axis, deg) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def render_depth(bowl_xy, c, R) -> np.ndarray:
    """The scene camera's depth image (m, the Orbbec's depth intrinsics) with the bowl at bowl_xy."""
    d_ = cfg.scene_camera.depth
    W, H = int(d_.width), int(d_.height)
    cam_axes = np.column_stack([-R[:, 1], R[:, 2], -R[:, 0]])   # MuJoCo's camera: x right, y up, looking along -z
    q = np.zeros(4); mujoco.mju_mat2Quat(q, cam_axes.flatten())

    def add_camera(s):
        cam = s.worldbody.add_camera(name="scene", pos=list(c), quat=list(q))
        cam.resolution = [W, H]
        cam.sensor_size = [W * 3e-6, H * 3e-6]
        cam.focal_pixel = [float(d_.fx), float(d_.fy)]
        cam.principal_pixel = [float(d_.cx) - W / 2, PRINCIPAL_V_SIGN * (float(d_.cy) - H / 2)]
    lay = Layout(TABLE_Z, tuple(bowl_xy), MOUTH, foods=[], width=W, height=H, laptop=False, hair=False,
                 bowl_round=(RIM_R, BOWL_H))
    m = build(lay, add_camera)
    d = new_data(m)
    d.qpos[arm_qpos_adr(m)] = np.radians(HOME)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, H, W)
    r.enable_depth_rendering()
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1] = opt.geomgroup[2] = 1
    r.update_scene(d, camera="scene", scene_option=opt)
    depth = r.render().astype(float)
    r.close()
    depth[(depth <= 0.05) | (depth > 6.0)] = 0.0
    return depth


def true_box(bowl_xy, c, R, rng) -> list[float] | None:
    """The bowl's outline in the colour image (true extrinsic), grown and shifted like a detector's box."""
    a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    pts = np.concatenate([np.column_stack([bowl_xy[0] + rr * np.cos(a), bowl_xy[1] + rr * np.sin(a), np.full(48, z)])
                          for rr, z in ((RIM_R, TABLE_Z + BOWL_H), (0.75 * RIM_R, TABLE_Z))])
    uv = LS.project_colour((pts - c) @ R)
    if np.isnan(uv).any():
        return None
    x1, y1 = uv.min(0); x2, y2 = uv.max(0)
    w, h = x2 - x1, y2 - y1
    col = cfg.scene_camera.colour
    if x2 < 0 or y2 < 0 or x1 > float(col.width) or y1 > float(col.height):
        return None
    g = rng.uniform(0, 0.10, 2) * [w, h]
    s = rng.uniform(-0.05, 0.05, 2) * [w, h]
    return [float(x1 - g[0] + s[0]), float(y1 - g[1] + s[1]), float(x2 + g[0] + s[0]), float(y2 + g[1] + s[1])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8, help="extrinsic errors per placement and bowl position")
    ap.add_argument("--out", default=str(ROOT / "out" / "studies" / "localization.json"))
    a = ap.parse_args()
    err_m = float(cfg.scene_camera.extrinsic.error_m)
    rng = np.random.default_rng(11)
    rows = []
    for name, (c, R) in PLACEMENTS.items():
        for bxy in BOWLS:
            truth = dict(bowl_xy=list(bxy), table_z=TABLE_Z, bowl_top_z=TABLE_Z + BOWL_H, rim_radius_m=RIM_R)
            row = dict(camera=name, truth=truth)
            box = true_box(bxy, c, R, rng)
            if box is None:
                rows.append(dict(row, outcome="out of view")); continue
            depth = render_depth(bxy, c, R)
            depth = np.where(depth > 0, depth * (1 + rng.normal(0, DEPTH_NOISE, depth.shape)), 0.0)
            p_link, _ = LS.depth_points(depth)
            uv = LS.project_colour(p_link)
            reports = []
            for k in range(a.samples + 1):
                if k == 0:                                    # the true extrinsic: the method's own error
                    Re, te = R, c
                else:
                    dv = rng.normal(size=3); dt = dv / np.linalg.norm(dv) * err_m * rng.uniform() ** (1 / 3)
                    Rerr = rot(rng.normal(size=3), rng.uniform(0, ROT_DEG))
                    Re, te = Rerr @ R, c + dt                 # what the calibration believes
                m = LS.measure(box, uv, p_link @ Re.T + te)
                if m.get("bowl_xy") is None:
                    reports.append(dict(rejected=m.get("rejected"))); continue
                reports.append(dict(bowl_xy=m["bowl_xy"], table_z=m["table_z"], bowl_top_z=m["bowl_top_z"],
                                    rim_radius_m=m["rim_radius_m"],
                                    xy_err_m=float(np.hypot(m["bowl_xy"][0] - bxy[0], m["bowl_xy"][1] - bxy[1])),
                                    table_err_m=float(m["table_z"] - TABLE_Z),
                                    top_err_m=float(m["bowl_top_z"] - (TABLE_Z + BOWL_H))))
            rows.append(dict(row, outcome="ok", reports=reports))
        print("  %s done" % name, flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    summary(rows)
    print("rows -> %s" % out)
    return 0


def summary(rows) -> None:
    print("\n%-32s %8s %10s   %-27s   %-27s" % ("camera", "in view", "located", "true extrinsic: xy / table / top",
                                                "with 5 cm error: xy p95 / max"))
    for name in PLACEMENTS:
        rs = [r for r in rows if r["camera"] == name]
        seen = [r for r in rs if r["outcome"] == "ok"]
        exact = [r["reports"][0] for r in seen if "xy_err_m" in r["reports"][0]]
        noisy = [p for r in seen for p in r["reports"][1:] if "xy_err_m" in p]
        f = lambda v, q: 1000 * np.percentile(v, q) if len(v) else float("nan")  # noqa: E731
        print("%-32s %4d/%-3d %5d/%-4d  xy %3.0f/%3.0f mm  tbl %+4.0f  top %+4.0f   %5.0f / %5.0f mm" % (
            name[:32], len(seen), len(rs), len(exact), len(seen),
            f([p["xy_err_m"] for p in exact], 50), f([p["xy_err_m"] for p in exact], 95),
            f([p["table_err_m"] for p in exact], 50), f([p["top_err_m"] for p in exact], 50),
            f([p["xy_err_m"] for p in noisy], 95), max([1000 * p["xy_err_m"] for p in noisy], default=float("nan"))))
    print("(xy: median / p95 of the bowl centre's error; tbl / top: median height errors, mm)")


if __name__ == "__main__":
    raise SystemExit(main())
