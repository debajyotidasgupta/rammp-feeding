#!/usr/bin/env python3
"""How wrong may the scene camera be? feeding_start <-> food_detect_start planned on what it reports, judged on the truth.

    python3 tests/scene_error_study.py [--trials 25] [--out study.json]       # in the curobo-eval container (GPU)

The scene camera's extrinsic is calibrated to ~5 cm (scene_camera.extrinsic.error_m). For each placement of the
camera and each sampled error -- the extrinsic off by up to error_m in position and ROT_DEG in angle (about the
camera), plus depth noise along the ray -- the bowl and table it would report are the true ones seen through that
error. The stack then runs as on the robot: food_detect_start computed over the REPORTED bowl, cuRobo planning in
the REPORTED world (grown by the margin under test), our tip floor. Each plan is judged against the TRUE scene:

  collides   any waypoint (and 3 points between each) invalid for cuRobo in the TRUE world (its padding: ~1-2 cm)
  tip        the utensil tip's lowest clearance over the TRUE table / bowl rim along the path
  framed     at food_detect_start the TRUE bowl centre is within the wrist camera's view, and within OVER's
             reach (acquisition.over_max_m) of where the lens points -- the wrist camera can take it from there
  peak       the tip's highest point above the true table (the user: lower is better)

for margins 0 (the reading trusted) and error_m, at overlook heights 0.40 and 0.45. The site's own placement
(the box-opening stack's calibration, 2026-09-24) is one of the cameras.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
RUNS = Path(tempfile.mkdtemp(prefix="scene_error_"))
os.environ.update(RAMMP_NO_LOCAL="1", RAMMP_CAMERA="realsense_d405", RAMMP_SCENE_CAMERA="orbbec_gemini336l",
                  RAMMP_SET="deployment.runs_dir=%s" % RUNS)

from settings import cfg  # noqa: E402
import fake_kortex as fk  # noqa: E402
import mujoco_kortex as mk  # noqa: E402

TRUTH = dict(bowl_xy=[0.290, 0.158], table_z=-0.024, bowl_top_z=0.046, rim_radius_m=0.054)   # site_20260930
MOUTH = [-0.277, 0.306, 0.397]
ROT_DEG = 1.5                     # the calibration's own mismatch: 1.17 deg
DEPTH_NOISE = 0.01                # of the range, 1 sigma
HOME = json.loads((ROOT / "calibration" / "home_pose.json").read_text())["joints_deg"]


def look(c, target) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(c, float), np.asarray(target, float)


BOWL3 = np.array([TRUTH["bowl_xy"][0], TRUTH["bowl_xy"][1], TRUTH["table_z"]])
PLACEMENTS = {                    # camera centre, the point it looks at
    "site (box-opening calibration)": look([-0.065, 0.642, 0.352], BOWL3),
    "across the table": look([1.10, 0.25, 0.45], BOWL3),
    "overhead": look([0.35, 0.20, 1.20], BOWL3),
    "low, from the side": look([0.35, 1.00, 0.10], BOWL3),
}


def rot(axis, deg) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def reported(c, rng, err_m) -> tuple[dict, float]:
    """The scene the camera at c would report with a random extrinsic error; and how far its bowl is off."""
    d = rng.normal(size=3); dt = d / np.linalg.norm(d) * err_m * rng.uniform() ** (1 / 3)   # uniform in the ball
    Re = rot(rng.normal(size=3), rng.uniform(0, ROT_DEG))

    def est(p):
        p = np.asarray(p, float)
        r = p - c
        p = p + rng.normal(0, DEPTH_NOISE * np.linalg.norm(r)) * r / np.linalg.norm(r)
        return Re @ (p - c) + c + dt
    b = est(BOWL3 + [0, 0, TRUTH["bowl_top_z"] - TRUTH["table_z"]])        # the rim's centre
    t = est(BOWL3 + [TRUTH["rim_radius_m"] + 0.08, 0, 0])                  # the table beside the bowl
    rep = dict(t=fk.clock.time(), bowl_xy=[float(b[0]), float(b[1])], table_z=float(t[2]),
               bowl_top_z=float(b[2]), rim_radius_m=TRUTH["rim_radius_m"])
    off = float(np.linalg.norm([rep["bowl_xy"][0] - TRUTH["bowl_xy"][0], rep["bowl_xy"][1] - TRUTH["bowl_xy"][1],
                                rep["table_z"] - TRUTH["table_z"]]))
    return rep, off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--out", default=str(RUNS / "study.json"))
    a = ap.parse_args()
    fk.install_clock()
    from kinova import scene
    from kinova.curobo_world import world
    from kinova.frames import R_from_kortex
    from kinova.motion import tip_in_tool
    from kinova.planner_server import Planners
    from feeding.camera import CAM_IN_TOOL, FXC
    arm = mk.KinematicArm(tool_z=float(cfg.deployment.arm.tool.transform_m[2]))
    conn = mk.KinematicConnection(arm)
    err_m = float(cfg.scene_camera.extrinsic.error_m)
    over_max = float(cfg.acquisition.over_max_m)
    state = RUNS / "scene_state.json"
    state.write_text(json.dumps(dict(TRUTH, t=time.time())))
    P = Planners()
    pl = P.by_load[False]
    truth_world = world(dict(TRUTH, bowl_radius=TRUTH["rim_radius_m"], error_m=0.0), MOUTH,
                        table_extent=dict(cfg.workspace.table_extent_m))["obstacles"]
    half_w = math.atan(float(cfg.camera.colour.width) / 2 / FXC)            # the wrist camera's half view, rad

    def tip(q):
        p = arm.fk_pose(q)
        return np.array(p[:3]) + R_from_kortex(*p[3:]) @ tip_in_tool()

    def true_clearance(t) -> float:
        bx, by = TRUTH["bowl_xy"]
        ext = cfg.workspace.table_extent_m
        if math.hypot(t[0] - bx, t[1] - by) < TRUTH["rim_radius_m"] + 0.02:
            return t[2] - TRUTH["bowl_top_z"]
        if ext.x[0] <= t[0] <= ext.x[1] and ext.y[0] <= t[1] <= ext.y[1]:
            return t[2] - TRUTH["table_z"]
        return math.inf

    rows = []
    rng = np.random.default_rng(7)
    samples = {name: [reported(c, rng, err_m) for _ in range(a.trials)] for name, (c, _) in PLACEMENTS.items()}
    for height in (0.40, 0.45):
        cfg["workspace"]["overlook_camera_height_m"] = height
        for margin in (0.0, err_m):
            for name, trials in samples.items():
                for rep, off in trials:
                    state.write_text(json.dumps(dict(rep, error_m=margin)))
                    scene._RESOLVED.clear()
                    row = dict(camera=name, height=height, margin=margin, off_m=off)
                    try:
                        qd = scene.overlook_joints(conn)
                    except Exception as e:  # noqa: BLE001
                        rows.append(dict(row, outcome="overlook unreachable: %s" % e)); continue
                    P.refresh()
                    paths = {}
                    for leg, (s, g) in dict(detect=(HOME, qd), home=(qd, HOME)).items():
                        r = P.plan(s, list(np.asarray(s) + (np.asarray(g) - np.asarray(s) + 180) % 360 - 180), False)
                        paths[leg] = np.array(r["path"]) if r.get("ok") else None
                    if any(p is None for p in paths.values()):
                        rows.append(dict(row, outcome="no plan")); continue
                    floor_ok = all(t[2] >= scene.tip_floor(t[0], t[1]) - 1e-6 for p in paths.values() for t in map(tip, p))
                    pl.update_world(truth_world)
                    dense = [q0 + (q1 - q0) * s for p in paths.values() for q0, q1 in zip(p, p[1:])
                             for s in np.linspace(0, 1, 4, endpoint=False)]
                    collides = sum(not pl.check_state_valid(list(np.radians(q)))[0] for q in dense)
                    tips = [tip(q) for q in dense]
                    # at food_detect_start: where the lens points on the true bowl's rim plane, vs the true bowl
                    pq = arm.fk_pose(qd)
                    R = R_from_kortex(*pq[3:])
                    lens = np.array(pq[:3]) + R @ CAM_IN_TOOL
                    ax = R[:, 2]
                    s_ = (TRUTH["bowl_top_z"] - lens[2]) / ax[2]
                    aim = lens + s_ * ax
                    miss = math.hypot(aim[0] - TRUTH["bowl_xy"][0], aim[1] - TRUTH["bowl_xy"][1])
                    in_view = math.atan2(miss, lens[2] - TRUTH["bowl_top_z"]) < half_w
                    rows.append(dict(row, outcome="ok", floor_ok=floor_ok, collides=int(collides),
                                     min_tip_clear_m=float(min(true_clearance(t) for t in tips)),
                                     peak_tip_m=float(max(t[2] for t in tips) - TRUTH["table_z"]),
                                     aim_miss_m=float(miss), framed=bool(in_view and miss <= over_max),
                                     lens_above_table_m=float(lens[2] - TRUTH["table_z"])))
                print("height %.2f margin %.2f done (%d rows)" % (height, margin, len(rows)), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    summary(rows)
    return 0


def summary(rows) -> None:
    print("\n%-32s %5s %6s  %6s %8s %8s %8s %8s %8s" % ("camera", "H", "margin", "plans", "collide", "min tip",
                                                       "peak tip", "framed", "off p95"))
    keys = sorted({(r["camera"], r["height"], r["margin"]) for r in rows}, key=lambda k: (k[1], k[2], k[0]))
    for k in keys:
        rs = [r for r in rows if (r["camera"], r["height"], r["margin"]) == k]
        ok = [r for r in rs if r["outcome"] == "ok"]
        col = sum(r["collides"] > 0 for r in ok)
        mt = min((r["min_tip_clear_m"] for r in ok), default=float("nan"))
        pk = max((r["peak_tip_m"] for r in ok), default=float("nan"))
        fr = sum(r["framed"] for r in ok)
        print("%-32s %5.2f %6.2f  %3d/%-3d %8d %8.3f %8.3f %5d/%-3d %7.3f" % (
            k[0][:32], k[1], k[2], len(ok), len(rs), col, mt, pk, fr, len(ok), np.percentile([r["off_m"] for r in rs], 95)))


if __name__ == "__main__":
    raise SystemExit(main())
