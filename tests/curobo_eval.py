#!/usr/bin/env python3
"""Could cuRobo plan our free-space moves? Plan them with it and score them against ours, on a site's geometry.

    tests/site_check.py --dump motions.json                        # (the Mac) our moves, executed on real kinematics
    python3 tests/curobo_eval.py motions.json [--out report.json]  # (a CUDA box, in the curobo-eval container)

For each free-space move of the bite (detect: feeding_start -> food_detect_start; the swing with the food;
the retract's return; home), the same start and goal joints are planned by cuRobo (RAMMP-CuRobo's planner:
the Gen3 + 2F-85 collision model, plus the spoon -- and the food for the swing -- as spheres on the wrist)
in a world built from the site file: the table, the bowl, the user's head and torso around the measured mouth,
the arm's pedestal. Both paths are then scored the same way: duration (and at our 70 deg/s), the spoon tip's
highest and lowest point, the clearance from the table / bowl / user, and how far the spoon tilts (from straight
down when empty, from level when it carries food). The swing is also planned holding the spoon's orientation.

Nothing here talks to an arm.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rammp_curobo  # noqa: E402
from rammp_curobo import CuRoboPlanner  # noqa: E402

CFG_DIR = Path(rammp_curobo.__file__).resolve().parent / "configs"
TOOL_Z = 0.12                         # the flange -> tool frame (ours and cuRobo's tool_frame alike)
DEG_LIMIT = 70.0                      # our joint speed limit, for a like-for-like duration


def fold(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------ the world, from the site file

def world(site: dict) -> dict:
    ws, mouth = site["workspace"], np.array(site["user"]["mouth_m"], float)
    tz = float(ws["table_z"])
    bx, by = ws["bowl_xy"]
    r = float(site.get("bowl_radius_m", 0.06))
    obstacles = [
        dict(name="pedestal", position=[0.0, 0.0, -0.05], dims=[0.14, 0.14, 0.04]),
        dict(name="table", position=[0.42, 0.25, tz - 0.025], dims=[0.68, 1.10, 0.05]),   # from x 0.08: clear of the base
        dict(name="bowl_floor", position=[bx, by, tz + 0.004], dims=[2 * r, 2 * r, 0.008]),
        dict(name="head", position=[mouth[0] - 0.09, mouth[1], mouth[2] + 0.05], dims=[0.20, 0.18, 0.26]),
        dict(name="torso", position=[mouth[0] - 0.12, mouth[1], mouth[2] - 0.30], dims=[0.28, 0.42, 0.45]),
    ]
    # the bowl is HOLLOW (a solid box puts the overlook, the spoon 20 mm over its opening, inside it): its wall as
    # 12 thin boxes round the rim, 6 mm thick, from the table to its top
    top = float(ws["bowl_top_z"])
    for k in range(12):
        a_ = 2 * math.pi * k / 12
        obstacles.append(dict(name="bowl_wall_%d" % k, position=[bx + r * math.cos(a_), by + r * math.sin(a_), (tz + top) / 2],
                              dims=[0.006, 2 * r * math.tan(math.pi / 12) + 0.004, top - tz], yaw=a_))
    for o in obstacles:                                      # plain floats: the planner reads this back as YAML
        o["position"], o["dims"] = [float(v) for v in o["position"]], [float(v) for v in o["dims"]]
        yaw = float(o.pop("yaw", 0.0))
        if yaw:                                               # a box turned about z (their schema: rpy_deg)
            o["rpy_deg"] = [0.0, 0.0, math.degrees(yaw)]
    return dict(base_frame="base_link", obstacles=obstacles, objects=[], targets=[])


# ------------------------------------------------------------------ the spoon (and the food) on the wrist

def spoon_spheres(tip_tool: np.ndarray, food: bool) -> list[dict]:
    """Spheres from the gripper pads to the spoon tip, in bracelet_link's frame (the flange is 61.5 mm out along
    -z, turned 180 deg about x: flange (x, y, z) = bracelet (x, -y, -0.061525 - z))."""
    tip_f = tip_tool + np.array([0.0, 0.0, TOOL_Z])          # in the flange frame
    out = []
    for s in np.linspace(0.0, 1.0, 5):
        p = np.array([0.0, 0.0, 0.17]) + s * (tip_f - np.array([0.0, 0.0, 0.17]))
        out.append(dict(center=[float(p[0]), float(-p[1]), float(-0.061525 - p[2])], radius=0.012 if s < 0.8 else 0.02))
    if food:                                                 # a 25 mm piece riding the tip
        p = tip_f - np.array([0.0, 0.0, 0.01])
        out.append(dict(center=[float(p[0]), float(-p[1]), float(-0.061525 - p[2])], radius=0.02))
    return out


def planner_for(site: dict, tip_tool: np.ndarray, food: bool, workdir: Path, tight: bool = False) -> CuRoboPlanner:
    robot = yaml.safe_load((CFG_DIR / "robot_gen3_2f85.yaml").read_text(encoding="utf-8"))
    k = robot["robot_cfg"]["kinematics"]
    k["collision_spheres"]["bracelet_link"] = list(k["collision_spheres"].get("bracelet_link", [])) + spoon_spheres(tip_tool, food)
    ign = k.setdefault("self_collision_ignore", {})
    ign["bracelet_link"] = sorted(set(ign.get("bracelet_link", [])) | {
        "robotiq_arg2f_base_link", "left_outer_finger", "right_outer_finger", "left_inner_finger_pad", "right_inner_finger_pad"})
    tag = "food" if food else "spoon"
    (workdir / ("robot_%s.yaml" % tag)).write_text(yaml.safe_dump(robot))
    (workdir / "world_site.yaml").write_text(yaml.safe_dump(world(site)))
    plan = yaml.safe_load((CFG_DIR / "gen3.yaml").read_text(encoding="utf-8"))
    plan["robot"], plan["world"] = "robot_%s.yaml" % tag, "world_site.yaml"
    if tight:          # our margins, not cuRobo's: the bowl unpadded, a 1 cm standoff (the default ~4.5 cm in all)
        plan["planner"]["collision_activation_distance"] = 0.01
        plan["planner"]["no_pad_names"] = ["pedestal", "bowl_floor"] + ["bowl_wall_%d" % k for k in range(12)]
        tag += "_tight"
    path = workdir / ("gen3_site_%s.yaml" % tag)
    path.write_text(yaml.safe_dump(plan))
    return CuRoboPlanner.from_config(str(path))


# ------------------------------------------------------------------ scoring

def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_dist(p: np.ndarray, b: dict) -> float:
    c, h = np.array(b["position"], float), np.array(b["dims"], float) / 2
    d = np.abs(p - c) - h
    return float(np.linalg.norm(np.maximum(d, 0.0)) + min(max(d[0], max(d[1], d[2])), 0.0))


def _unyaw(p: np.ndarray, o: dict) -> np.ndarray:
    """p in the frame of a box turned about z (its orientation wxyz), relative to its centre."""
    yaw = math.radians(o["rpy_deg"][2])
    d = np.asarray(p, float) - np.array(o["position"], float)
    c, s_ = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2]])


def score(pl: CuRoboPlanner, q_rad: np.ndarray, dt: np.ndarray, tip_tool: np.ndarray, site: dict, food: bool) -> dict:
    tips, tilts, wrists = [], [], []
    for q in q_rad:
        pos, quat = pl.fk(list(q), quat_order="wxyz")
        R = quat_R(*quat)
        tips.append(np.array(pos) + R @ tip_tool)
        wrists.append(np.array(pos) - R[:, 2] * TOOL_Z)
        ax = R[:, 2]                                         # the spoon's axis: the tool z
        tilts.append(abs(math.degrees(math.asin(np.clip(ax[2], -1, 1)))) if food else
                     math.degrees(math.acos(np.clip(-ax[2], -1, 1))))
    obs = {o["name"]: o for o in world(site)["obstacles"]}
    def dist(p, o):
        return box_dist(p, o) if "rpy_deg" not in o else box_dist(_unyaw(p, o), dict(o, position=[0, 0, 0]))
    raw = {n: min(min(dist(t, o) for t in tips), min(dist(w, o) for w in wrists)) for n, o in obs.items() if n != "pedestal"}
    clear = {}
    for n, v in raw.items():                                  # the bowl's pieces reported as one
        key = "bowl" if n.startswith("bowl") else n
        clear[key] = min(clear.get(key, 1e9), v)
    steps = np.abs(np.diff(np.degrees(q_rad), axis=0)).max(axis=1) if len(q_rad) > 1 else np.zeros(1)
    t_ours = float(np.sum(np.maximum(dt[1:], steps / DEG_LIMIT))) if len(dt) > 1 else float(dt.sum())
    z = np.array(tips)[:, 2]
    return dict(seconds=round(float(dt.sum()), 2), seconds_at_70dps=round(t_ours, 2), tip_z_max=round(float(z.max()), 3),
                tip_z_min=round(float(z.min()), 3), clearance_m={k: round(v, 3) for k, v in clear.items()},
                worst_tilt_deg=round(max(tilts), 1),
                joint_travel_deg=round(float(np.abs(np.diff(np.degrees(q_rad), axis=0)).sum()), 1))


def densify(start: list[float], points: list) -> tuple[np.ndarray, np.ndarray]:
    """Our executed move as dense joint states (a joint move is a straight line in joint space)."""
    qs, dts = [np.radians(fold(start))], [0.0]
    prev = fold(start)
    for q, dt in points:
        q = fold(q)
        n = max(1, int(math.ceil(np.abs(fold(q - prev)).max() / 2.0)))
        for k in range(1, n + 1):
            qs.append(np.radians(prev + fold(q - prev) * k / n)); dts.append(dt / n)
        prev = q
    return np.unwrap(np.array(qs), axis=0), np.array(dts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motions")
    ap.add_argument("--out", default="curobo_eval.json")
    ap.add_argument("--tight", action="store_true", help="our margins: the bowl unpadded, a 1 cm standoff")
    a = ap.parse_args()
    dump = json.loads(Path(a.motions).read_text(encoding="utf-8"))
    site = dict(workspace=dump["workspace"], user=dump["user"], bowl_radius_m=0.054)
    tip_tool = np.array(dump["tip_in_tool"], float)          # our tip model (tests/site_check.py --dump)
    groups: dict[str, list] = {}
    for m in dump["motions"]:
        groups.setdefault(m["step"].split(" --")[0], []).append(m)
    moves = [("detect: feeding_start -> food_detect_start", "sequence detect", False),
             ("swing: scoop end -> delivery (with food)", "deliver swing", True),
             ("retract: presented -> food_detect_start", "deliver retract", False),
             ("home: food_detect_start -> feeding_start", "sequence home", False)]
    work = Path(tempfile.mkdtemp(prefix="curobo_eval_"))
    planners = {}
    report = []
    for title, key, food in moves:
        ms = groups.get(key)
        if not ms:
            continue
        pl = planners.get(food) or planners.setdefault(food, planner_for(site, tip_tool, food, work, a.tight))
        start, goal = fold(ms[0]["start"]), fold(ms[-1]["points"][-1][0])
        ours_q, ours_dt = [], []
        for m in ms:
            q, dt = densify(m["start"], m["points"])
            ours_q.append(q); ours_dt.append(dt)
        oq = np.unwrap(np.concatenate(ours_q), axis=0)
        row = dict(move=title, ours=score(pl, oq, np.concatenate(ours_dt), tip_tool, site, food))
        for mode in (["joints", "hold orientation"] if food else ["joints"]):
            t0 = time.time()
            if mode == "joints":
                res = pl.plan_to_joints(list(np.radians(goal)), list(np.radians(start)))
            else:
                res = plan_holding(pl, start, goal)
            entry = dict(planning_s=round(time.time() - t0, 2), success=bool(res.success),
                         status=res.status, error=res.error)
            if res.success:
                tr = res.joint_traj
                entry.update(score(pl, np.unwrap(tr.positions, axis=0), np.full(tr.n_points, tr.dt), tip_tool, site, food))
            row["curobo " + mode] = entry
        report.append(row)
        print(json.dumps(row, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("report ->", a.out)
    return 0


def plan_holding(pl: CuRoboPlanner, start, goal):
    """The swing with the spoon's orientation held along the way (cuRobo's partial pose cost: the rotation held,
    the position free) -- to the goal's pose."""
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.types.math import Pose
    import torch
    pos, quat = pl.fk(list(np.radians(goal)), quat_order="wxyz")
    goal_pose = Pose(position=pl._tensor([list(pos)]), quaternion=pl._tensor([list(quat)]))
    cfg = pl._plan_config().clone() if hasattr(pl._plan_config(), "clone") else copy.copy(pl._plan_config())
    # the swing turns the spoon about the vertical, so only the two other rotation axes can be held: try each pair
    # (cuRobo's hold weights are in the goal frame: [rx, ry, rz, x, y, z])
    from rammp_curobo.types import PlanResult
    last = None
    for w in ([1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]):
        cfg.pose_cost_metric = PoseCostMetric(hold_partial_pose=True,
                                              hold_vec_weight=pl._tensor(w + [0.0, 0.0, 0.0]).view(-1))
        t0 = time.monotonic()
        try:
            result = pl._motion_gen.plan_single(pl._start_state(list(np.radians(start))), goal_pose, cfg)
        except Exception as e:  # noqa: BLE001
            last = PlanResult.failure("RAISED", repr(e)[:300]); continue
        res = pl._finish(result, t0)
        res.status = "%s (held rotation axes %s)" % (res.status, w)
        if res.success:
            return res
        last = res
    return last


if __name__ == "__main__":
    raise SystemExit(main())
