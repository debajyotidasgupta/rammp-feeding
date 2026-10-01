#!/usr/bin/env python3
"""Three planners on the same free-space moves: our scripted moves (until 2026-10-01), cuRobo (the stack's now),
and BLAST (Dynamium Lab's B-spline trajectory optimizer, the Kinova team's ask).

    python3 tests/planner_compare.py scripted.json curobo.json [--out compare.json]
        # in cornell/blast-eval (docker/Dockerfile.blast_eval) on a CUDA box: cuRobo judges collisions

The two dumps are tests/site_check.py --dump recordings of the same site's bite (scripted: commit 41d4a88, the last
with the scripted planner; curobo: now). The bite's free-space legs -- to food_detect_start, the swing to the user
with the piece, the return to food_detect_start (after the scripted back-off, which every planner shares), home --
are taken from the cuRobo recording (start and goal joints); BLAST plans each leg between the same joints, in the
same world (kinova/curobo_world.py: the table and bowl grown by the scene camera's error, the user, the wheelchair,
the pedestal), with the same utensil (and piece) as capsules on the bracelet. Every leg of every planner is then
judged alike: MuJoCo's Gen3 for the utensil tip (its peak above the table, its least clearance over the TRUE table /
rim, its tilt), cuRobo's collision check in the TRUE world (no margin), and the time the arm would take.

BLAST's Gen3 model (blast/manipulator/Kinova_Gen3.hpp @ fd12217) has the joint-4 link offset's lateral 6.4 mm on the
wrong side (0, -0.0064, -0.2104 against the URDF's 0, +0.006375, -0.21038): 13 mm at the flange. This uses the URDF's
offsets (checked against MuJoCo to 0.05 mm) and the arm's soft limits (61 deg/s, 45 deg/s^2 as the controller
measures it), so BLAST's time-optimal duration is directly the arm's.
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
SITE = ROOT / "tests" / "sites" / "site_20260930.yaml"
RUNS = Path(tempfile.mkdtemp(prefix="planner_compare_"))
os.environ.update(RAMMP_NO_LOCAL="1", RAMMP_CAMERA="realsense_d405", RAMMP_SCENE_CAMERA="orbbec_gemini336l",
                  RAMMP_SET="deployment.runs_dir=%s,workspace.scoop_end_follows=table" % RUNS)

import yaml  # noqa: E402

from settings import cfg  # noqa: E402
import fake_kortex as fk  # noqa: E402
import mujoco_kortex as mk  # noqa: E402

LEGS = [("to food_detect_start", "sequence detect", False), ("swing to the user (piece)", "deliver swing", True),
        ("back to food_detect_start", "deliver retract", False), ("home", "sequence home", False)]


def fold(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def legs_of(dump: dict) -> dict:
    """{leg: [(start, [[q, dt], ...]), ...]}: each leg's moves; the retract without its back-off (shared by all)."""
    out = {}
    for name, step, _ in LEGS:
        ms = [m for m in dump["motions"] if m["step"].startswith(step)]
        if step == "deliver retract":
            ms = ms[1:]
        out[name] = [(m["start"], m["points"]) for m in ms]
    return out


def joint_path(moves) -> tuple[np.ndarray, float]:
    """The leg's waypoints (unwrapped, deg) and its executed duration."""
    q, T = [], 0.0
    for start, pts in moves:
        cur = np.asarray(start, float) if not q else q[-1]
        if not q:
            q.append(cur)
        for p, dt in pts:
            cur = cur + fold(np.asarray(p, float) - cur)
            q.append(cur); T += float(dt)
    return np.array(q), T


# ------------------------------------------------------------------------------------------------ BLAST
def blast_gen3(blast, tip_tool, carrying: bool):
    """The Gen3 (URDF link offsets) at the arm's soft limits, with the 2F-85 and the utensil (and the piece) as
    capsules on the bracelet (frame 6 = bracelet_link, checked against MuJoCo)."""
    from kinova.curobo_world import spoon_spheres
    V, M = blast.Vec3, blast.Mat3
    lim = blast.ManipulatorLimits()
    jl = [math.radians(float(cfg.motion.joint_limits_deg[k]) - float(cfg.motion.plan_min_margin_deg)) for k in (2, 4, 6)]
    big = 50.0
    lim.position_max = blast.array([big, jl[0], big, jl[1], big, jl[2], big])
    lim.position_min = blast.array([-big, -jl[0], -big, -jl[1], -big, -jl[2], -big])
    v = math.radians(float(cfg.motion.free_space.dps))
    a = math.radians(float(cfg.motion.joint_accel_cap_dps2) / float(cfg.motion.controller_accel_factor))
    lim.velocity_max = blast.array([v] * 7)
    lim.acceleration_max = blast.array([a] * 7)
    lim.torque_max = blast.array([52.0, 52.0, 52.0, 52.0, 17.0, 17.0, 17.0])
    lim.tool_speed_max = 1.0
    kin = blast.ManipulatorKinematics()
    kin.joint_offsets = [V(0, 0.005375, -0.12838), V(0, -0.21038, -0.006375), V(0, 0.006375, -0.21038),
                         V(0, -0.20843, -0.006375), V(0, 0.00017505, -0.10593), V(0, -0.10593, -0.00017505),
                         V(0, 0, -0.0615)]
    kin.joint_axes = [V(0, 0, 1)] * 7
    rots = [M(1, 0, 0, 0, -1, 0, 0, 0, -1), M(1, 0, 0, 0, 0, 1, 0, -1, 0), M(1, 0, 0, 0, 0, -1, 0, 1, 0),
            M(1, 0, 0, 0, 0, 1, 0, -1, 0), M(1, 0, 0, 0, 0, -1, 0, 1, 0), M(1, 0, 0, 0, 0, 1, 0, -1, 0),
            M(1, 0, 0, 0, 0, -1, 0, 1, 0)]
    kin.static_rotations = rots + [M()] * (blast.MAX_JOINTS + 1 - len(rots))
    kin.first_joint_position = V(0, 0, 0.1564)
    dyn = blast.ManipulatorDynamics()
    dyn.link_masses = blast.array([1.377, 1.1636, 1.1636, 0.93, 0.678, 0.678, 0.364] + [0.0] * (blast.MAX_JOINTS - 7))
    I = [(0.004570, 0.000001, 0.000002, 0.000001, 0.004831, 0.000448, 0.000002, 0.000448, 0.001409),
         (0.011088, 0.000005, 0.0, 0.000005, 0.001072, -0.000691, 0.0, -0.000691, 0.011255),
         (0.010932, 0.0, -0.000007, 0.0, 0.011127, 0.000606, -0.000007, 0.000606, 0.001043),
         (0.008147, -0.000001, 0.0, -0.000001, 0.000631, -0.000500, 0.0, -0.000500, 0.008316),
         (0.001596, 0.0, 0.0, 0.0, 0.001607, 0.000256, 0.0, 0.000256, 0.000399),
         (0.001641, 0.0, 0.0, 0.0, 0.000410, -0.000278, 0.0, -0.000278, 0.001641),
         (0.000214, 0.0, 0.000001, 0.0, 0.000223, -0.000002, 0.000001, -0.000002, 0.000240)]
    dyn.inertia_tensors = [M(*t) for t in I] + [M()] * (blast.MAX_JOINTS - 7)
    dyn.cog_offsets = [V(-0.000023, -0.010364, -0.073360), V(-0.000044, -0.099580, -0.013278),
                       V(-0.000044, -0.006641, -0.117892), V(-0.000018, -0.075478, -0.015006),
                       V(0.000001, -0.009432, -0.063883), V(0.000001, -0.045483, -0.009650),
                       V(-0.000093, 0.000132, -0.022905)] + [V(0, 0, 0)] * (blast.MAX_JOINTS - 7)
    caps = blast.ManipulatorCapsules()
    base = blast.Sphere(); base.center = V(0, 0, 0.1214); base.radius = 0.14
    caps.base_sphere = base

    def cap(frame, p1, p2, r):
        c = blast.CollisionModelCapsule(); c.joint_frame = frame; c.p1 = V(*p1); c.p2 = V(*p2); c.radius = r
        return c
    lst = [cap(1, (0, 0.035, 0), (0, -0.425, 0), 0.06), cap(3, (0, 0, -0.025), (0, -0.3, -0.01), 0.06),
           cap(5, (0, 0, -0.015), (0, -0.15, -0.015), 0.055),
           cap(6, (0, 0, -0.075), (0, 0, -0.20), 0.045)]                     # the 2F-85, flange to fingertips
    sp = spoon_spheres(tip_tool, carrying)                                 # bracelet frame, as cuRobo's
    # BLAST is compiled for at most MAX_CAPSULES (7): the utensil as two -- its shaft and its head
    lst.append(cap(6, sp[0]["center"], sp[3]["center"], sp[0]["radius"]))
    lst.append(cap(6, sp[3]["center"], sp[4]["center"], sp[4]["radius"]))
    if carrying:
        lst.append(cap(6, sp[5]["center"], sp[5]["center"], sp[5]["radius"]))
    assert len(lst) <= blast.MAX_CAPSULES, "BLAST is compiled for %d capsules" % blast.MAX_CAPSULES
    caps.capsule_list = lst
    n = len(lst)
    tool = range(3, 6)                                                     # the gripper and the utensil
    caps.collision_base = blast.array([0, 0, 1] + [1] * 3 + [0] * (n - 6))
    cm = np.zeros((n, n), dtype=np.uint8)
    cm[2, 0] = cm[0, 2] = 1
    for i in tool:                                                         # the tool against the upper arm; the
        cm[i, 0] = cm[0, i] = 1                                            # piece at the tip only against the world
    caps.collision_matrix = cm
    return blast.Manipulator(7, lim, kin, dyn, caps)


def blast_world(blast, obstacles):
    from scipy.spatial.transform import Rotation
    w = blast.World()
    for o in obstacles:
        R = Rotation.from_euler("xyz", o.get("rpy_deg", [0, 0, 0]), degrees=True).as_matrix()
        w.add_box(blast.Vec3(*o["position"]), blast.Vec3(*(np.asarray(o["dims"]) / 2)), blast.Mat3(*R.T.flatten()))
    return w


def blast_plan(blast, robot, world, start_deg, goal_deg, attempts: int = 5):
    s = np.radians(np.asarray(start_deg, float))
    g = s + np.radians(fold(np.asarray(goal_deg, float) - np.asarray(start_deg, float)))
    opt = blast.Optimization(robot, blast.Task.stop_to_stop(blast.array(s), blast.array(g)))
    opt.bspline = blast.Bspline(16, 110, 5, 7)
    opt.world = world
    c = opt.constraints
    c.position = c.velocity = c.acceleration = True
    c.self_collisions = c.external_collisions = True
    opt.success_tolerance = 0.01
    opt.guess.type = blast.GuessType.random
    opt.guess.n_random_shots = 50
    t0, res = time.perf_counter(), None
    for k in range(attempts):
        res = blast.optimize(opt)
        if res.success:
            break
    ms = 1000 * (time.perf_counter() - t0)
    if res is None or not res.success:
        return None, ms, dict(tries=k + 1, max_constraint=float(getattr(res, "max_constraint_value", float("nan"))))
    tr = res.trajectory
    q = np.degrees(np.asarray(tr.pos).T)
    t = np.asarray(tr.t)
    return (q, float(t[-1] - t[0])), ms, dict(tries=k + 1)


# ------------------------------------------------------------------------------------------------ judging
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scripted"); ap.add_argument("curobo")
    ap.add_argument("--out", default=str(RUNS / "compare.json"))
    ap.add_argument("--repeats", type=int, default=10, help="plans per leg for cuRobo and BLAST: success rate, time")
    ap.add_argument("--dump-blast", help="write BLAST's bite as a site_check dump (its legs in the cuRobo bite's place)")
    a = ap.parse_args()
    fk.install_clock()
    import blast
    from kinova.curobo_world import world
    from kinova.frames import R_from_kortex
    from kinova.motion import _time_scale, tip_in_tool
    from kinova.planner_server import Planners
    site = yaml.safe_load(SITE.read_text())
    ws = site["workspace"]
    truth = dict(bowl_xy=ws["bowl_xy"], table_z=ws["table_z"], bowl_top_z=ws["bowl_top_z"], rim_radius_m=0.054)
    err = float(cfg.scene_camera.extrinsic.error_m)
    (RUNS / "scene_state.json").write_text(json.dumps(dict(truth, t=time.time(), error_m=err)))
    mouth = site["user"]["mouth_m"]
    ext = dict(cfg.workspace.table_extent_m)
    planning = world(dict(truth, bowl_radius=0.054, error_m=err), mouth, table_extent=ext)["obstacles"]
    true_w = world(dict(truth, bowl_radius=0.054, error_m=0.0), mouth, table_extent=ext)["obstacles"]
    arm = mk.KinematicArm(tool_z=float(cfg.deployment.arm.tool.transform_m[2]))
    P = Planners()
    tip_t = tip_in_tool()
    dumps = {"scripted (until 2026-10-01)": legs_of(json.loads(Path(a.scripted).read_text())),
             "cuRobo (now)": legs_of(json.loads(Path(a.curobo).read_text()))}
    vmax = float(cfg.motion.free_space.dps)
    amax = float(cfg.motion.joint_accel_cap_dps2) / float(cfg.motion.controller_accel_factor)

    def tip(q):
        p = arm.fk_pose(list(q))
        R = R_from_kortex(*p[3:])
        return np.array(p[:3]) + R @ tip_t, math.degrees(math.acos(max(-1.0, min(1.0, -R[2, 2]))))

    def clearance(t) -> float:
        bx, by = truth["bowl_xy"]
        if math.hypot(t[0] - bx, t[1] - by) < truth["rim_radius_m"] + 0.02:
            return t[2] - truth["bowl_top_z"]
        if ext["x"][0] <= t[0] <= ext["x"][1] and ext["y"][0] <= t[1] <= ext["y"][1]:
            return t[2] - truth["table_z"]
        return math.inf

    def judge(q: np.ndarray, carrying: bool) -> dict:
        dense = np.concatenate([q[i] + (q[i + 1] - q[i]) * s[:, None] for i in range(len(q) - 1)
                                for s in [np.linspace(0, 1, 5, endpoint=False)]] + [q[-1:]])
        pl = P.by_load[bool(carrying)]
        pl.update_world(true_w)
        bad = sum(not pl.check_state_valid(list(np.radians(x)))[0] for x in dense[::2])
        tt = [tip(x) for x in dense[::2]]
        travel = float(np.abs(np.diff(q, axis=0)).max(1).sum())
        return dict(collisions=int(bad), states=len(dense[::2]), travel_deg=travel,
                    peak_tip_above_table_m=float(max(t[0][2] for t in tt) - truth["table_z"]),
                    min_tip_clearance_m=float(min(clearance(t[0]) for t in tt)),
                    tilt_deg=[float(min(t[1] for t in tt)), float(max(t[1] for t in tt))])

    rows, reps, blast_paths = [], [], {}
    cur_legs = dumps["cuRobo (now)"]
    for leg, _, carrying in LEGS:
        q_cu, _ = joint_path(cur_legs[leg])
        start, goal = q_cu[0], q_cu[-1]
        for name, legs in dumps.items():
            q, T = joint_path(legs[leg])
            row = dict(leg=leg, planner=name, moves=len(legs[leg]), duration_s=T, **judge(q, carrying))
            if name.startswith("cuRobo"):
                P.refresh(); t0 = time.perf_counter()
                P.plan(list(start), list(start + fold(goal - start)), carrying)
                row["planning_ms"] = 1000 * (time.perf_counter() - t0)       # wall clock (time.time is simulated)
            rows.append(row)
        robot = blast_gen3(blast, tip_t, carrying)
        res, ms, info = blast_plan(blast, robot, blast_world(blast, planning), start, goal)
        if res is None:
            rows.append(dict(leg=leg, planner="BLAST", outcome="no trajectory", planning_ms=ms, **info))
        else:
            q, T = res
            q_thin = q[:: max(1, len(q) // 120)]
            blast_paths[leg] = (q_thin, T)
            T_ours = float(sum(_time_scale(q_thin, vmax, amax)))
            rows.append(dict(leg=leg, planner="BLAST", moves=1, duration_s=T, duration_retimed_s=T_ours,
                             planning_ms=ms, **info, **judge(q_thin, carrying)))
        robot_r = blast_gen3(blast, tip_t, carrying)
        world_r = blast_world(blast, planning)
        for k in range(a.repeats):                    # reliability: every plan judged by success and wall time
            r0 = blast_plan(blast, robot_r, world_r, start, goal, attempts=1)
            P.refresh(); t0 = time.perf_counter()
            rc = P.plan(list(start), list(start + fold(goal - start)), carrying)
            reps.append(dict(leg=leg, blast_ok=r0[0] is not None, blast_ms=r0[1], curobo_ok=bool(rc.get("ok")),
                             curobo_ms=1000 * (time.perf_counter() - t0)))
        print("  %s done" % leg, flush=True)
    Path(a.out).write_text(json.dumps(dict(legs=rows, repeats=reps), indent=1))
    table(rows)
    print("\nreliability, %d plans per leg (one attempt each):" % a.repeats)
    for leg, _, _ in LEGS:
        rr = [r for r in reps if r["leg"] == leg]
        if not rr:
            continue
        bm = [r["blast_ms"] for r in rr]; cm = [r["curobo_ms"] for r in rr]
        print("  %-27s cuRobo %2d/%d ok, %4.0f ms median   BLAST %2d/%d ok, %5.0f ms median (%5.0f-%5.0f)" % (
            leg, sum(r["curobo_ok"] for r in rr), len(rr), np.median(cm), sum(r["blast_ok"] for r in rr), len(rr),
            np.median(bm), min(bm), max(bm)))
    if a.dump_blast:
        d = json.loads(Path(a.curobo).read_text())
        motions, done = [], set()
        for m in d["motions"]:
            leg = next((n for n, step, _ in LEGS if m["step"].startswith(step)), None)
            if leg in blast_paths and not (leg == "back to food_detect_start" and not any(
                    x["step"].startswith("deliver retract") for x in motions)):
                if leg not in done:
                    q, T = blast_paths[leg]
                    dt = T / (len(q) - 1)
                    motions.append(dict(step=m["step"], start=[float(v) for v in fold(q[0])],
                                        points=[[[float(v) for v in fold(x)], dt] for x in q[1:]]))
                    done.add(leg)
                continue
            motions.append(m)
        d["motions"] = motions
        Path(a.dump_blast).write_text(json.dumps(d, indent=1))
    return 0


def table(rows) -> None:
    print("\n%-27s %-28s %5s %8s %9s %7s %9s %9s %11s %10s" % ("leg", "planner", "moves", "time s", "travel", "collide",
                                                               "tip peak", "tip clear", "tilt deg", "plan ms"))
    for r in rows:
        if r.get("outcome"):
            print("%-27s %-28s %s (%.0f ms, %s)" % (r["leg"], r["planner"], r["outcome"], r["planning_ms"],
                                                     r.get("max_constraint")))
            continue
        print("%-27s %-28s %5d %8.2f %8.0fd %7d %8.3fm %8.3fm %5.1f-%-5.1f %10s" % (
            r["leg"], r["planner"], r["moves"], r["duration_s"], r["travel_deg"], r["collisions"],
            r["peak_tip_above_table_m"], r["min_tip_clearance_m"], r["tilt_deg"][0], r["tilt_deg"][1],
            "%.0f" % r["planning_ms"] if "planning_ms" in r else "-"))
    for p in sorted({r["planner"] for r in rows}):
        rs = [r for r in rows if r["planner"] == p and not r.get("outcome")]
        print("  %-28s total %.1f s, %d legs planned, %d states in collision" % (
            p, sum(r["duration_s"] for r in rs), len(rs), sum(r["collisions"] for r in rs)))


if __name__ == "__main__":
    raise SystemExit(main())
