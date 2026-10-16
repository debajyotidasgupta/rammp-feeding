#!/usr/bin/env python3
"""site_check dumps -> one replay batch for sim/isaac/feed_replay.py (plain numpy: runs anywhere).

    python3 sim/isaac/make_batch.py out.npz dump1.json [dump2.json ...] [--copies N] [--seed 0]

Each dump (tests/site_check.py --dump) is one candidate's whole bite as the stack runs it: the joint path (every
executed move, real kinematics) and the scenario (workspace, mouth, utensil tip). Every dump is replicated --copies
times; each copy draws its own food (where it really is against where the wrist camera put it, its size, mass,
friction), so a batch row is (candidate, food draw). The stage boundaries (the pick, the swing, the approach) come
from the dump's step labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DT = 1.0 / 120.0          # the replay's control rate: the joint path sampled at this step
ROOT = Path(__file__).resolve().parent.parent.parent


def tool_frame(q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Kortex tool frame (flange + 0.12 m) at joints q: MuJoCo's Gen3 (tests/mujoco_kortex.py's model)."""
    import mujoco
    if not hasattr(tool_frame, "m"):
        tool_frame.m = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "assets" / "kinova_gen3" / "gen3.xml"))
        tool_frame.d = mujoco.MjData(tool_frame.m)
        tool_frame.s = mujoco.mj_name2id(tool_frame.m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
    m, d = tool_frame.m, tool_frame.d
    d.qpos[:7] = q_rad
    mujoco.mj_kinematics(m, d)
    R = d.site_xmat[tool_frame.s].reshape(3, 3).copy()
    return d.site_xpos[tool_frame.s] + R[:, 2] * 0.12, R


def fold(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def sample(dump: dict, gap_s: float = 0.4) -> tuple[np.ndarray, dict]:
    """The bite as joints (rad) every DT, and the time (s) each stage starts / ends."""
    t, q, marks = [0.0], [fold(dump["motions"][0]["start"])], {}
    for m in dump["motions"]:
        step = m["step"].split(" --")[0]
        cur = q[-1] + fold(np.asarray(m["start"], float) - q[-1])
        t.append(t[-1]); q.append(cur)
        marks.setdefault(step + ":start", t[-1])
        for p, dt in m["points"]:
            cur = cur + fold(np.asarray(p, float) - cur)
            t.append(t[-1] + float(dt)); q.append(cur)
        marks[step + ":end"] = t[-1]
        t.append(t[-1] + gap_s); q.append(cur)
    t, q = np.array(t), np.array(q)
    keep = np.r_[True, np.diff(t) > 1e-9]                  # pchip needs a strictly rising clock
    t, q = t[keep], q[keep]
    ts = np.arange(0.0, t[-1] + 1e-9, DT)
    from scipy.interpolate import PchipInterpolator       # no overshoot between waypoints, smooth like the controller's
    qs = PchipInterpolator(t, q, axis=0)(ts)
    return np.radians(qs).astype(np.float32), marks


def metrics(q: np.ndarray, ev, tip_t: np.ndarray, n_tool: np.ndarray) -> dict:
    """The bite's smoothness and how level it carries: the utensil tip's RMS jerk (m/s^3) and peak acceleration
    (m/s^2) while carrying (pick end -> approach end), the worst tilt of its food side off straight up (deg) then, and
    the tip's whole path length (m). From the joints at 60 Hz (the 120 Hz path, every second step)."""
    step = 2
    tips, tilt = [], []
    for k in range(0, len(q), step):
        p, R = tool_frame(q[k])
        tips.append(p + R @ tip_t)
        tilt.append(np.degrees(np.arccos(np.clip((R @ n_tool)[2], -1, 1))))
    tips, tilt, h = np.array(tips), np.array(tilt), DT * step
    a0, a1 = ev[1] // step, max(ev[1] // step + 4, ev[2] // step)
    acc = np.diff(tips, 2, axis=0) / h ** 2
    jerk = np.diff(tips, 3, axis=0) / h ** 3
    carry = slice(a0, a1)
    return dict(jerk_rms=float(np.sqrt((np.linalg.norm(jerk[carry], axis=1) ** 2).mean())),
                acc_peak=float(np.linalg.norm(acc[carry], axis=1).max()),
                carry_tilt_deg=float(tilt[a0:a1].max()),
                path_m=float(np.linalg.norm(np.diff(tips, axis=0), axis=1).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("dumps", nargs="+")
    ap.add_argument("--copies", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aim-sigma-m", type=float, default=0.004, help="the wrist camera's aim error, per axis")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for k, path in enumerate(a.dumps):
        d = json.loads(Path(path).read_text())
        q, marks = sample(d)
        rows.append((k, path, d, q, marks))
    T = max(len(r[3]) for r in rows)
    E = len(rows) * a.copies
    Q = np.zeros((len(rows), T, 7), np.float32)            # one path per candidate; row -> candidate below
    L = np.zeros(E, np.int32)
    ev = np.zeros((E, 4), np.int32)          # pick start, pick end (the swing starts), approach end, bite end
    cand = np.zeros(E, np.int32)
    bowl = np.zeros((E, 3), np.float32); table_z = np.zeros(E, np.float32); mouth = np.zeros((E, 3), np.float32)
    food_pos = np.zeros((E, 3), np.float32); food_size = np.zeros(E, np.float32)
    mass = np.zeros(E, np.float32); mu = np.zeros(E, np.float32); tip = np.zeros((E, 3), np.float32)
    normal = np.zeros((E, 3), np.float32); tool0 = np.zeros((E, 12), np.float32)
    per = []                                  # per candidate: duration and metrics()
    i = 0
    for k, path, d, q, marks in rows:
        ws = d["workspace"]
        idx = lambda key, default=0: int(round(marks.get(key, default) / DT))  # noqa: E731
        _, R_end = tool_frame(q[min(idx("acquire pick:end"), len(q) - 1)])
        n_tool = R_end.T @ np.array([0.0, 0.0, 1.0])          # the utensil's food side: up when the scoop ends
        p0, R0 = tool_frame(q[0])
        Q[k, :len(q)] = q; Q[k, len(q):] = q[-1]
        ev_k = [idx("acquire pick:start"), idx("acquire pick:end"), idx("deliver approach:end"),
                idx("deliver bite:end", marks.get("deliver approach:end", 0) + 2.0)]
        per.append(dict(name=Path(path).stem, duration_s=len(q) * DT, **metrics(q, ev_k, np.asarray(d["tip_in_tool"]), n_tool)))
        print("  %-40s %5.1f s  jerk %.1f m/s^3  carry tilt %.0f deg" % (per[-1]["name"], per[-1]["duration_s"],
                                                                       per[-1]["jerk_rms"], per[-1]["carry_tilt_deg"]), flush=True)
        for _ in range(a.copies):
            L[i] = len(q)
            ev[i] = ev_k
            cand[i] = k
            bowl[i] = [ws["bowl_xy"][0], ws["bowl_xy"][1], ws["bowl_top_z"]]
            table_z[i] = ws["table_z"]
            mouth[i] = d["user"]["mouth_m"]
            s = rng.uniform(0.018, 0.028)                      # a cut cube, 18-28 mm
            food_size[i] = s
            food_pos[i] = [ws["bowl_xy"][0] + rng.normal(0, a.aim_sigma_m), ws["bowl_xy"][1] + rng.normal(0, a.aim_sigma_m),
                           ws["table_z"] + 0.008 + s / 2]
            mass[i] = 1000.0 * s ** 3 * rng.uniform(0.85, 1.1)   # fruit: water's density, give or take
            mu[i] = rng.uniform(0.3, 0.9)                       # wet fruit to dry cheese
            tip[i] = d["tip_in_tool"]
            normal[i] = n_tool
            tool0[i] = np.concatenate([p0, R0.flatten()])
            i += 1
    np.savez_compressed(a.out, q=Q, length=L, events=ev, candidate=cand, bowl=bowl, table_z=table_z, mouth=mouth,
                        food_pos=food_pos, food_size=food_size, mass=mass, mu=mu, tip_in_tool=tip, dt=DT,
                        tray_normal_tool=normal, tool_at_start=tool0,
                        names=np.array([str(r[1]) for r in rows]))
    Path(a.out).with_suffix(".metrics.json").write_text(json.dumps(per, indent=1))
    print("%s: %d rows (%d candidates x %d food draws), %d steps (%.1f s)" % (a.out, E, len(rows), a.copies, T, T * DT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
