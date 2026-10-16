#!/usr/bin/env python3
"""The strategy search: a cross-entropy method over the stack's own tunables, scored on GPU food physics.

    python3 sim/isaac/search.py init DIR [--pop 12 --scenarios 8]      # generation 0 -> DIR/g000.json
    python3 sim/isaac/search.py score DIR 0                             # DIR/g000.out/status.json + results -> g001.json
    python3 sim/isaac/search.py curves DIR                              # DIR/curves.png, DIR/history.json

A strategy is a set of config keys (the PARAMS below: the overlook height, free-space speed, the plunge, the scoop, the
swing's lift and speed, the approach speed); its bites are run by the stack itself (sim/isaac/generate.py: site_check
with cuRobo) on randomized scenarios (where the bowl is, the table's height, where the user's mouth is), then each bite
is replayed with randomized food (sim/isaac/feed_replay.py). Hard constraints: the stack must not refuse, the utensil
must stop at least SAFE_STOP_M from the mouth. Score = the share of bites delivered to the mouth (picked and still on
the utensil there) - TIME_WEIGHT x the bite's duration (s): 1 s is worth 1 % of reliability. The elite quarter refits
the sampling distribution; the current config's values are always in generation 0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# continuous: key -> (current value, low, high); the free-space speed is not searched: it is at the arm's limits
PARAMS = {
    "workspace.overlook_camera_height_m": (0.45, 0.42, 0.55),
    "workspace.overlook_offset_m.x": (0.0, -0.06, 0.06),
    "workspace.overlook_offset_m.y": (0.0, -0.06, 0.06),
    "acquisition.plunge.fast_mps": (0.050, 0.030, 0.100),
    "acquisition.plunge.slow_mps": (0.008, 0.004, 0.020),
    "acquisition.plunge.slow_above_food_m": (0.020, 0.008, 0.030),
    "acquisition.scoop.speed_dps": (55.0, 25.0, 80.0),
    "acquisition.scoop.rot_exp": (0.8, 0.5, 1.4),
    "acquisition.scoop.backoff_from": (0.2, 0.0, 0.6),
    "delivery.lift_m": (0.175, 0.10, 0.22),
    "delivery.approach_mps": (0.077, 0.04, 0.10),
    "delivery.retract.back_off_mps": (0.20, 0.10, 0.30),
}
# structural: key -> the options (the first is the current); sampled from a probability per option
CHOICES = {
    "workspace.overlook_seed_delta_deg": [None, [30, 0, -30, 0, 0, 0, 0], [-30, 0, 30, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 40, 0, -40], [0, 0, 0, 0, -40, 0, 40]],
    "delivery.swing_standoff_m": [None, 0.12, 0.16, 0.20, 0.25],
    "delivery.retract.back_off_m": [None, 0.05, 0.08, 0.12],
}
TIME_WEIGHT = 0.01        # 1 s of bite = 1 % of reliability
JERK_WEIGHT = 0.02        # per m/s^3 of the utensil tip's RMS jerk while carrying (smoothness)
SAFE_STOP_M = 0.07        # the utensil must stop at least this short of the mouth
ELITE = 0.25


def to_set(vals: dict, picks: dict) -> dict:
    """A strategy's config keys (site_check's `set`) from its continuous values and its option picks."""
    out = {k: round(v, 4) for k, v in vals.items() if not k.startswith("workspace.overlook_offset_m")}
    out["workspace.overlook_offset_m"] = [round(vals["workspace.overlook_offset_m.x"], 3),
                                          round(vals["workspace.overlook_offset_m.y"], 3)]
    for k, i in picks.items():
        out[k] = CHOICES[k][i]
    return out


def scenarios(rng, n: int) -> list[dict]:
    """Where the bowl is (the workable region), the table's height, the user's mouth (around the site's)."""
    out = []
    for k in range(n):
        tz = float(rng.uniform(-0.040, -0.024))      # the site's table (-0.024) or lower: feeding_start is captured for it
        out.append(dict(id="c%02d" % k,
                        workspace=dict(bowl_xy=[round(float(rng.uniform(0.30, 0.48)), 3), round(float(rng.uniform(-0.03, 0.35)), 3)],
                                       table_z=round(tz, 3), bowl_top_z=round(tz + 0.07, 3)),
                        mouth_m=[round(-0.277 + float(rng.normal(0, 0.015)), 3), round(0.306 + float(rng.normal(0, 0.02)), 3),
                                 round(0.397 + float(rng.normal(0, 0.015)), 3)]))
    return out


def sample(rng, mu: dict, sigma: dict, probs: dict, n: int, keep_current: bool) -> list[dict]:
    out = []
    for k in range(n):
        if keep_current and k == 0:
            vals, picks = {key: v[0] for key, v in PARAMS.items()}, {key: 0 for key in CHOICES}
        else:
            vals = {key: float(np.clip(rng.normal(mu[key], sigma[key]), PARAMS[key][1], PARAMS[key][2])) for key in PARAMS}
            picks = {key: int(rng.choice(len(CHOICES[key]), p=probs[key])) for key in CHOICES}
        out.append(dict(id="s%02d" % k, vals=vals, picks=picks, set=to_set(vals, picks)))
    return out


def init(d: Path, pop: int, n_sc: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mu = {k: v[0] for k, v in PARAMS.items()}
    sigma = {k: (v[2] - v[1]) / 4 for k, v in PARAMS.items()}
    probs = {k: [1.0 / len(v)] * len(v) for k, v in CHOICES.items()}
    d.mkdir(parents=True, exist_ok=True)
    g = dict(gen=0, mu=mu, sigma=sigma, probs=probs, strategies=sample(rng, mu, sigma, probs, pop, True),
             scenarios=scenarios(rng, n_sc), seed=seed)
    (d / "g000.json").write_text(json.dumps(g, indent=1))
    print("g000: %d strategies x %d scenarios = %d bites" % (pop, n_sc, pop * n_sc))


def score(d: Path, gen: int) -> None:
    """Score generation gen (its bites' status, its batch's metrics, the replay's results), refit, write gen + 1."""
    g = json.loads((d / ("g%03d.json" % gen)).read_text())
    out = d / ("g%03d.out" % gen)
    status = json.loads((out / "status.json").read_text())
    names = [Path(str(x)).stem for x in np.load(out / "batch.npz")["names"]]       # <strategy>__<scenario>
    met = {m["name"]: m for m in json.loads((out / "batch.metrics.json").read_text())}
    res = [np.load(f) for f in sorted(out.glob("results_*.npz"))]
    cand = np.concatenate([r["candidate"] for r in res]); at = np.concatenate([r["at_mouth"] for r in res])
    picked = np.concatenate([r["picked"] for r in res]); stop = np.concatenate([r["min_tip_to_mouth_m"] for r in res])
    rows = []
    for st in g["strategies"]:
        sid = st["id"]
        runs = [k for k, v in status.items() if v["strategy"] == sid]
        ok = [k for k in runs if status[k]["ok"] and k in names]
        idx = [names.index(k) for k in ok]
        m = np.isin(cand, idx)
        # a refused bite delivers nothing: the delivery rate is over every scenario, each with its food draws
        delivered = float(at[m].sum()) / max(1, m.sum() // max(1, len(idx)) * len(runs)) if m.any() else 0.0
        mm = [met[k] for k in ok]
        mean = lambda key, dflt: float(np.mean([x[key] for x in mm])) if mm else dflt  # noqa: E731
        r = dict(id=sid, set=st["set"], vals=st["vals"], picks=st["picks"], bites=len(runs), refused=len(runs) - len(ok),
                 delivered=delivered, picked=float(picked[m].mean()) if m.any() else 0.0,
                 time_s=mean("duration_s", 99.0), jerk_rms=mean("jerk_rms", 99.0),
                 carry_tilt_deg=mean("carry_tilt_deg", 90.0), path_m=mean("path_m", 9.0),
                 min_stop_m=float(stop[m].min()) if m.any() else None,
                 why=sorted({status[k]["why"] for k in runs if not status[k]["ok"]})[:3])
        safe = r["min_stop_m"] is None or r["min_stop_m"] >= SAFE_STOP_M
        r["score"] = (delivered - TIME_WEIGHT * r["time_s"] - JERK_WEIGHT * r["jerk_rms"]) if safe and mm else -1.0
        rows.append(r)
    rows.sort(key=lambda r: -r["score"])
    hist_f = d / "history.json"
    hist = json.loads(hist_f.read_text()) if hist_f.exists() else []
    hist = [h for h in hist if h["gen"] != gen] + [dict(gen=gen, strategies=rows, arms=int(len(cand)))]
    hist_f.write_text(json.dumps(hist, indent=1))
    print("generation %d: %d arms, %d strategies" % (gen, len(cand), len(rows)))
    for r in rows[:5]:
        print("  %s score %+.3f  delivered %5.1f%%  bite %4.1f s  jerk %4.1f  tilt %3.0f deg  refused %d/%d  closest %.3f m" % (
            r["id"], r["score"], 100 * r["delivered"], r["time_s"], r["jerk_rms"], r["carry_tilt_deg"], r["refused"],
            r["bites"], r["min_stop_m"] or -1))
    # refit on the elite quarter: a normal per continuous key, a probability per option (smoothed), the best kept
    rng = np.random.default_rng(g["seed"] + gen + 1)
    elite = [r for r in rows[:max(2, int(math.ceil(ELITE * len(rows))))] if r["score"] > -1] or rows[:2]
    mu = {k: float(np.mean([e["vals"][k] for e in elite])) for k in PARAMS}
    sigma = {k: max(float(np.std([e["vals"][k] for e in elite])), (PARAMS[k][2] - PARAMS[k][1]) / 40) for k in PARAMS}
    probs = {}
    for k, opts in CHOICES.items():
        c = np.bincount([e["picks"][k] for e in elite], minlength=len(opts)) + 0.5
        probs[k] = list(0.7 * c / c.sum() + 0.3 * np.array(g["probs"][k]))
    best = dict(rows[0]); best["id"] = "s00"
    nxt = dict(gen=gen + 1, mu=mu, sigma=sigma, probs=probs, seed=g["seed"], scenarios=scenarios(rng, len(g["scenarios"])),
               strategies=[dict(id="s00", vals=best["vals"], picks=best["picks"], set=best["set"])]
               + sample(rng, mu, sigma, probs, len(rows) - 1, False))
    for k, s in enumerate(nxt["strategies"]):
        s["id"] = "s%02d" % k
    (d / ("g%03d.json" % (gen + 1))).write_text(json.dumps(nxt, indent=1))
    curves(d)


def curves(d: Path) -> None:
    """The learning curves: best and mean score, delivery rate and bite time per generation."""
    from PIL import Image, ImageDraw
    hist = json.loads((d / "history.json").read_text())
    W, H, pad = 1200, 700, 70
    im = Image.new("RGB", (W, H), (12, 14, 20)); dr = ImageDraw.Draw(im)
    gens = [h["gen"] for h in hist]
    series = {"best delivered %": ([100 * h["strategies"][0]["delivered"] for h in hist], (52, 211, 153)),
              "mean delivered %": ([100 * np.mean([s["delivered"] for s in h["strategies"]]) for h in hist], (56, 189, 248)),
              "best bite time s": ([h["strategies"][0]["time_s"] for h in hist], (251, 191, 36)),
              "best tip jerk m/s3": ([h["strategies"][0]["jerk_rms"] for h in hist], (244, 114, 182))}
    x = lambda g: pad + (W - 2 * pad) * (g / max(1, max(gens)))  # noqa: E731
    for name, (ys, col) in series.items():
        lo, hi = (0, 100) if "%" in name else (min(ys) - 2, max(ys) + 2)
        y = lambda v: H - pad - (H - 2 * pad) * (v - lo) / max(1e-6, hi - lo)  # noqa: E731
        pts = [(x(g), y(v)) for g, v in zip(gens, ys)]
        if len(pts) > 1:
            dr.line(pts, fill=col, width=3)
        for p in pts:
            dr.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=col)
    dr.text((pad, 20), "RAMMP strategy search: per generation (%d arms in the last)" % hist[-1]["arms"], fill=(230, 230, 240))
    for k, (name, (_, col)) in enumerate(series.items()):
        dr.text((pad + 270 * k, H - 40), name, fill=col)
    im.save(d / "curves.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "score", "curves"]); ap.add_argument("dir"); ap.add_argument("gen", nargs="?", type=int)
    ap.add_argument("--pop", type=int, default=12); ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.dir)
    if a.cmd == "init":
        init(d, a.pop, a.scenarios, a.seed)
    elif a.cmd == "score":
        score(d, a.gen)
    else:
        curves(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
