"""A recorded bite (feeding/lib.sh's run record) as a timeline the twin can replay.

    from sim.record import load_bite
    b = load_bite("~/rammp_runs/20260925_182224_feed", 3)
    b.q_at(12.0)            # the arm's joints (deg) 12 s into the bite
    b.phase_at(12.0)        # ("acquire", "Plunge", "25 mm below the top ...")

Everything comes from the record: the joints from each move's telemetry (arm/*_arm_*.npz: q at ~35 Hz, placed on
the clock by its `telemetry` event), the phases from events.jsonl, the food from perceive/detection.json mapped
to the table exactly as feeding/locate_target.py maps its target, the mouth from the approach's mouth_base, the
open-mouth signal from deliver/look_*/open_wait.json. Between moves the arm is still: the last sample holds.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from settings import cfg
from sim.scene import Food

FX, FY = float(cfg.camera.colour.fx), float(cfg.camera.colour.fy)


@dataclass
class Segment:
    """One recorded move: joint samples on the bite's clock."""
    label: str
    stage: str
    t: np.ndarray
    q: np.ndarray                   # deg, (n, 7)
    force: np.ndarray               # the arm's estimated tool force (n, 3)


@dataclass
class Phase:
    t0: float
    t1: float
    stage: str                      # perceive / acquire / deliver
    title: str
    detail: str = ""


@dataclass
class Bite:
    session: Path
    name: str
    t0: float                       # unix time of the bite's start (perceive's stage_start)
    duration: float
    segments: list[Segment]
    phases: list[Phase]
    foods: list[Food]
    picked: int                     # index into foods
    bowl_xy: tuple[float, float]
    bowl_half: tuple[float, float]
    mouth: tuple[float, float, float]
    events: dict[str, float] = field(default_factory=dict)      # named moments (s): plunged, bite, ...
    open_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    open_v: np.ndarray = field(default_factory=lambda: np.zeros(0))
    images: dict[str, Path] = field(default_factory=dict)
    wait_frames: list[tuple[float, Path]] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def q_at(self, t: float) -> np.ndarray:
        """Joints (deg) at t: interpolated inside a move, held between moves (unwrapped per move)."""
        prev = self.segments[0].q[0]
        for s in self.segments:
            if t < s.t[0]:
                return prev
            if t <= s.t[-1]:
                return np.array([np.interp(t, s.t, s.q[:, k]) for k in range(7)])
            prev = s.q[-1]
        return prev

    def phase_at(self, t: float) -> Phase:
        cur = self.phases[0]
        for p in self.phases:
            if p.t0 <= t:
                cur = p
        return cur

    def opening_at(self, t: float) -> float:
        if not len(self.open_t) or t < self.open_t[0]:
            return 0.0
        return float(np.interp(t, self.open_t, self.open_v))


def _events(paths: list[str], bite: str) -> list[dict]:
    ev = []
    for f in paths:
        if not os.path.exists(f):
            continue
        for line in open(f):
            e = json.loads(line)
            if e.get("bite", bite) == bite:
                ev.append(e)
    return sorted(ev, key=lambda e: e["t"])


def _vivid(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """The wrist camera runs hot (a melon reads (255, 248, 187)): keep the hue, restore saturation and value."""
    import colorsys
    h, s_, v = colorsys.rgb_to_hsv(*rgb)
    if 0.06 < h < 0.2:                  # clipped red pushes orange toward yellow: take back ~10 deg
        h -= 0.028
    return colorsys.hsv_to_rgb(h, min(0.85, max(0.5, s_ * 2.4)), min(0.95, max(0.55, v)))


def _foods(perceive: Path, target: dict) -> tuple[list[Food], int, tuple[float, float], tuple[float, float]]:
    """The detected pieces on the table, mapped as locate_target maps its target (the tip's vertical line p*)."""
    det = json.loads((perceive / "detection.json").read_text())
    ps, tool = target["p_star"], target["tool_now"]
    z = float(target["bowl_floor_mm"]) / 1000.0
    colour = np.asarray(Image.open(perceive / "colour.jpg").convert("RGB"), np.float32) / 255

    def to_base(u: float, v: float, zz: float = z) -> tuple[float, float]:
        return tool[0] + (ps[1] - v) * zz / FY, tool[1] + (ps[0] - u) * zz / FX

    foods = []
    rng = np.random.default_rng(len(det["food"]))
    for f in det["food"]:
        w, h = f["x2"] - f["x1"], f["y2"] - f["y1"]
        cu, cv = int(f["cx"]), int(f["cy"])
        patch = colour[max(0, cv - int(h / 6)):cv + int(h / 6) + 1, max(0, cu - int(w / 6)):cu + int(w / 6) + 1]
        rgb = tuple(float(c) for c in np.median(patch.reshape(-1, 3), axis=0)) if patch.size else (0.9, 0.6, 0.3)
        rgb = _vivid(rgb)
        size = float(np.clip(0.72 * min(w, h) * z / FX, 0.016, 0.032))
        foods.append(Food(to_base(f["cx"], f["cy"]), size, rgb, float(rng.uniform(-0.5, 0.5)),
                          tuple(float(x) for x in rng.uniform(0.82, 1.12, 3))))
    bu, bv = target["berry_bbox_centre"]
    picked = int(np.argmin([math.hypot(f["cx"] - bu, f["cy"] - bv) for f in det["food"]])) if foods else -1
    bowl = (det.get("bowl") or [None])[0]
    if bowl:
        zr = z - 0.055 * 0.6                                   # the box is the rim, nearer than the floor
        bxy = to_base(bowl["cx"], bowl["cy"])
        half = ((bowl["y2"] - bowl["y1"]) * zr / FY / 2, (bowl["x2"] - bowl["x1"]) * zr / FX / 2)
    else:
        bxy, half = tuple(map(float, cfg.workspace.bowl_centre_m)), (0.062, 0.074)
    # keep every piece inside the container's walls
    for f in foods:
        f.xy = (float(np.clip(f.xy[0], bxy[0] - half[0] + f.size * 0.7, bxy[0] + half[0] - f.size * 0.7)),
                float(np.clip(f.xy[1], bxy[1] - half[1] + f.size * 0.7, bxy[1] + half[1] - f.size * 0.7)))
    return foods, picked, bxy, half


def load_bite(session: str | Path, n: int) -> Bite:
    S = Path(os.path.expanduser(str(session)))
    name = "bite_%02d" % n
    B = S / name
    ev = _events([str(S / "events.jsonl")] + sorted(glob.glob(str(B / "*" / "arm" / "events.jsonl"))), name)
    t0 = next(e["t"] for e in ev if e["kind"] == "stage_start")
    t_end = max(e["t"] for e in ev if e["kind"] == "stage_end")
    rel = lambda t: float(t - t0)                                                  # noqa: E731

    segments = []
    for e in ev:
        if e["kind"] != "telemetry":
            continue
        f = B / e["stage"] / "arm" / e["file"]
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        end = rel(e["t"])
        t = np.asarray(z["t"], float)
        t = t - t[-1] + end                                    # the event is written as the move ends
        q = np.unwrap(np.radians(np.asarray(z["q"], float)), axis=0)
        segments.append(Segment(e["label"], e["stage"], t, np.degrees(q), np.asarray(z["force"], float)))
    # keep consecutive moves continuous through the +-180 wrap
    for a, b in zip(segments, segments[1:]):
        k = np.round((a.q[-1] - b.q[0]) / 360.0)
        b.q = b.q + 360.0 * k

    kinds = {}
    for e in ev:
        kinds.setdefault(e["kind"], []).append(e)

    def first(kind: str, stage: str | None = None) -> dict | None:
        return next((e for e in kinds.get(kind, []) if stage is None or e.get("stage") == stage), None)

    def seg(label_start: str) -> Segment | None:
        return next((s for s in segments if s.label.startswith(label_start)), None)

    target = json.loads((B / "perceive" / "target.json").read_text())
    foods, picked, bxy, half = _foods(B / "perceive", target)
    goal = first("approach_goal")
    mouth = tuple(goal["mouth_base"]) if goal else (-0.277, 0.306, 0.635)

    phases: list[Phase] = []

    def add(t0_: float, t1_: float, stage: str, title: str, detail: str = "") -> None:
        phases.append(Phase(t0_, t1_, stage, title, detail))

    p_end = first("stage_end", "perceive")
    add(0.0, rel(p_end["t"]) if p_end else 5.0, "perceive", "Perceive",
        "LocateAnything: %d pieces · depth picks the most central" % len(foods))
    over, plunge, scoop = seg("OVER"), seg("plunge"), seg("scoop")
    a_start = first("stage_start", "acquire")
    if a_start:
        add(rel(a_start["t"]), over.t[0] if over else rel(a_start["t"]) + 1, "acquire", "Aim",
            "the tip's vertical line through the top of the piece")
    if over:
        add(over.t[0], over.t[-1], "acquire", "Over the piece", "10 mm above its top, tines centred")
    pp = first("plunge_plan")
    if plunge:
        mode = (pp or {}).get("mode", "")
        add(over.t[-1] if over else plunge.t[0], plunge.t[-1], "acquire", "Plunge",
            ("%s · %.0f mm down" % (mode, 1000 * pp["down_m"])) if pp else "")
    if scoop:
        add(scoop.t[0], scoop.t[-1], "acquire", "Scoop", "tilt and lift: the piece rides the tines")
    sw = seg("swing to the user") or seg("lift + swing")       # the cuRobo swing; before 2026-10-01 the scripted one
    d_start = first("stage_start", "deliver")
    if sw:
        add(d_start and rel(d_start["t"]) or sw.t[0], sw.t[-1], "deliver", "Swing to the user",
            "to the user, the piece on the utensil")
    appr = seg("approach")
    ow_path = sorted(B.glob("deliver/look_*/open_wait.json"))
    open_t = open_v = np.zeros(0)
    wait_frames: list[tuple[float, Path]] = []
    if appr and sw:
        ow = json.loads(ow_path[-1].read_text()) if ow_path else None
        a0 = appr.t[0] - (appr.t[-1] - appr.t[0]) * 0 - 0.4
        if ow:
            span = float(ow["t"][-1])
            w0 = a0 - span - 0.2
            open_t = np.asarray(ow["t"], float) + w0
            open_v = np.asarray(ow["opening"], float)
            add(sw.t[-1], w0, "deliver", "Find the face", "MediaPipe on the resident colour stream + depth")
            add(w0, a0, "deliver", "Wait for an open mouth", "opening > %.2f held %.1f s" % (ow["threshold"], ow["hold_s"]))
            for f in sorted((ow_path[-1].parent / "wait_frames").glob("*.jpg")):
                wait_frames.append((w0 + float(f.stem.split("_")[0]), f))
        add(a0, appr.t[-1], "deliver", "Approach", "%s · stop %.0f mm from the mouth"
            % (appr.label.replace("approach at ", ""), 1000 * float((goal or {}).get("stop_m", 0.08))))
    bs = seg("bite")
    be = first("bite")
    if bs:
        detail = ("bite felt at +%.1f s · %.1f N" % (be["at_s"], be["peak_n"])) if be else ""
        add(bs.t[0], bs.t[-1], "deliver", "Bite", detail)
    back = seg("back off") or seg("back to food_detect_start")
    if back:
        add(back.t[0], t_end - t0, "deliver", "Retract", "back off from the mouth, then back to the overlook")

    events = {}
    if plunge:
        events["plunged"] = float(plunge.t[-1])
    if scoop:
        events["scooped"] = float(scoop.t[-1])
    if bs and be:
        events["bite"] = float(bs.t[0] + be["at_s"])
    if appr:
        events["presented"] = float(appr.t[-1])
        events["approach"] = float(appr.t[0])
    if len(open_t):
        above = np.nonzero(open_v > 0.2)[0]
        events["mouth_open"] = float(open_t[above[0]]) if above.size else float(open_t[-1])

    images = {k: B / p for k, p in {"colour": "perceive/colour.jpg", "detection": "perceive/detection.jpg",
                                    "target": "perceive/target.jpg", "face": "deliver/look_1/face_upright.jpg",
                                    "depth": "perceive/depth.png"}.items() if (B / p).exists()}
    over_png = sorted(B.glob("acquire/arm/*_depth_at_over.png"))
    if over_png:
        images["over_depth"] = over_png[0]
    summary = json.loads((B / "summary.json").read_text()) if (B / "summary.json").exists() else {}
    n_bites = sum(1 for line in open(S / "events.jsonl") if '"bite_start"' in line) or None
    return Bite(S, name, t0, t_end - t0, segments, sorted(phases, key=lambda p: p.t0), foods, picked, bxy, half,
                mouth, events, open_t, open_v, images, wait_frames,
                dict(summary=summary, target=target, n_bites=n_bites, n=n,
                     food_height=(pp or {}).get("food_height_m", 0.025)))
