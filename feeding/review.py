#!/usr/bin/env python3
"""Read a run record back: the timeline, the key numbers of every bite, and a contact sheet of its images.

    python -m feeding.review ~/rammp_runs/<session>            # every bite: timeline + numbers
    python -m feeding.review ~/rammp_runs/<session>/bite_02    # one bite
    python -m feeding.review <dir> --brief                     # one line per bite
    python -m feeding.review <bite> --write                    # + summary.json and contact_sheet.jpg in the bite
    make review [RUN=<dir>]                                    # the latest session

Reads only what the stages wrote (feeding/lib.sh has the layout): events.jsonl everywhere, the stage JSONs,
the images, the arm's telemetry (.npz). Runs in the face venv (PIL); nothing here touches the arm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def events(root: Path) -> list[dict]:
    """Every event under root, in time order."""
    out = []
    for f in root.rglob("events.jsonl"):
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
                e["_where"] = str(f.parent.relative_to(root)) if f.parent != root else "."
                out.append(e)
            except ValueError:
                pass
    return sorted(out, key=lambda e: float(e.get("t", 0)))


def _json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def bites(root: Path) -> list[Path]:
    if root.name.startswith("bite_"):
        return [root]
    return sorted(p for p in root.glob("bite_*") if p.is_dir())


# ------------------------------------------------------------------ numbers

def bite_summary(b: Path) -> dict[str, Any]:
    """The key numbers of one bite, from its files and events (its own, and the session's that name it)."""
    own = [e for e in events(b.parent) if e.get("_where") == "." and e.get("bite") == b.name]
    ev = sorted(own + events(b), key=lambda e: float(e.get("t", 0)))
    s: dict[str, Any] = dict(bite=b.name, stages={})
    starts = {}
    for e in ev:
        k = e.get("kind")
        if k == "stage_start":
            starts[e["stage"]] = e["t"]
        elif k == "stage_end":
            s["stages"][e["stage"]] = dict(rc=e.get("rc"), seconds=e.get("seconds"))
        elif k == "stopped":
            s.setdefault("stopped", []).append("%s %s: %s" % (e.get("stage"), e.get("command"), e.get("reason")))
        elif k in ("plunge_plan", "scoop", "scoop_plan", "swing", "approach_goal", "presented", "bite", "no_bite",
                   "retracted", "mouth_not_usable", "declined", "depth_fallback", "over"):
            s.setdefault("events", {}).setdefault(k, []).append({x: y for x, y in e.items() if x not in ("t", "_where")})
    t = _json(b / "perceive" / "target.json")
    if t:
        s["target"] = dict(tool_goal=t.get("tool_goal"), food_depth_m=t.get("berry_depth_m"), food=t.get("berry"))
    d = _json(b / "perceive" / "detection.json")
    if d:
        s["detected"] = dict(food=len(d.get("food", [])), bowl=bool(d.get("bowl")),
                             ms={k: v for k, v in d.items() if k.endswith("_ms") or k == "ms"})
    for name in ("under_over.json",):
        u = _json(b / "acquire" / name)
        if u:
            s["under_over"] = {k: u.get(k) for k in ("top", "floor", "gap_floor", "height", "filled_floor", "depth_source")}
    looks = sorted((b / "deliver").glob("look_*")) if (b / "deliver").exists() else []
    s["looks"] = []
    for lk in looks:
        f = _json(lk / "face.json") or {}
        w = _json(lk / "open_wait.json")
        s["looks"].append(dict(look=lk.name, nose=bool(f.get("nose")), mouth=bool(f.get("mouth")),
                               open_wait=(w or {}).get("result"),
                               open_after_s=(w["t"][-1] if w and w.get("t") else None)))
    return s


def _fmt(x: Any, unit: str = "mm") -> str:
    return "-" if x is None else ("%.0f %s" % (1000 * x, unit) if unit == "mm" else "%.1f %s" % (x, unit))


def print_bite(s: dict, brief: bool = False) -> None:
    st = s["stages"]
    total = sum((v.get("seconds") or 0) for v in st.values())
    E = s.get("events", {})
    bite = (E.get("bite") or [{}])[-1]
    result = "stopped" if s.get("stopped") else ("eaten" if bite else ("declined" if E.get("declined") else "?"))
    if brief:
        print("%s  %-8s %5.0f s  %s" % (s["bite"], result, total,
              "  ".join("%s %.0fs%s" % (k, v.get("seconds") or 0, "" if v.get("rc") in (0, None) else "(rc %s)" % v["rc"])
                        for k, v in st.items())))
        return
    print("\n== %s  (%s, %.0f s)" % (s["bite"], result, total))
    for k, v in st.items():
        print("   %-9s %6.1f s  rc %s" % (k, v.get("seconds") or 0, v.get("rc")))
    if s.get("detected"):
        print("   perceive   %d food piece(s), bowl %s" % (s["detected"]["food"], "yes" if s["detected"]["bowl"] else "NO"))
    if s.get("target"):
        print("   target     tool goal %s, food at %s depth" % (np.round(s["target"]["tool_goal"], 4).tolist(),
                                                               _fmt(s["target"]["food_depth_m"])))
    u = s.get("under_over")
    if u:
        print("   under      floor %s below the tip, food %s tall%s" % (_fmt(u.get("gap_floor")), _fmt(u.get("height")),
              ", %.0f%% of the floor filled (%s)" % (100 * u["filled_floor"], u.get("depth_source")) if u.get("filled_floor") else ""))
    for k, label in (("plunge_plan", "plunge"), ("scoop", "scoop"), ("approach_goal", "approach"),
                     ("presented", "presented"), ("bite", "bite"), ("no_bite", "no bite"), ("retracted", "retract")):
        for e in E.get(k, []):
            if k == "plunge_plan":
                print("   plunge     %s down, %.1f s planned" % (_fmt(e.get("down_m")), e.get("seconds", 0)))
            elif k == "scoop":
                print("   scoop      food rose %s, worst %.1f mm off its path" % (_fmt(e.get("rose_m")), 1000 * e.get("worst_off_path_m", 0)))
            elif k == "approach_goal":
                print("   approach   mouth from the %s, %s to go, stop %s short" % (e.get("source"), _fmt(e.get("distance_m")), _fmt(e.get("stop_m"))))
            elif k == "presented":
                print("   presented  utensil tip %s from the mouth" % _fmt(e.get("tip_to_mouth_m")))
            elif k == "bite":
                print("   bite       at %.1f s, peak %.1f N" % (e.get("at_s", 0), e.get("peak_n", 0)))
            elif k == "no_bite":
                print("   no bite    in %.0f s (peak %.1f N)" % (e.get("timeout_s", 0), e.get("peak_n", 0)))
            elif k == "retracted":
                print("   retract    lowest tip z %s" % _fmt(e.get("lowest_tip_z_m")))
    for lk in s.get("looks", []):
        print("   %-10s nose %s, mouth %s%s" % (lk["look"], "yes" if lk["nose"] else "no", "yes" if lk["mouth"] else "no",
              ", open-mouth wait: %s after %.1f s" % (lk["open_wait"], lk["open_after_s"] or 0) if lk["open_wait"] else ""))
    for k in ("depth_fallback", "mouth_not_usable", "declined"):
        for e in E.get(k, []):
            print("   note       %s %s" % (k.replace("_", " "), {x: y for x, y in e.items() if x not in ("kind", "stage")}))
    for r in s.get("stopped", []):
        print("   STOPPED    %s" % r)


# ------------------------------------------------------------------ images

def _plot(series: list[tuple[np.ndarray, np.ndarray, tuple]], title: str, size=(640, 240), hline: float | None = None):
    """A small line plot with PIL: [(t, y, colour)]."""
    from PIL import Image, ImageDraw
    W, H = size
    im = Image.new("RGB", size, (255, 255, 255))
    dr = ImageDraw.Draw(im)
    ts = np.concatenate([s[0] for s in series]) if series else np.zeros(1)
    ys = np.concatenate([s[1] for s in series] + ([np.array([hline])] if hline is not None else [])) if series else np.zeros(1)
    t0, t1 = float(ts.min()), float(max(ts.max(), ts.min() + 1e-3))
    y0, y1 = float(ys.min()), float(max(ys.max(), ys.min() + 1e-3))
    X = lambda t: 40 + (W - 50) * (t - t0) / (t1 - t0)          # noqa: E731
    Y = lambda y: H - 20 - (H - 40) * (y - y0) / (y1 - y0)       # noqa: E731
    dr.rectangle([40, 20, W - 10, H - 20], outline=(180, 180, 180))
    if hline is not None:
        dr.line([X(t0), Y(hline), X(t1), Y(hline)], fill=(200, 0, 0))
    for t, y, c in series:
        if len(t) > 1:
            dr.line(list(zip(map(X, t), map(Y, y))), fill=c, width=2)
    dr.text((44, 4), "%s   (%.1f s; %.2f .. %.2f)" % (title, t1 - t0, y0, y1), fill=(0, 0, 0))
    return im


def _arm_plots(b: Path) -> list:
    out = []
    for f in sorted(b.rglob("*_arm_*.npz")):
        try:
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            t = z["t"]
            if meta.get("label") == "bite":
                F = np.linalg.norm(z["force"] - np.median(z["force"][:40], 0), axis=1)
                out.append(_plot([(t, F, (0, 90, 200))], "bite: tool force above rest (N)", hline=meta.get("threshold_n")))
            else:
                tau = z["tau"] - z["tau"][0]
                cols = [(200, 0, 0), (0, 150, 0), (0, 0, 200), (200, 120, 0), (120, 0, 160), (0, 150, 150), (90, 90, 90)]
                out.append(_plot([(t, tau[:, j], cols[j]) for j in range(tau.shape[1])],
                                 "%s: joint torque change (N.m) -- %s" % (meta.get("label"), meta.get("outcome"))))
        except Exception:  # noqa: BLE001
            pass
    return out


def contact_sheet(b: Path, out: Path) -> Path | None:
    """Every image of the bite, and a plot per arm move, on one sheet."""
    from PIL import Image, ImageDraw
    try:                                     # the depth the arm fetched at OVER (pick) is only an .npz
        from feeding.viz import depth_png
        for f in b.glob("acquire/arm/*_depth_at_over.npz"):
            png = f.with_suffix(".png")
            if not png.exists():
                depth_png(np.load(f)["depth_mm"], png, text="depth at OVER")
    except Exception:  # noqa: BLE001
        pass
    pics = []
    for pat in ("perceive/detection.jpg", "perceive/target.jpg", "perceive/depth.png", "perceive/depth_filled.png",
                "acquire/*.png", "acquire/arm/*.png", "deliver/look_*/face_upright.jpg", "deliver/look_*/depth.png",
                "deliver/look_*/open_wait_last.jpg"):
        pics += sorted(b.glob(pat))
    tiles = []
    for p in pics:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((640, 360))
            tiles.append((str(p.relative_to(b)), im))
        except Exception:  # noqa: BLE001
            pass
    tiles += [("", im) for im in _arm_plots(b)]
    if not tiles:
        return None
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 650, rows * 390 + 30), (240, 240, 240))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 8), "%s   %s" % (b, time.strftime("%Y-%m-%d %H:%M")), fill=(0, 0, 0))
    for i, (name, im) in enumerate(tiles):
        x, y = 5 + (i % cols) * 650, 30 + (i // cols) * 390
        sheet.paste(im, (x, y))
        if name:
            dr.text((x, y + im.size[1] + 2), name, fill=(0, 0, 0))
    sheet.save(out, quality=88)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--write", action="store_true", help="also write summary.json and contact_sheet.jpg per bite")
    a = ap.parse_args()
    root = a.dir.expanduser().resolve()
    sess = root if not root.name.startswith("bite_") else root.parent
    info = _json(sess / "session.json") or {}
    if not a.brief or root == sess:
        print("session %s  (%s, commit %s%s, started %s)" % (sess.name, info.get("what", "?"), info.get("commit", "?"),
              " +%d dirty" % len(info["dirty"]) if info.get("dirty") else "", info.get("started", "?")))
    for b in bites(root):
        s = bite_summary(b)
        print_bite(s, brief=a.brief)
        if a.write:
            (b / "summary.json").write_text(json.dumps(s, indent=1, default=str))
            p = contact_sheet(b, b / "contact_sheet.jpg")
            if p and not a.brief:
                print("   sheet      %s" % p)
    if root == sess and not a.brief:
        end = [e for e in events(sess) if e.get("kind") == "session_end"]
        if end:
            print("\nsession end: %s after %s bite(s)" % (end[-1].get("why"), end[-1].get("bites")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
