"""The run record: what every stage leaves behind so a run can be debugged afterwards. Off unless record.enabled (a
debug setting; the node's `record` parameter): with it off nothing here touches the disk, and every function is a no-op
that returns None. The session's own writes (events, arrays, the log) never raise: a full disk must not stop the arm
mid-move. (A stage's images and depth, saved before it moves, do: the command then stops before moving.)

    <runs_dir>/<YYYYmmdd_HHMMSS>_<what>/      a session (rammp feed, a meal under the node, or a command on its own)
      session.json, config.yaml, log.txt, events.jsonl
      bite_01/<stage>/                        each stage's images, depth, plans, telemetry (<HHMMSS.mmm>_<name>.*)
    <runs_dir>/latest -> the current bite: `rammp acquire` after `rammp perceive` finds its target there
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rammp_feeding.config import ROOT, cfg, dump, runs_dir

log = logging.getLogger("rammp")
_dir: dict[str, Path | None] = dict(session=None, bite=None, stage=None)
_file_handler: list[logging.Handler] = []


def on() -> bool:
    """Whether the run record is kept (record.enabled)."""
    return bool(cfg.get("record.enabled", False))


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    log.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in log.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)


def session(what: str) -> Path | None:
    """Start a session directory (log.txt from here on), or keep the active one. None with the record off."""
    if not on():
        return None
    if _dir["session"] is None:
        d = runs_dir() / (time.strftime("%Y%m%d_%H%M%S") + "_" + what)
        d.mkdir(parents=True, exist_ok=True)
        _dir["session"] = d
        try:
            (d / "config.yaml").write_text(dump())
            git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,  # noqa: E731
                                            timeout=5).stdout.strip()
            (d / "session.json").write_text(json.dumps(dict(started=time.strftime("%F %T"), what=what, argv=sys.argv,
                                                            host=socket.gethostname(), commit=git("rev-parse", "--short", "HEAD"),
                                                            profiles=dict(cfg.profiles)), indent=1))
            fh = logging.FileHandler(d / "log.txt")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S"))
            log.addHandler(fh)
            _file_handler.append(fh)
        except (OSError, subprocess.SubprocessError):
            pass
    return _dir["session"]


def end_session() -> None:
    """Close the active session (the node: at the end of a meal), so the next one starts its own directory."""
    for h in _file_handler:
        log.removeHandler(h)
        h.close()
    _file_handler.clear()
    _dir.update(session=None, bite=None, stage=None)


def bite() -> Path | None:
    """The next bite directory of this session (runs_dir/latest points at it). None with the record off."""
    s = session("bite")
    if s is None:
        return None
    n = len(list(s.glob("bite_*")))
    d = s / ("bite_%02d" % (n + 1))
    d.mkdir(parents=True, exist_ok=True)
    _dir["bite"], _dir["stage"] = d, None
    latest = runs_dir() / "latest"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(d)
    except OSError:
        pass
    return d


def latest_bite() -> Path:
    """The active bite, else the latest one (a stage run on its own continues it): the hand-over between
    step-by-step commands, so it needs the record on."""
    if not on():
        raise FileNotFoundError("the run record is off (record.enabled): step-by-step commands hand over through it")
    if _dir["bite"] is None:
        latest = runs_dir() / "latest"
        if not latest.exists():
            raise FileNotFoundError("no bite yet -- rammp perceive first")
        _dir["bite"] = latest.resolve()
        _dir["session"] = _dir["session"] or _dir["bite"].parent
    return _dir["bite"]


def stage(name: str) -> Path | None:
    """This stage's directory in the current bite (created on first use). None with the record off."""
    if not on():
        return None
    d = latest_bite() / name
    d.mkdir(parents=True, exist_ok=True)
    _dir["stage"] = d
    return d


def here() -> Path | None:
    return (_dir["stage"] or _dir["bite"] or _dir["session"]) if on() else None


def event(kind: str, **data: Any) -> None:
    """A structured event, to the session's events.jsonl (and log.txt at DEBUG). Never printed."""
    s = _dir["session"] if on() else None
    if s is None:
        return
    try:
        line = json.dumps(dict(data, t=round(time.time(), 3), kind=kind), default=_jsonable)   # (the event's own time)
        with open(s / "events.jsonl", "a") as f:
            f.write(line + "\n")
        log.debug(line)
    except (OSError, TypeError, ValueError):
        pass


def _stamp() -> str:
    t = time.time()
    return time.strftime("%H%M%S", time.localtime(t)) + ".%03d" % (int(t * 1000) % 1000)


def save_arrays(name: str, d: Path | None = None, **arrays: Any) -> Path | None:
    d = d or here()
    if d is None:
        return None
    p = d / ("%s_%s.npz" % (_stamp(), name))
    try:
        np.savez_compressed(p, **{k: np.asarray(v) for k, v in arrays.items()})
        return p
    except (OSError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------- a stage's files (no-ops with no directory)
def save_json(d: Path | None, name: str, data: Any) -> None:
    if d is not None:
        (d / name).write_text(json.dumps(data, indent=1, default=_jsonable))


def save_text(d: Path | None, name: str, text: str) -> None:
    if d is not None:
        (d / name).write_text(text)


def save_image(d: Path | None, name: str, rgb, quality: int = 90) -> None:
    """An RGB array or a PIL image, as a JPEG."""
    if d is not None:
        from PIL import Image
        (rgb if hasattr(rgb, "save") else Image.fromarray(rgb)).save(d / name, quality=quality)


def save_npy(d: Path | None, name: str, a) -> None:
    if d is not None:
        np.save(d / name, a)
