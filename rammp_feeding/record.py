"""The run record: what every stage leaves behind so a run can be debugged afterwards. The session's own writes here
(events, arrays, the log) never raise: a full disk must not stop the arm mid-move. (A stage's images and depth, saved
before it moves, do: the command then stops before moving.)

    <runs_dir>/<YYYYmmdd_HHMMSS>_<what>/      a session (rammp feed, or a command on its own)
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
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)


def session(what: str) -> Path:
    """Start a session directory (log.txt from here on), or keep the active one."""
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
        except (OSError, subprocess.SubprocessError):
            pass
    return _dir["session"]


def bite() -> Path:
    """The next bite directory of this session (runs_dir/latest points at it)."""
    s = session("bite")
    n = len(list(s.glob("bite_*")))
    d = s / ("bite_%02d" % (n + 1))
    d.mkdir(parents=True, exist_ok=True)
    _dir["bite"] = d
    latest = runs_dir() / "latest"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(d)
    except OSError:
        pass
    return d


def latest_bite() -> Path:
    """The active bite, else the latest one (a stage run on its own continues it)."""
    if _dir["bite"] is None:
        latest = runs_dir() / "latest"
        if not latest.exists():
            raise FileNotFoundError("no bite yet -- rammp perceive first")
        _dir["bite"] = latest.resolve()
        _dir["session"] = _dir["session"] or _dir["bite"].parent
    return _dir["bite"]


def stage(name: str) -> Path:
    """This stage's directory in the current bite (created on first use)."""
    d = latest_bite() / name
    d.mkdir(parents=True, exist_ok=True)
    _dir["stage"] = d
    return d


def here() -> Path | None:
    return _dir["stage"] or _dir["bite"] or _dir["session"]


def event(kind: str, **data: Any) -> None:
    """A structured event, to the session's events.jsonl (and log.txt at DEBUG). Never printed."""
    s = _dir["session"]
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
