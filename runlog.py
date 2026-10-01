"""Logging and the run record: what every stage leaves behind so a run can be debugged afterwards.

    from runlog import setup, event, save_json, save_arrays, Telemetry

    log = setup("deliver")          # in main(): console output, plus the record when one is active
    log.info("    swing done")      # console: info -> stdout, warning/error -> stderr
    event("swing", retract_m=0.12)  # a structured event (events.jsonl), not printed
    save_json("face_goal", {...})   # a file in the record directory

The record is active when RAMMP_RECORD names a directory (the stage scripts set it: one directory per stage
of a bite, see feeding/lib.sh). Without it -- a module run by hand, the tests -- everything still logs to the
console and nothing is written. Recording never raises: a full disk must not stop the arm mid-move.

In the record directory:
    log.txt         every log line, timestamped (ms), with its module
    events.jsonl    one JSON object per event: t (unix s), stage, kind, and the event's fields
    <HHMMSS.mmm>_<name>.json / .npz    plans, measurements, and arm telemetry (Telemetry below)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from kinova.frames import fold

_STAGE = "rammp"


def record_dir() -> Path | None:
    """The active record directory (created on first use), or None."""
    d = os.environ.get("RAMMP_RECORD")
    if not d:
        return None
    try:
        p = Path(d).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        return None


def setup(stage: str) -> logging.Logger:
    """Configure the `rammp` loggers for one program (call once, in main) and return this stage's logger.

    Console: INFO and up, message only (what the operator and the shell scripts read).
    Record (when active): DEBUG and up to log.txt, timestamped.
    """
    global _STAGE
    _STAGE = stage
    root = logging.getLogger("rammp")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    plain = logging.Formatter("%(message)s")
    out = logging.StreamHandler(sys.stdout)
    out.setLevel(logging.INFO)
    out.addFilter(lambda r: r.levelno < logging.WARNING)
    out.setFormatter(plain)
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(logging.WARNING)
    err.setFormatter(plain)
    root.addHandler(out)
    root.addHandler(err)
    d = record_dir()
    if d is not None:
        try:
            fh = logging.FileHandler(d / "log.txt")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(name)-22s %(levelname)-7s %(message)s",
                                              datefmt="%H:%M:%S"))
            root.addHandler(fh)
        except OSError:
            pass
    return logging.getLogger("rammp." + stage)


def get(name: str) -> logging.Logger:
    """A module's logger (library modules: `log = runlog.get(__name__)`)."""
    return logging.getLogger("rammp." + name.rsplit(".", 1)[-1])


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "__dataclass_fields__"):
        return {k: getattr(x, k) for k in x.__dataclass_fields__}
    return str(x)


def _stamp() -> str:
    t = time.time()
    return time.strftime("%H%M%S", time.localtime(t)) + ".%03d" % (int(t * 1000) % 1000)


def event(kind: str, **data: Any) -> None:
    """Append a structured event to events.jsonl (and log.txt, at DEBUG). Never printed."""
    d = record_dir()
    if d is None:
        return
    rec = dict(t=round(time.time(), 3), stage=_STAGE, kind=kind, **data)
    try:
        line = json.dumps(rec, default=_jsonable)
        with open(d / "events.jsonl", "a") as f:
            f.write(line + "\n")
        logging.getLogger("rammp.event").debug(line)
    except (OSError, TypeError, ValueError):
        pass


def save_json(name: str, obj: Any) -> Path | None:
    """<record>/<time>_<name>.json; returns the path (None when not recording)."""
    d = record_dir()
    if d is None:
        return None
    p = d / ("%s_%s.json" % (_stamp(), name))
    try:
        p.write_text(json.dumps(obj, indent=1, default=_jsonable))
        return p
    except (OSError, TypeError, ValueError):
        return None


def save_arrays(name: str, **arrays: Any) -> Path | None:
    """<record>/<time>_<name>.npz (compressed); non-array values are stored as a JSON string `meta`."""
    d = record_dir()
    if d is None:
        return None
    p = d / ("%s_%s.npz" % (_stamp(), name))
    arr = {k: np.asarray(v) for k, v in arrays.items() if isinstance(v, (np.ndarray, list, tuple))}
    meta = {k: v for k, v in arrays.items() if k not in arr}
    try:
        np.savez_compressed(p, meta=np.array(json.dumps(meta, default=_jsonable)), **arr)
        return p
    except (OSError, TypeError, ValueError):
        return None


class Telemetry:
    """Arm feedback during one move: whatever the move's watch loop already reads, so recording adds no
    controller calls and no delay. `add(fb)` per sample, `save(...)` once at the end (success or not).

    Saved (npz): t (s from the first sample), q (deg, folded), dq (deg/s), tau (N.m), tool (x y z m, theta
    xyz deg), force (tool external force x y z, N), and `meta` (label, outcome, anything passed to save).
    """

    def __init__(self, label: str):
        self.label = label
        self.t: list[float] = []
        self.q: list[list[float]] = []
        self.dq: list[list[float]] = []
        self.tau: list[list[float]] = []
        self.tool: list[list[float]] = []
        self.force: list[list[float]] = []
        self._t0: float | None = None

    def add(self, fb: Any) -> None:
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        b = fb.base
        self.t.append(now - self._t0)
        self.q.append([fold(a.position) for a in fb.actuators])
        self.dq.append([getattr(a, "velocity", float("nan")) for a in fb.actuators])
        self.tau.append([a.torque for a in fb.actuators])
        self.tool.append([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                          b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z])
        self.force.append([b.tool_external_wrench_force_x, b.tool_external_wrench_force_y,
                           b.tool_external_wrench_force_z])

    def save(self, outcome: str = "done", **meta: Any) -> Path | None:
        if record_dir() is None or not self.t:
            return None
        safe = "".join(c if c.isalnum() else "_" for c in self.label).strip("_")[:40]
        p = save_arrays("arm_" + safe, t=self.t, q=self.q, dq=self.dq, tau=self.tau, tool=self.tool,
                        force=self.force, label=self.label, outcome=outcome, **meta)
        event("telemetry", label=self.label, outcome=outcome, samples=len(self.t),
              duration_s=round(self.t[-1], 3), file=p.name if p else None)
        return p
