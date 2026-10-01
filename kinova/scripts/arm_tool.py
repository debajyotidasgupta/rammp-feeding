#!/usr/bin/env python3
"""The arm's tool configuration (frame + mass): show it against config/, or set ours.

    python -m kinova.scripts.arm_tool            # make tool: what the arm has, and how it differs from ours
    python -m kinova.scripts.arm_tool --apply    # make tool APPLY=1: set deployment.arm.tool

The tool lives on the arm (Kinova ControlConfig) and another user of the arm may change it. Every move refuses
while it differs (safe_move.check_tool). --apply first writes the values it replaces to
<runs_dir>/arm_tool_before_<time>.json and prints them, so the other user's setting can be put back:

    python -m kinova.scripts.arm_tool --restore <runs_dir>/arm_tool_before_<time>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import runlog
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import tool_mismatch
from settings import cfg

log = runlog.get("arm_tool")


def _client(conn):
    from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
    return ControlConfigClient(conn._tcp_router)


def read(conn) -> dict:
    t = _client(conn).GetToolConfiguration()
    tr, mc = t.tool_transform, t.tool_mass_center
    return dict(transform_m=[tr.x, tr.y, tr.z], rotation_deg=[tr.theta_x, tr.theta_y, tr.theta_z],
                mass_kg=t.tool_mass, mass_centre_m=[mc.x, mc.y, mc.z])


def write(conn, tool: dict) -> None:
    from kortex_api.autogen.messages import ControlConfig_pb2 as CC
    t = CC.ToolConfiguration()
    t.tool_transform.x, t.tool_transform.y, t.tool_transform.z = map(float, tool["transform_m"])
    t.tool_transform.theta_x, t.tool_transform.theta_y, t.tool_transform.theta_z = map(float, tool["rotation_deg"])
    t.tool_mass = float(tool["mass_kg"])
    t.tool_mass_center.x, t.tool_mass_center.y, t.tool_mass_center.z = map(float, tool["mass_centre_m"])
    _client(conn).SetToolConfiguration(t)


def save_before(tool: dict) -> Path:
    """The arm's values, to <runs_dir>/arm_tool_before_<time>.json, before anything replaces them."""
    out = Path(str(cfg.deployment.runs_dir)).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    saved = out / time.strftime("arm_tool_before_%Y%m%d_%H%M%S.json")
    saved.write_text(json.dumps(dict(tool=tool, saved=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))
    return saved


def show(name: str, tool: dict) -> None:
    log.info("  %-6s frame %s m  rotation %s deg  mass %.3f kg  centre %s m", name,
             [round(v, 4) for v in tool["transform_m"]], [round(v, 2) for v in tool["rotation_deg"]],
             tool["mass_kg"], [round(v, 4) for v in tool["mass_centre_m"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="set deployment.arm.tool on the arm")
    ap.add_argument("--restore", help="set the values saved in this file (an arm_tool_before_*.json)")
    a = ap.parse_args()
    runlog.setup("arm_tool")
    ours = {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in cfg.deployment.arm.tool.items()
            if k != "tolerance"}
    with kinova_connection() as conn:
        now = read(conn)
        show("arm", now)
        show("ours", ours)
        if a.apply or a.restore:
            new = json.loads(Path(a.restore).read_text())["tool"] if a.restore else ours
            saved = save_before(now)
            log.info("  saved the arm's values to %s (put back: make tool RESTORE=%s)", saved, saved)
            write(conn, new)
            runlog.event("tool_set", before=now, after=new, saved=str(saved))
            show("set", read(conn))
        bad = tool_mismatch(conn)
        log.info("  %s", "the arm's tool is ours" if not bad else "DIFFERS from ours: " + "; ".join(bad) +
                 " -- every move refuses until `make tool APPLY=1`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
