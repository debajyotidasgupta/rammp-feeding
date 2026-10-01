"""Where the table and the bowl are now: the scene camera's latest reading, else rammp.yaml's workspace.

Plain data (no Kortex, no torch): read by kinova/scene.py on the arm side and kinova/planner_server.py in the
planner's container alike.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from settings import cfg


def workspace() -> dict:
    """bowl_xy, table_z, bowl_top_z, bowl_radius, error_m (how far the reading may be off), source."""
    w = cfg.workspace
    if cfg.profiles.get("scene_camera"):
        f = Path(os.path.expanduser(str(cfg.deployment.runs_dir))) / "scene_state.json"
        try:
            st = json.loads(f.read_text())
            if time.time() - float(st["t"]) < float(w.scene_max_age_s):
                return dict(bowl_xy=[float(v) for v in st["bowl_xy"]], table_z=float(st["table_z"]),
                            bowl_top_z=float(st.get("bowl_top_z", st["table_z"])),
                            bowl_radius=float(st.get("rim_radius_m") or w.bowl_radius_m),
                            error_m=float(st.get("error_m", cfg.get("scene_camera.extrinsic.error_m", 0.05))),
                            source="scene camera (%s)" % time.strftime("%H:%M:%S", time.localtime(st["t"])))
        except (OSError, ValueError, KeyError):
            pass
    return dict(bowl_xy=[float(v) for v in w.bowl_centre_m], table_z=float(w.table_height_m),
                bowl_top_z=float(w.table_height_m) + float(w.bowl_height_m), bowl_radius=float(w.bowl_radius_m),
                error_m=0.0, source="config")
