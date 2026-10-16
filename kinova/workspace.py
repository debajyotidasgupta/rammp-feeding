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
    """bowl_xy, table_z, bowl_top_z, bowl_radius, error_m (how far the reading may be off), age_s, source.

    With a scene camera configured, its latest reading whatever its age (a coarse position does not go wrong with
    time; rammp.yaml's own workspace is another site's) -- `make detect` refuses one older than scene_max_age_s
    (the bowl may have moved), not this. rammp.yaml's workspace only without a scene camera, or before its first
    reading."""
    w = cfg.workspace
    if cfg.profiles.get("scene_camera"):
        f = Path(os.path.expanduser(str(cfg.deployment.runs_dir))) / "scene_state.json"
        try:
            st = json.loads(f.read_text())
            age = time.time() - float(st["t"])
            return dict(bowl_xy=[float(v) for v in st["bowl_xy"]], table_z=float(st["table_z"]),
                        bowl_top_z=float(st.get("bowl_top_z", st["table_z"])),
                        bowl_radius=float(st.get("rim_radius_m") or w.bowl_radius_m),
                        error_m=float(st.get("error_m", cfg.get("scene_camera.extrinsic.error_m", 0.05))), age_s=age,
                        source="scene camera (%s%s)" % (time.strftime("%H:%M:%S", time.localtime(st["t"])),
                                                        ", %.0f min old" % (age / 60) if age > float(w.scene_max_age_s) else ""))
        except (OSError, ValueError, KeyError):
            pass
    return dict(bowl_xy=[float(v) for v in w.bowl_centre_m], table_z=float(w.table_height_m),
                bowl_top_z=float(w.table_height_m) + float(w.bowl_height_m), bowl_radius=float(w.bowl_radius_m),
                error_m=0.0, age_s=0.0, source="config")


def mouth_now() -> list[float]:
    """The user's mouth (arm base frame, m): the last delivery's measurement, else motion.free_space.user_mouth_m."""
    try:
        f = Path(os.path.expanduser(str(cfg.deployment.runs_dir))) / "mouth_state.json"
        return [float(v) for v in json.loads(f.read_text())["mouth_base"]]
    except (OSError, ValueError, KeyError):
        return [float(v) for v in cfg.motion.free_space.user_mouth_m]
