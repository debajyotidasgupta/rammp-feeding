"""The offline tests' site, set up before rammp_feeding is imported: a temporary state_dir holding the arm poses
(tests/poses: abra's captures) and the wrist camera's mount and the fork's pixels as rammp calibrate-wrist leaves them
(abra's D405), the fork's ruler reading, and the run record beside it."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
WORK = Path(tempfile.mkdtemp(prefix="rammp_test_"))
STATE = WORK / "state"
STATE.mkdir(parents=True)
for p in (ROOT / "tests" / "poses").glob("*.json"):
    shutil.copy(p, STATE / p.name)
(STATE / "wrist_mount.json").write_text(json.dumps(dict(
    camera="realsense_d405", utensil="fork", camera_in_tool_m=[0.0, 0.075, -0.125], rotation_quat_xyzw=[0.0, 0.0, 1.0, 0.0],
    tip_px=[456, 374], image_region_px=[280, 365, 660, 480])))
os.environ.update(RAMMP_NO_LOCAL="1", RAMMP_STATE_DIR=str(STATE),
                  RAMMP_SET="utensil.glass_to_tip_m=0.233,deployment.runs_dir=%s,planner.world_file=%s,"
                            "planner.world_file_in_planner=" % (WORK / "runs", STATE / "world.yaml"))
