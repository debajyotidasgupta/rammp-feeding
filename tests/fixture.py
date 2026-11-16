"""The offline tests' site, set up before rammp_feeding is imported: a temporary state_dir holding the arm poses and the
wrist camera's mount and the fork's pixels as rammp calibrate-wrist leaves them, the site's values (the fork's ruler
reading, the table, where the bowl is laid), and the run record beside it.

RAMMP_TEST_SITE picks the site (tests/fake_robot.py renders it):
  level   (default) the arm level on its mount, a D405 on the wrist, the user to the arm's left (tests/poses)
  tilt30  the arm leaning 30 deg forward on its mount, an OAK-D Pro on the wrist, the user to the arm's right, no scene
          camera, a narrow deep cup on a low table, the scoop turning late and the other way (tests/poses_tilt30: the
          overlook captured on that robot; a fork-down home from the overlook's arm shape, and the delivery look straight
          in front of the mouth, both computed -- the planner refuses a look from the far side, over its occupant; its
          measured mount and the fork's pixels as calibrate-wrist found them)"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
SITE = os.environ.get("RAMMP_TEST_SITE", "level")
WORK = Path(tempfile.mkdtemp(prefix="rammp_test_"))
STATE = WORK / "state"
STATE.mkdir(parents=True)
for p in (ROOT / "tests" / ("poses_tilt30" if SITE == "tilt30" else "poses")).glob("*.json"):
    shutil.copy(p, STATE / p.name)
RUNS = "deployment.runs_dir=%s,record.enabled=true,planner.world_file=%s,planner.world_file_in_planner=" % (
    WORK / "runs", STATE / "world.yaml")
if SITE == "tilt30":
    (STATE / "wrist_mount.json").write_text(json.dumps(dict(
        camera="oak_d_pro", utensil="fork", camera_in_tool_m=[0.00565, 0.07312, -0.10241],
        rotation_quat_xyzw=[-0.003915, 0.010272, 0.999934, 0.003192], tip_px=[357.7, 308.5],
        image_region_px=[313, 291, 401, 400])))
    SET = ("profiles.camera=oak_d_pro,utensil.glass_to_tip_m=0.254,workspace.table_extent_m.x=[0.50, 1.00],"
           "workspace.table_extent_m.y=[-0.60, 0.40],workspace.table_height_m=0.095,workspace.bowl_centre_m=[0.639, -0.223],"
           "workspace.overlook_camera_height_m=0.40,perception.scene.use=false,bowl.rim_radius_m=0.045,bowl.height_m=0.08,"
           "acquisition.scoop.turn_deg=-90.0,acquisition.scoop.rot_exp=3.0,acquisition.scoop.speed_dps=25.0,"
           "delivery.second_look_m=[0.0, 0.05, 0.0],")
else:
    (STATE / "wrist_mount.json").write_text(json.dumps(dict(
        camera="realsense_d405", utensil="fork", camera_in_tool_m=[0.0, 0.075, -0.125],
        rotation_quat_xyzw=[0.0, 0.0, 1.0, 0.0], tip_px=[456, 374], image_region_px=[280, 365, 660, 480])))
    SET = ("utensil.glass_to_tip_m=0.233,workspace.table_extent_m.x=[0.08, 0.76],"
           "workspace.table_extent_m.y=[-0.30, 0.80],workspace.bowl_centre_m=[0.290, 0.158],")
os.environ.update(RAMMP_NO_LOCAL="1", RAMMP_STATE_DIR=str(STATE), RAMMP_SET=SET + RUNS)
