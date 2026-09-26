#!/usr/bin/env python3
"""The twin's arm against the real one: MuJoCo FK of recorded joints vs the controller's own tool pose.

    .venv/bin/python tests/twin_check.py [SESSION]     # default: the newest feed session under runs_dir

Every move of every bite: the tool frame (flange + 120 mm) in the twin vs the `tool` column the telemetry
recorded at the same instant. Fails above 10 mm / 2 deg -- the twin's overlays and the TWIN camera tile are
only as right as this. Also builds one bite's scene, so a broken asset shows up here rather than mid-render.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from sim.scene import ASSETS  # noqa: E402

MAX_MM, MAX_DEG = 10.0, 2.0


def main() -> int:
    from sim.twin import bites_in, latest_session
    session = Path(os.path.expanduser(sys.argv[1])) if len(sys.argv) > 1 else latest_session()
    m = mujoco.MjModel.from_xml_path(str(ASSETS / "kinova_gen3" / "gen3.xml"))
    d = mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
    pos, rot = [], []
    for f in sorted(glob.glob(str(session / "bite_*" / "*" / "arm" / "*_arm_*.npz"))):
        z = np.load(f, allow_pickle=True)
        for i in range(0, len(z["t"]), max(1, len(z["t"]) // 6)):
            d.qpos[:7] = np.radians(z["q"][i])
            mujoco.mj_kinematics(m, d)
            R = d.site_xmat[site].reshape(3, 3)
            p = d.site_xpos[site] + R[:, 2] * 0.12
            tool = z["tool"][i]
            Rt = Rotation.from_euler("xyz", tool[3:], degrees=True).as_matrix()
            pos.append(1000 * np.linalg.norm(p - tool[:3]))
            rot.append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rt.T @ R).as_rotvec())))
    if not pos:
        print("no telemetry under %s" % session)
        return 1
    pos, rot = np.array(pos), np.array(rot)
    print("FK vs the controller, %d samples: position mean %.1f / max %.1f mm, rotation max %.2f deg"
          % (len(pos), pos.mean(), pos.max(), rot.max()))
    from sim.record import load_bite
    from sim.twin import Twin
    b = load_bite(session, bites_in(session)[0])
    tw = Twin(b)
    tw.pose(b.events.get("presented", 0.0))
    gap = 1000 * np.linalg.norm(tw.d.site_xpos[tw.tip_id] - np.array(b.mouth))
    print("scene built: %s, %d pieces; tip to mouth when presented %.0f mm (the stack stops at 80)"
          % (b.name, len(b.foods), gap))
    ok = pos.max() <= MAX_MM and rot.max() <= MAX_DEG and 60 <= gap <= 100
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
