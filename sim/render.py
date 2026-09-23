#!/usr/bin/env python3
"""Render the feeding scene from named cameras. Offscreen, no display needed."""
from __future__ import annotations
import argparse, pathlib
import mujoco
import numpy as np
from PIL import Image

# Gen3 'retract' keyframe from the menagerie model - the arm's real stow pose.
RETRACT = [0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="sim/feeding_scene.xml")
    ap.add_argument("--cams", nargs="+", default=["side", "front", "arm_wrist"])
    ap.add_argument("--out", default="sim/out")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--qpos", nargs="*", type=float, default=None,
                    help="7 arm joint angles (rad); default = retract")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.scene)
    d = mujoco.MjData(m)
    q = a.qpos if a.qpos else RETRACT
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(m, height=a.height, width=a.width) as r:
        for cam in a.cams:
            r.update_scene(d, camera=cam)
            Image.fromarray(r.render()).save(out / f"{cam}.png")
            print(f"  rendered {cam}.png")

    tcp = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm_tcp")]
    mouth = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mouth")]
    print("  tcp   %s" % np.round(tcp, 3))
    print("  mouth %s   |tcp-mouth| = %.3f m" % (np.round(mouth, 3), np.linalg.norm(tcp - mouth)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
