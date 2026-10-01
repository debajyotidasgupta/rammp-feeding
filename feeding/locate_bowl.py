#!/usr/bin/env python3
"""Where the bowl and the table are, from a perceive run at food_detect_start -- to fill config/ (workspace).

    python -m feeding.locate_bowl RUN_DIR [--tool X Y Z]      # make locate-bowl [RUN=...]

RUN_DIR holds colour.jpg, depth.npy (sensor) and detection.json (the bowl box) from `make perceive`; the tool
position defaults to the run's target.json (tool_now), else the captured food_detect_start. Prints the bowl
centre (x, y) and floor, the table height where the table is visible beside the bowl, and the YAML to put in
config/local.yaml (or rammp.yaml) if they moved. Everything bowl-dependent then follows (kinova/scene.py).

The bowl's box is often cut by the image edge (the rim does not fit): its centre is then taken from the
uncut sides, assuming a round bowl. Positions are good to a few mm -- enough, because only the DIFFERENCE
from where the poses were captured is used, and both are measured this same way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kinova.frames import R_from_kortex

from settings import cfg
from feeding.camera import CXC, CYC, FXC, FYC, cam_to_tool, colour_to_depth, load_depth


def to_base(u: float, v: float, z: float, T: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Colour pixel (u, v) at colour-frame depth z -> base frame (the camera's mount: feeding.camera.cam_to_tool)."""
    X, Y = (u - CXC) * z / FXC, (v - CYC) * z / FYC
    return T + R @ cam_to_tool([X, Y, z])


def depth_at(D: np.ndarray, box: tuple, z_guess: float) -> float | None:
    """Median valid depth (m, colour frame) over a colour-image box."""
    u1, v1, u2, v2 = box
    a = colour_to_depth(u1, v1, z_guess)
    b = colour_to_depth(u2, v2, z_guess)
    x1, x2 = sorted((int(round(a[0])), int(round(b[0]))))
    y1, y2 = sorted((int(round(a[1])), int(round(b[1]))))
    H, W = D.shape
    patch = D[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    patch = patch[patch > 0]
    return float(np.median(patch)) / 1000.0 if patch.size > 50 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tool", type=float, nargs=3)
    a = ap.parse_args()
    run = Path(a.run)
    det = json.loads((run / "detection.json").read_text())
    if not det.get("bowl"):
        raise SystemExit("no bowl in %s/detection.json" % run)
    b = det["bowl"][0]
    W, H = det.get("width", 1280), det.get("height", 720)
    pose = cfg.overlook_pose()["tool_pose"]
    if a.tool:
        T = np.array(a.tool)
    elif (run / "target.json").exists():
        T = np.array(json.loads((run / "target.json").read_text())["tool_now"])
    else:
        T = np.array([pose["x_m"], pose["y_m"], pose["z_m"]])
    R = R_from_kortex(pose["theta_x_deg"], pose["theta_y_deg"], pose["theta_z_deg"])
    D = load_depth(run / "depth_sensor.npy" if (run / "depth_sensor.npy").exists() else run / "depth.npy")

    # centre: a round bowl cut by the image edge keeps its full width across the cut direction
    x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
    w, h = x2 - x1, y2 - y1
    cu, cv = (x1 + x2) / 2, (y1 + y2) / 2
    cut = None
    if h < 0.95 * w and y1 <= 2:
        cut, cv = "top", y2 - w / 2
    elif h < 0.95 * w and y2 >= H - 2:
        cut, cv = "bottom", y1 + w / 2
    elif w < 0.95 * h and x1 <= 2:
        cut, cu = "left", x2 - h / 2
    elif w < 0.95 * h and x2 >= W - 2:
        cut, cu = "right", x1 + h / 2
    r = min(w, h) * 0.15
    floor = depth_at(D, (cu - r, cv - r, cu + r, cv + r), 0.43)
    if floor is None:
        raise SystemExit("no depth at the bowl's centre")
    centre = to_base(cu, cv, floor, T, R)
    # the table: beside the bowl box, where the image shows it
    table = []
    for box in ((0, 0, x1 - 20, H), (x2 + 20, 0, W, H)):
        if box[2] - box[0] > 40:
            z = depth_at(D, box, floor + 0.02)
            if z is not None:
                table.append(to_base((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, z, T, R)[2])
    ws, cap = cfg.workspace, cfg.bowl.captured_at
    print("bowl   box (%.0f %.0f %.0f %.0f)%s -> centre pixel (%.0f, %.0f)"
          % (x1, y1, x2, y2, "  cut at the %s" % cut if cut else "", cu, cv))
    print("bowl   centre (%.4f, %.4f) m, floor z %.4f m (depth %.3f m)" % (centre[0], centre[1], centre[2], floor))
    if table:
        print("table  z %.4f m (beside the bowl, %d side%s)" % (np.median(table), len(table), "s" if len(table) > 1 else ""))
    else:
        print("table  not visible beside the bowl -- measure it (ruler from the utensil tip at food_detect_start)")
    print("config workspace: bowl_centre_m [%.4f, %.4f], table_height_m %.4f;  %s profile captured_at: [%.4f, %.4f], %.4f"
          % (ws.bowl_centre_m[0], ws.bowl_centre_m[1], ws.table_height_m, cfg.profiles.bowl,
             cap.bowl_centre_m[0], cap.bowl_centre_m[1], cap.table_height_m))
    dx, dy = centre[0] - float(ws.bowl_centre_m[0]), centre[1] - float(ws.bowl_centre_m[1])
    print("        measured - configured: bowl (%+.0f, %+.0f) mm%s" % (1000 * dx, 1000 * dy,
          ", table %+.0f mm" % (1000 * (np.median(table) - float(ws.table_height_m))) if table else ""))
    print("\nif it moved, in config/local.yaml:\nworkspace:\n  bowl_centre_m: [%.4f, %.4f]%s"
          % (centre[0], centre[1], "\n  table_height_m: %.4f" % np.median(table) if table else ""))
    Path(run / "bowl_location.json").write_text(json.dumps(dict(
        centre_pixel=[cu, cv], cut=cut, floor_depth_m=floor, bowl_centre_m=centre[:2].tolist(), bowl_floor_z_m=centre[2],
        table_height_m=float(np.median(table)) if table else None, tool=T.tolist()), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
