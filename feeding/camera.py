"""The wrist camera, from the selected camera profile (config/camera/<name>.yaml) -- whichever camera it is.

One place for the intrinsics, the depth-to-colour extrinsic, the pixel mappings between the two sensors, and
the depth correction (scale / offset) -- used by perception, the floor measurement, the face and the depth fill.
"""

from __future__ import annotations

import numpy as np

from settings import cfg

_c, _d = cfg.camera.colour, cfg.camera.depth
FXC, FYC, CXC, CYC = float(_c.fx), float(_c.fy), float(_c.cx), float(_c.cy)
COLOUR_SIZE = (int(_c.width), int(_c.height))
VP = np.array(_c.vertical_vp, dtype=float)                 # image of true vertical at the overlook
FXD, FYD, CXD, CYD = float(_d.fx), float(_d.fy), float(_d.cx), float(_d.cy)
DEPTH_SHAPE = (int(_d.height), int(_d.width))
T_D2C = np.array(_d.depth_to_colour_m, dtype=float)
DEPTH_SCALE, DEPTH_OFFSET_M = float(_d.scale), float(_d.offset_m)
CAM_IN_TOOL = np.array(cfg.camera.mount.camera_in_tool_m, dtype=float)
COLOUR_URL = cfg.camera.streams.colour_url             # RTSP from the camera, or the camera server's MJPEG


def corrected(depth_raw):
    """Sensor depth (mm, 0 = no return) with the profile's correction applied; 0 stays 0."""
    d = np.asarray(depth_raw, dtype=np.float64)
    if DEPTH_SCALE == 1.0 and DEPTH_OFFSET_M == 0.0:
        return d
    return np.where(d > 0, DEPTH_SCALE * d + 1000.0 * DEPTH_OFFSET_M, 0.0)


def load_depth(path):
    return corrected(np.load(path))


def colour_to_depth(u, v, z_m):
    """Colour pixel at colour-frame depth z_m -> depth-frame pixel."""
    X = (u - CXC) * z_m / FXC - T_D2C[0]
    Y = (v - CYC) * z_m / FYC - T_D2C[1]
    Z = z_m - T_D2C[2]
    return FXD * X / Z + CXD, FYD * Y / Z + CYD


def depth_to_colour(ud, vd, z_m):
    """Depth-frame pixel at depth-frame z_m -> colour pixel and colour-frame z."""
    X = (ud - CXD) * z_m / FXD + T_D2C[0]
    Y = (vd - CYD) * z_m / FYD + T_D2C[1]
    Z = z_m + T_D2C[2]
    return FXC * X / Z + CXC, FYC * Y / Z + CYC, Z


def tip_line_px(tip_px, z_m, z_tip_m):
    """Where the utensil tip's vertical line crosses depth z_m (through the image of true vertical)."""
    return VP + (np.asarray(tip_px, dtype=float) - VP) * z_tip_m / z_m
