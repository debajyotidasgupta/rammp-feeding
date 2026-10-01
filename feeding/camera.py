"""The wrist camera, from the selected camera profile (config/camera/<name>.yaml) -- whichever camera it is.

One place for the intrinsics, the depth-to-colour extrinsic, the pixel mappings between the two sensors, and
the depth correction (scale / offset) -- used by perception, the floor measurement, the face and the depth fill.
"""

from __future__ import annotations

import os

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
DEFAULT_ROT = np.diag([-1.0, -1.0, 1.0])     # the Kinova module (and the site D405): 180 deg about the tool axis


def _quat(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.round(R, 12) + 0.0                # exact -1 / 0 / 1 for a half turn: the default stays bit-identical


def _mount() -> tuple[np.ndarray, np.ndarray]:
    """The colour camera in the TOOL frame: (position, rotation: the optical frame's axes -- x right, y down, z out of
    the lens -- in tool coordinates). From camera.mount.file when set -- another stack's calibration, read in place
    (format rammp_box_opening: end_effector_link = the flange, mount_xyz, mount_quat_xyzw) -- else the profile's
    camera_in_tool_m and rotation_quat_xyzw (default: a half turn about the tool axis)."""
    m = cfg.camera.mount
    path = str(m.get("file", "") or "")
    if not path:
        q = m.get("rotation_quat_xyzw", None)
        return np.array(m.camera_in_tool_m, dtype=float), (_quat(q) if q is not None else DEFAULT_ROT.copy())
    import yaml
    doc = yaml.safe_load(open(os.path.expanduser(path)))
    if doc.get("parent_frame") != "end_effector_link":
        raise ValueError("%s: mount relative to %r, expected end_effector_link" % (path, doc.get("parent_frame")))
    tool = np.array(cfg.deployment.arm.tool.transform_m, dtype=float)     # the flange -> tool offset
    return np.array(doc["mount_xyz"], dtype=float) - tool, _quat(doc["mount_quat_xyzw"])


CAM_IN_TOOL, CAM_ROT = _mount()
DEFAULT_MOUNT = bool(np.array_equal(CAM_ROT, DEFAULT_ROT))


def cam_to_tool(p_cam) -> np.ndarray:
    """A point in the colour camera's optical frame (m) -> the tool frame."""
    return CAM_IN_TOOL + CAM_ROT @ np.asarray(p_cam, dtype=float)
COLOUR_URL = cfg.camera.streams.colour_url             # RTSP from the camera, or the camera server's MJPEG


def corrected(depth_raw):
    """Sensor depth (mm, 0 = no return) with the profile's correction applied; 0 stays 0."""
    d = np.asarray(depth_raw, dtype=np.float64)
    if DEPTH_SCALE == 1.0 and DEPTH_OFFSET_M == 0.0:
        return d
    return np.where(d > 0, DEPTH_SCALE * d + 1000.0 * DEPTH_OFFSET_M, 0.0)


def load_depth(path):
    return corrected(np.load(path))


def colour_to_depth(u, v, z_m, cx=None, cy=None):
    """Colour pixel at colour-frame depth z_m -> depth-frame pixel (cx, cy: another colour principal point)."""
    cx, cy = CXC if cx is None else cx, CYC if cy is None else cy
    X = (u - cx) * z_m / FXC - T_D2C[0]
    Y = (v - cy) * z_m / FYC - T_D2C[1]
    Z = z_m - T_D2C[2]
    return FXD * X / Z + CXD, FYD * Y / Z + CYD


def depth_to_colour(ud, vd, z_m, cx=None, cy=None):
    """Depth-frame pixel at depth-frame z_m -> colour pixel and colour-frame z."""
    cx, cy = CXC if cx is None else cx, CYC if cy is None else cy
    X = (ud - CXD) * z_m / FXD + T_D2C[0]
    Y = (vd - CYD) * z_m / FYD + T_D2C[1]
    Z = z_m + T_D2C[2]
    return FXC * X / Z + cx, FYC * Y / Z + cy, Z

