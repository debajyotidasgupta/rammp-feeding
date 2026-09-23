"""Cameras, with the failure modes that actually bit us on hardware.

Three sensors, matching Sec. 1 and Sec. 4.3 of the architecture:

  wrist colour   1280x720  -- food and plate detection
  wrist depth     480x270  16UC1 millimetres -- metric lookup, plane fit
  face camera     640x480  -- static, on the chair; the wrist camera CANNOT
                              track a face during approach

Rendered on demand, never on a fixed clock, because perception here is episodic
(Sec. 2): food detection runs about once per 15 s bite cycle, not at 30 Hz.

The depth model is not decoration. Every effect below cost real debugging time:

  * invalid pixels at depth discontinuities. A food cube sits on a plate on a
    table -- two occlusion edges right where the detector wants to sample. The
    architecture's rule "reject a box whose depth patch is >30 percent invalid
    rather than interpolate" only has teeth if the sim produces those holes.
  * a minimum range. Below it the sensor returns nothing, and the wrist camera
    is INSIDE that range for much of the skewer descent.
  * the stream stalling. Measured on the real arm: the RTSP session dies at
    roughly 62 s because rtspsrc refreshes the session slower than the server's
    30 s timeout. It is patched, not cured, and a node that assumes frames keep
    arriving will hang on hardware.
"""
from __future__ import annotations

import numpy as np
import mujoco

# --- intrinsics --------------------------------------------------------------
# Read from the ARM ITSELF (VisionConfig.GetIntrinsicParametersProfile, vision
# device 9) via kinova/scripts/camera_intrinsics.py -- not from the generic
# files kinova_vision ships:
#
#   colour 1280x720  fx=1297.673 fy=1298.631 cx=620.914 cy=238.280
#   depth   480x270  fx= 334.810 fy= 334.810 cx=238.536 cy=139.694
#
# The colour values happen to match the shipped defaults to 3 dp at every
# resolution. The DEPTH values do not: the shipped file says 360.013, which is
# 7.53% larger than this unit, a scale error on every deprojected point worth
# about 8 mm at the plate edge. calibration/depth_480x270.ini carries the real
# numbers and the `depth` Makefile target now passes them to the driver.
#
# The MuJoCo cameras are set to the matching fovy, so `intrinsics()` below
# reproduces fx/fy to within a rounding error. Two caveats that matter:
#
#  * These are Kinova's GENERIC defaults, not this unit's calibration. The
#    driver only reads per-unit values if you pass color_camera_info_url /
#    depth_camera_info_url. Nothing here is a substitute for calibrating.
#  * cy in the 16:9 COLOUR profiles is not near the image centre and does not
#    scale with resolution: cy/h is 0.491 at 320x240 and 0.484 at 640x480, but
#    0.331 at 1280x720 and 0.090 at 1920x1080, while cx/w stays ~0.49
#    throughout. Either those files encode an asymmetric vertical crop or they
#    are simply wrong. Until that is settled against the device, the sim keeps a
#    CENTRED principal point -- deliberately not reproducing an anomaly we
#    cannot yet explain.
COLOUR = dict(name="arm_wrist", width=1280, height=720)
DEPTH = dict(name="arm_wrist_depth", width=480, height=270)
FACE = dict(name="face_cam", width=640, height=480)

# Factory depth->colour extrinsic on the Gen3 module, read from the device:
#   t = [-0.02706, -0.00997, -0.00471] m  (|t| = 29.2 mm), R = identity.
# NOT applied to the camera poses here: both sim cameras still sit at the colour
# origin. Applying it needs the optical-frame convention pinned down against the
# real published tf, and a 29 mm offset inserted with the wrong sign is a 58 mm
# error -- worse than the 29 mm of leaving it out. Nothing in this sim currently
# samples the depth image at a pixel located in the colour image, which is the
# only operation that would notice. Revisit before anything does.
DEPTH_TO_COLOUR_M = (-0.02706, -0.00997, -0.00471)


def intrinsics(model, cam_name: str, width: int, height: int) -> dict:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy = float(model.cam_fovy[cid])
    fy = (height / 2.0) / np.tan(np.radians(fovy) / 2.0)
    return dict(fx=fy, fy=fy, cx=width / 2.0, cy=height / 2.0,
                width=width, height=height, fovy=fovy)


def deproject(depth_m: np.ndarray, K: dict) -> np.ndarray:
    """Depth image -> Nx3 points in the camera frame (x right, y down, z forward)."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    ok = np.isfinite(z) & (z > 0)
    x = (u - K["cx"]) * z / K["fx"]
    y = (v - K["cy"]) * z / K["fy"]
    return np.stack([x[ok], y[ok], z[ok]], axis=-1)


class DepthModel:
    """Noise, dropout and range limits for the Gen3's depth stream."""

    # Measured on the real arm: 87-91 percent of pixels valid.
    TARGET_VALID = 0.89
    # Stereo disparity noise grows with the square of range.
    NOISE_COEFF = 0.002          # sigma [m] = NOISE_COEFF * z^2
    MIN_RANGE = 0.25             # m -- UNVERIFIED, Sec. 9 open question
    MAX_RANGE = 3.0              # m
    INVALID = 0                  # 16UC1 convention: 0 means no reading

    def __init__(self, rng):
        self.rng = rng

    def apply(self, depth_m: np.ndarray) -> np.ndarray:
        """float metres -> uint16 millimetres, with the sensor's warts."""
        z = depth_m.astype(np.float64).copy()
        bad = ~np.isfinite(z) | (z < self.MIN_RANGE) | (z > self.MAX_RANGE)

        # Occlusion edges: a stereo sensor cannot match a pixel it can only see
        # from one camera, so discontinuities come back as holes. This is the
        # dominant, and structured, source of invalid pixels.
        # Clip before differencing: the far plane sits at ~114 m, and the raw
        # step from a surface to the sky swamps every real occlusion edge.
        gy, gx = np.gradient(np.clip(np.nan_to_num(z, nan=self.MAX_RANGE), 0, self.MAX_RANGE))
        edge = np.hypot(gx, gy) > 0.01
        edge = _dilate(edge)
        bad |= edge

        # Top up with speckle to the measured valid-pixel rate.
        deficit = (1.0 - self.TARGET_VALID) - bad.mean()
        if deficit > 0:
            bad |= self.rng.random(z.shape) < deficit

        z += self.rng.normal(0.0, self.NOISE_COEFF * np.clip(z, 0, self.MAX_RANGE) ** 2)
        mm = np.clip(z * 1000.0, 0, 65535).astype(np.uint16)
        mm[bad] = self.INVALID
        return mm


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for ax in (0, 1):
        out |= np.roll(mask, 1, axis=ax) | np.roll(mask, -1, axis=ax)
    return out


class Cameras:
    """Renders the three streams. Renderers are kept open; they are expensive."""

    # Measured on hardware: the RTSP session drops at ~62 s and takes about a
    # second to come back on TCP (6 s on UDP, which is why we run TCP).
    STALL_PERIOD = 62.0
    STALL_RECOVERY = 1.0

    def __init__(self, world, stalls: bool = False):
        self.w = world
        self.stalls = stalls
        self.depth_model = DepthModel(world.rng)
        self.K = {k: intrinsics(world.model, c["name"], c["width"], c["height"])
                  for k, c in (("colour", COLOUR), ("depth", DEPTH), ("face", FACE))}
        self._r = {}
        self._last_good = {}

    def _renderer(self, key, spec, depth=False):
        if key not in self._r:
            r = mujoco.Renderer(self.w.model, spec["height"], spec["width"])
            if depth:
                r.enable_depth_rendering()
            self._r[key] = r
        return self._r[key]

    def stalled(self) -> bool:
        """Is the stream in one of its periodic outages right now?"""
        if not self.stalls:
            return False
        phase = self.w.t % self.STALL_PERIOD
        return phase > (self.STALL_PERIOD - self.STALL_RECOVERY)

    def colour(self) -> np.ndarray | None:
        if self.stalled():
            return None
        r = self._renderer("colour", COLOUR)
        r.update_scene(self.w.data, camera=COLOUR["name"])
        return r.render()

    def face(self) -> np.ndarray | None:
        r = self._renderer("face", FACE)
        r.update_scene(self.w.data, camera=FACE["name"])
        return r.render()          # separate camera, not on the arm's RTSP link

    def depth(self) -> np.ndarray | None:
        """uint16 millimetres, 0 = invalid. None while the stream is stalled."""
        if self.stalled():
            return None
        r = self._renderer("depth", DEPTH, depth=True)
        r.update_scene(self.w.data, camera=DEPTH["name"])
        return self.depth_model.apply(r.render())

    def close(self) -> None:
        for r in self._r.values():
            r.close()
        self._r.clear()
