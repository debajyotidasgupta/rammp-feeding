"""Stand-in for the GPU perception nodes -- ground truth, degraded honestly.

The real stack runs LocateAnything-3B for food and plate and a keypoint model
for the face. Neither can run yet: triphammer has the GPU but no
nvidia-container-toolkit, so nothing CUDA reaches a container. Waiting for that
would block the state machine and the safety monitor, which are the parts that
must exist before anyone sits in front of this arm.

So detections come from the simulator's own state -- but they are NOT handed
over as exact numbers. Each one is put through the same steps the real pipeline
uses, and fails the same ways:

  * project the object into the DEPTH image and sample the patch there;
  * if too much of the patch is invalid, REJECT the detection rather than
    interpolate it (architecture Sec. 4.2: a missed cube costs one retry, a
    phantom cube drives the fork into the plate);
  * otherwise deproject the sampled depth, so range error is the depth
    sensor's real error, not zero;
  * return nothing at all while the stream is stalled.

What this deliberately does NOT model is the detector being WRONG about what a
thing is -- a shadow read as food, two touching cubes read as one. Those are
label errors, they need the real model to characterise, and `Faults` can inject
them explicitly instead of pretending we know their statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensors import DEPTH, FACE


@dataclass
class FoodPiece:
    """One detected piece.

    `surface` is what the sensor actually measured: the point on the piece's
    TOP FACE under the detection's centroid. `position` is the inferred centre,
    the surface pushed back along the viewing ray by the piece's half-thickness.

    The distinction is worth keeping. Skewering aims at the centre of the
    thickest cross-section, so a planner that treated the measured surface as
    the centre would enter the piece one half-extent too shallow -- 7.5 mm on a
    15 mm cube, which is the difference between the tines gripping the piece and
    scraping its top.
    """
    index: int
    position: np.ndarray          # inferred CENTRE, world frame, metres
    surface: np.ndarray           # measured top-face point, world frame
    half_extents: np.ndarray
    confidence: float
    valid_fraction: float


@dataclass
class FaceTrack:
    mouth: np.ndarray
    nose: np.ndarray
    confidence: float


def project(world, cam_name: str, K: dict, p_world: np.ndarray):
    """World point -> (u, v, range). MuJoCo cameras look down their own -Z."""
    cid = world._id(__import__("mujoco").mjtObj.mjOBJ_CAMERA, cam_name)
    t = world.data.cam_xpos[cid]
    R = world.data.cam_xmat[cid].reshape(3, 3)
    p = R.T @ (np.asarray(p_world, float) - t)
    z = -p[2]
    if z <= 1e-6:
        return None
    u = K["cx"] + K["fx"] * (p[0] / z)
    v = K["cy"] - K["fy"] * (p[1] / z)
    return float(u), float(v), float(z)


def deproject_pixel(world, cam_name: str, K: dict, u: float, v: float, z: float) -> np.ndarray:
    cid = world._id(__import__("mujoco").mjtObj.mjOBJ_CAMERA, cam_name)
    t = world.data.cam_xpos[cid]
    R = world.data.cam_xmat[cid].reshape(3, 3)
    p_cam = np.array([(u - K["cx"]) * z / K["fx"], -(v - K["cy"]) * z / K["fy"], -z])
    return R @ p_cam + t


class Perception:
    PATCH = 5                      # half-size of the depth patch sampled, px
    MIN_VALID = 0.70               # architecture Sec. 4.2: reject above 30% invalid

    def __init__(self, world, cameras, faults=None):
        self.w = world
        self.cam = cameras
        self.faults = faults

    # ------------------------------------------------------------------ food
    def detect_food(self, depth=None) -> list[FoodPiece]:
        """One episodic detection pass over the plate."""
        if self.faults and self.faults.detection_blackout:
            return []
        depth = self.cam.depth() if depth is None else depth
        if depth is None:                      # stream stalled
            return []

        K = self.cam.K["depth"]
        out: list[FoodPiece] = []
        for i, bid in enumerate(self.w.food_bid):
            if self.w.data.qpos[self.w.model.jnt_qposadr[self.w.model.body_jntadr[bid]] + 2] < 0.6:
                continue                        # on the floor; gone
            pr = project(self.w, DEPTH["name"], K, self.w.data.xpos[bid])
            if pr is None:
                continue
            u, v, _ = pr
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < K["width"] and 0 <= vi < K["height"]):
                continue
            p = self.PATCH
            patch = depth[max(0, vi - p):vi + p + 1, max(0, ui - p):ui + p + 1]
            if patch.size == 0:
                continue
            valid = patch > 0
            frac = float(valid.mean())
            if frac < self.MIN_VALID:
                continue                        # reject, do not interpolate
            z = float(np.median(patch[valid])) / 1000.0
            surface = deproject_pixel(self.w, DEPTH["name"], K, u, v, z)
            gid = self.w.model.body_geomadr[bid]
            he = self.w.model.geom_size[gid].copy()
            # Push back along the viewing ray to the centre. On hardware the
            # extent comes from the 2D box plus the depth patch; here it is
            # taken from the model, which makes this estimate better than the
            # real one will be. Flagged rather than hidden.
            cid = self.w._id(__import__("mujoco").mjtObj.mjOBJ_CAMERA, DEPTH["name"])
            ray = surface - self.w.data.cam_xpos[cid]
            ray /= np.linalg.norm(ray)
            centre = surface + ray * float(np.abs(ray) @ he)
            out.append(FoodPiece(i, centre, surface, he, min(1.0, frac), frac))

        if self.faults:
            out = self.faults.corrupt_food(out, self.w)
        return out

    # ------------------------------------------------------------------ face
    def track_face(self) -> FaceTrack | None:
        """Continuous mouth/nose track from the STATIC chair camera.

        Deliberately not from the wrist camera: during approach the fork and
        food occlude the lower half of that frame exactly when the mouth matters
        (architecture Sec. 4.3).
        """
        if self.faults and self.faults.face_lost:
            return None
        K = self.cam.K["face"]
        mouth_w, nose_w = self.w.site_pos("mouth"), self.w.site_pos("nose")
        for p in (mouth_w, nose_w):
            pr = project(self.w, FACE["name"], K, p)
            if pr is None or not (0 <= pr[0] < K["width"] and 0 <= pr[1] < K["height"]):
                return None                     # out of frame -> no track
        # Range comes from the NOSE: the mouth changes shape when it opens, the
        # nose does not, so it is the stable landmark for standoff.
        sigma = 0.004
        n = self.w.rng.normal(0, sigma, 3)
        return FaceTrack(mouth_w + n, nose_w + n, 0.9)

    # ----------------------------------------------------------------- table
    def table_plane(self, depth=None):
        """RANSAC plane fit -> the hard floor for the skewer descent (Sec. 4.1)."""
        depth = self.cam.depth() if depth is None else depth
        if depth is None:
            return None
        from sensors import deproject
        K = self.cam.K["depth"]
        z = np.where(depth > 0, depth / 1000.0, np.nan)
        pts_cam = deproject(z, K)
        if len(pts_cam) < 100:
            return None
        cid = self.w._id(__import__("mujoco").mjtObj.mjOBJ_CAMERA, DEPTH["name"])
        R = self.w.data.cam_xmat[cid].reshape(3, 3)
        t = self.w.data.cam_xpos[cid]
        pts = (R @ (pts_cam * np.array([1.0, -1.0, -1.0])).T).T + t

        best, best_n = None, 0
        rng = self.w.rng
        for _ in range(120):
            idx = rng.choice(len(pts), 3, replace=False)
            a, b, c = pts[idx]
            nrm = np.cross(b - a, c - a)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9:
                continue
            nrm /= ln
            d = -nrm @ a
            inl = np.abs(pts @ nrm + d) < 0.004
            if inl.sum() > best_n:
                best_n, best = int(inl.sum()), (nrm, d, inl)
        if best is None:
            return None
        nrm, d, inl = best
        if nrm[2] < 0:
            nrm, d = -nrm, -d
        return dict(normal=nrm, offset=float(d), inliers=best_n,
                    fraction=float(best_n / len(pts)),
                    height=float(-d / nrm[2]) if abs(nrm[2]) > 1e-6 else float("nan"))
