"""The user's face from the wrist camera, MediaPipe face landmarks on the CPU (~25 ms a frame).

- The image is upright after a quarter turn at the delivery pose (camera.mount.image_rotated_at_delivery).
- A far face (a user sitting back: a few dozen pixels) is below the landmarker's own detector: with none in the whole
  frame, half-overlapping zoomed crops are searched (delivery.face.zoom_grid), at most every zoom_every_s.
- The mouth's range: the lips' own depth (each lip landmark's 5 x 5 depth window), unless that is far beyond what the
  face's size allows (the depth past the sensor's range, or the food in front of the lips); else the face's size --
  the outer eye corners' spacing (delivery.face.eye_corners_m): a turned head looks narrower, so it reads FARTHER.
- open = inner-lip gap / mouth width.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rammp_feeding import record
from rammp_feeding.config import cfg

log = logging.getLogger("rammp")
DL, FACE = cfg.delivery, cfg.delivery.face
LIPS = [0, 13, 14, 17, 37, 61, 78, 84, 267, 291, 308, 314]
EYES = (33, 263)


class Face:
    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision
        self.det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(cfg.path("face.model"))), num_faces=1))
        self.rotated = bool(cfg.camera.mount.get("image_rotated_at_delivery", False))
        self._crop, self._zoomed = None, 0.0

    def _one(self, img: np.ndarray):
        import mediapipe as mp
        r = self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
        if not r.face_landmarks:
            return None
        h, w = img.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])

    def upright(self, rgb: np.ndarray) -> np.ndarray:
        return np.rot90(rgb, -1) if self.rotated else rgb

    def landmarks(self, rgb: np.ndarray):
        """Face landmarks in the upright image (pixels): the whole frame, else its zoomed crops (a far face)."""
        up = self.upright(rgb)
        L = self._one(up)
        n = int(FACE.zoom_grid)
        if L is not None or n < 2 or time.time() - self._zoomed < float(FACE.zoom_every_s):
            return L
        h, w = up.shape[:2]
        ch, cw = int(2 * h / (n + 1)), int(2 * w / (n + 1))
        crops = [(int(i * (h - ch) / (n - 1)), int(j * (w - cw) / (n - 1))) for i in range(n) for j in range(n)]
        if self._crop in crops:
            crops.remove(self._crop); crops.insert(0, self._crop)
        for y0, x0 in crops:
            L = self._one(up[y0:y0 + ch, x0:x0 + cw])
            if L is not None:
                self._crop = (y0, x0)
                return L + np.array([x0, y0])
        self._zoomed = time.time()
        return None

    def to_image(self, x, y, H: int):
        """An upright pixel -> the camera image's."""
        return (y, H - 1 - x) if self.rotated else (x, y)

    def mouth(self, rgb: np.ndarray, depth: np.ndarray | None, cam) -> dict | None:
        """The mouth in the camera's optical frame (cam: its model): cam_xyz (m), source "depth" (the lips) or "size"
        (eye spacing), open (the opening ratio); None without a face."""
        L = self.landmarks(rgb)
        if L is None:
            return None
        H = rgb.shape[0]
        z_eyes = float(cam.colour.fx * float(FACE.eye_corners_m) / max(1e-6, np.linalg.norm(L[EYES[0]] - L[EYES[1]])))
        z, src = z_eyes, "size"
        if depth is not None:
            Hd, Wd = depth.shape
            zs = []
            for i in LIPS:                                  # each lip point's depth: a 5 x 5 window where it images
                u, v = self.to_image(*L[i], H)
                zi = z_eyes
                for _ in range(4):
                    ud, vd = (int(round(x)) for x in cam.colour_to_depth(u, v, zi))
                    win = depth[max(0, vd - 2):min(Hd, vd + 3), max(0, ud - 2):min(Wd, ud + 3)]
                    win = win[win > 0]
                    if win.size < 3:
                        zi = None
                        break
                    zi = cam.depth_to_colour(ud, vd, float(np.percentile(win, 20)) / 1000.0)[2]
                if zi is not None:
                    zs.append(zi)
            med = float(np.median(zs)) if len(zs) >= 3 else None
            if med is not None and med <= max(float(FACE.depth_beyond_eyes_factor) * z_eyes, z_eyes + 0.3):
                z, src = med, "depth"
        u, v = self.to_image(*L[[13, 14]].mean(0), H)
        opening = float(np.linalg.norm(L[13] - L[14]) / max(1e-6, np.linalg.norm(L[61] - L[291])))
        return dict(cam_xyz=[float(x) for x in cam.ray(u, v, z)], source=src, open=opening,
                    range_from_eyes_m=z_eyes, px=[float(u), float(v)])

    def wait_open(self, cam, timeout_s: float, out: Path) -> tuple[float, np.ndarray] | None:
        """Watch the wrist camera until the mouth has been open open_mouth_hold_s (frames with no face found are
        skipped, up to open_mouth_gap_s: a far face is not found in every frame): (that frame's time, RGB), or None
        (the user declined). open_wait.json keeps the trace."""
        t0, since, seen, last, trace = time.time(), None, -1e9, time.time(), []
        log.info("    waiting up to %.0f s for an open mouth (> %.2f for %.1f s)", timeout_s, float(DL.open_mouth_ratio),
                 float(DL.open_mouth_hold_s))
        try:
            while time.time() - t0 < timeout_s:
                t, rgb = cam.frame(after=last, wait_s=2.0)
                last = t
                L = self.landmarks(rgb)
                o = float(np.linalg.norm(L[13] - L[14]) / max(1e-6, np.linalg.norm(L[61] - L[291]))) if L is not None else 0.0
                trace.append((round(t - t0, 3), round(o, 3), L is not None))
                if L is not None:
                    seen = time.time()
                if o > float(DL.open_mouth_ratio):
                    since = since or time.time()
                    if time.time() - since >= float(DL.open_mouth_hold_s):
                        Image.fromarray(rgb).save(out / "colour.jpg", quality=95)
                        log.info("    mouth open (%.2f) after %.1f s", o, time.time() - t0)
                        return t, rgb
                elif L is not None or time.time() - seen > float(DL.open_mouth_gap_s):
                    since = None
            log.info("    no open mouth in %.0f s", timeout_s)
            return None
        finally:
            try:
                (out / "open_wait.json").write_text(json.dumps(dict(t=[r[0] for r in trace], open=[r[1] for r in trace],
                                                                    face=[r[2] for r in trace])))
            except OSError:
                pass
            record.event("open_wait", frames=len(trace), seconds=round(time.time() - t0, 2))
