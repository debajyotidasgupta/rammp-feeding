"""The wrist camera's colour stream, kept open: always the newest frame, no stream set-up per use, and a log of
every time the camera goes quiet.

Owned by the resident face server (feeding/detect_mouth_mp.py --serve): the open-mouth wait reads frames as they
arrive, and GET /frame hands a fresh JPEG to feeding/grab_frame.py. Opening the RTSP stream for each use cost
~1.6 s a look, and on 2026-09-25 the colour stream stopped delivering for over a minute with the RTSP server still
answering -- a per-use open cannot tell that from a slow start; a resident reader logs it.

Dropout log (config perception.colour_stream): runs_dir/colour_stream.jsonl, one JSON object per event --
{"t": unix s, "kind": "connected" | "gap" | "lost" | "reconnect", "seconds": ...}.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from settings import cfg
from feeding.camera import COLOUR_URL

CS = cfg.perception.colour_stream
GAP_S = float(CS.gap_s)                  # no frame for this long = a gap, logged
RECONNECT_S = float(CS.reconnect_after_s)  # no frame for this long = reopen the stream


class ColourStream:
    """A background reader. frame_after(n) blocks for the first frame newer than frame number n."""

    def __init__(self, url: str = COLOUR_URL, log_path: Path | None = None):
        self.url = url
        self.log_path = log_path or Path(os.path.expanduser(str(cfg.deployment.runs_dir))) / "colour_stream.jsonl"
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None     # BGR
        self.t = 0.0                             # arrival time of self.frame
        self.n = 0                               # frames received since start
        self.connected = False
        threading.Thread(target=self._run, daemon=True).start()

    def _log(self, kind: str, **data) -> None:
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(dict(t=round(time.time(), 3), kind=kind, **data)) + "\n")
        except OSError:
            pass

    def _run(self) -> None:
        # a silent stream must fail the read, not block it forever: FFmpeg's socket timeout (us), both spellings
        us = int(1e6 * RECONNECT_S)
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "stimeout;%d|timeout;%d" % (us, us))
        import cv2
        while True:
            t_open = time.time()
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                self._log("reconnect", ok=False, seconds=round(time.time() - t_open, 2))
                time.sleep(1.0)
                continue
            self._log("connected", seconds=round(time.time() - t_open, 2))
            self.connected = True
            last = time.time()
            while True:
                ok, bgr = cap.read()                       # blocks until a frame (or the stream fails)
                now = time.time()
                if not ok:
                    self._log("lost", after_s=round(now - last, 2))
                    break
                if now - last > GAP_S:
                    self._log("gap", seconds=round(now - last, 2))
                    if now - last > RECONNECT_S:           # a long silence: reopen, the server may have restarted
                        self._log("reconnect", after_gap_s=round(now - last, 2))
                        break
                last = now
                with self.cond:
                    self.frame, self.t, self.n = bgr, now, self.n + 1
                    self.cond.notify_all()
            self.connected = False
            cap.release()
            time.sleep(0.2)

    def frame_after(self, n: int, timeout: float) -> tuple[int, float, np.ndarray] | None:
        """(frame number, arrival time, BGR) of the first frame numbered above n, or None within timeout."""
        end = time.time() + timeout
        with self.cond:
            while self.n <= n:
                left = end - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)
            return self.n, self.t, self.frame

    def fresh(self, timeout: float = 3.0) -> tuple[int, float, np.ndarray] | None:
        """The first frame that arrives after this call."""
        return self.frame_after(self.n, timeout)

    def status(self) -> dict:
        return dict(connected=self.connected, frames=self.n,
                    last_frame_age_s=round(time.time() - self.t, 3) if self.t else None)
