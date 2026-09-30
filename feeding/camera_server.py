#!/usr/bin/env python3
"""The camera, kept open, behind one HTTP port -- whatever the camera is. Runs where rclpy is (a ROS 2 container,
host network), subscribed to the camera profile's topics (config/camera/<name>.yaml, `make camera`):

    GET /median?frames=15&wait=3&since=T&min=5  -> .npy bytes (uint16 mm, 0 = no return): the median of the first
                               15 depth frames that arrived after unix time T (default: this request) -- T = when the
                               arm stopped, so frames buffered before a stream dropout still count; after `wait` s,
                               the median of what there is if at least `min` frames, else 503
    GET /frame                 -> the newest colour frame, JPEG (header X-Frame-Age: its age in s)      (--colour)
    GET /mjpeg                 -> the colour frames as an MJPEG stream (the face server's colour stream,
                                  grab_frame: OpenCV opens it like the RTSP stream of a camera that has one)
    GET /ping                  -> ok

Depth topics can be several, merged (the Kinova vision module ends each RTSP session every ~62 s: two sessions
31 s apart cover it). Colour is optional: the Kinova wrist camera's colour comes over RTSP, not ROS.
feeding/depth_grab.py is the depth client. The median is exactly depth_probe.py's.
"""

from __future__ import annotations

import argparse
import io
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

DEPTH = deque(maxlen=150)              # (arrival time, frame): 5 s at 30 Hz
COLOUR: dict = {}                      # "jpg", "t", "n": the newest colour frame, encoded once
LOCK = threading.Condition()


def median(frames):
    stack = np.stack(frames)
    m = np.where(stack > 0, stack, np.nan).astype(np.float32)
    with np.errstate(all="ignore"):
        d = np.nanmedian(m, axis=0)
    return np.nan_to_num(d, nan=0.0).astype(np.uint16)


def to_array(msg) -> np.ndarray:
    """sensor_msgs/Image -> numpy (16-bit depth as uint16; colour as BGR uint8, for OpenCV)."""
    enc = msg.encoding.lower()
    if "16" in enc:
        return np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width)
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, -1)
    return a[..., ::-1] if enc.startswith("rgb") else a                      # rgb8 -> BGR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--depth", nargs="+", required=True, help="depth topic(s), merged")
    ap.add_argument("--colour", help="colour topic (omit: no /frame, /mjpeg)")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    a = ap.parse_args()
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    class Sub(Node):
        def __init__(self):
            super().__init__("camera_server")
            for t in a.depth:
                self.create_subscription(Image, t, self.depth, qos_profile_sensor_data)
            if a.colour:
                self.create_subscription(Image, a.colour, self.colour, qos_profile_sensor_data)

        def depth(self, msg):
            with LOCK:
                DEPTH.append((time.time(), to_array(msg).copy()))

        def colour(self, msg):
            ok, jpg = cv2.imencode(".jpg", to_array(msg), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            if ok:
                with LOCK:
                    COLOUR.update(jpg=jpg.tobytes(), t=time.time(), n=COLOUR.get("n", 0) + 1)
                    LOCK.notify_all()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/ping":                           # ok = depth frames actually flowing
                with LOCK:
                    ok = bool(DEPTH) and time.time() - DEPTH[-1][0] < 2.0
                self.send_response(200 if ok else 503); self.end_headers()
                self.wfile.write(b"ok" if ok else b"no frames"); return
            if u.path == "/median":
                n, wait, t_req = int(q.get("frames", 15)), float(q.get("wait", 10)), time.time()
                since, least = float(q.get("since", t_req)), int(q.get("min", n))
                while True:
                    with LOCK:
                        fresh = [f for t, f in DEPTH if t > since]
                    if len(fresh) >= n or time.time() - t_req >= wait:
                        break
                    time.sleep(0.02)
                if len(fresh) < max(1, least):
                    self.send_response(503); self.end_headers(); self.wfile.write(b"no fresh frames"); return
                buf = io.BytesIO(); np.save(buf, median(fresh[:n]))
                self.send_response(200); self.end_headers(); self.wfile.write(buf.getvalue()); return
            if u.path == "/frame" and a.colour:
                with LOCK:
                    jpg, t = COLOUR.get("jpg"), COLOUR.get("t", 0.0)
                if jpg is None:
                    self.send_response(503); self.end_headers(); return
                self.send_response(200); self.send_header("Content-Type", "image/jpeg")
                self.send_header("X-Frame-Age", "%.3f" % (time.time() - t)); self.end_headers()
                self.wfile.write(jpg); return
            if u.path == "/mjpeg" and a.colour:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
                seen = 0
                try:
                    while True:
                        with LOCK:
                            LOCK.wait_for(lambda: COLOUR.get("n", 0) > seen, timeout=2.0)
                            jpg, seen = COLOUR.get("jpg"), COLOUR.get("n", 0)
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_response(404); self.end_headers()

    rclpy.init()
    node = Sub()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    print("camera server on :%d  depth %s  colour %s" % (a.port, a.depth, a.colour or "-"), flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
