#!/usr/bin/env python3
"""Keep the depth stream open: a per-pixel median of FRESH frames in ~0.5 s instead of ~5 s.

Runs INSIDE the kinova_vision container (rclpy), host network, port 8767:

    GET /median?frames=15&wait=3&since=T&min=5  -> .npy bytes (uint16 mm, 0 = no return): the median of the first
                               15 frames that arrived after unix time T (default: this request) -- T = when the arm
                               stopped, so frames buffered before a stream dropout still count; after `wait` s, the
                               median of what there is if at least `min` frames, else 503.
    (was: the next 15 frames
                                  that ARRIVE AFTER the request -- never frames from before the arm stopped
    GET /ping                  -> ok

feeding/depth_grab.py is the host-side client (falls back to depth_probe.py if this is not up). Each
depth_probe run started ROS, a node and a subscription from scratch: ~5 s, twice a bite (perceive and the
floor re-measure after OVER). The median itself is exactly depth_probe.py's.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

PORT = 8767                            # --port / --topic: `make models` passes them from config/
TOPIC = "/camera/depth/image_raw"
BUF = deque(maxlen=150)                # (arrival time, frame): 5 s at 30 Hz
LOCK = threading.Lock()


def median(frames):
    stack = np.stack(frames)
    m = np.where(stack > 0, stack, np.nan).astype(np.float32)
    with np.errstate(all="ignore"):
        d = np.nanmedian(m, axis=0)
    return np.nan_to_num(d, nan=0.0).astype(np.uint16)


def main() -> int:
    global PORT, TOPIC
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--topic", nargs="+", default=[TOPIC], help="one or more depth topics, merged (staggered sessions)")
    a = ap.parse_args(); PORT, TOPIC = a.port, a.topic
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    class Sub(Node):
        def __init__(self):
            super().__init__("depth_server")
            for t in TOPIC:          # the arm ends each RTSP session every ~62 s: two sessions, 31 s apart, cover it
                self.create_subscription(Image, t, self.cb, qos_profile_sensor_data)

        def cb(self, msg):
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
            with LOCK:
                BUF.append((time.time(), arr))

    rclpy.init(); node = Sub()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/ping":
                ok = bool(BUF) and time.time() - BUF[-1][0] < 2.0     # frames actually flowing
                self.send_response(200 if ok else 503); self.end_headers(); self.wfile.write(b"ok" if ok else b"no frames")
                return
            q = parse_qs(u.query)
            n = int(q.get("frames", ["15"])[0]); wait = float(q.get("wait", ["10"])[0]); t_req = time.time()
            since = float(q.get("since", [t_req])[0]); least = int(q.get("min", [n])[0])
            while time.time() - t_req < wait:
                with LOCK:
                    fresh = [f for t, f in BUF if t > since]
                if len(fresh) >= n:
                    break
                time.sleep(0.02)
            if len(fresh) < max(1, least):
                self.send_response(503); self.end_headers(); self.wfile.write(b"no fresh frames"); return
            b = io.BytesIO(); np.save(b, median(fresh[:n]))
            self.send_response(200); self.end_headers(); self.wfile.write(b.getvalue())

    print("depth server on :%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
