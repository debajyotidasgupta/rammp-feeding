#!/usr/bin/env python3
"""Keep the depth stream open: a per-pixel median of FRESH frames in ~0.5 s instead of ~5 s.

Runs INSIDE the kinova_vision container (rclpy), host network, port 8767:

    GET /median?frames=15      -> .npy bytes (uint16 mm, 0 = no return): the median of the next 15 frames
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

PORT = 8767
BUF = deque(maxlen=60)                 # (arrival time, frame)
LOCK = threading.Lock()


def median(frames):
    stack = np.stack(frames)
    m = np.where(stack > 0, stack, np.nan).astype(np.float32)
    with np.errstate(all="ignore"):
        d = np.nanmedian(m, axis=0)
    return np.nan_to_num(d, nan=0.0).astype(np.uint16)


def main() -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    class Sub(Node):
        def __init__(self):
            super().__init__("depth_server")
            self.create_subscription(Image, "/camera/depth/image_raw", self.cb, qos_profile_sensor_data)

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
            n = int(parse_qs(u.query).get("frames", ["15"])[0]); t_req = time.time()
            while time.time() - t_req < 10.0:
                with LOCK:
                    fresh = [f for t, f in BUF if t > t_req]
                if len(fresh) >= n:
                    break
                time.sleep(0.02)
            if not fresh:
                self.send_response(503); self.end_headers(); self.wfile.write(b"no fresh frames"); return
            b = io.BytesIO(); np.save(b, median(fresh[:n]))
            self.send_response(200); self.end_headers(); self.wfile.write(b.getvalue())

    print("depth server on :%d" % PORT, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
