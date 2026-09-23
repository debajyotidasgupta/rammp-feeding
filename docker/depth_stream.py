"""Colorize /camera/depth/image_raw and write raw rgb24 frames to stdout.

Pipe into ffmpeg/ffplay. Fixed mm ramp so colours are stable frame to frame.
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

ANCH = np.array([[48, 18, 59], [70, 134, 251], [27, 229, 181], [170, 250, 60],
                 [249, 183, 52], [231, 80, 21], [122, 4, 3]], dtype=np.float32)
LO, HI = 350.0, 900.0


def cmap(n):
    x = np.clip(n, 0, 1) * (len(ANCH) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(ANCH) - 2)
    t = (x - i)[..., None]
    return (ANCH[i] * (1 - t) + ANCH[i + 1] * t).astype(np.uint8)


class Stream(Node):
    """Colorize depth to stdout, and shout on stderr if the feed goes stale.

    The RTSP session on the arm drops periodically (the server advertises a 30s
    timeout that rtspsrc refreshes too slowly -- see kinova.scripts.rtsp_probe).
    A stalled feed is easy to miss: the picture simply stops changing, which
    looks identical to a stationary scene. Recording that silently produced a
    70s "video" of one frozen frame once already, so warn loudly instead.
    """

    def __init__(self):
        super().__init__("depth_stream")
        self.create_subscription(Image, "/camera/depth/image_raw", self.cb, 1)
        self._prev = None
        self._repeat = 0
        self._last_msg = time.time()

    def cb(self, m):
        now = time.time()
        if now - self._last_msg > 2.0:
            print("[depth_stream] WARNING: %.1fs gap in depth frames"
                  % (now - self._last_msg), file=sys.stderr, flush=True)
        self._last_msg = now

        d = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.width)

        # Byte-identical consecutive frames mean the pipeline is serving a stale
        # buffer -- a live sensor always has some noise.
        if self._prev is not None and np.array_equal(d, self._prev):
            self._repeat += 1
            if self._repeat in (15, 60, 240):
                print("[depth_stream] WARNING: %d consecutive identical frames "
                      "- feed is frozen" % self._repeat, file=sys.stderr, flush=True)
        else:
            if self._repeat >= 15:
                print("[depth_stream] feed recovered after %d frozen frames"
                      % self._repeat, file=sys.stderr, flush=True)
            self._repeat = 0
        self._prev = d
        img = cmap((d.astype(np.float32) - LO) / (HI - LO))
        img[d == 0] = 0
        img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
        sys.stdout.buffer.write(img.tobytes())
        sys.stdout.buffer.flush()


def main():
    rclpy.init()
    rclpy.spin(Stream())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
