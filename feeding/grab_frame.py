#!/usr/bin/env python3
"""One colour frame from the wrist camera, in ~1 s.

    ~/venvs/mp/bin/python -m feeding.grab_frame OUT.jpg

Replaces recording the RTSP stream through gst-launch until a 22 s timeout killed it and keeping the last
frame: that cost 22 s on every perceive (user, 2026-09-24: 'why is the detection taking so much time').
Skips the first few decoded frames so the H.264 decoder has a full picture and exposure has settled.
"""
import sys
import time

import cv2

from feeding.camera import COLOUR_RTSP as RTSP
SKIP = 8


def main() -> int:
    cap = cv2.VideoCapture(RTSP); t0 = time.time(); n = 0; frame = None
    while time.time() - t0 < 8.0:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.02); continue
        n += 1; frame = f
        if n > SKIP:
            break
    cap.release()
    if frame is None:
        print("no frame from %s in 8 s" % RTSP, file=sys.stderr)
        return 1
    cv2.imwrite(sys.argv[1], frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("frame after %d decoded, %.1f s" % (n, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
