#!/usr/bin/env python3
"""Keep LocateAnything loaded between detections.

    ~/venvs/la/bin/python -m feeding.detector_server          # start (make detector), stays up
    from feeding.detector_server import connect; la = connect()   # a client, or a local model if none is up

Loading the model took 13.6 s -- on EVERY `make perceive`, i.e. every bite (user, 2026-09-24: "keep it on").
The server holds it on the GPU and answers locate / locate_one over localhost; the client has the same
methods as LocateAnything, so perceive.py does not care which it got.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

from feeding.locate_anything import Box

HOST, PORT = "127.0.0.1", 8765


class _Remote:
    """LocateAnything's interface, answered by the server."""

    def _call(self, image, fn, phrase, **kw):
        path = getattr(image, "filename", None)
        if not path:
            raise RuntimeError("the server needs an image opened from a file")
        req = urllib.request.Request("http://%s:%d/%s" % (HOST, PORT, fn), method="POST",
                                     data=json.dumps(dict(image=path, phrase=phrase, kw=kw)).encode())
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        return [Box(**b) for b in out]

    def locate(self, image, phrase, **kw):
        return self._call(image, "locate", phrase, **kw)

    def locate_one(self, image, phrase, **kw):
        boxes = self._call(image, "locate_one", phrase, **kw)
        return boxes[0] if boxes else None


def connect():
    """The running server if there is one, else the model loaded here (the old behaviour)."""
    try:
        with urllib.request.urlopen("http://%s:%d/ping" % (HOST, PORT), timeout=2) as r:
            if r.read() == b"ok":
                print("  (detector server: model already loaded)")
                return _Remote()
    except OSError:
        pass
    print("  (no detector server -- loading the model here; `make detector` keeps it loaded)")
    from feeding.locate_anything import LocateAnything
    return LocateAnything()


def main() -> int:
    from PIL import Image
    from feeding.locate_anything import LocateAnything
    t0 = time.time(); la = LocateAnything()
    print("model loaded in %.1f s" % (time.time() - t0), flush=True)
    # warm it up before the first real request: the first inference after loading pays one-time GPU setup
    import glob
    shots = sorted(glob.glob(str(Path.home() / "rammp_runs" / "2*" / "colour.jpg")))
    if shots:
        t = time.time(); la.locate(Image.open(shots[-1]).convert("RGB"), "food pieces", max_new_tokens=128)
        print("warm-up detection %.1f s" % (time.time() - t), flush=True)
    print("serving on %s:%d" % (HOST, PORT), flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            try:
                q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                img = Image.open(q["image"]).convert("RGB")
                t = time.time()
                if self.path == "/locate_one":
                    b = la.locate_one(img, q["phrase"], **q["kw"]); boxes = [b] if b else []
                else:
                    boxes = la.locate(img, q["phrase"], **q["kw"])
                print("%s %r: %d box(es) in %.1f s" % (self.path, q["phrase"], len(boxes), time.time() - t), flush=True)
                body = json.dumps([dict(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, label=b.label) for b in boxes]).encode()
                self.send_response(200)
            except Exception as e:  # noqa: BLE001 -- report, keep serving
                body = json.dumps(dict(error=str(e))).encode(); self.send_response(500)
            self.end_headers(); self.wfile.write(body)

    HTTPServer((HOST, PORT), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
