#!/usr/bin/env python3
"""Keep the GPU models loaded and warm: LocateAnything (detection) and Depth Anything 3 (depth fallback).

    ~/venvs/la/bin/python -m feeding.detector_server          # start (make detector), stays up
    from feeding.detector_server import connect; la = connect()   # a client, or a local model if none is up

Loading LocateAnything took 13.6 s -- on EVERY `make perceive`, i.e. every bite (user, 2026-09-24: "keep it
on"); Depth Anything 3 takes 10.4 s to load and 0.8 s for its first inference, then ~0.1 s. The server loads
both once (8.5 GB of the Jetson's 61 GB together), runs a warm-up inference on each, and only then answers:
  /locate, /locate_one   LocateAnything; the client has the same methods, so perceive.py does not care
  /depth                 DA3 metric depth of a colour image, written to an .npy (metres, colour resolution)
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


def da3_depth(image_path: str, out_path: str) -> bool:
    """DA3 depth of a colour image into out_path via the server; False if the server is not up."""
    try:
        req = urllib.request.Request("http://%s:%d/depth" % (HOST, PORT), method="POST",
                                     data=json.dumps(dict(image=image_path, out=out_path)).encode())
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("ok", False)
    except OSError:
        return False


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
    import glob
    import numpy as np
    import torch
    from PIL import Image
    from feeding.locate_anything import LocateAnything
    t0 = time.time(); la = LocateAnything()
    print("LocateAnything loaded in %.1f s" % (time.time() - t0), flush=True)
    da3 = None
    try:
        from depth_anything_3.api import DepthAnything3
        t0 = time.time(); da3 = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()
        print("Depth Anything 3 loaded in %.1f s" % (time.time() - t0), flush=True)
    except Exception as e:  # noqa: BLE001 -- detection still works without the depth fallback
        print("Depth Anything 3 NOT loaded (%s) -- no depth fallback" % e, flush=True)

    from feeding.run_depth_anything3 import FX_COLOUR, FY_COLOUR

    def depth(img):
        """DA3 metric depth in metres at colour resolution -- exactly as run_depth_anything3.py:
        depth_m = focal_px_at_processed_size * output / 300, processed at 1008 px, upsampled back."""
        W, H = img.size
        with torch.no_grad():
            pred = da3.inference([img], process_res=1008, process_res_method="upper_bound_resize")
        raw = np.asarray(pred.depth[0], dtype=np.float32); h, w = raw.shape
        metric = 0.5 * (FX_COLOUR + FY_COLOUR) * (w / W) * raw / 300.0
        return torch.nn.functional.interpolate(torch.from_numpy(metric)[None, None], size=(H, W),
                                               mode="bilinear", align_corners=False)[0, 0].numpy()

    # warm both up before the first real request: the first inference pays one-time GPU setup
    shots = sorted(glob.glob(str(Path.home() / "rammp_runs" / "2*" / "colour.jpg")))
    if shots:
        img = Image.open(shots[-1]).convert("RGB")
        t = time.time(); la.locate(img, "food pieces", max_new_tokens=128)
        print("LocateAnything warm-up %.1f s" % (time.time() - t), flush=True)
        if da3 is not None:
            t = time.time(); depth(img); depth(img)
            print("Depth Anything 3 warm-up %.1f s" % (time.time() - t), flush=True)
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
                if self.path == "/depth":
                    if da3 is None:
                        raise RuntimeError("Depth Anything 3 is not loaded")
                    np.save(q["out"], depth(img).astype(np.float32))
                    print("/depth: %.2f s" % (time.time() - t), flush=True)
                    body = json.dumps(dict(ok=True)).encode(); self.send_response(200)
                    self.end_headers(); self.wfile.write(body); return
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
