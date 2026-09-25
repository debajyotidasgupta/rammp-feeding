#!/usr/bin/env python3
"""Face, nose and mouth by MediaPipe face landmarks -- ~25 ms a frame on the Jetson CPU (LocateAnything: 30-40 s).

    python -m feeding.detect_mouth_mp RUN_DIR                    # colour.jpg + depth.npy -> face.json, mediapipe.jpg
    python -m feeding.detect_mouth_mp RUN_DIR --wait-open 60     # watch the stream; save colour.jpg once the mouth opens

Runs in ~/venvs/mp (mediapipe, opencv). Writes the same face.json as detect_mouth.py (LocateAnything), which
deliver.sh falls back to when no face is found here (exit 5).

- The image is rolled 90 deg at the delivery pose; it is rotated upright first, as in detect_mouth.py.
- On 8 saved delivery frames it found the face every time, including the two where LocateAnything's "mouth"
  box was the whole frame; where both worked they agreed to 2-10 px.
- DEPTH: when the food covers the mouth, the depth at the mouth's pixel is the FOOD. Depth is read at many
  face points instead; points far nearer than the face (the food, ~250 mm nearer) are dropped; the mouth
  takes the lips' own depth where they are visible, else the nose tip's + 5 mm, else the face's.
- mouth_open = inner-lip gap / mouth width: closed read <= 0.02 on every saved frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from feeding.measure_under import CXD, CYD, FXD, FYD, T_D2C

CAL = Path(__file__).resolve().parent.parent / "calibration"
MODEL = Path.home() / "models" / "face_landmarker.task"
RTSP = "rtsp://192.168.2.67/color"
NOSE = [1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 327]
LIPS = [0, 13, 14, 17, 37, 61, 78, 84, 267, 291, 308, 314]
FACE_PTS = list(range(0, 468, 6))
OPEN_RATIO = 0.20              # mouth_open above this counts as open (closed <= 0.02 measured)
OPEN_HOLD_S = 0.30
NO_FACE = 5


def landmarker():
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), num_faces=1, output_face_blendshapes=True))


def landmarks(det, upright: np.ndarray):
    import mediapipe as mp
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(upright)))
    if not r.face_landmarks:
        return None, None
    h, w = upright.shape[:2]
    L = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
    jaw = next((c.score for c in r.face_blendshapes[0] if c.category_name == "jawOpen"), float("nan"))
    return L, jaw


def opening(L) -> float:
    return float(np.linalg.norm(L[13] - L[14]) / max(1e-6, np.linalg.norm(L[61] - L[291])))


def wait_open(D: Path, timeout: float, det=None) -> int:
    """Watch the colour stream; save the first frame after the mouth has been open OPEN_HOLD_S."""
    import cv2
    own = det is None
    det = det or landmarker(); cap = cv2.VideoCapture(RTSP)
    try:
        return _wait(det, cap, D, timeout)
    finally:
        cap.release()
        if own:
            det.close()


def _wait(det, cap, D, timeout):
    t0 = time.time(); since = None; last = -1.0
    print("    waiting up to %.0f s for an open mouth (opening > %.2f for %.1f s)" % (timeout, OPEN_RATIO, OPEN_HOLD_S))
    while time.time() - t0 < timeout:
        ok, bgr = cap.read()
        if not ok:
            time.sleep(0.05); continue
        rgb = bgr[:, :, ::-1]
        L, _ = landmarks(det, np.rot90(rgb, -1))            # == PIL ROTATE_270 (90 deg clockwise)
        o = opening(L) if L is not None else 0.0
        if time.time() - last > 2.0:
            print("      mouth opening %.2f" % o); last = time.time()
        if o > OPEN_RATIO:
            since = since or time.time()
            if time.time() - since >= OPEN_HOLD_S:
                Image.fromarray(rgb).save(D / "colour.jpg")
                print("    mouth open (%.2f) after %.1f s" % (o, time.time() - t0))
                return 0
        else:
            since = None
    print("    no open mouth in %.0f s" % timeout)
    return 4


def detect(D: Path, det=None) -> int:
    c = json.loads((CAL / "colour_intrinsics.json").read_text())
    cx, cy, fx, fy = c["cx"], c["cy"], c["fx"], c["fy"]
    img = Image.open(D / "colour.jpg").convert("RGB"); H = img.size[1]
    up = img.transpose(Image.ROTATE_270)
    dep = np.load(D / "depth.npy").astype(float); yy, xx = np.mgrid[0:dep.shape[0], 0:dep.shape[1]]
    mp_ = D / "depth_filled_mask.npy"
    filled = np.load(mp_) if mp_.exists() else np.zeros(dep.shape, bool)
    sensor_dep = np.where(filled, 0.0, dep)                  # prefer what the sensor itself measured
    own = det is None
    det = det or landmarker()
    try:
        t0 = time.time(); L, jaw = landmarks(det, np.asarray(up)); dt = time.time() - t0
    finally:
        if own:
            det.close()
    if L is None:
        print("mediapipe: no face (%.0f ms)" % (1000 * dt))
        return NO_FACE

    def to_orig(x, y):
        return y, H - 1 - x

    def depth_at(u, v, z=0.55, dep=dep):
        for _ in range(6):
            X = (u - cx) * z / fx - T_D2C[0]; Y = (v - cy) * z / fy - T_D2C[1]; Zd = z - T_D2C[2]
            ud, vd = FXD * X / Zd + CXD, FYD * Y / Zd + CYD
            m = (np.hypot(xx - ud, yy - vd) <= 3) & (dep > 0)
            if m.sum() < 3:
                return None
            z = np.percentile(dep[m], 20) / 1000.0 + T_D2C[2]
        return z

    def zs(idx):
        """Sensor depth at each landmark; the filled (Depth Anything 3) value only where the sensor has none."""
        out = {}
        for i in idx:
            z = depth_at(*to_orig(*L[i]), dep=sensor_dep)
            out[i] = z if z is not None else depth_at(*to_orig(*L[i]))
        return out
    lip_sensor = sum(depth_at(*to_orig(*L[i]), dep=sensor_dep) is not None for i in LIPS)

    fz = [z for z in zs(FACE_PTS).values() if z]
    if len(fz) < 5:
        print("mediapipe: face found but almost no depth on it")
        return NO_FACE
    med = np.median(fz); face_z = float(np.median([z for z in fz if z > med - 0.05]))   # drop the food
    ok = lambda z: z is not None and face_z - 0.04 < z < face_z + 0.06
    nose_z = [z for z in zs(NOSE).values() if ok(z)]; lip_z = [z for z in zs(LIPS).values() if ok(z)]
    nose_depth = float(np.median(nose_z)) if nose_z else None
    if len(lip_z) >= 3:
        mouth_depth, how = float(np.median(lip_z)), "lips"
    elif nose_depth:
        mouth_depth, how = nose_depth + 0.005, "nose + 5 mm"
    else:
        mouth_depth, how = face_z, "face"

    def entry(idx, centre, z, prompt):
        x1, y1 = L[idx].min(0); x2, y2 = L[idx].max(0)
        u, v = to_orig(*centre)
        return dict(box_upright=[float(x1), float(y1), float(x2), float(y2)], px=[float(u), float(v)],
                    depth_m=z, prompt=prompt,
                    cam_xyz=None if z is None else [(u - cx) * z / fx, (v - cy) * z / fy, z])

    mouth_c = L[[13, 14]].mean(0)
    out = dict(face=entry(list(range(468)), L.mean(0), face_z, "mediapipe"),
               nose=entry(NOSE, L[1], nose_depth, "mediapipe"),
               mouth=dict(entry(LIPS, mouth_c, mouth_depth, "mediapipe, depth from %s" % how),
                          depth_filled=bool(filled.any() and lip_sensor < 3)),
               mouth_open=opening(L), jaw_open=jaw, detector="mediapipe", ms=1000 * dt)
    (D / "face.json").write_text(json.dumps(out, indent=1))
    d = ImageDraw.Draw(up)
    for x, y in L[::4]:
        d.point((x, y), fill=(0, 255, 0))
    for name, col in (("nose", "cyan"), ("mouth", "red")):
        d.rectangle(out[name]["box_upright"], outline=col, width=3)
    d.ellipse([mouth_c[0] - 6, mouth_c[1] - 6, mouth_c[0] + 6, mouth_c[1] + 6], outline="yellow", width=3)
    up.save(D / "face_upright.jpg")
    print("nose  depth %s  [mediapipe]" % ("none" if nose_depth is None else "%.0f mm" % (1000 * nose_depth)))
    print("mouth depth %.0f mm  [mediapipe, from %s; open %.2f; %.0f ms]" % (1000 * mouth_depth, how, out["mouth_open"], 1000 * dt))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--wait-open", type=float, default=0.0, help="seconds to watch for an open mouth first (0: don't)")
    if sys.argv[1:] == ["--serve"]:
        return serve()
    a = ap.parse_args()
    D = Path(a.run_dir)
    rc = _via_server(D, a.wait_open)           # the resident landmarker (make models), if it is up
    if rc is not None:
        return rc
    if a.wait_open:
        return wait_open(D, a.wait_open)
    return detect(D)


FACE_PORT = 8766


def _via_server(D: Path, wait: float):
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:%d/%s" % (FACE_PORT, "wait_open" if wait else "detect"),
                                     method="POST", data=json.dumps(dict(run_dir=str(D), timeout=wait)).encode())
        with urllib.request.urlopen(req, timeout=wait + 30) as r:
            out = json.loads(r.read())
    except OSError:
        return None
    sys.stdout.write(out.get("log", "")); sys.stdout.flush()
    return out["rc"]


def serve() -> int:
    """Keep the face landmarker loaded (make models): /detect and /wait_open, as the CLI does."""
    import contextlib, io
    from http.server import BaseHTTPRequestHandler, HTTPServer
    det = landmarker()
    shots = sorted(Path.home().glob("rammp_runs/deliver_*/colour.jpg"))
    if shots:                                  # warm-up
        t = time.time(); landmarks(det, np.asarray(Image.open(shots[-1]).convert("RGB").transpose(Image.ROTATE_270)))
        print("face landmarker warm-up %.0f ms" % (1000 * (time.time() - t)), flush=True)
    print("serving on 127.0.0.1:%d" % FACE_PORT, flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    rc = wait_open(Path(q["run_dir"]), q["timeout"], det) if self.path == "/wait_open" \
                        else detect(Path(q["run_dir"]), det)
                except Exception as e:  # noqa: BLE001
                    print("face server error: %s" % e); rc = 1
            body = json.dumps(dict(rc=rc, log=buf.getvalue())).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)

    HTTPServer(("127.0.0.1", FACE_PORT), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
