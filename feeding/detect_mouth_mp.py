#!/usr/bin/env python3
"""Face, nose and mouth by MediaPipe face landmarks -- ~25 ms a frame on the Jetson CPU (LocateAnything: 30-40 s).

    python -m feeding.detect_mouth_mp RUN_DIR                    # colour.jpg + depth.npy -> face.json, mediapipe.jpg
    python -m feeding.detect_mouth_mp RUN_DIR --wait-open 60     # watch the stream; save colour.jpg once the mouth opens

Runs in ~/venvs/mp (mediapipe, opencv). Writes the same face.json as detect_mouth.py (LocateAnything), which
deliver.sh falls back to when no face is found here (exit 5).

- The image is rolled 90 deg at the delivery pose; it is rotated upright first, as in detect_mouth.py.
- DEPTH: when the food covers the mouth, the depth at the mouth's pixel is the FOOD. Depth is read at many
  face points instead; points far nearer than the face (the food, ~250 mm nearer) are dropped; the mouth
  takes the lips' own depth where they are visible, else the nose tip's + 5 mm, else the face's.
- mouth_open = inner-lip gap / mouth width.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from settings import cfg
from feeding.camera import COLOUR_URL, CXC, CXD, CYC, CYD, FXC, FXD, FYC, FYD, T_D2C, corrected

MODEL = cfg.path("deployment.models.face_landmarker")
URL = COLOUR_URL
NOSE = [1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 327]
LIPS = [0, 13, 14, 17, 37, 61, 78, 84, 267, 291, 308, 314]
FACE_PTS = list(range(0, 468, 6))
OPEN_RATIO = float(cfg.delivery.open_mouth_ratio)
OPEN_HOLD_S = float(cfg.delivery.open_mouth_hold_s)
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


def wait_open(D: Path, timeout: float, det=None, stream=None) -> int:
    """Watch the colour stream; save the first frame after the mouth has been open OPEN_HOLD_S.
    stream: the resident ColourStream (the face server's) -- only frames arriving after the wait starts are used;
    without one, the stream is opened here (~1.6 s)."""
    own = det is None
    det = det or landmarker()
    try:
        if stream is not None:
            last = [stream.n]

            def read():
                got = stream.frame_after(last[0], 1.0)
                if got is None:
                    return False, None, None
                last[0] = got[0]
                return True, got[2], round(time.time() - got[1], 3)      # the frame's age when read
            return _wait(det, read, D, timeout, open_s=0.0, source="resident stream")
        import cv2
        t_call = time.time()
        cap = cv2.VideoCapture(URL)
        try:
            return _wait(det, lambda: (*cap.read(), round(float(cap.get(cv2.CAP_PROP_POS_MSEC)), 1)), D, timeout,
                         open_s=time.time() - t_call, source="rtsp opened for the wait")
        finally:
            cap.release()
    finally:
        if own:
            det.close()


def _wait(det, read, D, timeout, open_s=0.0, source=""):
    """The open-mouth wait. Records, to tell "declined" from "face not seen" from "the stream is late":
    open_wait.json -- per frame: the time read (s from the wait's start), the stream's own timestamp (ms), the
    opening, whether a face was found; and how long the stream took to open. read() -> (ok, BGR, stamp): the
    stamp is the frame's age when read (resident stream) or the stream's own timestamp. wait_frames/ -- a small upright
    snapshot every 0.5 s, its opening in the name. open_wait_last.jpg -- the last frame, if never opened."""
    t0 = time.time(); since = None; last = -1.0; trace = []; rgb = None; snap_t = -1.0; live_t = -1.0
    live = Path(str(cfg.deployment.runs_dir)).expanduser() / "open_wait_live.txt"   # followed live (tail -F)
    try:
        live.write_text("== wait started %s (%s)\n" % (time.strftime("%H:%M:%S"), D))
    except OSError:
        live = None
    frames_dir = D / "wait_frames"
    frames_dir.mkdir(exist_ok=True)

    def record(result):
        try:
            (D / "open_wait.json").write_text(json.dumps(dict(result=result, threshold=OPEN_RATIO, hold_s=OPEN_HOLD_S,
                                                              stream_open_s=round(open_s, 3), source=source,
                                                              t=[r[0] for r in trace], opening=[r[1] for r in trace],
                                                              face=[r[2] for r in trace],
                                                              frame_stamp=[r[3] for r in trace])))
            if result != "open" and rgb is not None:
                Image.fromarray(rgb).save(D / "open_wait_last.jpg")
        except Exception:  # noqa: BLE001 -- a debug record
            pass
    print("    waiting up to %.0f s for an open mouth (opening > %.2f for %.1f s; %s%s)"
          % (timeout, OPEN_RATIO, OPEN_HOLD_S, source, ", opened in %.1f s" % open_s if open_s else ""))
    while time.time() - t0 < timeout:
        ok, bgr, stamp = read()
        if not ok:
            time.sleep(0.05); continue
        rgb = bgr[:, :, ::-1]
        L, _ = landmarks(det, np.rot90(rgb, -1))            # == PIL ROTATE_270 (90 deg clockwise)
        o = opening(L) if L is not None else 0.0
        now = time.time() - t0
        trace.append((round(now, 3), round(o, 4), L is not None, stamp))
        if live is not None and now - live_t >= 0.25:
            live_t = now
            try:
                with open(live, "a") as f:
                    f.write("%s.%03d  +%5.2f s  opening %.2f  face %s\n" % (time.strftime("%H:%M:%S"), int(time.time() * 1000) % 1000,
                                                                       now, o, "yes" if L is not None else "NO"))
            except OSError:
                pass
        if now - snap_t >= 0.5:
            snap_t = now
            try:
                Image.fromarray(np.rot90(rgb, -1)).resize((180, 320)).save(frames_dir / ("%05.1f_%.2f.jpg" % (now, o)))
            except Exception:  # noqa: BLE001 -- a debug record
                pass
        if time.time() - last > 2.0:
            print("      mouth opening %.2f" % o); last = time.time()
        if o > OPEN_RATIO:
            since = since or time.time()
            if time.time() - since >= OPEN_HOLD_S:
                Image.fromarray(rgb).save(D / "colour.jpg")
                print("    mouth open (%.2f) after %.1f s" % (o, time.time() - t0))
                record("open")
                return 0
        else:
            since = None
    print("    no open mouth in %.0f s" % timeout)
    record("no open mouth")
    return 4


def detect(D: Path, det=None) -> int:
    cx, cy, fx, fy = CXC, CYC, FXC, FYC
    img = Image.open(D / "colour.jpg").convert("RGB"); H = img.size[1]
    up = img.transpose(Image.ROTATE_270)
    dep = corrected(np.load(D / "depth.npy")); yy, xx = np.mgrid[0:dep.shape[0], 0:dep.shape[1]]
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


FACE_PORT = int(cfg.deployment.servers.face_port)


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
    """Keep the face landmarker loaded and the colour stream open (make models): POST /detect and /wait_open, as
    the CLI does; GET /frame -> a fresh JPEG (feeding/grab_frame.py); GET /stream -> the stream's status."""
    import contextlib, io
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from feeding.colour_stream import ColourStream
    stream = ColourStream()
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
            if self.path.startswith("/frame"):
                import cv2
                got = stream.fresh(3.0)
                if got is None:
                    self.send_response(503); self.end_headers(); self.wfile.write(b"no fresh colour frame in 3 s")
                    return
                ok, jpg = cv2.imencode(".jpg", got[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.send_response(200); self.send_header("X-Frame-Age", "%.3f" % (time.time() - got[1]))
                self.end_headers(); self.wfile.write(jpg.tobytes())
                return
            if self.path.startswith("/stream"):
                self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(stream.status()).encode())
                return
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    rc = wait_open(Path(q["run_dir"]), q["timeout"], det, stream) if self.path == "/wait_open" \
                        else detect(Path(q["run_dir"]), det)
                except Exception as e:  # noqa: BLE001
                    print("face server error: %s" % e); rc = 1
            body = json.dumps(dict(rc=rc, log=buf.getvalue())).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)

    HTTPServer(("127.0.0.1", FACE_PORT), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
