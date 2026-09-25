#!/usr/bin/env python3
"""How far the wrist camera's colour picture lags the arm: time a known motion in both.

    ~/venvs/mp/bin/python -m feeding.camera_latency [--deg 5] [--dps 10]     # ON THE JETSON, arm at feeding_start

The arm turns ONLY its wrist (J7) by --deg and back (the utensil stays pointing down; the fork tip moves a few
mm) while this samples frames from the resident colour stream (the face server's GET /frame, each with its
arrival time) and the driver container logs J7 against the same clock. The camera's delay = when the image
starts (and stops) changing minus when the joint does. Everything goes to runs_dir/<time>_camera_latency/.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from settings import cfg

ARM = r'''
import json, threading, time
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import goto, joints, fold, MoveError
DEG, DPS = %f, %f
home = json.load(open("calibration/home_pose.json"))["joints_deg"]
with kinova_connection() as c:
    q0 = joints(c)
    if max(abs(fold(a - b)) for a, b in zip(q0, home)) > 1.0:
        raise SystemExit("not at feeding_start -- not moving")
    log, stop = [], []
    def rec():
        while not stop:
            log.append((time.time(), fold(c.base_cyclic.RefreshFeedback().actuators[6].position)))
            time.sleep(0.005)
    th = threading.Thread(target=rec, daemon=True); th.start()
    time.sleep(0.5)
    q1 = list(q0); q1[6] = fold(q0[6] + DEG)
    try:
        goto(c, q1, speed=DPS, label="J7 +%%.0f deg" %% DEG, verbose=False)
        time.sleep(1.0)
        goto(c, q0, speed=DPS, label="J7 back", verbose=False)
    except MoveError as e:
        print("STOPPED", e)
    time.sleep(0.5); stop.append(1); th.join()
    print("JOINT", json.dumps(log))
'''


def sample(stop: list, frames: list) -> None:
    url = "http://127.0.0.1:%d/frame" % int(cfg.deployment.servers.face_port)
    while not stop:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data, age = r.read(), float(r.headers.get("X-Frame-Age", "0"))
            t = time.time() - age
            g = np.asarray(Image.open(io.BytesIO(data)).convert("L").resize((160, 90)), dtype=np.float32)
            frames.append((t, g, data))
        except Exception:  # noqa: BLE001
            time.sleep(0.05)


def onset(t: np.ndarray, x: np.ndarray, base_until: float) -> tuple[float | None, float | None]:
    """First and last time x rises above its baseline (before base_until) by 6 sigma (at least a floor)."""
    b = x[t < base_until]
    thr = (b.mean() + max(6 * b.std(), 0.5)) if b.size > 3 else x.max() / 2
    above = np.nonzero(x > thr)[0]
    return (float(t[above[0]]), float(t[above[-1]])) if above.size else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, default=5.0)
    ap.add_argument("--dps", type=float, default=10.0)
    a = ap.parse_args()
    out = Path(str(cfg.deployment.runs_dir)).expanduser() / (time.strftime("%Y%m%d_%H%M%S") + "_camera_latency")
    out.mkdir(parents=True)
    stop, frames = [], []
    th = threading.Thread(target=sample, args=(stop, frames), daemon=True)
    th.start()
    time.sleep(1.5)                                          # a baseline of still frames
    t_cmd = time.time()
    r = subprocess.run(["docker", "exec", "-e", "PYTHONPATH=/work", "-w", "/work", str(cfg.deployment.containers.driver),
                        "python3", "-c", ARM % (a.deg, a.dps)], capture_output=True, text=True)
    time.sleep(3.0)
    stop.append(1)
    th.join()
    line = [l for l in r.stdout.splitlines() if l.startswith("JOINT ")]
    if not line:
        print("arm: %s %s" % (r.stdout[-500:], r.stderr[-500:]))
        return 1
    J = np.array(json.loads(line[0][6:]))
    jt, jq = J[:, 0], np.abs(J[:, 1] - J[0, 1])
    moving = np.nonzero(jq > 0.05)[0]
    j_on, j_off = float(jt[moving[0]]), float(jt[moving[-1]])
    ft = np.array([f[0] for f in frames])
    diff = np.array([0.0] + [float(np.abs(frames[i][1] - frames[i - 1][1]).mean()) for i in range(1, len(frames))])
    i_on, i_off = onset(ft, diff, j_on - 0.2)
    for k, (t, _, data) in enumerate(frames):                 # keep the frames, named by their time from the joint's start
        (out / ("%+07.2f.jpg" % (t - j_on))).write_bytes(data)
    res = dict(frames=len(frames), fps=round((len(ft) - 1) / (ft[-1] - ft[0]), 1) if len(ft) > 1 else None,
               joint_start=j_on - t_cmd, joint_end=j_off - t_cmd,
               image_start=None if i_on is None else i_on - t_cmd, image_end=None if i_off is None else i_off - t_cmd,
               delay_start_s=None if i_on is None else round(i_on - j_on, 3),
               delay_end_s=None if i_off is None else round(i_off - j_off, 3))
    (out / "latency.json").write_text(json.dumps(dict(res, joint=J.tolist(), frame_t=ft.tolist(), frame_diff=diff.tolist()), indent=1))
    print("frames sampled: %d (%.1f/s)" % (len(frames), res["fps"] or 0))
    print("joint J7 moved     from %+.2f s to %+.2f s (after the command)" % (res["joint_start"], res["joint_end"]))
    if i_on is None:
        print("the image never changed measurably -- nothing to time")
    else:
        print("the image changed  from %+.2f s to %+.2f s" % (res["image_start"], res["image_end"]))
        print("CAMERA DELAY: %.2f s at the start, %.2f s at the end" % (res["delay_start_s"], res["delay_end_s"]))
    print("record: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
