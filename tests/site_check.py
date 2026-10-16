#!/usr/bin/env python3
"""Is a site's geometry workable? The whole bite, planned and run against REAL kinematics, without the arm.

    .venv/bin/python tests/site_check.py                      # the 2026-09-30 site (tests/sites/site_20260930.yaml)
    .venv/bin/python tests/site_check.py tests/sites/<x>.yaml [--render] [--dump motions.json]
        [--planner stub|curobo|server]                        # stub (default): tests/stub_planner.py, no GPU;
                                                              # curobo: start the real planner server (a CUDA box);
                                                              # server: one already running on --planner-port, whose
                                                              # runs dir is SITE_RUNS (it re-reads the world first)
        [--camera <profile>] [--scene-camera <profile>|none]  # another wrist / scene camera than the site's
        [--user-offset dx,dy,dz]                              # the user's mouth moved by this (m)
        [--head-sway M]                                       # the head sways +-M sideways while tracked
        [--face-lost-at S]                                    # ... and is lost S s into the tracking
        [--track-hz F]                                        # the tracker yields a new face only F times a second

The arm stages (sequence / acquire / deliver) run as on the robot -- their own checks, their own refusals --
against tests/mujoco_kortex.py (the Gen3's kinematics: IK / FK anywhere, unlike fake_kortex, which only knows the
recorded deployment) and a site file: the cameras, where the bowl and table are (as the scene camera would
report them), the overlook height, where the user's mouth is. Prints each step's outcome and the line that
says why; exit 1 if the bite does not go through. --render: a MuJoCo twin frame of the overlook and the
presented pose (out/sim/site_check_*.png).

What it cannot tell: anything about contact, the food, or the cameras' images (the depth it feeds the stages is
synthetic, consistent with the site's numbers). It is the geometry: reachability, joint limits, every path check.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fake_kortex as fk  # noqa: E402

SITE = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") else \
    ROOT / "tests" / "sites" / "site_20260930.yaml"
RENDER = "--render" in sys.argv
site = yaml.safe_load(SITE.read_text())

fk.install_clock()
WORK = Path(os.environ["SITE_WORK"]) if os.environ.get("SITE_WORK") else Path(tempfile.mkdtemp(prefix="site_check_"))
STATE, RUNS = WORK / "state", Path(os.environ.get("SITE_RUNS", WORK / "runs"))
shutil.rmtree(STATE, ignore_errors=True)
STATE.mkdir(parents=True); RUNS.mkdir(parents=True, exist_ok=True)
for stale in ("mouth_state.json",):                 # a reused runs dir: no mouth from the last bite
    (RUNS / stale).unlink(missing_ok=True)
os.environ["RAMMP_NO_LOCAL"] = "1"
os.environ["RAMMP_STATE_DIR"] = str(STATE)


def _arg(name: str, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if _arg("--user-offset"):                        # the user elsewhere than the site's mouth (sat back, leaning, ...)
    site["user"]["mouth_m"] = [float(m) + float(o) for m, o in zip(site["user"]["mouth_m"], _arg("--user-offset").split(","))]
profiles = dict(site.get("profiles", {}))
profiles["camera"] = _arg("--camera", profiles.get("camera", "kinova_wrist"))
profiles["scene_camera"] = _arg("--scene-camera", profiles.get("scene_camera", ""))
if profiles["scene_camera"] == "none":
    profiles["scene_camera"] = ""
for k, v in profiles.items():
    os.environ["RAMMP_" + k.upper()] = str(v)
PLANNER = _arg("--planner", "stub")
sets = dict(site.get("set", {}), **{"deployment.runs_dir": str(RUNS),
                                     "delivery.track.watchdog": False})   # no second session on the fake arm
if _arg("--planner-port"):
    sets["deployment.servers.planner_port"] = int(_arg("--planner-port"))
if not profiles["scene_camera"]:                 # no scene camera: the site's workspace is the config's
    sets.update({"workspace.table_height_m": site["workspace"]["table_z"],
                 "workspace.bowl_centre_m": site["workspace"]["bowl_xy"],
                 "workspace.bowl_height_m": round(site["workspace"]["bowl_top_z"] - site["workspace"]["table_z"], 4)})
os.environ["RAMMP_SET"] = ",".join("%s=%s" % (k, json.dumps(v) if isinstance(v, list) else v) for k, v in sets.items())

from settings import cfg  # noqa: E402
import mujoco_kortex as mk  # noqa: E402

ws = site["workspace"]
(RUNS / "scene_state.json").write_text(json.dumps(dict(t=fk.clock.time(), bowl_xy=ws["bowl_xy"],
                                                       table_z=ws["table_z"], bowl_top_z=ws["bowl_top_z"])))
HOME = json.loads((ROOT / "calibration" / "home_pose.json").read_text())
ARM = mk.KinematicArm(tool_z=float(cfg.deployment.arm.tool.transform_m[2]))


@contextlib.contextmanager
def _conn():
    yield mk.KinematicConnection(ARM)


def depth_frame() -> np.ndarray:
    """What the wrist camera would see from wherever the arm is now: the table, the bowl floor, one food piece
    under the utensil's vertical line (consistent with the site's heights)."""
    from kinova.frames import R_from_kortex
    from feeding.camera import CAM_IN_TOOL
    c, d = cfg.camera.colour, cfg.camera.depth
    p = ARM.pose
    cam_z = p[2] + (R_from_kortex(*p[3:]) @ CAM_IN_TOOL)[2]
    floor = cam_z - (ws["table_z"] + 0.007)
    top = cam_z - (ws["table_z"] + 0.007 + float(site["food"]["height_m"]))
    vp, tip, zt = np.array(c.vertical_vp, float), np.array(cfg.utensil.tip_px, float), float(cfg.utensil.camera_to_tip_m)
    from feeding.camera import colour_to_depth
    px = colour_to_depth(*(vp + (tip - vp) * zt / top), top)   # the colour pixel, in the depth frame
    yy, xx = np.mgrid[0:int(d.height), 0:int(d.width)]
    out = np.full((int(d.height), int(d.width)), 1000 * floor)
    r = float(site["food"]["size_m"]) / 2 * float(d.fx) / top
    out[np.hypot(xx - px[0], yy - px[1]) < r] = 1000 * top
    return out.astype(np.uint16)


import urllib.request as _ur  # noqa: E402
_REAL_URLOPEN = _ur.urlopen


def fake_urlopen(url, timeout=None):
    target = url if isinstance(url, str) else url.full_url
    if any(":%d/" % int(p) in target for p in (cfg.deployment.servers.planner_port, cfg.deployment.servers.face_port)):
        return _REAL_URLOPEN(url, timeout=timeout)                   # the planner server, the face tracker: real
    b = io.BytesIO()
    np.save(b, depth_frame())
    return contextlib.closing(io.BytesIO(b.getvalue()))


RESULTS: list[tuple[str, int, str]] = []


def run(module: str, args: list[str], note: str = "") -> int:
    mod = __import__(module, fromlist=["main"])
    ARM.step = "%s %s" % (module.split(".")[-1], " ".join(a for a in args if not a.startswith(str(WORK))))
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", [module] + args), mock.patch.object(mod, "kinova_connection", _conn), \
            mock.patch("urllib.request.urlopen", fake_urlopen), contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(err):
        try:
            rc = mod.main()
        except SystemExit as e:
            rc = e.code
        except Exception as e:  # noqa: BLE001 -- a crash is an outcome here
            rc, err = 99, io.StringIO("CRASH: %r" % e)
    lines = (out.getvalue() + err.getvalue()).splitlines()
    why = next((l.strip() for l in reversed(lines) if "STOPPED" in l or "REFUSED" in l or "CRASH" in l), "")
    if not why and lines:
        why = next((l.strip() for l in reversed(lines) if l.strip() and "Deprecat" not in l), "")
    label = "%s %s" % (module.split(".")[-1], " ".join(a for a in args if not a.startswith(str(WORK))))
    RESULTS.append((label + ("  (%s)" % note if note else ""), rc, why))
    print("  %-46s rc=%-3s %s" % (label[:46], rc, why[:150]), file=sys.__stderr__)
    if os.environ.get("SITE_VERBOSE"):
        print("\n".join("      " + l for l in lines), file=sys.__stderr__)
    return rc


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj))


def face_tracker():
    """A stand-in for the face server's /track: the mouth as the wrist camera would see it from the arm's pose now
    -- from depth within 0.75 m, else from the face's size, 15% long -- the head swaying by --head-sway."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from kinova.frames import R_from_kortex
    from feeding.camera import cam_to_tool
    M = np.array([cam_to_tool(e) - cam_to_tool([0.0, 0.0, 0.0]) for e in np.eye(3)]).T   # camera -> tool, linear
    sway, m0 = float(_arg("--head-sway", 0.0)), np.array(site["user"]["mouth_m"], float)
    lost_at, t_start = float(_arg("--face-lost-at", 1e9)), [None]
    period, last = 1.0 / float(_arg("--track-hz", 1e9)), {}      # a slow tracker: its last frame, re-served

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            p = ARM.pose
            R = R_from_kortex(*p[3:])
            t = fk.clock.time()
            t_start[0] = t_start[0] or t
            if last and t - last["t"] < period:                 # no new frame yet: the last one again
                body = json.dumps(last).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
                return
            if t - t_start[0] > lost_at:
                body = json.dumps(dict(t=t, face=False)).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
                return
            mouth = m0 + np.array([0.0, sway * np.sin(2 * np.pi * t / 4.0), 0.0])
            c = np.linalg.solve(M, R.T @ (mouth - np.array(p[:3])) - cam_to_tool([0.0, 0.0, 0.0]))
            src = "depth" if np.linalg.norm(c) < 0.75 else "size"
            if src == "size":
                c = c * 1.15
            last.update(t=t, face=True, cam_xyz=[float(v) for v in c], source=src, opening=0.4)
            body = json.dumps(last).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
        do_GET = do_POST = _reply

    srv = ThreadingHTTPServer(("127.0.0.1", int(cfg.deployment.servers.face_port)), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def face_fixture(name: str) -> list[str]:
    """A face as the delivery's look would report it, the mouth where the site says, seen from the arm's pose now."""
    from kinova.frames import R_from_kortex
    from feeding.camera import CAM_IN_TOOL
    p = ARM.pose
    R = R_from_kortex(*p[3:])
    cam = np.array(p[:3]) + R @ CAM_IN_TOOL
    m = np.array(site["user"]["mouth_m"], float)
    v = R.T @ (m - cam)                                         # the tool frame, from the camera
    cam_xyz = [-v[0], -v[1], v[2]]                               # the camera's optical frame (180 deg about z)
    d = WORK / name
    d.mkdir(exist_ok=True)
    (d / "colour.jpg").write_bytes(b"jpg")
    f = dict(nose=dict(box_upright=[330, 475, 390, 525], depth_m=cam_xyz[2] - 0.01),
             mouth=dict(box_upright=[320, 530, 400, 570], depth_m=cam_xyz[2], cam_xyz=cam_xyz, prompt="mouth"),
             depth_from_size=bool(cam_xyz[2] > 1.0))            # past the wrist depth's range: from the face's size
    (d / "face.json").write_text(json.dumps(f))
    now = fk.clock.time()
    os.utime(d / "colour.jpg", (now - 2, now - 2))
    os.utime(d / "face.json", (now - 1, now - 1))
    return ["--face", str(d / "face.json"), "--colour", str(d / "colour.jpg")]


def render(tag: str) -> None:
    if not RENDER:
        return
    try:
        import mujoco
        from PIL import Image
        from sim.scene import Food, Layout, arm_qpos_adr, build, new_data
        lay = Layout.from_config(mouth=tuple(site["user"]["mouth_m"]), bowl_xy=tuple(ws["bowl_xy"]),
                                 foods=[Food(tuple(ws["bowl_xy"]))], width=1280, height=720)
        lay.table_z = ws["table_z"]
        m = build(lay); d = new_data(m)
        d.qpos[arm_qpos_adr(m)] = np.radians(ARM.q); mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 720, 1280); opt = mujoco.MjvOption(); opt.geomgroup[:] = 0
        opt.geomgroup[1] = opt.geomgroup[2] = 1
        cam = mujoco.MjvCamera(); cam.lookat[:] = (0.05, 0.1, 0.25); cam.distance, cam.azimuth, cam.elevation = 1.8, 150, -20
        r.update_scene(d, camera=cam, scene_option=opt)
        out = ROOT / "out" / "sim" / ("site_check_%s.png" % tag); out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(r.render()).save(out); print("    rendered %s" % out, file=sys.__stderr__)
    except Exception as e:  # noqa: BLE001
        print("    (no render: %s)" % e, file=sys.__stderr__)


SEQ, ACQ, DEL = "kinova.scripts.sequence", "kinova.scripts.acquire", "kinova.scripts.deliver"


def stub_via(start, goal, carrying):
    """The stub planner's corners: up from the start, across, down onto the goal -- each corner the start's / goal's
    tool position raised to above both (and the table's clearance), the orientation held, by this arm's IK."""
    a, b = ARM.fk_pose(start), ARM.fk_pose(goal)
    top = max(a[2], b[2], ws["table_z"] + 0.25) + 0.05

    def ladder(p, q, k=6):                       # straight up in steps (a joint line between far poses bows)
        out = []
        for z in np.linspace(p[2], top, k + 1)[1:]:
            q = ARM.solve([p[0], p[1], z] + p[3:], q)
            if q is None:
                break
            out.append(q)
        return out
    up, down = ladder(a, start), ladder(b, goal)[::-1]
    across = []
    if up and down:                              # over at the top: the joint line's poses, held at that height
        qa, qb = np.array(up[-1]), np.array(down[0])
        for s in np.linspace(0, 1, 12)[1:-1]:
            qs = qa + (qb - qa) * s
            p = ARM.fk_pose(list(qs))
            q = ARM.solve([p[0], p[1], top] + p[3:], list(qs))
            if q is not None:
                across.append(q)
    return up + across + down


def start_planner():
    """--planner curobo: the planner server, on this run's config and runs dir (a CUDA box: the curobo-eval
    container); stub: tests/stub_planner.py in a thread. The bite waits until it answers."""
    if PLANNER == "stub":
        import stub_planner
        stub_planner.serve(int(cfg.deployment.servers.planner_port), stub_via)
        return None
    if PLANNER == "server":                       # already warm: just this scenario's world
        from kinova import curobo_client
        curobo_client.refresh_world()
        return None
    import subprocess
    import urllib.request
    log = open(WORK / "planner_server.log", "w")
    proc = subprocess.Popen([sys.executable, "-m", "kinova.planner_server"], cwd=ROOT, env=dict(os.environ),
                            stdout=log, stderr=subprocess.STDOUT)
    url = "http://127.0.0.1:%d/ping" % int(cfg.deployment.servers.planner_port)
    for _ in range(600):
        try:
            if b"ok" in urllib.request.urlopen(url, timeout=2).read():
                print("  planner server warm (cuRobo)", file=sys.__stderr__)
                return proc
        except OSError:
            if proc.poll() is not None:
                raise SystemExit("the planner server died: %s" % (WORK / "planner_server.log"))
        import threading
        threading.Event().wait(1.0)              # REAL time: time.sleep is the fake controller's simulated clock here
    raise SystemExit("the planner server did not come up in 600 s")


def main() -> int:
    print("site check: %s  (planner: %s; camera %s, scene camera %s)" % (SITE.name, PLANNER, profiles["camera"],
          profiles["scene_camera"] or "none"), file=sys.__stderr__)
    planner = start_planner()
    print("  bowl %s, table z %.3f, bowl top z %.3f, overlook camera %.2f m above the table, mouth %s" % (
        ws["bowl_xy"], ws["table_z"], ws["bowl_top_z"], float(cfg.workspace.overlook_camera_height_m),
        site["user"]["mouth_m"]), file=sys.__stderr__)
    ARM.place(HOME["joints_deg"])
    ok = run(SEQ, ["detect", "--plan"]) == 0 and run(SEQ, ["detect"]) == 0
    render("overlook")
    if ok:
        from kinova.frames import R_from_kortex
        from kinova.motion import tip_in_tool
        p = ARM.pose
        tip = np.array(p[:3]) + R_from_kortex(*p[3:]) @ tip_in_tool()
        goal = [p[0] + ws["bowl_xy"][0] - tip[0], p[1] + ws["bowl_xy"][1] - tip[1], p[2]]   # the tip over the bowl
        write(STATE / "target.json", dict(tool_goal=goal))
        ok = run(ACQ, ["pick"], "over + plunge + scoop") == 0
    if ok:
        ok = run(DEL, ["swing", "--validate-only"]) == 0 and run(DEL, ["swing"]) == 0
    if ok:
        if cfg.features.track_approach:
            srv = face_tracker()
            ok = run(DEL, ["track"], "tracked approach") == 0
            srv.shutdown()
        else:
            ok = run(DEL, ["approach"] + face_fixture("face")) == 0
        render("presented")
    if ok:
        t0 = fk.clock.time()
        ARM.force = lambda t: [0.0, 0.0, 3.0 if t0 + 1.5 < t < t0 + 1.9 else 0.0]   # a bite, over bite_n
        ok = run(DEL, ["bite"]) == 0
        ARM.force = lambda t: [0.0, 0.0, 0.0]
    if ok:
        ok = run(DEL, ["retract"]) == 0
    if ok:
        ok = run(SEQ, ["home"]) == 0
    print("%s  (%d IK calls; record %s)" % ("THE BITE GOES THROUGH" if ok else "STOPPED: see the step above",
                                            ARM.ik_calls, WORK), file=sys.__stderr__)
    if planner is not None:
        planner.terminate()
    if "--dump" in sys.argv:                   # every executed move, for sim/compare_planners.py
        out = Path(sys.argv[sys.argv.index("--dump") + 1])
        from kinova.motion import tip_in_tool
        out.write_text(json.dumps(dict(site=str(SITE), workspace=ws, user=site["user"], tip_in_tool=list(tip_in_tool()),
                                       motions=ARM.motions), indent=1))
        print("motions -> %s (%d)" % (out, len(ARM.motions)), file=sys.__stderr__)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
