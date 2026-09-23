#!/usr/bin/env python3
"""Log arm state during a teleop demo, over the HIGH-LEVEL API.

Architecture Sec. 10 step 2: instrument the gamepad path so trials you are
running anyway become data. This is the state half; video is captured alongside
by gstreamer and aligned on wall-clock time.

Runs on the high-level channel deliberately. The ROS driver owns a LOW-LEVEL
cyclic session and the gamepad switches servoing mode out from under it -- the
driver aborts with WRONG_SERVOING_MODE. `BaseCyclic.RefreshFeedback()` over TCP
reads the same per-actuator telemetry without owning servoing, so it coexists
with whoever is driving.

The valuable column is TORQUE. A demo of the tines entering fruit gives the
force signature that the skewer needs in order to stop on contact rather than on
a commanded depth -- the threshold in feeding/scan_node.py is currently a guess
and this is what replaces it with a measurement.

    python -m kinova.scripts.record_demo --out /tmp/demo1 --seconds 180
"""
from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys
import time

from kinova.connection import kinova_connection

STOP = False


def _sigint(*_):
    global STOP
    STOP = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/demo")
    ap.add_argument("--hz", type=float, default=40.0)
    ap.add_argument("--seconds", type=float, default=300.0)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "state.jsonl"

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    n = 0
    t_start = time.time()
    period = 1.0 / a.hz
    with kinova_connection() as conn, path.open("w") as fh:
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        bc = BaseCyclicClient(conn._tcp_router)
        print("[rec] logging to %s at %.0f Hz; Ctrl-C or %.0f s to stop"
              % (path, a.hz, a.seconds))
        sys.stdout.flush()
        last_report = 0.0
        while not STOP and time.time() - t_start < a.seconds:
            t0 = time.time()
            try:
                fb = bc.RefreshFeedback()
            except Exception as e:                              # noqa: BLE001
                print("[rec] feedback error: %s" % str(e)[:70])
                sys.stdout.flush()
                time.sleep(0.2)
                continue
            rec = dict(
                t=time.time(),
                joints=[dict(pos=act.position, vel=act.velocity,
                             torque=act.torque, current=act.current_motor)
                        for act in fb.actuators],
                tool=dict(x=fb.base.tool_pose_x, y=fb.base.tool_pose_y,
                          z=fb.base.tool_pose_z,
                          tx=fb.base.tool_pose_theta_x,
                          ty=fb.base.tool_pose_theta_y,
                          tz=fb.base.tool_pose_theta_z),
                wrench=dict(fx=fb.base.tool_external_wrench_force_x,
                            fy=fb.base.tool_external_wrench_force_y,
                            fz=fb.base.tool_external_wrench_force_z,
                            tx=fb.base.tool_external_wrench_torque_x,
                            ty=fb.base.tool_external_wrench_torque_y,
                            tz=fb.base.tool_external_wrench_torque_z),
            )
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
            now = time.time()
            if now - last_report > 5.0:
                tq = [round(x.torque, 2) for x in fb.actuators]
                print("[rec] %5.1fs  %6d samples  tool z=%.3f  torque %s"
                      % (now - t_start, n, fb.base.tool_pose_z, tq))
                sys.stdout.flush()
                last_report = now
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    print("[rec] stopped: %d samples over %.1f s -> %s"
          % (n, time.time() - t_start, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
