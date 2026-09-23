#!/usr/bin/env python3
"""Record fork-tip pivot-calibration poses over the HIGH-LEVEL Kortex API.

Why not the ROS driver: it holds a LOW-LEVEL cyclic session, and jogging with
the gamepad switches the arm's servoing mode out from under it. The driver then
aborts with

    Device error, Error sub type=WRONG_SERVOING_MODE
    description: Wrong servoing mode, must be low level servoing mode

so "gamepad jogs while the driver reports pose" cannot work. The high-level
request/response channel (TCP 10000) reads pose without owning servoing, which
does coexist with the gamepad.

Euler convention: Kortex reports orientation as theta_x/y/z in DEGREES and the
convention is not something to assume. Pivot calibration is exquisitely
sensitive to getting it wrong -- a wrong convention still "converges", just to
nonsense. So every candidate convention is solved and the one with the smallest
residual wins, which is self-validating: only the true convention makes a single
physical point look stationary across varied orientations.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import deque

import numpy as np

from kinova.connection import kinova_connection

OUT = pathlib.Path("/tmp/tcp_poses_hl.json")


def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

CONVENTIONS = {
    "extrinsic xyz (Rz@Ry@Rx)": lambda a, b, c: Rz(c) @ Ry(b) @ Rx(a),
    "intrinsic XYZ (Rx@Ry@Rz)": lambda a, b, c: Rx(a) @ Ry(b) @ Rz(c),
    "extrinsic zyx (Rx@Ry@Rz)": lambda a, b, c: Rx(a) @ Ry(b) @ Rz(c),
    "intrinsic ZYX (Rz@Ry@Rx)": lambda a, b, c: Rz(c) @ Ry(b) @ Rx(a),
    "extrinsic xyz swapped   ": lambda a, b, c: Rz(a) @ Ry(b) @ Rx(c),
}


def solve(poses, conv):
    A, b, Rs = [], [], []
    for p in poses:
        R = conv(*np.radians(p["rpy_deg"]))
        Rs.append(R)
        A.append(np.hstack([R, -np.eye(3)]))
        b.append(-np.asarray(p["pos"], float))
    sol, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    p_tool, p_fixed = sol[:3], sol[3:]
    tips = np.array([R @ p_tool + np.asarray(p["pos"], float)
                     for R, p in zip(Rs, poses)])
    resid = np.linalg.norm(tips - p_fixed, axis=1)
    spread = max(float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b_) - 1) / 2, -1, 1))))
                 for a in Rs for b_ in Rs)
    return p_tool, p_fixed, resid, spread


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=5)
    ap.add_argument("--still-m", type=float, default=0.0015)
    ap.add_argument("--min-sep-deg", type=float, default=20.0)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--require-move", action="store_true", default=True,
                    help="ignore the pose the arm is already parked in")
    a = ap.parse_args()

    poses: list[dict] = []
    buf: deque = deque(maxlen=a.window)
    last_cap = 0.0
    moved = not a.require_move
    origin = None

    with kinova_connection() as conn:
        # What frame is the arm reporting? If a tool transform is configured on
        # the arm, the reported pose ALREADY includes it and the offset we solve
        # is relative to that, not to the flange.
        try:
            from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
            cc = ControlConfigClient(conn._tcp_router)
            tc = cc.GetToolConfiguration()
            t = tc.tool_transform
            print("[info] arm tool transform: xyz=(%.4f, %.4f, %.4f) rpy=(%.2f, %.2f, %.2f) mass=%.3f"
                  % (t.x, t.y, t.z, t.theta_x, t.theta_y, t.theta_z, tc.tool_mass))
            if max(abs(t.x), abs(t.y), abs(t.z)) > 1e-6:
                print("[info] NON-ZERO tool transform -- the solved offset is relative to THAT frame")
        except Exception as e:                                  # noqa: BLE001
            print("[info] could not read tool configuration: %s" % str(e)[:70])

        print("[ready] touch ONE fixed point with the tine tips, hold ~2 s.")
        print("[ready] need %d poses, each >=%.0f deg from the others." % (a.need, a.min_sep_deg))
        sys.stdout.flush()

        period = 1.0 / a.hz
        while True:
            time.sleep(period)
            p = conn.base.GetMeasuredCartesianPose()
            pos = np.array([p.x, p.y, p.z])
            rpy = np.array([p.theta_x, p.theta_y, p.theta_z])
            if origin is None:
                origin = pos.copy()
            if not moved and np.linalg.norm(pos - origin) > 0.03:
                moved = True
                print("[info] arm has moved; now accepting captures")
                sys.stdout.flush()
            buf.append(pos)
            if not moved or len(buf) < buf.maxlen:
                continue
            if time.time() - last_cap < a.cooldown:
                continue
            if np.ptp(np.array(buf), axis=0).max() > a.still_m:
                continue
            R = CONVENTIONS["extrinsic xyz (Rz@Ry@Rx)"](*np.radians(rpy))
            novel = True
            for q in poses:
                Rq = CONVENTIONS["extrinsic xyz (Rz@Ry@Rx)"](*np.radians(q["rpy_deg"]))
                ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ Rq) - 1) / 2, -1, 1)))
                if ang < a.min_sep_deg:
                    novel = False
                    break
            if not novel:
                continue
            poses.append(dict(pos=pos.tolist(), rpy_deg=rpy.tolist()))
            last_cap = time.time()
            OUT.write_text(json.dumps(poses, indent=2))
            print("[pose %d/%d] pos %s  rpy %s"
                  % (len(poses), a.need, np.round(pos, 4), np.round(rpy, 1)))
            sys.stdout.flush()

            if len(poses) >= a.need:
                print("")
                print("[RESULT] solving under each candidate Euler convention;")
                print("[RESULT] the true one makes one physical point look stationary.")
                best = None
                for name, conv in CONVENTIONS.items():
                    pt, pf, res, spread = solve(poses, conv)
                    print("[RESULT]   %-26s residual mean %6.2f mm  max %6.2f mm  |p|=%6.1f mm"
                          % (name, res.mean() * 1000, res.max() * 1000,
                             np.linalg.norm(pt) * 1000))
                    if best is None or res.max() < best[1]:
                        best = (name, res.max(), pt, pf, res, spread)
                name, _, pt, pf, res, spread = best
                print("")
                print("[RESULT] best convention: %s" % name.strip())
                print("[RESULT] fork tip vs the reported tool frame:")
                print("[RESULT]   x %+.4f  y %+.4f  z %+.4f  m   |p| = %.1f mm"
                      % (*pt, np.linalg.norm(pt) * 1000))
                print("[RESULT]   residual mean %.2f mm  max %.2f mm   spread %.0f deg"
                      % (res.mean() * 1000, res.max() * 1000, spread))
                if spread < 45:
                    print("[RESULT]   *** spread too small -- not observable, take more varied poses")
                elif res.max() > 3.0:
                    print("[RESULT]   *** residual too large to trust for 15 mm food")
                else:
                    print("[RESULT]   usable.")
                sys.stdout.flush()
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
