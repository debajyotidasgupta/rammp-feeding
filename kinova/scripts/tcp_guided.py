#!/usr/bin/env python3
"""Guided fork-tip calibration: the script rotates, the operator translates.

Pivot calibration needs the tine tips on ONE point across several wrist
ORIENTATIONS. The gamepad here only does translation, so the operator cannot
supply the orientation spread -- but the high-level API can.

Both sides therefore run high-level. The ROS driver is deliberately NOT used:
it holds a low-level cyclic session and the gamepad switches servoing mode out
from under it (WRONG_SERVOING_MODE, observed). `PlayCartesianTrajectory` is a
high-level command and coexists with gamepad jogging.

Cycle, repeated once per pose:
  1. command an orientation, holding position
  2. wait for the operator to jog the tips onto the reference point
  3. capture when the arm goes quiet
Orientation is unchanged by translation, so the commanded attitude survives
while the operator moves.
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

# NOT under /work: that is the synced repo, and sync.sh runs with
# --delete, which silently removed an earlier capture file.
OUT = pathlib.Path("/work_out/tcp_poses.json")

# Offsets applied to the STARTING orientation, in degrees (theta_x, theta_y, theta_z).
# Kept modest so the wrist stays clear of the table while still giving spread.
# Rotations about Y ONLY.
#
# The operator's gamepad gives horizontal + vertical translation (X and Z) and
# no lateral Y. A rotation about X swings the tines sideways in Y, which they
# then cannot correct -- so they bring the fork down vertically instead and the
# tips land on the mat somewhere OTHER than the reference corner. Measured
# exactly that: pose 1 -> pose 2 moved dz=+28.7 mm with dx=dy=0 after a +25 deg
# X rotation. Touching a surface at varied angles is the ill-conditioned case
# (13-41 mm error in synthetic tests), so those poses are worthless.
#
# Rotating about Y displaces the tips in X and Z, both of which the operator
# CAN correct. The cost is that p_y becomes unobservable -- rotations about a
# single axis never constrain the offset component along that axis. The fork is
# symmetric so p_y ~ 0, and the number that matters is p_z, the length.
ATTITUDES = [
    (0.0, 0.0, 0.0),
    (0.0, 22.0, 0.0),
    (0.0, -22.0, 0.0),
    (0.0, 35.0, 0.0),
    (0.0, -35.0, 0.0),
    (0.0, 14.0, 0.0),
]


def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def R_of(rpy_deg): a, b, c = np.radians(rpy_deg); return Rz(c) @ Ry(b) @ Rx(a)


def angle(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def solve(poses):
    A, b, Rs = [], [], []
    for p in poses:
        R = R_of(p["rpy_deg"]); Rs.append(R)
        A.append(np.hstack([R, -np.eye(3)])); b.append(-np.asarray(p["pos"], float))
    sol, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    pt, pf = sol[:3], sol[3:]
    tips = np.array([R @ pt + np.asarray(p["pos"], float) for R, p in zip(Rs, poses)])
    return pt, pf, np.linalg.norm(tips - pf, axis=1), max(angle(a, b) for a in Rs for b in Rs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=5)
    ap.add_argument("--still-m", type=float, default=0.0015)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=15.0, help="deg/s for the rotations")
    ap.add_argument("--lift", type=float, default=0.040,
                    help="metres to retract before each rotation")
    ap.add_argument("--min-move", type=float, default=0.015,
                    help="metres the operator must move before a capture counts")
    a = ap.parse_args()

    # Resume: poses already captured by hand are just as good as ours.
    poses: list[dict] = json.loads(OUT.read_text()) if OUT.exists() else []
    if poses:
        print("[info] resuming with %d pose(s) already captured" % len(poses))
    with kinova_connection() as conn:
        from kortex_api.autogen.messages import Base_pb2
        p0 = conn.base.GetMeasuredCartesianPose()
        # Anchor the rotations to the attitude of the FIRST GOOD TOUCH, not to
        # wherever the arm happens to be sitting. A previous aborted run left it
        # 25 deg off in theta_x; starting from there would displace the tines
        # laterally before the operator ever touched, in the one axis they
        # cannot correct -- silently reproducing the failure we just diagnosed.
        if poses:
            base_rpy = np.array(poses[0]["rpy_deg"], float)
            print("[info] anchoring to pose 1's attitude rpy=(%.1f, %.1f, %.1f)"
                  % tuple(base_rpy))
            cur = conn.base.GetMeasuredCartesianPose()
            if abs(cur.theta_x - base_rpy[0]) > 1.0 or abs(cur.theta_z - base_rpy[2]) > 1.0:
                print("[info] restoring that attitude first (arm is %.1f deg off in theta_x)"
                      % (cur.theta_x - base_rpy[0]))
                sys.stdout.flush()
                back = Base_pb2.ConstrainedPose()
                back.target_pose.x = cur.x; back.target_pose.y = cur.y
                back.target_pose.z = cur.z
                back.target_pose.theta_x = base_rpy[0]
                back.target_pose.theta_y = cur.theta_y
                back.target_pose.theta_z = base_rpy[2]
                back.constraint.speed.translation = 0.05
                back.constraint.speed.orientation = a.speed
                for _ in range(40):
                    try:
                        conn.base.PlayCartesianTrajectory(back); break
                    except Exception as e:                    # noqa: BLE001
                        if "SESSION_NOT_IN_CONTROL" not in str(e):
                            print("[warn] restore failed: %s" % str(e)[:100]); break
                        time.sleep(3.0)
                time.sleep(5.0)
                print("[info] attitude restored; re-touch the corner for the NEXT pose")
                sys.stdout.flush()
        else:
            base_rpy = np.array([p0.theta_x, p0.theta_y, p0.theta_z])
        print("[info] start pose xyz=(%.4f, %.4f, %.4f) rpy=(%.1f, %.1f, %.1f)"
              % (p0.x, p0.y, p0.z, *base_rpy))
        sys.stdout.flush()

        for k, off in enumerate(ATTITUDES[len(poses):a.need]):
            # LIFT CLEAR BEFORE ROTATING. Rotating about the tool origin while
            # the tines are resting on a surface drives them INTO it -- the
            # clearance closes as the fork swings, and the fork bends. Observed
            # on the real arm: the tines started folding. Never rotate in
            # contact; retract, rotate in free space, then come back down.
            cur = conn.base.GetMeasuredCartesianPose()
            lift = Base_pb2.ConstrainedPose()
            lift.target_pose.x = cur.x
            lift.target_pose.y = cur.y
            lift.target_pose.z = cur.z + a.lift
            lift.target_pose.theta_x = cur.theta_x
            lift.target_pose.theta_y = cur.theta_y
            lift.target_pose.theta_z = cur.theta_z
            lift.constraint.speed.translation = 0.03
            lift.constraint.speed.orientation = a.speed
            print("[lift] retracting %.0f mm before rotating (never rotate in contact)"
                  % (a.lift * 1000))
            sys.stdout.flush()
            for _ in range(40):
                try:
                    conn.base.PlayCartesianTrajectory(lift); break
                except Exception as e:                        # noqa: BLE001
                    if "SESSION_NOT_IN_CONTROL" not in str(e):
                        print("[warn] lift failed: %s" % str(e)[:100]); break
                    time.sleep(3.0)
            time.sleep(a.lift / 0.03 + 2.0)

            cur = conn.base.GetMeasuredCartesianPose()
            tgt = Base_pb2.ConstrainedPose()
            tgt.target_pose.x = cur.x
            tgt.target_pose.y = cur.y
            tgt.target_pose.z = cur.z
            tgt.target_pose.theta_x = base_rpy[0] + off[0]
            tgt.target_pose.theta_y = base_rpy[1] + off[1]
            tgt.target_pose.theta_z = base_rpy[2] + off[2]
            # CartesianTrajectoryConstraint has a `speed` submessage directly,
            # inside a oneof named "type" -- not a repeated `oneof_type` list.
            tgt.constraint.speed.translation = 0.05
            tgt.constraint.speed.orientation = a.speed
            print("[rotate %d/%d] to rpy offset (%+.0f, %+.0f, %+.0f) -- holding position"
                  % (k + 1, a.need, *off))
            sys.stdout.flush()
            # Only ONE session may command the arm. While the operator is on the
            # gamepad, ours is refused with SESSION_NOT_IN_CONTROL -- which is
            # not an error so much as "wait your turn". Retry until they let go.
            ok = False
            for attempt in range(40):
                try:
                    conn.base.PlayCartesianTrajectory(tgt)
                    ok = True
                    break
                except Exception as e:                       # noqa: BLE001
                    if "SESSION_NOT_IN_CONTROL" not in str(e):
                        print("[warn] rotate failed: %s" % str(e)[:110])
                        sys.stdout.flush()
                        break
                    if attempt == 0:
                        print("[info] gamepad still has control -- let go and I will rotate")
                        sys.stdout.flush()
                    time.sleep(3.0)
            if not ok:
                print("[warn] could not command rotation %d; skipping" % (k + 1))
                sys.stdout.flush()
                continue
            time.sleep(max(3.0, abs(max(off, key=abs)) / a.speed + 2.0))

            print("[touch %d/%d] now jog X/Y so the TINE TIPS rest on the point, then hold still"
                  % (k + 1, a.need))
            sys.stdout.flush()

            # The arm holds the TOOL-frame origin fixed while it rotates, so if
            # the operator does not move, every pose has an identical position
            # and only the orientation differs. That data is degenerate: with
            # t_i constant, R_i @ p = const has no solution except p along the
            # rotation axis, and the solver quietly returns ~0. So require real
            # movement after the rotation before anything is captured.
            buf: deque = deque(maxlen=a.window)
            t0 = time.time()
            captured = False
            anchor = np.array([conn.base.GetMeasuredCartesianPose().x,
                               conn.base.GetMeasuredCartesianPose().y,
                               conn.base.GetMeasuredCartesianPose().z])
            moved = False
            while not captured and time.time() - t0 < 240:
                time.sleep(1.0 / a.hz)
                p = conn.base.GetMeasuredCartesianPose()
                pos = np.array([p.x, p.y, p.z])
                buf.append(pos)
                if len(buf) < buf.maxlen:
                    continue
                if not moved:
                    if np.linalg.norm(pos - anchor) > a.min_move:
                        moved = True
                        print("[info] movement seen; will capture when you hold still")
                        sys.stdout.flush()
                    continue
                if np.ptp(np.array(buf), axis=0).max() > a.still_m:
                    continue
                rpy = [p.theta_x, p.theta_y, p.theta_z]
                poses.append(dict(pos=pos.tolist(), rpy_deg=rpy))
                OUT.write_text(json.dumps(poses, indent=2))
                sp = 0.0
                if len(poses) > 1:
                    Rs = [R_of(q["rpy_deg"]) for q in poses]
                    sp = max(angle(x, y) for x in Rs for y in Rs)
                print("[pose %d/%d] captured at %s   spread now %.0f deg"
                      % (len(poses), a.need, np.round(pos, 4), sp))
                sys.stdout.flush()
                captured = True
            if not captured:
                print("[warn] no capture at attitude %d (timed out)" % (k + 1))
                sys.stdout.flush()

        if len(poses) >= 4:
            pt, pf, res, spread = solve(poses)
            print("")
            print("[RESULT] fork tip vs the reported tool frame (which is already")
            print("[RESULT] 120 mm past the flange -- see the tool transform):")
            print("[RESULT]   x %+.4f  y %+.4f  z %+.4f  m   |p| = %.1f mm"
                  % (*pt, np.linalg.norm(pt) * 1000))
            print("[RESULT]   (y is UNOBSERVABLE from Y-axis rotations alone --")
            print("[RESULT]    treat it as 0; x and z are the measured ones)")
            print("[RESULT]   residual mean %.2f mm  max %.2f mm   spread %.0f deg"
                  % (res.mean() * 1000, res.max() * 1000, spread))
            print("[RESULT]   => flange to tine tip ~ %.1f mm"
                  % ((0.120 + pt[2]) * 1000))
            if spread < 40:
                print("[RESULT]   *** spread too small to trust")
            elif res.max() > 3.0:
                print("[RESULT]   *** residual too large for 15 mm food")
            else:
                print("[RESULT]   usable.")
        else:
            print("[RESULT] only %d poses captured; need >= 4" % len(poses))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
