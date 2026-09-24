#!/usr/bin/env python3
"""Food acquisition, one step at a time. Frozen 2026-09-23 after it ran end to end.

    python -m kinova.scripts.acquire over    --target /tmp/target.json
    python -m kinova.scripts.acquire descend --under /tmp/under.json --to-above 0.030
    python -m kinova.scripts.acquire punch   --under /tmp/under.json
    python -m kinova.scripts.acquire scoop   [--speed 20]

feeding/acquire.sh runs them in order with a depth re-measure between each, and
is what `make acquire` calls. Every motion goes through safe_move.

  over     at the overlook height, translate until the utensil tip's vertical
           line passes through the target food. Refuses unless at
           food_detect_start and the move is small and level.
  descend  straight down until the tip is --to-above over the food's TOP, using
           a depth reading taken from WHERE THE ARM IS (measure_under.json).
           Refuses a descent that would end below 10 mm (approach_stop_m).
  punch    straight down to bowl floor + 5 mm, slowly.
  scoop    the food rises straight up while the utensil turns vertical ->
           horizontal about it, pointing +Y (toward the user). Planned here as a
           continuous joint path (a single Cartesian move is unreachable: J6
           hits its limit), validated by the controller, run as one trajectory.

Measured on the arm, 2026-09-23, fork on a strawberry:
  over     berry lands 1.2 mm from the fork's vertical line
  approach 30 mm hover re-measured, final stop 10 mm above: user-confirmed
  punch    35.0 mm down, stopped ~11 mm above the floor as planned
  scoop    berry point rose 155.7 mm, 0.2 mm off vertical, fork horizontal.
           Geometry "perfectly correct" (user). The berry FELL: the scoop
           was too slow (25 s) -- food slides under gravity as the utensil
           tilts. The user also placed the aim 5 mm off along Y (applied in
           OVER from calibration) and said the punch can go deeper: floor + 5.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from kortex_api.autogen.messages import Base_pb2

from kinova.connection import kinova_connection
from kinova.scripts.safe_move import (MoveError, TorqueGuard, _events, fold, goto_cartesian,
                                      joints, tilt_deg, tool_pose)

CAL = Path(__file__).resolve().parent.parent.parent / "calibration"
MAX_OVER_M = 0.10          # OVER stays inside the bowl, never a transit (central berries sit ~70-80 mm out + the aim offset)
BACKOFF_FROM = 0.2         # fraction of the scoop before the back-off starts
PUNCH_FLOOR_M = 0.0        # nominal; the real gap is ~5-10 mm (camera_to_tip stored long, floor read beside the food)
LIMITS = {1: 128.9, 3: 147.8, 5: 120.3}
# Kinova's model (gen3_7dof.urdf): the colour camera sits at (0, 56.39, -3.05) mm
# in the end-effector (flange) frame; the configured tool point is 120 mm out
# along z. So in the TOOL frame the camera is at (0, 56.39, -123.05) mm.
CAM_IN_TOOL = np.array([0.0, 0.05639, -0.12305])


def cal():
    return (json.loads((CAL / "food_detect_pose.json").read_text()),
            json.loads((CAL / "colour_intrinsics.json").read_text()))


def tip_in_tool() -> np.ndarray:
    """Utensil tip in the tool frame, from calibration only.

    tip = camera + camera_to_tip along the axis + the tip's sideways offset,
    read off the image as (tip pixel - image of true vertical) scaled to that
    distance. Image +u is tool -x and image +v is tool -y at the overlook
    orientation (from motion probes). For the fork this gives (0, 8.8, 142) mm.
    """
    pose, c = cal()
    zt = pose["camera_to_tip_m"]
    d = (np.array(pose["utensil_tip_px"], float) - np.array(c["vertical_vp"])) * zt / c["fx"]
    return CAM_IN_TOOL + np.array([-d[0], -d[1], zt])


def rot_x(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def R_from_kortex(tx, ty, tz):
    tx, ty, tz = (math.radians(v) for v in (tx, ty, tz))
    Rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    Rz = np.array([[math.cos(tz), -math.sin(tz), 0], [math.sin(tz), math.cos(tz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def kortex_from_R(R):
    ty = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    return (math.degrees(math.atan2(R[2, 1], R[2, 2])), ty,
            math.degrees(math.atan2(R[1, 0], R[0, 0])))


def _vertical_move(conn, down_m, speed, label):
    p = tool_pose(conn)
    goto_cartesian(conn, (p[0], p[1], p[2] - down_m), tuple(p[3:]), speed=speed, label=label)


# ------------------------------------------------------------------ steps

def over(conn, a):
    pose, _ = cal()
    off = max(abs(fold(x - y)) for x, y in zip(joints(conn), pose["joints_deg"]))
    if off > 1.0:
        raise MoveError("not at food_detect_start (%.1f deg away) -- run: make detect" % off)
    goal = list(json.loads(Path(a.target).read_text())["tool_goal"])
    off_xy = pose.get("aim_offset_base_m", [0.0, 0.0])     # measured aim correction, base frame
    goal[0] += off_xy[0]; goal[1] += off_xy[1]
    p = tool_pose(conn)
    d = math.hypot(goal[0] - p[0], goal[1] - p[1])
    if abs(goal[2] - p[2]) > 0.002 or d > MAX_OVER_M:
        raise MoveError("target is not a small level move (%.0f mm across) -- re-run make perceive"
                        % (1000 * d))
    goto_cartesian(conn, goal, tuple(p[3:]), speed=0.02, label="OVER")


def descend(conn, a):
    pose, _ = cal()
    u = json.loads(Path(a.under).read_text())
    stop = max(a.to_above, pose.get("approach_stop_m", 0.010))
    down = u["gap_top"] - stop
    if down <= 0:
        print("    already %.1f mm above the food, nothing to do" % (1000 * u["gap_top"]))
        return
    if down > 0.200:
        raise MoveError("descent of %.0f mm is not an approach -- re-measure" % (1000 * down))
    # fast to within 20 mm of the stop, then slow
    fast = max(0.0, down - 0.020)
    if fast > 0:
        _vertical_move(conn, fast, 0.015, "descend")
    _vertical_move(conn, down - fast, 0.005, "descend, slow")
    print("    tip now ~%.1f mm above the food top" % (1000 * (u["gap_top"] - down)))


def punch(conn, a):
    u = json.loads(Path(a.under).read_text())
    if "floor" not in u:
        raise MoveError("no floor reading under the tip -- cannot punch safely")
    stop_above_floor = a.floor_m
    down = u["gap_floor"] - stop_above_floor
    if not 0.010 < down < 0.060:
        raise MoveError("punch depth %.1f mm is outside the sane range" % (1000 * down))
    print("    food %.1f mm tall: stop %.1f mm above the bowl floor -> %.1f mm down"
          % (1000 * u["height"], 1000 * stop_above_floor, 1000 * down))
    first = min(0.020, down)
    _vertical_move(conn, first, 0.005, "punch")
    if down > first:
        _vertical_move(conn, down - first, 0.005, "punch, finish")
    Path("/tmp/punch_done.json").write_text(json.dumps(dict(height=u["height"], floor_m=a.floor_m)))


PLUNGE_FAST = 0.020        # m/s down to SLOW_ABOVE the food top
PLUNGE_SLOW = 0.005        # m/s through the food, to the floor
SLOW_ABOVE = 0.020         # the overlook reads the food top 4-7 mm high; this covers it


def plunge(conn, a):
    """OVER -> punch in ONE smooth move (user, 2026-09-24: no stopping on the way down).

    Aimed at the FLOOR read from the overlook, which matched the later close re-measures to within 1 mm
    on every run (the food TOP read 4-7 mm high -- why the stepped route re-measured; the slow zone
    starts SLOW_ABOVE the top to cover that). One joint trajectory along the vertical, fast then easing
    to 5 mm/s through the food, validated by the controller, watched for torque."""
    u = json.loads(Path(a.under).read_text())
    if "floor" not in u:
        raise MoveError("no floor reading under the tip -- cannot punch safely")
    down = u["gap_floor"] - a.floor_m
    if not 0.080 < down < 0.250:
        raise MoveError("plunge of %.0f mm is outside the sane range (80-250) -- re-measure" % (1000 * down))
    if not 0.010 < u["height"] < 0.060:
        raise MoveError("food %.0f mm tall is implausible -- re-measure" % (1000 * u["height"]))
    slow_at = max(0.0, u["gap_top"] - SLOW_ABOVE)          # distance travelled when the slow zone starts
    p = tool_pose(conn); q0 = joints(conn); T0 = np.array(p[:3])
    n = int(math.ceil(down / 0.004)); path = [q0]; prev = q0; d = [0.0]
    for k in range(1, n + 1):
        dk = down * k / n
        q = _ik(conn, (T0[0], T0[1], T0[2] - dk), tuple(p[3:]), prev)
        if q is None or max(abs(fold(x - y)) for x, y in zip(q, prev)) > 8:
            raise MoveError("no continuous plunge path at %.0f mm -- nothing moved" % (1000 * dk))
        prev = q; path.append(q); d.append(dk)

    def v(x):                                               # smooth: ramp in, fast, ease to slow
        ramp = min(1.0, 0.25 + x / 0.015)
        w = min(1.0, max(0.0, (x - (slow_at - 0.015)) / 0.015))
        w = w * w * (3 - 2 * w)
        return ramp * (PLUNGE_FAST + (PLUNGE_SLOW - PLUNGE_FAST) * w)
    raw = [x.position for x in conn.base_cyclic.RefreshFeedback().actuators]
    for scale in (1.0, 0.75, 0.5):
        wl = Base_pb2.WaypointList(); wl.duration = 0.0; wl.use_optimal_blending = False
        cur, pq = list(raw), path[0]; total = 0.0
        for k, q in enumerate(path[1:], 1):
            cur = [c + fold(x - y) for c, x, y in zip(cur, q, pq)]
            dt = (d[k] - d[k - 1]) / (scale * v(0.5 * (d[k] + d[k - 1]))); total += dt
            w_ = wl.waypoints.add(); w_.name = "p%02d" % k
            w_.angular_waypoint.angles.extend(cur); w_.angular_waypoint.duration = dt; pq = q
        errs = list(conn.base.ValidateWaypointList(wl).trajectory_error_report.trajectory_error_elements)
        if not errs:
            break
        print("    x%.2f speed refused by the controller: %s" % (scale, " ".join(str(errs[0]).split())))
    else:
        raise MoveError("controller refused the plunge -- nothing moved")
    print("    food %.1f mm tall: %.0f mm down to the floor + %.0f mm, slow from %.0f mm; %.1f s"
          % (1000 * u["height"], 1000 * down, 1000 * a.floor_m, 1000 * slow_at, total))
    if a.validate_only:
        print("    validate-only: controller accepted it, nothing moved")
        return
    ev = _events(conn); ev.clear(); guard = TorqueGuard(); t0 = time.time()
    conn.base.ExecuteWaypointTrajectory(wl)
    while time.time() - t0 < total + 15:
        time.sleep(0.02)
        hit = guard.update([x.torque for x in conn.base_cyclic.RefreshFeedback().actuators])
        if hit or any(e[0] == "ACTION_ABORT" for e in ev):
            conn.base.Stop()
            raise MoveError("plunge stopped: %s" % ("joint %d torque step" % (hit[0] + 1) if hit else "controller aborted"))
        if any(e[0] == "ACTION_END" for e in ev):
            break
    print("    plunge done in %.1f s" % (time.time() - t0))
    Path("/tmp/punch_done.json").write_text(json.dumps(dict(height=u["height"], floor_m=a.floor_m)))


def _ik(conn, T, rpy, g):
    d = Base_pb2.IKData(); p = d.cartesian_pose
    p.x, p.y, p.z = map(float, T); p.theta_x, p.theta_y, p.theta_z = map(float, rpy)
    for v in g:
        d.guess.joint_angles.add().value = v % 360.0
    try:
        return [fold(j.value) for j in conn.base.ComputeInverseKinematics(d).joint_angles]
    except Exception:  # noqa: BLE001 -- unreachable is an answer
        return None


def _timing(path, peak, ramp):
    """Waypoint durations for a smooth speed profile along the joint path.

    Progress is measured by the leading joint's travel. Speed eases in over the
    first `ramp` of the time (sin^2, so acceleration is continuous), cruises at
    `peak` deg/s, and eases out over the last `ramp`.
    """
    steps = [max(abs(fold(x - y)) for x, y in zip(q, p)) for p, q in zip(path, path[1:])]
    L = sum(steps)
    tau = np.linspace(0.0, 1.0, 4001)
    v = np.ones_like(tau)
    up, dn = tau < ramp, tau > 1 - ramp
    v[up] = np.sin(0.5 * math.pi * tau[up] / ramp) ** 2
    v[dn] = np.sin(0.5 * math.pi * (1 - tau[dn]) / ramp) ** 2
    prog = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tau))])
    T = L / (peak * prog[-1])                 # cruise speed = peak
    t = np.interp(np.cumsum([0.0] + steps) / L, prog / prog[-1], tau) * T
    return [max(0.02, d) for d in np.diff(t)]


def _waypoints(conn, path, top_speed, ramp, what):
    """The joint path as one WaypointList the controller has validated; slower if refused."""
    raw = [x.position for x in conn.base_cyclic.RefreshFeedback().actuators]
    for speed in (top_speed, top_speed * 0.75, top_speed * 0.5, top_speed * 0.35):
        durs = _timing(path, speed, ramp)
        wl = Base_pb2.WaypointList(); wl.duration = 0.0; wl.use_optimal_blending = False
        cur, prev = list(raw), path[0]
        for k, (q, dt) in enumerate(zip(path[1:], durs), 1):
            cur = [c + fold(x - y) for c, x, y in zip(cur, q, prev)]
            w = wl.waypoints.add(); w.name = "w%02d" % k
            w.angular_waypoint.angles.extend(cur)
            w.angular_waypoint.duration = dt
            prev = q
        errs = list(conn.base.ValidateWaypointList(wl).trajectory_error_report.trajectory_error_elements)
        if not errs:
            return wl, speed
        print("    %.0f deg/s refused by the controller: %s" % (speed, " ".join(str(errs[0]).split())))
    raise MoveError("controller refused the %s even at %.0f deg/s -- nothing moved" % (what, top_speed * 0.35))



def scoop(conn, a):
    pose, _ = cal()
    done = json.loads(Path("/tmp/punch_done.json").read_text()) if Path("/tmp/punch_done.json").exists() else {}
    h = done.get("height", 0.030)
    tip = tip_in_tool()
    # the punch leaves the tip floor_m above the floor; the food's centre is at h/2
    food_up = max(0.0, h / 2.0 - done.get("floor_m", PUNCH_FLOOR_M))
    o_piv = tip - np.array([0, 0, food_up])
    p = tool_pose(conn); q0 = joints(conn)
    R0 = R_from_kortex(*p[3:]); T0 = np.array(p[:3])
    piv0 = T0 + R0 @ o_piv
    end_z = pose["tool_pose"]["z_m"] + (R0 @ tip)[2]          # tip height at the overlook
    rise = end_z - piv0[2]
    m = lambda q: min(LIMITS[i] - abs(q[i]) for i in LIMITS)
    # With a calibrated scoop_end, the food also backs off to where that pose holds it: the rotation still
    # starts at once, the back-off mostly once the utensil is past steep (from BACKOFF_FROM of the scoop).
    se = pose.get("scoop_end")
    back = np.zeros(3)
    if se:
        pivE = np.array(se["tool_m"]) + rot_x(90.0) @ R0 @ o_piv
        back = pivE - piv0; rise = back[2]; back[2] = 0.0
        if np.linalg.norm(back) > 0.300:
            raise MoveError("backing off %.0f mm is too far -- not the bowl this was built for" % (1000 * np.linalg.norm(back)))

    def piv_at(s):
        x = min(1.0, max(0.0, (s - BACKOFF_FROM) / (1 - BACKOFF_FROM)))
        return piv0 + np.array([0, 0, rise * s]) + x * x * (3 - 2 * x) * back

    def at(s):
        th = 90.0 * s ** a.rot_exp
        R = rot_x(th) @ R0
        return piv_at(s) - R @ o_piv, kortex_from_R(R)

    # 1. reachable end shapes, nearest the current one first
    T, r = at(1.0); ends = []
    for j1, j2, j3, j4, j6, fl in itertools.product((q0[0], 150, 120, 100), (-30, 0, 30, 60),
                                                    (q0[2], 30, 60, 85), (-135, -110),
                                                    (-85, -50, 50, 85), (0, 1)):
        g = [j1, j2, j3, j4, q0[4], j6, q0[6]]
        if fl:
            g[4], g[6] = fold(g[4] + 180), fold(g[6] + 180)
        q = _ik(conn, T, r, g)
        if q and m(q) > 8:
            ends.append(q)
    ends.sort(key=lambda q: max(abs(fold(x - y)) for x, y in zip(q, q0)))
    if se:
        ends.insert(0, se["joints_deg"])                     # the shape the downstream stages were built from
    # 2. a continuous joint path, IK guess blended from start to end shape
    path = None
    for qe in ends[:10]:
        prev, cand = q0, [q0]
        for k in range(1, a.n + 1):
            s = k / a.n
            T, r = at(s)
            q = _ik(conn, T, r, [fold(x + fold(y - x) * s) for x, y in zip(q0, qe)])
            if q is None or max(abs(fold(x - y)) for x, y in zip(q, prev)) > 12:
                break
            prev = q; cand.append(q)
        else:
            path = cand; break
    if path is None:
        raise MoveError("no continuous scoop path from here (%d end shapes tried)" % min(10, len(ends)))
    print("    plan: food rises %.0f mm%s while turning 90 deg; %d waypoints, min joint margin %.1f deg"
          % (1000 * rise, ", backing off %.0f mm" % (1000 * np.linalg.norm(back)) if se else "", a.n, min(m(q) for q in path)))
    # 3. one smooth trajectory, validated by the controller; slow down until it passes.
    #    Timed with a smooth speed profile (ease in, cruise, ease out): giving every
    #    waypoint the cruise speed asks for full speed within the first ~0.1 s, and
    #    the controller refuses that at any speed (INVALID_JOINT_ACCELERATION).
    wl, speed = _waypoints(conn, path, a.speed, a.ramp, "scoop")
    total = sum(w.angular_waypoint.duration for w in wl.waypoints)
    print("    cruising at %.0f deg/s (ramps %.0f%% each end): %.1f s" % (speed, 100 * a.ramp, total))
    if a.validate_only:
        print("    validate-only: controller accepted it, nothing moved")
        return
    # 4. watch: the food must stay on its planned path (its vertical line, if not backing off) and not drop
    plan_piv = np.array([piv_at(k / 400) for k in range(401)])
    ev = _events(conn); ev.clear(); guard = TorqueGuard(); log = []; why = None; worst = 0.0
    conn.base.ExecuteWaypointTrajectory(wl)
    t0 = time.time()
    while time.time() - t0 < total + 15:
        time.sleep(0.02)
        fb = conn.base_cyclic.RefreshFeedback(); b = fb.base
        pv = np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z]) + \
            R_from_kortex(b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z) @ o_piv
        near = plan_piv[np.argmin(np.linalg.norm(plan_piv - pv, axis=1))]
        off = math.hypot(*(pv[:2] - near[:2])); worst = max(worst, off)
        log.append(dict(t=time.time() - t0, piv=pv.tolist(), off=off))
        if off > 0.015:
            why = "food point %.1f mm off its planned path" % (1000 * off)
        elif pv[2] < near[2] - 0.010:
            why = "food point %.1f mm below its planned path" % (1000 * (near[2] - pv[2]))
        hit = guard.update([x.torque for x in fb.actuators])
        if hit and not why:
            why = "joint %d torque step" % (hit[0] + 1)
        if any(e[0] == "ACTION_ABORT" for e in ev) and not why:
            why = "controller aborted"
        if why:
            conn.base.Stop()
            break
        if any(e[0] == "ACTION_END" for e in ev):
            break
    Path("/tmp/scoop_log.json").write_text(json.dumps(log))
    p = tool_pose(conn)
    ax = R_from_kortex(*p[3:]) @ np.array([0, 0, 1.0])
    if why:
        raise MoveError("scoop stopped: " + why)
    print("    food rose %.1f mm, worst %.1f mm off its path, utensil axis (%.2f %.2f %.2f), %.1f s"
          % (1000 * (pv[2] - piv0[2]), 1000 * worst, *ax, time.time() - t0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["over", "descend", "punch", "plunge", "scoop"])
    ap.add_argument("--target", default="/tmp/target.json")
    ap.add_argument("--under", default="/tmp/under.json", help="measure_under.py output")
    ap.add_argument("--to-above", type=float, default=0.010, help="descend: stop this far above the food top")
    ap.add_argument("--floor-m", type=float, default=PUNCH_FLOOR_M, help="punch: stop this far above the bowl floor")
    ap.add_argument("--speed", type=float, default=20.0,
                    help="scoop: deg/s. Fast on purpose -- at 6 deg/s (25 s) the berry slid off")
    ap.add_argument("--rot-exp", type=float, default=0.8,
                    help="scoop: turn angle = 90 * progress**rot_exp. <1 starts turning the moment the food lifts "
                         "(user, 2026-09-24: at 1.6 it rose steep first and gravity works on the food)")
    ap.add_argument("--n", type=int, default=40, help="scoop waypoints")
    ap.add_argument("--ramp", type=float, default=0.25, help="scoop: fraction of the time spent easing in, and out")
    ap.add_argument("--validate-only", action="store_true", help="scoop: plan and let the controller validate; do not move")
    a = ap.parse_args()
    with kinova_connection() as conn:
        try:
            dict(over=over, descend=descend, punch=punch, plunge=plunge, scoop=scoop)[a.step](conn, a)
            p = tool_pose(conn)
            print("  %s done: tool (%.4f %.4f %.4f), utensil %.1f deg off vertical"
                  % (a.step, p[0], p[1], p[2], tilt_deg(conn)))
        except MoveError as e:
            print("\n  STOPPED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
