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

from settings import cfg
from kinova.motion import (A_MAX, CAM_IN_TOOL, LIMITS, R_from_kortex, _ik, _time_scale, _waypoints,  # noqa: F401
                           kortex_from_R, rot_x, tip_in_tool)

A_ = cfg.acquisition
MAX_OVER_M = float(A_.over_max_m)             # OVER stays inside the bowl, never a transit
BACKOFF_FROM = float(A_.scoop.backoff_from)   # fraction of the scoop before the back-off starts
PUNCH_FLOOR_M = float(cfg.utensil.punch_floor_margin_m)   # nominal; really ~5-10 mm (camera_to_tip stored long)


def cal():
    """(the bowl's overlook pose + the utensil's aim settings, the camera's colour intrinsics) -- kept for callers
    written against the old calibration files; new code reads cfg directly."""
    pose = dict(cfg.overlook_pose())
    pose.update(aim_offset_base_m=list(cfg.utensil.aim_offset_base_m), camera_to_tip_m=cfg.utensil.camera_to_tip_m,
                utensil_tip_px=list(cfg.utensil.tip_px), approach_stop_m=A_.stepped.approach_stop_m,
                scoop_end=dict(cfg.bowl.scoop_end))
    return pose, dict(cfg.camera.colour)










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
    goto_cartesian(conn, goal, tuple(p[3:]), speed=float(A_.over_speed_mps), label="OVER")


def descend(conn, a):
    pose, _ = cal()
    u = json.loads(Path(a.under).read_text())
    stop = max(a.to_above, pose.get("approach_stop_m", 0.010))
    down = u["gap_top"] - stop
    if down <= 0:
        print("    already %.1f mm above the food, nothing to do" % (1000 * u["gap_top"]))
        return
    if down > float(A_.stepped.descend_max_m):
        raise MoveError("descent of %.0f mm is not an approach -- re-measure" % (1000 * down))
    # fast to within 20 mm of the stop, then slow
    fast = max(0.0, down - 0.020)
    if fast > 0:
        _vertical_move(conn, fast, float(A_.stepped.fast_mps), "descend")
    _vertical_move(conn, down - fast, float(A_.stepped.slow_mps), "descend, slow")
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
    _vertical_move(conn, first, float(A_.stepped.slow_mps), "punch")
    if down > first:
        _vertical_move(conn, down - first, float(A_.stepped.slow_mps), "punch, finish")
    Path("/tmp/punch_done.json").write_text(json.dumps(dict(height=u["height"], floor_m=a.floor_m)))


PLUNGE_FAST = float(A_.plunge.fast_mps)            # m/s down to SLOW_ABOVE the food top
PLUNGE_SLOW = float(A_.plunge.slow_mps)            # m/s through the food, to the floor
SLOW_ABOVE = float(A_.plunge.slow_above_food_m)    # the overlook reads the food top 4-7 mm high; this covers it


def plunge(conn, a):
    """OVER -> punch in ONE smooth move (user, 2026-09-24: no stopping on the way down).

    Aimed at the FLOOR read from the overlook, which matched the later close re-measures to within 1 mm
    on every run (the food TOP read 4-7 mm high -- why the stepped route re-measured; the slow zone
    starts SLOW_ABOVE the top to cover that). One joint trajectory along the vertical, fast then easing
    to 5 mm/s through the food, validated by the controller, watched for torque."""
    t_plan = time.time()
    u = json.loads(Path(a.under).read_text())
    if "floor" not in u:
        raise MoveError("no floor reading under the tip -- cannot punch safely")
    floor_m = a.floor_m
    if u.get("depth_source") == "da3-raw" and (u.get("filled_floor", 0) > 0 or u.get("filled_top", 0) > 0):
        raise MoveError("the floor under the tip was measured on RAW Depth Anything 3 (no sensor depth to fit it to) "
                        "-- not accurate enough to stop at the bowl floor; nothing moved")
    if u.get("filled_floor", 0) > float(A_.plunge.refuse_if_floor_filled_over):
        # the fallback gets the level right (median ~1 mm) but its local shape can be 25-60 mm off: never stop
        # a plunge on it
        raise MoveError("%.0f%% of the floor reading under the tip is the Depth Anything 3 fallback -- its local "
                        "error (up to ~60 mm) is too large to stop at the bowl floor; nothing moved" % (100 * u["filled_floor"]))
    down = u["gap_floor"] - floor_m
    lo_, hi_ = A_.plunge.sane_range_m
    if not lo_ < down < hi_:
        raise MoveError("plunge of %.0f mm is outside the sane range (80-250) -- re-measure" % (1000 * down))
    lo_, hi_ = A_.plunge.food_height_range_m
    if not lo_ < u["height"] < hi_:
        raise MoveError("food %.0f mm tall is implausible -- re-measure" % (1000 * u["height"]))
    slow_at = max(0.0, u["gap_top"] - SLOW_ABOVE)          # distance travelled when the slow zone starts
    p = tool_pose(conn); q0 = joints(conn); T0 = np.array(p[:3])
    n = int(math.ceil(down / float(A_.plunge.ik_step_m))); path = [q0]; prev = q0; d = [0.0]   # fewer IK calls: planning is dead time
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
          % (1000 * u["height"], 1000 * down, 1000 * floor_m, 1000 * slow_at, total))
    scoop_plan = None
    if a.then_scoop:                                          # plan the scoop NOW, from where the plunge will end
        Te = np.array([T0[0], T0[1], T0[2] - down])
        scoop_plan = plan_scoop(conn, a, path[-1], Te, R_from_kortex(*p[3:]),
                                dict(height=u["height"], floor_m=floor_m))
    print("    plunge%s planned in %.1f s" % (" + scoop" if scoop_plan else "", time.time() - t_plan))
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
    Path("/tmp/punch_done.json").write_text(json.dumps(dict(height=u["height"], floor_m=floor_m)))
    if scoop_plan is not None:
        run_scoop(conn, a, scoop_plan)                        # no pause at the bottom: already planned


class DepthUnavailable(Exception):
    """No usable sensor depth under the tip after OVER: the shell falls back to the filled measurement."""


def pick(conn, a):
    """OVER -> fresh depth from the resident depth server -> plan plunge + scoop -> plunge -> scoop, all in
    THIS process: no program restarts, no reconnection, so the arm does not stand still between them
    (user, 2026-09-25). The floor is still re-measured from above the berry after OVER: read from the overlook
    frame instead it disagreed by up to 11 mm on 2 of 17 runs."""
    import io
    import urllib.request
    from feeding.measure_under import measure
    over(conn, a)
    t = time.time()
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/median?frames=%d" % (int(cfg.deployment.servers.depth_port), int(cfg.perception.depth_frames)), timeout=15) as r:
            d = np.load(io.BytesIO(r.read()))
    except Exception as e:  # noqa: BLE001
        raise DepthUnavailable("depth server: %s" % e)
    try:
        u = measure(d, 0.41)
    except SystemExit as e:                    # measure_under's own "no depth return under the tip"
        raise DepthUnavailable(str(e))
    if "floor" not in u:
        raise DepthUnavailable("no floor reading beside the food")
    print("  under the tip: top %.1f mm (tip %.1f mm above it), floor %.1f mm (tip %.1f mm above it), food %.1f mm tall"
          "   [fresh depth in %.1f s]" % (1000 * u["top"], 1000 * u["gap_top"], 1000 * u["floor"], 1000 * u["gap_floor"],
                                           1000 * u["height"], time.time() - t))
    Path(a.under).write_text(json.dumps(u))
    try:
        Path("/work_out/under_over.json").write_text(json.dumps(u))      # for the run's record (host /tmp/demo_state)
    except OSError:
        pass
    a.then_scoop = True
    plunge(conn, a)






SCOOP_TORQUE_NM = float(cfg.guards.torque_step_nm.scoop)   # 2.5 tripped on the arm's own acceleration






def plan_scoop(conn, a, q0, T0, R0, done):
    """Plan the scoop from a given arm state (the current one, or where the plunge WILL end -- so the
    plunge can flow straight into it with no planning pause at the bottom)."""
    pose, _ = cal()
    h = done.get("height", 0.030)
    tip = tip_in_tool()
    # the punch leaves the tip floor_m above the floor; the food's centre is at h/2
    food_up = max(0.0, h / 2.0 - done.get("floor_m", PUNCH_FLOOR_M))
    o_piv = tip - np.array([0, 0, food_up])
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
        if np.linalg.norm(back) > float(A_.scoop.max_backoff_m):
            raise MoveError("backing off %.0f mm is too far -- not the bowl this was built for" % (1000 * np.linalg.norm(back)))

    def piv_at(s):
        x = min(1.0, max(0.0, (s - BACKOFF_FROM) / (1 - BACKOFF_FROM)))
        return piv0 + np.array([0, 0, rise * s]) + x * x * (3 - 2 * x) * back

    def at(s):
        th = 90.0 * s ** a.rot_exp
        R = rot_x(th) @ R0
        return piv_at(s) - R @ o_piv, kortex_from_R(R)

    def along(qe):
        """A continuous joint path, the IK guess blended from the start toward the end shape qe."""
        prev, cand = q0, [q0]
        for k in range(1, a.n + 1):
            s = k / a.n
            T, r = at(s)
            q = _ik(conn, T, r, [fold(x + fold(y - x) * s) for x, y in zip(q0, qe)])
            if q is None or max(abs(fold(x - y)) for x, y in zip(q, prev)) > 12:
                return None
            prev = q; cand.append(q)
        return cand

    # the calibrated end shape first (the one delivery was built from) -- the search only if it fails
    path = along(se["joints_deg"]) if se else None
    if path is None:
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
        for qe in ends[:10]:
            path = along(qe)
            if path:
                break
    if path is None:
        raise MoveError("no continuous scoop path from here -- nothing moved")
    print("    scoop planned: food rises %.0f mm%s while turning 90 deg; %d waypoints, min joint margin %.1f deg"
          % (1000 * rise, ", backing off %.0f mm" % (1000 * np.linalg.norm(back)) if se else "", a.n, min(m(q) for q in path)))
    return dict(path=path, o_piv=o_piv, piv_at=piv_at, piv0=piv0)


def scoop(conn, a):
    done = json.loads(Path("/tmp/punch_done.json").read_text()) if Path("/tmp/punch_done.json").exists() else {}
    p = tool_pose(conn)
    run_scoop(conn, a, plan_scoop(conn, a, joints(conn), np.array(p[:3]), R_from_kortex(*p[3:]), done))


def run_scoop(conn, a, plan):
    path, o_piv, piv_at, piv0 = plan["path"], plan["o_piv"], plan["piv_at"], plan["piv0"]
    path = [joints(conn)] + path[1:]                         # start from where the arm actually is
    # one smooth trajectory, validated by the controller; slower only if refused
    wl, speed = _waypoints(conn, path, a.speed, a.ramp, "scoop")
    total = sum(w.angular_waypoint.duration for w in wl.waypoints)
    print("    scoop at %.0f deg/s: %.1f s" % (speed, total))
    if a.validate_only:
        print("    validate-only: controller accepted it, nothing moved")
        return
    # 4. watch: the food must stay on its planned path (its vertical line, if not backing off) and not drop
    plan_piv = np.array([piv_at(k / 400) for k in range(401)])
    ev = _events(conn); ev.clear(); guard = TorqueGuard(limit=SCOOP_TORQUE_NM); log = []; why = None; worst = 0.0
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
        if off > float(cfg.guards.scoop_food_off_path_m):
            why = "food point %.1f mm off its planned path" % (1000 * off)
        elif pv[2] < near[2] - float(cfg.guards.scoop_food_drop_m):
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
    ap.add_argument("step", choices=["over", "descend", "punch", "plunge", "scoop", "pick"])
    ap.add_argument("--target", default="/tmp/target.json")
    ap.add_argument("--under", default="/tmp/under.json", help="measure_under.py output")
    ap.add_argument("--to-above", type=float, default=float(A_.stepped.approach_stop_m), help="descend: stop this far above the food top")
    ap.add_argument("--floor-m", type=float, default=PUNCH_FLOOR_M, help="punch: stop this far above the bowl floor")
    ap.add_argument("--speed", type=float, default=float(A_.scoop.speed_dps),
                    help="scoop: deg/s. Fast on purpose -- at 6 deg/s (25 s) the berry slid off")
    ap.add_argument("--rot-exp", type=float, default=float(A_.scoop.rot_exp),
                    help="scoop: turn angle = 90 * progress**rot_exp. <1 starts turning the moment the food lifts "
                         "(user, 2026-09-24: at 1.6 it rose steep first and gravity works on the food)")
    ap.add_argument("--n", type=int, default=int(A_.scoop.waypoints), help="scoop waypoints")
    ap.add_argument("--ramp", type=float, default=0.25, help="scoop: fraction of the time spent easing in, and out")
    ap.add_argument("--validate-only", action="store_true", help="scoop: plan and let the controller validate; do not move")
    ap.add_argument("--then-scoop", action="store_true", help="plunge: plan the scoop first and run it straight after")
    a = ap.parse_args()
    with kinova_connection() as conn:
        try:
            dict(over=over, descend=descend, punch=punch, plunge=plunge, scoop=scoop, pick=pick)[a.step](conn, a)
            p = tool_pose(conn)
            print("  %s done: tool (%.4f %.4f %.4f), utensil %.1f deg off vertical"
                  % (a.step, p[0], p[1], p[2], tilt_deg(conn)))
        except DepthUnavailable as e:
            print("\n  no sensor depth under the tip (%s) -- the shell falls back to the filled measurement" % e, file=sys.stderr)
            return 6
        except MoveError as e:
            print("\n  STOPPED: %s" % e, file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
