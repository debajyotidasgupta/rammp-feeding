#!/usr/bin/env python3
"""Plan the food acquisition -- WITHOUT moving the arm.

    python -m feeding.acquire_plan --target ~/rammp_runs/<run>/target.json --out plan

There is no execute path in this file. It builds the waypoints of each step and
asks the controller's own inverse kinematics whether each is reachable, by which
joints, and with what margins. Every call it makes is read-only
(ComputeInverseKinematics / ComputeForwardKinematics / RefreshFeedback).

THE STEPS, each planned and checked on its own:

  1 OVER     translate at the overlook height until the utensil tip's vertical
             line passes through the chosen food piece (locate_target's goal)
  2 PUNCH    straight down: fast to just above the food, then slow through the
             flesh, stopping MARGIN above the bowl floor. The stop is a planned
             depth; the torque guard is the backstop, not the terminator.
  3 SCOOP    the food moves STRAIGHT UP in the world frame and nothing else,
             while the utensil rotates vertical -> horizontal about the food.
             Implemented as the tool frame rotating about a pivot at the food's
             centre (a little up the utensil from its tip) while the pivot
             rises. Ends at ~the overlook height with the utensil horizontal.

DIRECTIONS, from measurements rather than assumption:
  * In the food_detect_start image, image-left is +Y in the arm base frame and
    image-up is +X (tool-motion probes).
  * feeding_start, which points away from the user, has the tool at -Y. So the
    user is toward +Y -- image-left, where the user's laptop and hand appear.
  * "Scoop toward the human, head on the left" => the utensil ends horizontal
    pointing +Y. That is a +90 deg rotation of the tool about world +X:
    approach (0,0,-1) -> (0,+1,0). The gripper swings to -Y, AWAY from the user.

GEOMETRY THAT IS NOT MEASURED YET, so it is swept rather than assumed:
  --z-tip      camera -> utensil tip along the axis. 0.115 m, measured by the user along the spoon from the camera case (2026-09-23).
  --tip-len    configured tool point -> utensil tip along the axis. Unknown;
               the configured tool point is 120 mm past the flange.
  --lateral    utensil tip offset from the tool axis. Unknown; this is what
               swept the fork into joint 6.
The PUNCH depth depends on --z-tip; the SCOOP pivot depends on --tip-len. The
5 mm floor margin cannot be honoured until --z-tip is measured to ~1 mm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ rotations

def rot_x(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_from_kortex(tx, ty, tz):
    """Kortex tool orientation: fixed-frame XYZ Euler, R = Rz @ Ry @ Rx."""
    return rot_z(tz) @ rot_y(ty) @ rot_x(tx)


def kortex_from_R(R):
    ty = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    tx = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    tz = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return tx, ty, tz


def fold(d):
    return (d + 180.0) % 360.0 - 180.0


LIMITS = {1: 128.9, 3: 147.8, 5: 120.3}        # J2, J4, J6 (0-indexed)

# ------------------------------------------------------------------ planning

def plan_steps(tool0, rpy0, goal_xy, top_depth, floor_depth, a):
    """Waypoints as (step, label, tool_xyz, tool_rpy, extra) -- tool frame poses."""
    R0 = R_from_kortex(*rpy0)
    o = np.array([a.lateral[0], a.lateral[1], a.tip_len])   # tip, in the TOOL frame
    wps = []

    # 1 OVER -- same height, same orientation
    over = np.array([goal_xy[0], goal_xy[1], tool0[2]])
    wps.append(("OVER", "over the food", over, rpy0, {}))

    # 2 PUNCH -- descent distances depend on the camera->tip distance only:
    # the camera and tip move together, depths are measured from the camera.
    d_top = top_depth - a.z_tip                       # tip reaches the food surface
    # Stop height = floor + (food height / 3): the user's rule. Not a fixed
    # margin -- it scales with the food, keeping the tip in the lower third of
    # the piece for purchase while staying well clear of the bowl.
    food_h = floor_depth - top_depth
    stop_above_floor = food_h / 3.0 if a.margin is None else a.margin
    d_end = floor_depth - stop_above_floor - a.z_tip
    d_slow = max(0.0, d_top - a.slow_before)
    n_fast = max(1, int(math.ceil(d_slow / 0.03)))
    for k in range(1, n_fast + 1):
        d = d_slow * k / n_fast
        lbl = "fast, down %3.0f mm" % (1000 * d) if k < n_fast else "fast ends %.0f mm above food" % (1000 * a.slow_before)
        wps.append(("PUNCH", lbl, over - np.array([0, 0, d]), rpy0, dict(descent_m=d)))
    for lbl, d in (("slow: food surface", d_top),
                   ("slow: stop %.0f mm above floor" % (1000 * stop_above_floor), d_end)):
        wps.append(("PUNCH", lbl, over - np.array([0, 0, d]), rpy0, dict(descent_m=d)))

    # 3 SCOOP -- rotate about the PIVOT (food centre, up the utensil from its
    # tip) while the pivot rises straight up.
    T_end = wps[-1][2]
    tip0 = T_end + R0 @ o
    axis0 = R0 @ np.array([0, 0, 1.0])                # approach, ~(0,0,-1)
    pivot0 = tip0 - a.food_up * axis0                 # a little up the utensil
    o_piv = o - np.array([0, 0, a.food_up])           # pivot in the tool frame
    rise = (tool0[2] + (R0 @ o)[2] - a.food_up * axis0[2]) - pivot0[2]  # back to overlook tip height
    for k in range(1, a.n + 1):
        s = k / a.n
        th = 90.0 * (s ** a.rot_exp)                  # rotate late: stay steep inside the bowl
        R = rot_x(th) @ R0
        piv = pivot0 + np.array([0, 0, rise * s])
        T = piv - R @ o_piv
        wps.append(("SCOOP", "theta %5.1f deg" % th, T, kortex_from_R(R),
                    dict(theta=th, pivot=piv.tolist(), axis=(R @ np.array([0, 0, 1.0])).tolist())))
    return wps, dict(d_top=d_top, d_end=d_end, stop_above_floor=stop_above_floor, pivot0=pivot0.tolist(), rise=float(rise),
                     o_piv=o_piv.tolist())


def check_ik(conn, wps, q_start):
    from kortex_api.autogen.messages import Base_pb2
    out, guess = [], list(q_start)
    for step, lbl, T, rpy, extra in wps:
        ik = Base_pb2.IKData()
        p = ik.cartesian_pose
        p.x, p.y, p.z = map(float, T)
        p.theta_x, p.theta_y, p.theta_z = map(float, rpy)
        for v in guess:
            ik.guess.joint_angles.add().value = v % 360.0
        try:
            sol = conn.base.ComputeInverseKinematics(ik)
            q = [fold(j.value) for j in sol.joint_angles]
            jump = max(abs(fold(x - y)) for x, y in zip(q, guess))
            margin = min(LIMITS[i] - abs(q[i]) for i in LIMITS)
            worst_j = min(LIMITS, key=lambda i: LIMITS[i] - abs(q[i])) + 1
            out.append(dict(step=step, label=lbl, tool=list(map(float, T)), rpy=list(map(float, rpy)),
                            ok=True, joints=q, jump=jump, margin=margin, tightest=worst_j, **extra))
            guess = q
        except Exception as e:  # noqa: BLE001 -- unreachable is an answer, not a crash
            out.append(dict(step=step, label=lbl, tool=list(map(float, T)), rpy=list(map(float, rpy)),
                            ok=False, error=str(e).split("=>")[-1].strip()[:80], **extra))
    return out


def scoop_joint_path(conn, wps, a, n=60):
    """The SCOOP as a continuous joint-space path.

    As one Cartesian move the scoop is unreachable past ~57 deg: on the branch
    the arm is on, the rotation is done by J6, which starts the scoop at ~-52
    and hits its -120.3 limit. The end pose itself is reachable in hundreds of
    OTHER arm shapes (7-DoF redundancy): J2 leaning forward and J1/J3 turning
    take over the pitch. So: find the reachable end shape nearest the punch-end
    shape, then solve IK at n points along the scoop with the guess blended
    from one to the other, keeping the one continuous branch.
    """
    import itertools
    from kortex_api.autogen.messages import Base_pb2
    start = [w for w in wps if w["step"] == "PUNCH"][-1]
    qs = start["joints"]
    R0 = R_from_kortex(*start["rpy"])
    G = a._geo
    o_piv, piv0, rise = np.array(G["o_piv"]), np.array(G["pivot0"]), G["rise"]

    def pose(s_):
        th = 90.0 * s_ ** a.rot_exp
        R = rot_x(th) @ R0
        return np.array(piv0) + [0, 0, rise * s_] - R @ o_piv, kortex_from_R(R), th

    def ik(T, rpy, g):
        d = Base_pb2.IKData(); pp = d.cartesian_pose
        pp.x, pp.y, pp.z = map(float, T)
        pp.theta_x, pp.theta_y, pp.theta_z = map(float, rpy)
        for v in g:
            d.guess.joint_angles.add().value = v % 360.0
        try:
            return [fold(j.value) for j in conn.base.ComputeInverseKinematics(d).joint_angles]
        except Exception:  # noqa: BLE001
            return None

    T, r, _ = pose(1.0)
    ends = []
    for j1, j2, j3, j4, j6, fl in itertools.product(
            (qs[0], 150, 120, 100), (-30, 0, 30, 60), (qs[2], 30, 60, 85),
            (-135, -110), (-85, -50, 50, 85), (0, 1)):
        g = [j1, j2, j3, j4, qs[4], j6, qs[6]]
        if fl:
            g[4], g[6] = fold(g[4] + 180), fold(g[6] + 180)
        q = ik(T, r, g)
        if q and min(LIMITS[i] - abs(q[i]) for i in LIMITS) > 8:
            ends.append(q)
    ends.sort(key=lambda q: max(abs(fold(x - y)) for x, y in zip(q, qs)))
    for qe in ends[:10]:
        prev, path, worst, maxstep = qs, [qs], 1e9, 0.0
        for k in range(1, n + 1):
            s_ = k / n
            T, r, th = pose(s_)
            q = ik(T, r, [fold(x + fold(y - x) * s_) for x, y in zip(qs, qe)])
            if q is None:
                break
            step = max(abs(fold(x - y)) for x, y in zip(q, prev))
            if step > 12:
                break
            maxstep = max(maxstep, step)
            worst = min(worst, min(LIMITS[i] - abs(q[i]) for i in LIMITS))
            prev = q
            path.append(q)
        else:
            return dict(path=path, max_step=maxstep, min_margin=worst, n_end_candidates=len(ends))
    return dict(path=None, n_end_candidates=len(ends))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target.json from make perceive")
    ap.add_argument("--pose", default="calibration/food_detect_pose.json")
    ap.add_argument("--z-tip", type=float, default=0.115, help="camera -> tip along the utensil axis, m (user-measured 115 mm)")
    ap.add_argument("--tip-len", type=float, default=0.09, help="tool point -> tip, m (GUESS)")
    ap.add_argument("--lateral", type=float, nargs=2, default=(0.0, 0.0), help="tip offset, tool x y, m")
    ap.add_argument("--margin", type=float, default=None,
                    help="override: stop this far above the floor, m. Default floor + height/3")
    ap.add_argument("--food-up", type=float, default=0.012, help="pivot above the tip, m")
    ap.add_argument("--slow-before", type=float, default=0.020, help="go slow this far above the food, m")
    ap.add_argument("--n", type=int, default=12, help="scoop waypoints")
    ap.add_argument("--rot-exp", type=float, default=1.6,
                    help="theta = 90*s^exp; >1 stays steep while still inside the bowl")
    ap.add_argument("--wrist-flip", action="store_true",
                    help="plan on the flipped wrist branch: J5+180, J6 -> -J6, J7+180. Same tool "
                         "pose, J6 on the positive side -- which is the side the scoop unwinds")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="plan")
    a = ap.parse_args()

    tg = json.loads(Path(a.target).read_text())
    pose = json.loads(Path(a.pose).read_text())
    tp = pose["tool_pose"]
    tool0 = np.array([tp["x_m"], tp["y_m"], tp["z_m"]])
    rpy0 = (tp["theta_x_deg"], tp["theta_y_deg"], tp["theta_z_deg"])
    goal = tg["tool_goal"]
    top = tg["berry_depth_m"] - 0.008                 # locate_target stores the cap centre
    floor = tg.get("floor_beside_m") or tg["bowl_floor_mm"] / 1000.0

    wps, geo = plan_steps(tool0, rpy0, goal[:2], top, floor, a)
    a._geo = geo

    from kinova.connection import kinova_connection
    with kinova_connection() as conn:
        q_start = list(pose["joints_deg"])
        if a.wrist_flip:
            q_start[4] = fold(q_start[4] + 180.0)
            q_start[5] = -q_start[5]
            q_start[6] = fold(q_start[6] + 180.0)
        res = check_ik(conn, wps, q_start)
        joint_scoop = scoop_joint_path(conn, res, a) if all(
            r["ok"] for r in res if r["step"] != "SCOOP") else dict(path=None)

    print("BRANCH          %s" % ("wrist FLIPPED (J6 positive side)" if a.wrist_flip else "as recorded (J6 negative side)"))
    print("GEOMETRY USED   z_tip %.0f mm (user-measured)  tip_len %.0f mm (guess)  lateral %s mm"
          % (1000 * a.z_tip, 1000 * a.tip_len, [round(1000 * v) for v in a.lateral]))
    print("  food surface %.0f mm below the camera, floor beside it %.0f mm"
          % (1000 * top, 1000 * floor))
    print("  punch: %.0f mm down to the food, %.0f mm to the stop (%.0f mm above the floor = height/3)"
          % (1000 * geo["d_top"], 1000 * geo["d_end"], 1000 * geo["stop_above_floor"]))
    print("  scoop: pivot rises %.0f mm while rotating 0 -> 90 deg about world +X" % (1000 * geo["rise"]))
    print()
    print("  %-6s %-30s %-24s %s" % ("step", "", "tool xyz (m)", "IK"))
    last = None
    for r in res:
        if r["step"] != last:
            print("  " + "-" * 92); last = r["step"]
        xyz = "(%.3f %.3f %.3f)" % tuple(r["tool"])
        if r["ok"]:
            flag = "" if (r["jump"] < 25 and r["margin"] > 3) else "   <-- CHECK"
            print("  %-6s %-30s %-24s ok  jump %5.1f deg  tightest J%d margin %5.1f deg%s"
                  % (r["step"], r["label"], xyz, r["jump"], r["tightest"], r["margin"], flag))
        else:
            print("  %-6s %-30s %-24s UNREACHABLE: %s" % (r["step"], r["label"], xyz, r["error"]))
    print()
    if joint_scoop.get("path"):
        P = joint_scoop["path"]
        rng = ["J%d %.0f..%.0f" % (i + 1, min(q[i] for q in P), max(q[i] for q in P)) for i in range(7)]
        print("  SCOOP as a joint-space path: CONTINUOUS over %d waypoints, max step %.1f deg, "
              "min joint margin %.1f deg" % (len(P) - 1, joint_scoop["max_step"], joint_scoop["min_margin"]))
        print("    (the Cartesian rows above hit J6's limit; this path reshapes the arm instead)")
        print("    " + "  ".join(rng))
    else:
        print("  SCOOP: no continuous joint-space path found (%d end candidates)"
              % joint_scoop.get("n_end_candidates", 0))
    Path(a.out + ".json").write_text(json.dumps(dict(args={k: v for k, v in vars(a).items() if k != "_geo"},
                                                     geometry=geo, waypoints=res, scoop_joint_path=joint_scoop),
                                                indent=2, default=list))
    print("\n  wrote %s.json   (nothing was moved)" % a.out)
    ok_pre = all(r["ok"] for r in res if r["step"] != "SCOOP")
    return 0 if (ok_pre and joint_scoop.get("path")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
