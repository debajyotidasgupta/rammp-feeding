"""Small moves by hand, one per command, slow, from wherever the arm is: for setting up and for easing the arm off a
spot. Each is checked against the world like any of our own paths (Arm.check_world: what the gripper holds kept over
our floor and off our boxes, the joints inside their limits; into: the bowl, the stand or the user the move goes into,
where its own guard applies instead) and runs under the contact guard.

    rammp move DX DY DZ     the tool straight by (DX, DY, DZ) m, base frame, its orientation kept
    rammp up D | down D     straight up / down D m
    rammp turn DEG          turned DEG about the vertical through the tip (counter-clockwise seen from above)
    rammp point-down        turned about the tip until the tool points straight down (its turn about the vertical kept)
    rammp joint J DEG       joint J (1..7) alone by DEG

What the gripper holds is the utensil's state's (rammp_feeding/fork.py): the fork, or, where the fork is picked from a
stand and is on it, the empty gripper (its closed fingers' ends the tip). A move longer than jog.max_m or turning more
than jog.max_deg is refused (metres typed for centimetres).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from rammp_feeding.arm import Arm, MoveError
from rammp_feeding.config import cfg
from rammp_feeding.geometry import joint_delta, rodrigues, rot, rot_log

log = logging.getLogger("rammp")
ARGS = {"move": 3, "up": 1, "down": 1, "turn": 1, "point-down": 0, "joint": 2}   # each command's numbers


def J():
    return cfg.motion.jog


def held():
    """What the gripper holds now, for Arm(held=...): None (the fork, the meal's checks) unless a stand is used and the
    fork is on it -- then the empty gripper, in the stand's column where the stand is known."""
    from rammp_feeding import fork
    if not fork.used() or fork.holding():
        return None
    s = fork.state()
    at = s.get("stand") or s.get("hole")
    return fork.AtStand(fork.Hole.from_json(at).centre if at else None)


def move(arm: Arm, d, into=()) -> None:
    """The tool straight by d (m, base frame), its orientation kept."""
    d = np.asarray(d, float)
    if np.linalg.norm(d) > float(J().max_m):
        raise MoveError("a move of %.0f mm: at most %.0f (motion.jog.max_m) -- metres, not centimetres?"
                        % (1000 * np.linalg.norm(d), 1000 * float(J().max_m)))
    T, R = arm.tool()
    arm.straight(T + d, R, float(J().mps), "moved (%+.1f, %+.1f, %+.1f) mm" % tuple(1000 * d), inside=into)


def turn_to(arm: Arm, Rg, label: str, into=()) -> None:
    """The tool turned to orientation Rg about the tip (it stays where it is), evenly: a continuous IK path."""
    q0 = arm.q()
    T0, R0 = arm.tool(q0)
    tip = T0 + R0 @ arm.tip_tool
    w = rot_log(np.asarray(Rg) @ R0.T)
    deg = math.degrees(float(np.linalg.norm(w)))
    if deg < 0.5:
        log.info("    %-28s already there", label)
        return
    if deg > float(J().max_deg):
        raise MoveError("%s: a turn of %.0f deg, at most %.0f (motion.jog.max_deg)" % (label, deg, float(J().max_deg)))

    def at(s):
        Rs = rodrigues(w * s) @ R0
        return tip - Rs @ arm.tip_tool, Rs
    path, why = arm.homotopy(q0, at, max(4, math.ceil(deg / 2.0)))
    if path is None:
        raise MoveError("%s: %s -- not moving" % (label, why))
    arm.check_world(path, label, into)
    arm.run(path, label, dps=float(J().dps))


def turn(arm: Arm, deg: float, into=()) -> None:
    turn_to(arm, rot("z", deg) @ arm.tool()[1], "turned %+.1f deg about the vertical" % deg, into)


def point_down(arm: Arm, into=()) -> None:
    """The least turn that points the tool's axis (z) straight down."""
    R = arm.tool()[1]
    z, down = R[:, 2], np.array([0.0, 0.0, -1.0])
    axis = np.cross(z, down)
    ang = math.atan2(float(np.linalg.norm(axis)), float(z @ down))
    w = np.zeros(3) if np.linalg.norm(axis) < 1e-9 else axis / np.linalg.norm(axis) * ang
    turn_to(arm, rodrigues(w) @ R, "pointed straight down (%.1f deg)" % math.degrees(ang), into)


def joint(arm: Arm, j: int, deg: float, into=()) -> None:
    """Joint j (1..7) alone by deg, the tool's sweep checked against the world first."""
    if not 1 <= j <= 7:
        raise ValueError("joint 1 .. 7, not %d" % j)
    if abs(deg) > float(J().max_deg):
        raise MoveError("joint %d by %.0f deg: at most %.0f (motion.jog.max_deg)" % (j, deg, float(J().max_deg)))
    q0 = arm.q()
    goal = list(q0)
    goal[j - 1] += deg
    n = max(2, math.ceil(abs(deg) / 1.0))
    path = [list(np.asarray(q0) + joint_delta(q0, goal) * k / n) for k in range(n + 1)]
    label = "joint %d by %+.1f deg" % (j, deg)
    arm.check_world(path, label, into)
    arm.goto(goal, label, dps=float(J().dps))


def run(arm: Arm, command: str, args: list[float], into=()) -> None:
    if command == "move":
        move(arm, args, into)
    elif command in ("up", "down"):
        move(arm, [0.0, 0.0, args[0] if command == "up" else -args[0]], into)
    elif command == "turn":
        turn(arm, args[0], into)
    elif command == "point-down":
        point_down(arm, into)
    elif command == "joint":
        joint(arm, int(args[0]), args[1], into)
