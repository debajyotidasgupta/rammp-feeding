"""6-DoF IK for the Gen3 in the feeding scene.

Position-only IK is not enough here: the fork has a direction. Food is taken off
the FRONT of the tines, so at PRESENT the tine axis must point AT the mouth, and
at SKEWER it must point into the plate. Orientation is a requirement, not a
nicety -- solving position alone presents the food edge-on.

Solved on a SCRATCH MjData, never the live one. The previous version iterated
directly on `d.qpos` and left the solution there, so every caller silently
teleported the arm to the goal before starting to servo toward it -- the physics
then ran from a corrupted initial condition at each phase boundary.
"""
from __future__ import annotations

import mujoco
import numpy as np

RETRACT = np.array([0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633])


def _wrap(a: np.ndarray) -> np.ndarray:
    """Map angles into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def frame_from_axis(axis: np.ndarray, up_hint=np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Rotation whose local +z is `axis` (the fork's handle->tines direction)."""
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    if abs(z @ up_hint) > 0.95:                 # degenerate; pick another hint
        up_hint = np.array([1.0, 0.0, 0.0])
    x = np.cross(up_hint, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


class IK:
    """Holds the scratch state so repeated solves do not reallocate."""

    # Where to push the elbow when the arm has a choice. The Gen3 is 7-DoF, so
    # a position+orientation goal leaves exactly one free dimension, and that
    # dimension IS "where the elbow goes". Left unconstrained, damped least
    # squares put the elbow at (-0.02, -0.09, 0.84) -- inside the seated user's
    # chest -- while the fork sat obediently on target. The arm is mounted on
    # the user's right at -Y, so -Y is its own side of the chair.
    ELBOW_BODY = "arm_forearm_link"
    ELBOW_PUSH = np.array([0.0, -1.0, 0.0])

    def __init__(self, world):
        self.w = world
        self.scratch = mujoco.MjData(world.model)
        self.jacp = np.zeros((3, world.model.nv))
        self.jacr = np.zeros((3, world.model.nv))
        self.jace = np.zeros((3, world.model.nv))
        self.elbow_bid = world.bid(self.ELBOW_BODY)

    def solve(self, pos, R_des=None, q0=None, site: str = "arm_tcp",
              iters: int = 800, pos_tol: float = 8e-4, rot_tol: float = 0.02,
              w_rot: float = 0.6, elbow_gain: float = 0.6, restarts: int = 6):
        """Solve, retrying from other seeds if the first attempt stalls.

        Damped least squares on a 7-DoF arm with an orientation constraint has
        local minima, and it does not announce them: it returns a pose tens of
        millimetres and several degrees off, which downstream looks like a
        slightly sloppy move rather than a failed solve. Retrying from a handful
        of seeds and keeping the best costs a few milliseconds and turns a silent
        near-miss into either a good solution or an honest failure the caller can
        see in the residual.
        """
        best = None
        seeds = [q0 if q0 is not None else RETRACT, RETRACT, self.HOME]
        for k in range(max(1, restarts)):
            seed = seeds[k] if k < len(seeds) else \
                np.clip(seeds[0] + self.w.rng.normal(0, 0.8, 7), self.w.arm_lo, self.w.arm_hi)
            q, pe, re = self._solve_once(pos, R_des, seed, site, iters, pos_tol,
                                         rot_tol, w_rot, elbow_gain)
            score = pe + np.radians(re) * 0.05
            if best is None or score < best[0]:
                best = (score, q, pe, re)
            if pe < pos_tol * 2 and re < 0.5:
                break
        return best[1], best[2], best[3]

    HOME = np.array([0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633])

    def _solve_once(self, pos, R_des=None, q0=None, site: str = "arm_tcp",
                    iters: int = 800, pos_tol: float = 8e-4, rot_tol: float = 0.02,
                    w_rot: float = 0.6, elbow_gain: float = 0.6):
        """Damped least-squares IK on the 7 arm joints.

        Returns (q, pos_err_m, rot_err_deg). The live sim state is untouched.
        """
        w, sd = self.w, self.scratch
        sd.qpos[:] = w.data.qpos          # start from the world as it is, so
        sd.qvel[:] = 0.0                  # the user's lean is accounted for
        sid = w.sid(site)
        q = (RETRACT if q0 is None else np.asarray(q0, float)).copy()
        aa = np.zeros(3)

        for _i in range(iters):
            sd.qpos[w.arm_qadr] = q
            mujoco.mj_kinematics(w.model, sd)
            mujoco.mj_comPos(w.model, sd)
            e_pos = np.asarray(pos, float) - sd.site_xpos[sid]
            if R_des is None:
                mujoco.mj_jacSite(w.model, sd, self.jacp, None, sid)
                err, J = e_pos, self.jacp[:, w.arm_dadr]
            else:
                R = sd.site_xmat[sid].reshape(3, 3)
                qq = np.zeros(4)
                mujoco.mju_mat2Quat(qq, (R_des @ R.T).flatten())
                mujoco.mju_quat2Vel(aa, qq, 1.0)          # axis-angle of the error
                err = np.concatenate([e_pos, w_rot * aa])
                mujoco.mj_jacSite(w.model, sd, self.jacp, self.jacr, sid)
                J = np.vstack([self.jacp[:, w.arm_dadr], w_rot * self.jacr[:, w.arm_dadr]])
            if np.linalg.norm(e_pos) < pos_tol and (R_des is None or np.linalg.norm(aa) < rot_tol):
                break
            n = J.shape[0]
            Jpinv = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(n))
            dq = Jpinv @ err
            # Decay the secondary objective to zero over the solve. The damped
            # pseudo-inverse's null projector is only approximate, so a constant
            # bias leaks into the primary task and the solve stalls tens of mm
            # off target. Shaping the redundancy early and finishing on the pure
            # task gives both.
            g = elbow_gain * max(0.0, 1.0 - 2.0 * _i / iters)
            if g:
                # Secondary objective in the null space: it cannot move the
                # fork, so the tool goal is never traded away for it.
                mujoco.mj_jacBody(w.model, sd, self.jace, None, self.elbow_bid)
                grad = self.jace[:, w.arm_dadr].T @ self.ELBOW_PUSH
                dq = dq + (np.eye(7) - Jpinv @ J) @ (g * grad)
            q = np.clip(q + dq * 0.5, w.arm_lo, w.arm_hi)

        # Unwind the continuous joints. Gen3 joints 1/3/5/7 have no limits, and
        # damped least squares happily walks one of them through several full
        # turns: a solve seeded at LIFT came back with joint 3 at 502 deg, a
        # 380 deg journey to a pose 130 deg away. Kinematically it is the same
        # arm; as a PATH it swept the upper arm 161 mm through the user's chest.
        # Snap every unlimited joint onto the branch nearest the seed.
        seed = (RETRACT if q0 is None else np.asarray(q0, float))
        free = ~w.arm_limited
        q[free] = seed[free] + _wrap(q[free] - seed[free])

        sd.qpos[w.arm_qadr] = q
        mujoco.mj_kinematics(w.model, sd)
        mujoco.mj_comPos(w.model, sd)
        pe = float(np.linalg.norm(np.asarray(pos, float) - sd.site_xpos[sid]))
        if R_des is None:
            return q, pe, 0.0
        R = sd.site_xmat[sid].reshape(3, 3)
        cos = (np.trace(R_des @ R.T) - 1) / 2
        return q, pe, float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
