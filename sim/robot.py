"""The arm, as the feeding stack sees it.

Mirrors what `kinova-gen3-ros2` gives us on hardware -- joint position, velocity
and MEASURED TORQUE at the cyclic rate, plus position commands -- and adds the
one derived quantity the bite detector is built on: the gravity-compensated
torque residual (docs/feeding-architecture.md Sec. 6).

The residual is the whole point of this module, so it is worth being precise
about what it is:

    tau_measured  = motor torque the joint actually applied
                  = servo output + the driver's gravity feed-forward
    tau_predicted = the DRIVER's model of gravity + Coriolis at this (q, qd)
    residual      = tau_measured - tau_predicted

Both the feed-forward and the prediction come from `world.est`, a deliberately
imperfect copy of the plant. That matters: if the estimator used the true model
the residual would be identically zero and any threshold would "work". The
achievable threshold is set by model error, not by sensor noise, and this is
where that shows up.
"""
from __future__ import annotations

import mujoco
import numpy as np


class Arm:
    # Reported by kinova_gen3_node on startup, max_ref_speed [rad/s]
    VMAX = np.array([1.40, 1.40, 1.40, 1.40, 1.22, 1.22, 1.22])
    RETRACT = np.array([0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633])
    HOME = np.array([0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633])

    # Torque-sensor noise, 1 sigma per joint. A PLACEHOLDER: measure it on the
    # real arm by logging /joint_states at rest for 60 s and taking the per-joint
    # std. Until then the detector thresholds derived here are provisional.
    TORQUE_NOISE = 0.15          # Nm

    def __init__(self, world, gravity_comp: bool = True):
        self.w = world
        self.gravity_comp = gravity_comp
        self.q_cmd = self.RETRACT.copy()
        self._jacp = np.zeros((3, world.model.nv))
        self._jacr = np.zeros((3, world.model.nv))
        self.reset()

    # ----------------------------------------------------------------- state
    @property
    def q(self) -> np.ndarray:
        return self.w.data.qpos[self.w.arm_qadr].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.w.data.qvel[self.w.arm_dadr].copy()

    def tcp(self) -> tuple[np.ndarray, np.ndarray]:
        """Fork tine-tip pose: (position, rotation). +z of the frame is the tine axis."""
        return self.w.site_pos("arm_tcp"), self.w.site_mat("arm_tcp")

    def reset(self, q=None) -> None:
        q = self.RETRACT if q is None else np.asarray(q)
        self.w.data.qpos[self.w.arm_qadr] = q
        self.w.data.qvel[self.w.arm_dadr] = 0.0
        self.w.data.ctrl[self.w.arm_uadr] = q
        self.q_cmd = q.copy()
        mujoco.mj_forward(self.w.model, self.w.data)

    # -------------------------------------------------------------- commands
    def command(self, q) -> None:
        self.q_cmd = np.asarray(q, float).copy()
        self.w.data.ctrl[self.w.arm_uadr] = self.q_cmd

    def duration_for(self, q_from, q_to, fraction: float = 0.35, floor: float = 0.4) -> float:
        """Seconds a move needs at `fraction` of the arm's rated joint speed.

        Commanding a 115 deg wrist roll in 1.2 s asks ~96 deg/s of a joint rated
        for 70; the servo just lags and the fork is not where the plan thinks.
        """
        travel = np.abs(np.asarray(q_to) - np.asarray(q_from))
        return max(floor, float(np.max(travel / (self.VMAX * fraction))))

    # --------------------------------------------------------------- torques
    def _est_bias(self) -> np.ndarray:
        """Gravity + Coriolis at the current state, from the DRIVER's model."""
        ed = self.w.est_data
        ed.qpos[:] = self.w.data.qpos
        ed.qvel[:] = self.w.data.qvel
        mujoco.mj_forward(self.w.est, ed)
        return ed.qfrc_bias[self.w.arm_dadr].copy()

    def apply_feedforward(self) -> None:
        """Push the driver's gravity feed-forward into the plant. Call each step.

        Without it the position servos sit in a steady-state droop -- about 28
        deg with the arm extended -- that no amount of extra settling removes,
        because it is sag, not lag.
        """
        if self.gravity_comp:
            self.w.data.qfrc_applied[self.w.arm_dadr] = self._est_bias()

    def measured_torque(self) -> np.ndarray:
        """What the joint torque sensors report: servo output + feed-forward + noise."""
        tau = (self.w.data.qfrc_actuator[self.w.arm_dadr]
               + self.w.data.qfrc_applied[self.w.arm_dadr])
        return tau + self.w.rng.normal(0.0, self.TORQUE_NOISE, size=7)

    def residual(self, tau_measured=None) -> np.ndarray:
        """tau_measured - tau_predicted, per joint.

        Valid as an external-force estimate only while the arm is near-static;
        under acceleration the servo's inertial torque lands here too. PRESENT
        holds still, which is exactly why bite detection is done there.
        """
        tau = self.measured_torque() if tau_measured is None else tau_measured
        return tau - self._est_bias()

    def jacobian(self) -> np.ndarray:
        """6x7 TCP Jacobian for the arm joints only."""
        mujoco.mj_jacSite(self.w.model, self.w.data, self._jacp, self._jacr,
                          self.w.sid("arm_tcp"))
        return np.vstack([self._jacp[:, self.w.arm_dadr],
                          self._jacr[:, self.w.arm_dadr]])

    def external_wrench(self, tau_residual=None) -> np.ndarray:
        """External wrench at the TCP inferred from the residual, world frame.

        Sign is verified against the F/T sensor in tests/test_residual.py rather
        than argued from MuJoCo's force conventions.
        """
        r = self.residual() if tau_residual is None else tau_residual
        J = self.jacobian()
        return -np.linalg.lstsq(J.T, r, rcond=None)[0]

    def true_wrench(self) -> np.ndarray:
        """GROUND TRUTH from the simulated wrist F/T sensor.

        The real Gen3 has no wrist F/T. Nothing that will ever run on hardware
        may read this -- it exists to check `external_wrench` against.
        """
        return np.concatenate([self.w.sensor("arm_fork_force"),
                               self.w.sensor("arm_fork_torque")])
