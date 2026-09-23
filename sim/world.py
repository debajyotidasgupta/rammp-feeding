"""The simulated feeding cell: one MuJoCo model, and named views onto it.

Everything downstream addresses the model BY NAME through this object. That is
not stylistic. Adding the user's lean hinge moved the arm's joints from index
0..6 to 1..7, and every `d.qpos[:7]` in the codebase silently kept running
against the wrong DoF. Index arithmetic over a model that is still growing is a
bug waiting for a scene edit.
"""
from __future__ import annotations

import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
SCENE = HERE / "feeding_scene.xml"

ARM_JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
FOOD_BODIES = [f"food{i}" for i in range(4)]


class World:
    """Owns the model and data. Views (Arm, Human, Cameras) hold a reference."""

    def __init__(self, xml: str | pathlib.Path = SCENE, seed: int = 0,
                 model_error: float = 0.03):
        missing = [m for m in ("human_lower.obj", "human_upper.obj")
                   if not (HERE / "assets" / "human" / m).exists()]
        if missing:
            raise FileNotFoundError(
                "The posed human meshes are missing: %s.\n"
                "They are derivatives of the licensed SMPL-X body model and are "
                "not kept in the repo. Regenerate them with:\n"
                "    python sim/pose_human.py\n"
                "which needs torch + smplx and sim/assets/smplx/SMPLX_NEUTRAL.npz."
                % ", ".join(missing))
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.dt = float(self.model.opt.timestep)

        # --- named lookups, resolved once -----------------------------------
        self.arm_jid = np.array([self.jid(n) for n in ARM_JOINTS])
        self.arm_qadr = np.array([self.model.jnt_qposadr[j] for j in self.arm_jid])
        # Joint limits, addressed by JOINT id (not qpos address -- they only
        # coincide while every joint is 1-DoF) and with the unlimited joints
        # left unbounded: Gen3 joints 1/3/5/7 are continuous, and their
        # jnt_range is [0, 0], which clamps them to zero if taken literally.
        self.arm_limited = self.model.jnt_limited[self.arm_jid].astype(bool)
        self.arm_lo = np.where(self.arm_limited, self.model.jnt_range[self.arm_jid, 0], -np.inf)
        self.arm_hi = np.where(self.arm_limited, self.model.jnt_range[self.arm_jid, 1], np.inf)
        self.arm_dadr = np.array([self.model.jnt_dofadr[j] for j in self.arm_jid])
        self.arm_uadr = np.array([self.aid(n) for n in ARM_JOINTS])
        self.food_bid = np.array([self.bid(n) for n in FOOD_BODIES])
        self.food_eq = np.array([self.eqid(f"hold_{n}") for n in FOOD_BODIES])

        # --- the estimator's model of the plant -----------------------------
        # The bite detector subtracts a PREDICTED gravity/Coriolis torque from
        # the measured one. On hardware that prediction comes from the driver's
        # Pinocchio model, which is never exactly the real arm: link masses are
        # nominal, the fork and food are not in it at all. Giving the estimator
        # a deliberately imperfect copy is what makes the simulated residual
        # honest -- with a perfect model the detection threshold could be set
        # arbitrarily low, which is precisely the wrong lesson to carry to
        # hardware.
        self.model_error = model_error
        self.est = mujoco.MjModel.from_xml_path(str(xml))
        if model_error:
            scale = 1.0 + self.rng.normal(0.0, model_error, size=self.est.body_mass.shape)
            self.est.body_mass[:] *= np.abs(scale)
            self.est.body_inertia[:] *= np.abs(scale)[:, None]
        self.est_data = mujoco.MjData(self.est)

        self.reset()

    # ------------------------------------------------------------------ ids
    def _id(self, objtype, name: str) -> int:
        i = mujoco.mj_name2id(self.model, objtype, name)
        if i < 0:
            raise KeyError(f"{name!r} not in the model")
        return i

    def jid(self, n): return self._id(mujoco.mjtObj.mjOBJ_JOINT, n)
    def bid(self, n): return self._id(mujoco.mjtObj.mjOBJ_BODY, n)
    def sid(self, n): return self._id(mujoco.mjtObj.mjOBJ_SITE, n)
    def gid(self, n): return self._id(mujoco.mjtObj.mjOBJ_GEOM, n)
    def aid(self, n): return self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def eqid(self, n): return self._id(mujoco.mjtObj.mjOBJ_EQUALITY, n)
    def sensor(self, n: str) -> np.ndarray:
        i = self._id(mujoco.mjtObj.mjOBJ_SENSOR, n)
        a = self.model.sensor_adr[i]
        return self.data.sensordata[a:a + self.model.sensor_dim[i]]

    # ---------------------------------------------------------------- state
    @property
    def t(self) -> float:
        return float(self.data.time)

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.sid(name)].copy()

    def site_mat(self, name: str) -> np.ndarray:
        return self.data.site_xmat[self.sid(name)].reshape(3, 3).copy()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def steps_for(self, seconds: float) -> int:
        return max(1, int(round(seconds / self.dt)))
