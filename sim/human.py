"""The person being fed: what they do, and what the robot must detect.

Two behaviours, and they are the two that the system cannot be allowed to get
wrong:

  bite  -- a transient tug on the food along the fork axis. Modelled as a real
           force on the food body, which propagates through the weld into the
           arm and shows up in the joint-torque residual. The detector is not
           told a bite happened; it has to see it. (Sec. 6)
  lean  -- the user closing on the fork under their own power. The scenario the
           whole safety architecture is built around. (Sec. 0, Sec. 10)

Neither is scripted into the robot's state machine. Both are things the world
does TO the robot, which is the only arrangement in which a safety monitor can
actually be tested.
"""
from __future__ import annotations

import numpy as np


class Human:
    # Force needed to pull a soft piece off fork tines. Assistive-feeding rigs
    # report single-newton pull-off forces; a firm piece is a few N. Treated as
    # a range to sweep, not a constant to trust -- Sec. 9 lists characterising
    # real bite force as an open question, and it should be measured.
    BITE_FORCE_RANGE = (1.0, 5.0)

    def __init__(self, world, arm):
        self.w = world
        self.arm = arm
        self.lean_u = world.aid("user_lean")
        self._bite = None          # (body_id, direction, peak, t0, rise, hold)
        self.lean_cmd = 0.0

    # ------------------------------------------------------------------ lean
    def set_lean(self, radians: float) -> None:
        """Command a lean angle. Positive pitches the torso forward."""
        lo, hi = self.w.model.actuator_ctrlrange[self.lean_u]
        self.lean_cmd = float(np.clip(radians, lo, hi))
        self.w.data.ctrl[self.lean_u] = self.lean_cmd

    @property
    def lean(self) -> float:
        return float(self.w.data.qpos[self.w.model.jnt_qposadr[self.w.jid("user_lean")]])

    def mouth(self) -> np.ndarray:
        """Ground-truth mouth position. Moves with the lean."""
        return self.w.site_pos("mouth")

    def nose(self) -> np.ndarray:
        """Ground-truth nose tip -- the stable landmark for range estimation.

        The mouth changes shape when it opens; the nose does not.
        """
        return self.w.site_pos("nose")

    def head_sphere(self) -> tuple[np.ndarray, float]:
        gid = self.w.gid("head_nogo")
        return self.w.data.geom_xpos[gid].copy(), float(self.w.model.geom_size[gid, 0])

    # ------------------------------------------------------------------ bite
    def start_bite(self, food_body: int, peak: float | None = None,
                   rise: float = 0.12, hold: float = 0.25) -> None:
        """Begin tugging the piece off the fork, along the fork's own axis.

        A real bite is a pull, not an impulse: lips close, then the head draws
        back. Rise time is what the detector's minimum-duration test keys on --
        an instantaneous step would be indistinguishable from a bump.
        """
        if peak is None:
            peak = float(self.w.rng.uniform(*self.BITE_FORCE_RANGE))
        _, R = self.arm.tcp()
        axis = R[:, 2] / np.linalg.norm(R[:, 2])     # tine direction
        self._bite = dict(body=int(food_body), dir=axis, peak=float(peak),
                          t0=self.w.t, rise=rise, hold=hold)

    @property
    def biting(self) -> bool:
        return self._bite is not None

    def bite_force(self) -> float:
        if not self._bite:
            return 0.0
        b = self._bite
        dt = self.w.t - b["t0"]
        if dt < 0:
            return 0.0
        if dt < b["rise"]:
            return b["peak"] * dt / b["rise"]
        if dt < b["rise"] + b["hold"]:
            return b["peak"]
        return 0.0

    def apply(self) -> None:
        """Push this step's human-generated forces into the sim. Call each step."""
        if not self._bite:
            return
        b = self._bite
        f = self.bite_force()
        self.w.data.xfrc_applied[b["body"], :3] = b["dir"] * f
        if self.w.t - b["t0"] > b["rise"] + b["hold"]:
            self.w.data.xfrc_applied[b["body"]] = 0.0
            self._bite = None

    def stop_bite(self) -> None:
        if self._bite:
            self.w.data.xfrc_applied[self._bite["body"]] = 0.0
        self._bite = None
