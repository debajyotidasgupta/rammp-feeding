"""Detect that the user has taken the bite, from the arm's own joint torques.

Structure and most of the tuning logic is adapted from
`empriselab/feeding-deployment`'s `joint_torque_bite_detector`, which was fit on
~300 bites across 20 days of real deployment. What is NOT adapted is which
joints to watch, and that difference is the whole point:

    their fork is PERPENDICULAR to the tool axis, on an actuated wrist. The tip
    sits ~0.255 m off joint 7's axis, so joint 7 has the largest moment arm of
    any joint AND the quietest baseline (0.021 Nm sd, SNR ~19). They measured
    that joint 7 alone beats all seven on every axis.

    ours is INLINE with the tool axis. Measured on this arm with a 5 N axial
    pull at the tines: joints 5, 6 and 7 carry 0.0% of the signal. The Gen3's
    spherical wrist has its three axes intersecting, so an axial force's line of
    action passes through that point and makes no moment about any of them.
    Joint 2 carries 33%, joint 1 29%, joint 3 23%, joint 4 15%.

So the best channel there is the worst channel here, from fork orientation
alone. We watch joints 1-4. They are noisier (0.021-0.045 Nm vs joint 7's
0.010) and carry more model error, but they are the only joints that see the
load at all.

No ROS, no numpy beyond arrays: the version validated offline must be the
version that runs on the robot.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

N_ARM_JOINTS = 7


@dataclass
class BiteDetectorConfig:
    # --- settling gate ------------------------------------------------------
    # The arm must be STILL before a baseline means anything. While it settles
    # into the transfer pose the torque sd is 0.3-0.85 Nm -- the same size as a
    # bite -- so no threshold can separate them. Refuse to arm instead of
    # guessing, and report it via `ready`.
    settle_speed: float = 0.02      # rad/s; a held arm reads ~0.0001
    settle_time: float = 0.35       # s of continuous stillness

    # --- baseline / current windows (s) -------------------------------------
    # span + gap sets how soon after arriving a bite can be caught, and that is
    # the binding constraint in practice, not the threshold: upstream measured
    # that shortening 0.50/0.20 to 0.25/0.15 took the fraction of real bites the
    # detector could even ATTEMPT from 31% to 75%, at equal accuracy. Users bite
    # sooner than a long baseline allows.
    baseline_span: float = 0.25
    # Dead time between the baseline and now. Must exceed the bite's rise time
    # or the bite contaminates its own reference. ~150 ms measured upstream;
    # our own demo showed a ~120 ms rise.
    baseline_gap: float = 0.15
    current_span: float = 0.04

    # --- trigger ------------------------------------------------------------
    # An ABSOLUTE floor separates the populations; k*sigma scales with local
    # noise and shrinks signal and disturbance together. Upstream swept this on
    # 246 settled bites and found detection RISES with the floor, because
    # suppressing an early-contact fire leaves the detector armed for the real
    # bite rather than burning the window.
    #
    # Their 0.25 Nm was tuned for joint 7 with a 0.255 m moment arm. Ours is a
    # different channel entirely, so this is a STARTING POINT, not a measured
    # value: from our sim, a 5 N axial pull moves joint 2 by ~1.9 Nm, and the
    # measured resting noise on joints 1-4 is 0.021-0.045 Nm. Re-sweep against
    # real bites before trusting it.
    abs_min_nm: float = 0.25
    k_sigma: float = 5.0
    sigma_floor: float = 0.01
    confirm_samples: int = 3        # at our 40 Hz logging rate, ~75 ms

    # Joints watched, 0-indexed. See the module docstring: the wrist is blind to
    # an axial pull on an inline fork.
    joints: tuple[int, ...] = (0, 1, 2, 3)

    # The Kinova reports all-zero torques in joint-compliant mode; a stray zero
    # frame must not read as a multi-Nm step.
    zero_torque_eps: float = 1e-6

    # Freeze the baseline once settled rather than sliding it. A sliding
    # baseline treats sustained load as the new normal, so a user who bites and
    # HOLDS stops being detected after a few hundred ms.
    freeze_baseline: bool = True

    timeout_s: float = 20.0


@dataclass
class BiteResult:
    outcome: str                    # "bite" | "timeout"
    t: float = 0.0
    joint: int = -1
    deviation_nm: float = 0.0
    baseline_nm: float = 0.0


class JointTorqueBiteDetector:
    """Feed it (t, q_dot, tau). It tells you when the user took the bite."""

    def __init__(self, cfg: BiteDetectorConfig | None = None):
        self.cfg = cfg or BiteDetectorConfig()
        self.reset(0.0)

    def reset(self, t: float) -> None:
        self._t0 = t
        self._hist: collections.deque = collections.deque(maxlen=4096)
        self._still_since: float | None = None
        self._baseline: dict[int, tuple[float, float]] = {}
        self._over = 0
        self._fired = False

    @property
    def ready(self) -> bool:
        """True once the arm has been still long enough to trust a baseline."""
        return bool(self._baseline)

    def update(self, t: float, qdot, tau) -> BiteResult | None:
        c = self.cfg
        if max(abs(float(v)) for v in tau) < c.zero_torque_eps:
            return None                      # all-zero frame; ignore
        self._hist.append((t, [float(x) for x in tau]))

        moving = max(abs(float(v)) for v in qdot) > c.settle_speed
        if moving:
            self._still_since = None
            self._baseline = {}
            self._over = 0
        elif self._still_since is None:
            self._still_since = t

        if self._still_since is None or t - self._still_since < c.settle_time:
            return self._timeout(t)

        # baseline window sits entirely inside the settled stretch, and ends
        # `gap` before now so the bite's leading edge stays out of it
        if not self._baseline or not c.freeze_baseline:
            lo, hi = t - c.baseline_gap - c.baseline_span, t - c.baseline_gap
            if lo < self._still_since:
                return self._timeout(t)
            for j in c.joints:
                vals = [v[j] for (ts, v) in self._hist if lo <= ts <= hi]
                if len(vals) < 3:
                    return self._timeout(t)
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                self._baseline[j] = (mean, max(var ** 0.5, c.sigma_floor))

        cur_lo = t - c.current_span
        hit = None
        for j in c.joints:
            vals = [v[j] for (ts, v) in self._hist if ts >= cur_lo]
            if not vals:
                continue
            cur = sum(vals) / len(vals)
            base, sd = self._baseline[j]
            dev = abs(cur - base)
            if dev >= max(c.abs_min_nm, c.k_sigma * sd):
                if hit is None or dev > hit[1]:
                    hit = (j, dev, base)
        if hit is None:
            self._over = 0
            return self._timeout(t)
        self._over += 1
        if self._over >= c.confirm_samples:
            return BiteResult("bite", t, hit[0], hit[1], hit[2])
        return self._timeout(t)

    def _timeout(self, t: float) -> BiteResult | None:
        if t - self._t0 >= self.cfg.timeout_s:
            return BiteResult("timeout", t)
        return None
