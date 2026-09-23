"""Fault injection: the reason this simulator exists.

docs/feeding-architecture.md Sec. 10 is explicit that the highest-value use of
simulation here is not policy training, it is "testing the failure paths that
cannot be safely tested on a person". It lists five, and each maps to something
here:

  detection returns nothing            -> detection_blackout
  detection returns a phantom cube     -> phantom_food
  face track lost mid-transport        -> face_lost
  the depth stream stalls (~62 s)      -> Cameras(stalls=True), measured on hardware
  the bite never comes                 -> simply never call Human.start_bite
  the user leans into the fork         -> lean_schedule, driven through Human

Faults are set from OUTSIDE the robot's control flow and may change at any
instant. Nothing in the feeding stack is allowed to read this object: a stack
that can see the fault is not being tested against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Faults:
    # --- perception ---------------------------------------------------------
    detection_blackout: bool = False
    face_lost: bool = False
    phantom_food: bool = False
    drop_rate: float = 0.0          # fraction of true detections silently lost
    position_bias: np.ndarray | None = None   # metres, added to every detection

    # --- the user -----------------------------------------------------------
    # (time_s, lean_radians) steps, applied by the harness through Human.
    lean_schedule: list = field(default_factory=list)

    _fired: set = field(default_factory=set)

    # ------------------------------------------------------------------------
    def corrupt_food(self, pieces, world):
        """Apply the detection-layer faults to a list of true detections."""
        out = list(pieces)

        if self.drop_rate > 0 and out:
            keep = world.rng.random(len(out)) >= self.drop_rate
            out = [p for p, k in zip(out, keep) if k]

        if self.position_bias is not None:
            for p in out:
                p.position = p.position + np.asarray(self.position_bias, float)

        if self.phantom_food:
            # A phantom sits ABOVE the plate rim with plausible geometry and a
            # high confidence score -- the dangerous kind. A detector that only
            # ever invented obviously-bad boxes would not be worth testing
            # against, because the rim check alone would catch it.
            from perception import FoodPiece
            base = out[0].position.copy() if out else np.array([0.72, -0.10, 0.782])
            ghost = base + np.array([0.03, 0.03, 0.0])
            out.append(FoodPiece(index=-1, position=ghost, surface=ghost,
                                 half_extents=np.array([0.0075] * 3),
                                 confidence=0.95, valid_fraction=1.0))
        return out

    # ------------------------------------------------------------------------
    def lean_for(self, t: float) -> float | None:
        """Lean angle to command at time `t`, or None if nothing is scheduled."""
        due = [(ts, a) for ts, a in self.lean_schedule if ts <= t]
        return max(due)[1] if due else None
