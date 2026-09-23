"""Where to put the fork into a food piece, and how to keep it there.

Two things the naive version got wrong:

1. Aiming at the 2D centroid and touching the surface does not skewer anything.
   The tines must pass THROUGH the thickest cross-section, entering at the
   centre of the face presented to the approach and travelling deep enough that
   the food is gripped along the tine shafts, not balanced on the points.

2. A tilted fork sheds its load. Immediately after insertion the fork must
   rotate so the tines are level (or slightly tip-up), so gravity and the tine
   shafts retain the piece instead of sliding it off the points.
"""
from __future__ import annotations
import numpy as np

# Fraction of the food's thickness the tines should travel through. Too little
# and it falls off the points; too much and the tines hit the plate.
INSERT_FRACTION = 0.70
TINE_LEN = 0.030          # matches gen3_fork.xml


def thickest_axis(half_extents: np.ndarray) -> int:
    """Index of the food's thickest dimension - the one to skewer along."""
    return int(np.argmax(half_extents))


def skewer_plan(center: np.ndarray, half_extents: np.ndarray,
                approach_axis: np.ndarray, plate_z: float):
    """Entry pose and insertion depth for one piece.

    `approach_axis` is the fork's tine direction (unit, pointing into the food).
    Returns (pre_pose, insert_pose, depth, clipped) where the poses are TCP
    (tine-tip) positions and `clipped` says the plate limited the depth.
    """
    a = np.asarray(approach_axis, float)
    a /= np.linalg.norm(a)
    he = np.asarray(half_extents, float)

    # Thickness of the piece along the approach direction: support function of a
    # box is |a . he| summed per-axis, so the half-thickness along `a` is:
    half_thickness = float(np.abs(a) @ he)
    desired = 2.0 * half_thickness * INSERT_FRACTION

    # Entry point: centre of the face the approach hits, i.e. the surface point
    # on the far side of the centre along -a.
    entry = center - a * half_thickness

    # Do not drive the tips into the plate.
    tip_at_full = entry + a * desired
    clipped = False
    if tip_at_full[2] < plate_z + 0.002:
        desired = max(0.004, (plate_z + 0.002 - entry[2]) / a[2]) if a[2] < 0 else desired
        clipped = True

    depth = min(desired, TINE_LEN * 0.9)
    pre = entry - a * 0.06                    # 6 cm standoff along the approach
    insert = entry + a * depth
    return pre, insert, float(depth), clipped


def retention_axis(approach_axis: np.ndarray, tip_up_deg: float = 8.0,
                   heading_deg: float = 0.0) -> np.ndarray:
    """Fork axis to rotate to right after insertion, so the piece is retained.

    Level, with a few degrees of tip-up so the food settles back against the
    tine shafts rather than creeping toward the points.

    What retention requires is that the fork be HORIZONTAL. Which way it points
    in plan is mechanically free, and `heading_deg` rotates it about the
    vertical. That freedom matters: pinning the heading to the approach
    direction left two of the four pieces on the plate with no reachable
    retention pose at all, while a rotated heading solves every one. Treating a
    free parameter as fixed is the difference between "cannot reach that piece"
    and "reach it from a different angle".
    """
    a = np.asarray(approach_axis, float)
    horiz = np.array([a[0], a[1], 0.0])
    if np.linalg.norm(horiz) < 1e-6:
        horiz = np.array([1.0, 0.0, 0.0])
    horiz /= np.linalg.norm(horiz)
    c, s = np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg))
    horiz = np.array([c * horiz[0] - s * horiz[1], s * horiz[0] + c * horiz[1], 0.0])
    t = np.radians(tip_up_deg)
    return np.array([horiz[0] * np.cos(t), horiz[1] * np.cos(t), np.sin(t)])


RETENTION_HEADINGS = (0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0)


def weld_here(m, d, eq_id: int, body1: int, body2: int) -> None:
    """Pin a weld to the CURRENT relative pose, then activate it.

    MuJoCo stores a weld's relative pose in the model, defaulting to the bodies'
    configuration at compile time. Activating it without updating that yanks the
    food from wherever it was to wherever it started - which looks like the piece
    being flung away. Set it from the live state instead.
    """
    import mujoco
    import numpy as np

    # pose of body2 expressed in body1's frame
    p1, r1 = d.xpos[body1], d.xmat[body1].reshape(3, 3)
    p2, r2 = d.xpos[body2], d.xmat[body2].reshape(3, 3)
    rel_p = r1.T @ (p2 - p1)
    rel_r = r1.T @ r2
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rel_r.flatten())

    m.eq_data[eq_id, 0:3] = 0.0          # anchor, in body2 frame
    m.eq_data[eq_id, 3:6] = rel_p
    m.eq_data[eq_id, 6:10] = quat
    m.eq_data[eq_id, 10] = 1.0           # torquescale
    d.eq_active[eq_id] = 1
