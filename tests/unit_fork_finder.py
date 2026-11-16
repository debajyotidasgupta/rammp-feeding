"""The fork pass's utensil finder on real frames (python3 tests/unit_fork_finder.py): a wrist OAK-D Pro on an arm leaning 30 deg, the
ChArUco board standing in front of the fork (tests/data/fork_endon/wrist_fork0..4.jpg, as calibrate-wrist saved them),
so the fork is seen end-on, its tines pointing right in the image. The tines' tips read by eye at (360, 316) px."""

from __future__ import annotations

import math
import unittest

from fixture import ROOT  # noqa: E402  (the test site: before rammp_feeding is imported)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from rammp_feeding import wrist_calibration as wc  # noqa: E402
from rammp_feeding.camera import CameraModel, Intrinsics  # noqa: E402
from rammp_feeding.config import cfg  # noqa: E402
from rammp_feeding.geometry import quat_matrix  # noqa: E402

DATA = ROOT / "tests" / "data" / "fork_endon"
OAK = Intrinsics(398.443, 398.350, 337.992, 203.154, 640, 400)       # its camera_info, 640 x 400, no distortion
TIPS = (360.0, 316.0)


def frames() -> list[np.ndarray]:
    return [np.array(Image.open(DATA / ("wrist_fork%d.jpg" % i)).convert("RGB")) for i in range(5)]


def mount() -> np.ndarray:
    """tool <- camera, as calibrate-wrist measured it on that arm."""
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = quat_matrix([-0.003915, 0.010272, 0.999934, 0.003192]), [0.00565, 0.07312, -0.10241]
    return X


class EndOn(unittest.TestCase):
    def setUp(self):
        self.ctt = cfg.utensil.camera_to_tip_m
        cfg["utensil"]["camera_to_tip_m"] = 0.254                       # its ruler reading (the OAK's glass: 0)
        self.cam = CameraModel(OAK, OAK, np.eye(4))

    def tearDown(self):
        cfg["utensil"]["camera_to_tip_m"] = self.ctt

    def test_the_tips_of_a_fork_seen_end_on(self):
        """The side-on search finds no utensil in these frames; the end-on one finds the tines' tips within 12 px of
        where they are, the same every time, and a box round the fork down to the image's bottom edge (the gripper)."""
        g = np.stack([f.mean(2) for f in frames()])
        Xi = np.linalg.inv(mount())
        a, b = (np.array(self.cam.pixel(Xi[:3, :3] @ np.array([0.0, 0.0, z]) + Xi[:3, 3])) for z in (0.0, 0.25))
        self.assertIsNone(wc._fork_side_on(g, a, (b - a) / np.linalg.norm(b - a), float(np.linalg.norm(b - a))))
        got = wc.fork_in_image(frames(), mount(), self.cam)
        self.assertIsNotNone(got)
        tip, box = got
        self.assertLess(math.hypot(tip[0] - TIPS[0], tip[1] - TIPS[1]), 12.0, tip)
        self.assertEqual(wc.fork_in_image(frames(), mount(), self.cam), got)
        x1, y1, x2, y2 = box
        for u, v in (TIPS, (345, 370), (340, 330)):                    # the tips, the shaft, the tines' base
            self.assertTrue(x1 <= u <= x2 and y1 <= v <= y2, (box, u, v))
        self.assertEqual(y2, OAK.height)

    def test_nothing_moving_is_refused(self):
        """The same frame five times: nothing moves, so nothing tells the utensil from the board."""
        f = frames()[0]
        self.assertIsNone(wc.fork_in_image([f] * 5, mount(), self.cam))

    def test_no_board_moving_beyond_the_tips_is_refused(self):
        """The board held still just beyond the tines (every frame's patch there the first's): no tip read off it."""
        fs = frames()
        for f in fs[1:]:
            f[290:345, 355:430] = fs[0][290:345, 355:430]
        self.assertIsNone(wc.fork_in_image(fs, mount(), self.cam))


if __name__ == "__main__":
    unittest.main()
