"""Unit tests: no ROS, no arm, no GPU (python3 tests/unit_test.py)."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path

from fixture import ROOT, WORK  # noqa: E402  (the test site: before rammp_feeding is imported)
import numpy as np  # noqa: E402

from fake_robot import FakeRobot, Scene  # noqa: E402
from rammp_feeding import geometry as g, scene  # noqa: E402
from rammp_feeding.arm import Arm, MoveError, TorqueGuard, chain, smooth_timing, time_scale  # noqa: E402
from rammp_feeding.config import cfg, state_dir  # noqa: E402
from rammp_feeding.ros import Detection  # noqa: E402

RNG = np.random.default_rng(0)


def over_the_bowl(robot, arm, sc: Scene) -> None:
    """The arm as OVER leaves it: from food_detect_start, slid level until the tip is over the bowl's centre."""
    q = scene.overlook_joints(arm)
    T, R = arm.tool(q)
    tip = arm.tip(q)
    robot.raw = np.radians(arm.ik(T + np.array([sc.bowl_xy[0] - tip[0], sc.bowl_xy[1] - tip[1], 0.0]), R, q))


def site(sc: Scene) -> None:
    (state_dir() / "scene_state.json").write_text(json.dumps(dict(
        bowl_xy=list(sc.bowl_xy), table_z=sc.table_z, bowl_top_z=sc.table_z + 0.06, rim_radius_m=sc.bowl_radius, t=time.time())))


class Geometry(unittest.TestCase):
    def test_rotations(self):
        for _ in range(50):
            w = RNG.normal(size=3) * RNG.uniform(0, 3)
            R = g.rodrigues(w)
            self.assertTrue(np.allclose(g.rodrigues(g.rot_log(R)), R, atol=1e-9))
            self.assertTrue(np.allclose(g.quat_matrix(g.matrix_quat(R)), R, atol=1e-9))

    def test_fold(self):
        self.assertAlmostEqual(g.fold(190.0), -170.0)
        self.assertAlmostEqual(g.max_step([179] + [0] * 6, [-179] + [0] * 6), 2.0)


class Kinematics(unittest.TestCase):
    kin = chain(FakeRobot(Scene(), [0.0] * 7))

    def test_the_model_is_the_served_one_rooted_at_arm_mount(self):
        """The model is the description node's (rooted at arm.base_frame, its joints arm_joint_1..7); none, or one
        rooted elsewhere, is refused."""
        from unittest import mock
        self.assertEqual(self.kin.base, "arm_mount")
        self.assertEqual([j.name for j in self.kin.arm], ["arm_joint_%d" % i for i in range(1, 8)])
        robot = FakeRobot(Scene(), [0.0] * 7)
        with mock.patch.object(robot, "robot_description", lambda: None):
            with self.assertRaisesRegex(RuntimeError, "no arm model"):
                chain(robot)
        from fake_robot import MODEL
        with mock.patch.object(robot, "robot_description", lambda: MODEL.replace('"arm_mount"', '"world"')):
            with self.assertRaisesRegex(RuntimeError, "rooted at world"):
                chain(robot)

    def test_jacobian_matches_finite_differences(self):
        q = scene.captured("overlook")
        J = self.kin.jacobian(q)
        p0, R0 = self.kin.fk(q)
        for i in range(7):
            dq = np.zeros(7); dq[i] = 1e-4
            p1, R1 = self.kin.fk(np.asarray(q) + dq)
            num = np.concatenate([(p1 - p0), g.rot_log(R1 @ R0.T)]) / np.radians(1e-4)
            self.assertTrue(np.allclose(J[:, i], num, atol=1e-4), "joint %d" % (i + 1))

    def test_ik_round_trip(self):
        for name in ("home", "overlook"):
            q0 = scene.captured(name)
            n = 0
            while n < 10:
                qt = np.asarray(q0) + RNG.normal(scale=6, size=7)
                if any(not lo <= qt[i] <= hi for i, (lo, hi) in self.kin.limits.items()):
                    continue                                  # past a limit: no solution is the right answer
                n += 1
                T, R = self.kin.fk(qt)
                q = self.kin.ik(T, R, q0)
                self.assertIsNotNone(q)
                T2, R2 = self.kin.fk(q)
                self.assertLess(np.linalg.norm(T2 - T), 2e-4)
                self.assertLess(np.linalg.norm(g.rot_log(R2 @ R.T)), 2e-3)

    def test_tool_frame_is_flange_plus_tcp(self):
        q = scene.captured("home")
        F = self.kin.flange(q)
        T, _ = self.kin.fk(q)
        self.assertTrue(np.allclose(T, F[:3, 3] + F[:3, :3] @ np.array(cfg.arm.tool_from_flange_m)))


class Motion(unittest.TestCase):
    @staticmethod
    def _profile(Q, dt):
        """Every sample's joint speeds, and every change of speed from one sample to the next (from rest, to rest)."""
        V = np.vstack([np.zeros(7), np.diff(np.array(Q), axis=0) / np.array(dt)[:, None], np.zeros(7)])
        return np.abs(V), np.abs(np.diff(V, axis=0))

    def test_smooth_timing_caps_speed_and_acceleration(self):
        """A planned path from rest to rest under vmax and amax, sampled every sample_s, its speed never jumping:
        time_scale's timing changes speed in steps at its waypoints, the jolt the arm feels."""
        path = [[0.0] * 7, [4.0] + [0] * 6, [30.0, 5.0] + [0] * 5, [60.0, 5.0] + [0] * 5]
        Q, dt = smooth_timing(path, 20.0, 45.0, 0.02)
        self.assertTrue(np.allclose(Q[0], path[0]) and np.allclose(Q[-1], path[-1]))
        self.assertGreaterEqual(sum(dt), 3.0)                    # 60 deg at <= 20 deg/s
        self.assertLessEqual(max(dt), 0.02 + 1e-9)
        V, dV = self._profile(Q, dt)
        self.assertLessEqual(V.max(), 1.1 * 20.0)
        self.assertLessEqual(dV.max(), 1.1 * 45.0 * 0.02)        # at most amax over one sample, anywhere
        old = time_scale(path, 20.0, 45.0)
        Vo, dVo = self._profile(np.array(path), old)
        self.assertGreater(dVo.max(), 5 * dV.max())              # (the stepped timing's jump at a waypoint)

    def test_smooth_timing_across_the_fold_and_where_a_joint_turns_back(self):
        """A continuous joint crossing +-180 runs the short way; a joint turning back stops there, no overshoot."""
        Q, dt = smooth_timing([[170.0] + [0] * 6, [-170.0] + [0] * 6], 30.0, 45.0, 0.02)
        self.assertAlmostEqual(float(Q[-1][0] - Q[0][0]), 20.0, places=6)
        Q, dt = smooth_timing([[0.0] * 7, [20.0] + [0] * 6, [10.0] + [0] * 6], 60.0, 45.0, 0.02)
        self.assertLessEqual(max(q[0] for q in Q), 20.0 + 1e-9)
        V, dV = self._profile(Q, dt)
        self.assertLessEqual(dV.max(), 1.1 * 45.0 * 0.02)

    def test_torque_guard(self):
        G = TorqueGuard(2.5)
        for k in range(40):                                      # a slow ramp: no contact
            self.assertIsNone(G.update([0.05 * k] + [0] * 6))
        hits = [G.update([2.0 + 3.0] + [0] * 6) for _ in range(3)]
        self.assertTrue(any(hits))

    def test_plunge_guard_is_zoned(self):
        """A torque step above the food stops the plunge; the same step once the tines are in the food does not."""
        from rammp_feeding.acquire import plunge
        sc = Scene()
        site(sc)
        for step, stops in ((4.0, True), (5.0, False)):
            robot = FakeRobot(sc, scene.captured("home"))
            arm = Arm(robot)
            robot.raw = np.radians(scene.overlook_joints(arm))
            tip = arm.tip()
            H = tip[2] - sc.table_z                               # the tip over the table (a clear bowl reads it)
            u = dict(top_z=sc.table_z + 0.055, gap_top=H - 0.055, floor_z=sc.table_z, gap_floor=H, height=0.055)   # 55 mm
            robot.inject[0] = (0.15 if stops else 0.97, [step] + [0] * 6)  # early (by time): above the food; at the end: in it
            out = Path(tempfile.mkdtemp(dir=WORK))
            if stops:
                with self.assertRaises(MoveError):
                    plunge(arm, u, out, scoop_after=False)
            else:
                plunge(arm, u, out, scoop_after=False)
                kept = json.loads((out / "food.json").read_text())["height"]       # for --start-at scoop: its top
                self.assertAlmostEqual(kept, 0.055 - float(cfg.bowl.floor_above_table_m), places=6)   # over the bowl's floor
                self.assertLess(arm.tip()[2], tip[2] - 0.1)


class Restored(unittest.TestCase):
    """What the port keeps from the stack before it."""

    def _at_overlook(self):
        sc = Scene(floor_lift=0.007)
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        robot.raw = np.radians(scene.overlook_joints(arm))          # food_detect_start
        sc.piece_xy = tuple(sc.bowl_xy)                             # a piece in the middle of the bowl
        return sc, robot, arm

    @staticmethod
    def slide_over(arm, xy) -> dict:
        """The target perceive would give: the level slide that puts the tip over xy."""
        T, tip = arm.tool()[0], arm.tip()
        return dict(tool_goal=[float(T[0] + xy[0] - tip[0]), float(T[1] + xy[1] - tip[1]), float(T[2])])

    def test_bite_waits_for_the_force_to_settle(self):
        """3 N at the tool for 0.4 s (a bite), then gone: detected, and the wait ends once it has settled."""
        from rammp_feeding.deliver import bite
        sc, robot, arm = self._at_overlook()
        J = arm.kin.jacobian(arm.q())[:3]
        t0 = time.time()
        f = lambda t: J.T @ np.array([0, 0, 3.0]) if 1.0 < t - t0 < 1.4 else np.zeros(7)   # noqa: E731
        robot.effort_fn = f
        self.assertEqual(bite(robot, arm, 5.0), 0)
        self.assertLess(time.time() - t0, 3.5)                   # settled soon after, not bite_settle_s
        robot.effort_fn = lambda t: np.zeros(7)
        self.assertEqual(bite(robot, arm, 1.0), 4)               # no bite

    def test_hover_then_rise(self):
        """The aim check for tip_px: down to a 10 mm hover over the piece, then straight back up."""
        from rammp_feeding.acquire import acquire, rise
        sc, robot, arm = self._at_overlook()
        z0 = arm.tip()[2]
        acquire(robot, arm, self.slide_over(arm, sc.piece_xy), Path(tempfile.mkdtemp()), stop_after="hover10")
        self.assertAlmostEqual(arm.tip()[2] - (sc.table_z + sc.floor_lift + sc.piece_height), 0.010, delta=0.002)
        rise(arm)
        self.assertAlmostEqual(arm.tip()[2], z0, delta=0.002)

    def test_food_detect_start_centres_the_bowl_in_the_wrist_image(self):
        """The bowl's centre in the middle of the wrist image, the camera overlook_camera_height_m above the table."""
        from fake_robot import D405
        sc, robot, arm = self._at_overlook()
        robot.raw = np.radians(scene.overlook_joints(arm))
        u, v = robot.project([sc.bowl_xy[0], sc.bowl_xy[1], sc.table_z])
        self.assertLess(np.hypot(u - D405.cx, v - D405.cy), 3.0)
        self.assertAlmostEqual(robot.camera_pose()[0][2] - sc.table_z, float(cfg.workspace.overlook_camera_height_m),
                               delta=0.002)

    def test_direct_home(self):
        from rammp_feeding.cli import home
        sc, robot, arm = self._at_overlook()
        home(robot, arm, direct=True)
        self.assertLess(g.max_step(arm.q(), scene.captured("home")), 0.1)

    def test_the_hover_never_goes_under_the_bowls_floor(self):
        """Nothing under the tines and a clear bowl: the depth reads the table, and the hover stops over the bowl's own
        floor instead."""
        from rammp_feeding.acquire import acquire
        sc, robot, arm = self._at_overlook()
        sc.floor_lift, sc.piece_xy = 0.0, (sc.bowl_xy[0] + 0.5, sc.bowl_xy[1])         # the piece out of the way
        acquire(robot, arm, self.slide_over(arm, sc.bowl_xy), Path(tempfile.mkdtemp()), stop_after="hover10")
        self.assertAlmostEqual(arm.tip()[2], sc.table_z + float(cfg.bowl.floor_above_table_m) + 0.010, delta=0.002)

    def test_a_scoop_on_its_own_starts_only_where_its_plunge_ended(self):
        from rammp_feeding.acquire import acquire
        sc, robot, arm = self._at_overlook()
        out = Path(tempfile.mkdtemp(dir=WORK))
        with self.assertRaisesRegex(FileNotFoundError, "no plunge"):
            acquire(robot, arm, {}, out, start_at="scoop")
        (out / "food.json").write_text(json.dumps(dict(height=0.03, floor_m=0.008, joints_deg=scene.captured("home"))))
        with self.assertRaisesRegex(MoveError, "not where this bite's plunge ended"):
            acquire(robot, arm, {}, out, start_at="scoop")
        self.assertEqual(robot.moves, [])


class Delivery(unittest.TestCase):
    def _at_look(self):
        from fake_robot import FakeFace
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("deliver"))
        scene.measured_mouth(None)                                  # (no mouth measured yet in this command)
        return sc, robot, Arm(robot), FakeFace(robot)

    def test_a_compliant_move_never_takes_a_continuous_joint_across_180(self):
        """The driver's impedance mode runs a continuous joint the long way round past +-180: refused before moving;
        the same path in position control runs, the short way."""
        sc, robot, arm, face = self._at_look()
        q0 = np.array(arm.q())
        self.assertLess(q0[0], -170.0)                           # (the delivery look: joint 1 near -180)
        path = [list(q0 + [-15.0 * k / 10, 0, 0, 0, 0, 0, 0]) for k in range(11)]
        n = len(robot.moves)
        with self.assertRaisesRegex(MoveError, "joint 1 would cross"):
            arm.run(path, "approach", dps=10.0, impedance="approach")
        self.assertEqual(len(robot.moves), n)                    # nothing sent
        arm.run(path, "approach", dps=10.0)
        self.assertLess(g.max_step(arm.q(), path[-1]), 0.1)
        self.assertEqual(robot.moves[-1], "position")
        self.assertFalse(cfg.delivery.approach_impedance)        # the approach runs in position control

    def test_the_swing_starts_from_a_fork_tipped_past_its_limit(self):
        """A swing from where a stopped move left the fork tipped up past swing_max_tilt_deg runs (no further than it
        starts), back to the delivery look."""
        from rammp_feeding.deliver import swing
        sc, robot, arm, face = self._at_look()
        look = np.array(scene.captured("deliver"))
        q = [90.54, -100.81, -91.98, -141.43, -16.78, -29.70, 98.96]      # where a faulted approach left the arm
        self.assertGreater(g.tilt_deg(arm.tool(q)[1]), float(cfg.delivery.swing_max_tilt_deg) + 2)
        robot.raw = np.radians(q)
        swing(arm)
        self.assertLess(g.max_step(arm.q(), look), 0.1)

    def test_no_face_from_the_delivery_look_then_the_second(self):
        from rammp_feeding.deliver import present
        sc, robot, arm, face = self._at_look()
        look1, finds = arm.tool()[0], []
        face.find = lambda cam, s: finds.append(arm.tool()[0]) or len(finds) > 1    # no face from the first look
        self.assertEqual(present(robot, arm, face, Path(tempfile.mkdtemp())), 0)
        self.assertLess(np.linalg.norm(finds[1] - (look1 + np.array(cfg.delivery.second_look_m))), 0.002)
        mouth = np.array(sc.mouth) + [0, 0, float(cfg.delivery.aim_dz_m)]
        self.assertAlmostEqual(np.linalg.norm(arm.tip() - mouth), float(cfg.delivery.stop_m), delta=0.01)

    def test_no_bite_is_exit_4_after_retracting(self):
        from unittest import mock
        from rammp_feeding import deliver as dv
        sc, robot, arm, face = self._at_look()
        with mock.patch.dict(dv.DL, {"bite_timeout_s": 0.3}):
            self.assertEqual(dv.deliver(robot, arm, face, Path(tempfile.mkdtemp()), start_at="bite"), 4)
        self.assertLess(g.max_step(arm.q(), scene.overlook_joints(arm)), 1.0)        # retracted

    def test_the_retract_backs_off_straight_only_from_in_front_of_the_look(self):
        """From the scoop's end (level, over the bowl, behind the look) the return is planned, never a straight line."""
        from rammp_feeding.acquire import Food, plan_scoop
        from rammp_feeding.deliver import retract
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        over_the_bowl(robot, arm, sc)
        q = arm.q()
        T, R = arm.tool(q)
        robot.raw = np.radians(plan_scoop(arm, q, T, R, Food(0.02, 0.005))["path"][-1])
        arm.straight = lambda *a, **k: self.fail("a straight back-off from the scoop end")
        retract(robot, arm)
        self.assertLess(g.max_step(arm.q(), scene.overlook_joints(arm)), 1.0)

    def test_a_free_space_move_settles_onto_a_level_goal(self):
        """Within a degree of the delivery look (the utensil level), travel settles onto it, not refuses it."""
        sc, robot, arm, face = self._at_look()
        robot.raw = np.radians(np.asarray(scene.captured("deliver")) + [0.5, 0, 0, 0, 0, 0, 0])
        arm.travel(scene.captured("deliver"), "swing to the user")
        self.assertLess(g.max_step(arm.q(), scene.captured("deliver")), 0.1)

    def test_no_face_from_either_look_is_refused(self):
        from rammp_feeding.deliver import present
        sc, robot, arm, face = self._at_look()
        face.find = lambda cam, s: False
        with self.assertRaises(MoveError):
            present(robot, arm, face, Path(tempfile.mkdtemp()))

    def test_the_user_in_the_planners_world_is_where_the_mouth_was_measured(self):
        """No user in the planner's world until the mouth is measured from the delivery look; then the user round it,
        wherever they sit (here 15 cm from where the look points), for the rest of the command and for this bite's
        later stages run on their own."""
        from rammp_feeding import deliver as dv
        sc, robot, arm, face = self._at_look()
        self.assertFalse([o for o in scene.world()["obstacles"] if o["name"].startswith("user_")])
        sc.mouth = tuple(np.array(sc.mouth) + [0.05, -0.10, 0.10])
        out = Path(tempfile.mkdtemp())
        self.assertEqual(dv.present(robot, arm, face, out), 0)
        m = np.array(json.loads((out / "mouth.json").read_text())["mouth"])
        self.assertLess(np.linalg.norm(m - np.array(sc.mouth)), 0.01)
        head = next(o for o in scene.world()["obstacles"] if o["name"] == "user_head")
        self.assertLess(np.linalg.norm(np.array(head["position"]) - (m + [-0.09, 0.0, 0.05])), 1e-9)
        scene.measured_mouth(None)                                  # a new command: --start-at retract of this bite
        self.assertEqual(dv.deliver(robot, arm, face, out, start_at="retract"), 0)
        self.assertLess(np.linalg.norm(scene.user_mouth() - m), 1e-9)


class Perception(unittest.TestCase):
    def test_food_filters(self):
        from rammp_feeding.perception import food
        rgb = np.full((480, 848, 3), (200, 170, 90), np.uint8)

        def box(score, b):                                     # a detection whose mask fills its box
            m = np.zeros(rgb.shape[:2], bool)
            m[b[1]:b[3] + 1, b[0]:b[2] + 1] = True
            return Detection("f", score, b, m)
        bowl = box(1.0, [300, 100, 600, 400]).mask
        dets = [box(0.9, [400, 200, 440, 240]),                 # kept
                box(0.8, [402, 202, 441, 241]),                 # the same piece again
                box(0.7, [450, 380, 480, 420]),                 # inside the utensil's region
                box(0.6, [100, 100, 140, 140]),                 # outside the bowl
                box(0.5, [500, 200, 505, 260]),                 # a sliver
                box(0.4, [320, 120, 560, 380])]                 # bowl-sized
        self.assertEqual([d.box for d in food(rgb, dets, bowl)], [[400, 200, 440, 240]])

    def test_no_bowl_in_view_is_refused(self):
        from rammp_feeding.perception import perceive
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        robot.raw = np.radians(scene.overlook_joints(arm))
        found = robot.detect
        robot.detect = lambda role, prompts, t: [d for d in found(role, prompts, t) if d.label == cfg.food.prompt]
        with self.assertRaisesRegex(MoveError, "no SAM3 bowl mask"):
            perceive(robot, arm, Path(tempfile.mkdtemp(dir=WORK)))

    def test_the_bowl_is_the_mask_round_the_measured_centre(self):
        """SAM3's best-scored bowl with an empty mask, and a "plate" mask over the whole table round the bowl: perceive
        takes the bowl's own mask (the smallest round the centre rammp detect measured) and finds the piece."""
        from rammp_feeding.perception import perceive
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        robot.raw = np.radians(scene.overlook_joints(arm))
        found = robot.detect

        def detect(role, prompts, t):
            dets = found(role, prompts, t)
            m = next(d for d in dets if d.label in cfg.perception.bowl_prompts).mask
            return dets + [Detection("bowl", 0.99, [0, 0, 1, 1], np.zeros_like(m)),
                           Detection("plate", 0.95, [0, 0, m.shape[1], m.shape[0]], np.ones_like(m))]
        robot.detect = detect
        tgt = perceive(robot, arm, Path(tempfile.mkdtemp(dir=WORK)))
        self.assertLess(np.linalg.norm(np.array(tgt["piece"]["xy"]) - np.array(sc.piece_xy)), 0.01)

    def test_the_bowl_is_the_circle_through_its_rim(self):
        """Depth points of a bowl on a table (its rim 60 mm up, 80 mm round, the near half only, as from the side):
        the centre and radius of the whole rim; with no rim seen, refused."""
        from rammp_feeding.scene_camera import bowl_at
        rng = np.random.default_rng(3)
        a = rng.uniform(-math.pi / 2, math.pi / 2, 400)              # the near half of the rim
        rim = np.c_[0.4 + 0.08 * np.cos(a), 0.1 + 0.08 * np.sin(a), np.full(400, 0.06)]
        table = np.c_[rng.uniform(0.2, 0.6, 2000), rng.uniform(-0.1, 0.3, 2000), np.zeros(2000)]
        p = np.vstack([rim, table])
        uv = np.c_[np.r_[np.full(400, 50), np.full(2000, 65)], np.full(2400, 50)]   # the rim in its mask, the table round it
        mask = np.zeros((100, 300), bool)
        mask[40:60, 40:60] = True
        got = bowl_at(mask, [40, 40, 60, 60], uv, p)
        np.testing.assert_allclose(got["bowl_xy"], [0.4, 0.1], atol=1e-6)
        self.assertAlmostEqual(got["rim_radius_m"], 0.08, places=3)
        self.assertAlmostEqual(got["table_z"], 0.0, places=6)
        got = bowl_at(mask, [40, 40, 60, 60], uv, np.vstack([rim * [1, 1, 0], table]))      # nothing above the table
        self.assertIn("rejected", got)

    def test_an_empty_bowl_reads_empty_with_the_utensil_over_it(self):
        """The inner floor alone counts: its walls rising toward the rim, and the utensil hanging over the bowl (above
        its rim), are not food; a piece standing on the floor is."""
        from rammp_feeding.perception import food_left
        rng = np.random.default_rng(4)
        ws = dict(bowl_xy=[0.5, 0.0], bowl_radius=0.08, bowl_top_z=0.07)
        r = 0.08 * np.sqrt(rng.uniform(0, 1, 6000))
        a = rng.uniform(0, 2 * np.pi, 6000)
        floor = np.c_[0.5 + r * np.cos(a), r * np.sin(a), 0.07 * (r / 0.08) ** 3]          # a bowl: flat, then its walls
        fork = np.c_[rng.uniform(0.48, 0.52, 800), rng.uniform(-0.01, 0.01, 800), np.full(800, 0.30)]
        uv = np.zeros((6800, 2), int)
        mask = np.ones((1, 1), bool)
        self.assertLess(food_left(np.vstack([floor, fork]), uv, mask, ws), float(cfg.perception.food_left_min_frac))
        piece = np.c_[rng.uniform(0.49, 0.51, 600), rng.uniform(-0.01, 0.01, 600), np.full(600, 0.025)]
        self.assertGreater(food_left(np.vstack([floor, fork, piece]), np.zeros((7400, 2), int), mask, ws),
                           float(cfg.perception.food_left_min_frac))

    def test_under_tip_from_the_rendered_depth(self):
        from rammp_feeding.perception import under_tip
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        over_the_bowl(robot, arm, sc)
        tip = arm.tip()
        sc.piece_xy = tuple(tip[:2])                              # the piece right under the tip
        u = under_tip(robot.render_depth(), arm)
        self.assertAlmostEqual(u["top_z"], sc.table_z + sc.piece_height, delta=0.002)
        self.assertAlmostEqual(u["gap_top"], tip[2] - (sc.table_z + sc.piece_height), delta=0.002)
        self.assertAlmostEqual(u["gap_floor"], tip[2] - sc.table_z, delta=0.002)


class Plunge(unittest.TestCase):
    def test_never_under_the_bowls_own_floor(self):
        from rammp_feeding.acquire import plunge_depth
        sc = Scene()
        site(sc)
        tip_z = sc.table_z + 0.20
        u = dict(gap_top=0.175, gap_floor=0.20, height=0.025, top_z=sc.table_z + 0.025, floor_z=sc.table_z)   # the floor through the glass
        down, food = plunge_depth(u, tip_z)
        self.assertAlmostEqual(tip_z - down, sc.table_z + float(cfg.bowl.floor_above_table_m)
                               + float(cfg.utensil.punch_floor_margin_m), places=6)

    def test_no_floor_seen_is_refused(self):
        from rammp_feeding.acquire import plunge_depth
        sc = Scene()
        site(sc)
        with self.assertRaises(MoveError):
            plunge_depth(dict(gap_top=0.15, top_z=sc.table_z + 0.05), sc.table_z + 0.20)


class World(unittest.TestCase):
    def test_world(self):
        sc = Scene()
        site(sc)
        w = scene.world()
        names = [o["name"] for o in w["obstacles"]]
        self.assertEqual(sum(n.startswith("bowl_wall") for n in names), 12)
        table = next(o for o in w["obstacles"] if o["name"] == "table")
        top = table["position"][2] + table["dims"][2] / 2
        self.assertAlmostEqual(top, sc.table_z + float(cfg.scene_camera.error_m) + float(cfg.motion.free_space.min_tip_above_table_m))

    def test_the_scoop_ends_level_over_the_bowls_centre_above_its_rim(self):
        """The scoop's end, computed from the bowl: the piece lift_m over the free-space floor above the bowl's centre
        (so the swing may start there), the utensil level and pointing along the base's +y."""
        from rammp_feeding.acquire import Food, plan_scoop
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        over_the_bowl(robot, arm, sc)
        q = arm.q()
        T, R = arm.tool(q)
        plan = plan_scoop(arm, q, T, R, Food(0.02, 0.005))
        Te, Re = arm.tool(plan["path"][-1])
        piv = Te + Re @ plan["o_piv"]
        np.testing.assert_allclose(piv, [*sc.bowl_xy, scene.tip_floor(*sc.bowl_xy) + float(cfg.acquisition.scoop.lift_m)],
                                   atol=2e-3)
        self.assertGreater(Re[:, 2][1], 0.99)

    def test_the_planners_bowl_floor_is_where_the_plunge_stops_above(self):
        """One height for the bowl's floor: the planner's floor box ends at bowl.floor_above_table_m over the table
        (plus the scene's error), the height the plunge never goes under."""
        sc = Scene()
        site(sc)
        box = next(o for o in scene.world()["obstacles"] if o["name"] == "bowl_floor")
        self.assertAlmostEqual(box["position"][2] + box["dims"][2] / 2,
                               sc.table_z + float(cfg.scene_camera.error_m) + float(cfg.bowl.floor_above_table_m))
        self.assertAlmostEqual(float(cfg.bowl.floor_above_table_m), 0.005)

    def test_the_bowl_is_padded_by_the_camera_that_read_it(self):
        """The scene camera's reading pads the bowl and the table by its error_m; once the wrist camera has read them
        (rammp detect), by wrist_bowl_error_m -- the bowl's top, still the scene camera's, keeps that camera's error."""
        sc = Scene()
        site(sc)
        f = state_dir() / "scene_state.json"
        walls = lambda: [o for o in scene.world()["obstacles"] if o["name"].startswith("bowl_wall")]    # noqa: E731
        top = lambda: max(o["position"][2] + o["dims"][2] / 2 for o in walls())                     # noqa: E731
        over = scene.tip_floor(*sc.bowl_xy)
        self.assertAlmostEqual(walls()[0]["dims"][0], 0.006 + 2 * float(cfg.scene_camera.error_m), places=9)
        top0 = top()
        f.write_text(json.dumps(dict(json.loads(f.read_text()), refined_by="the wrist camera")))
        self.assertAlmostEqual(walls()[0]["dims"][0], 0.006 + 2 * float(cfg.perception.wrist_bowl_error_m), places=9)
        self.assertAlmostEqual(top(), top0, places=9)
        self.assertAlmostEqual(scene.tip_floor(*sc.bowl_xy), over, places=9)          # the scoop's end unchanged
        table = next(o for o in scene.world()["obstacles"] if o["name"] == "table")      # its near edge as before:
        near = table["position"][0] - table["dims"][0] / 2                                # the planner's padded table
        self.assertAlmostEqual(near, float(cfg.workspace.table_extent_m.x[0]) + float(cfg.scene_camera.error_m))
        self.assertGreater(near - 0.02, 0.075 + 0.005)            # clear of its base sphere (r 0.075 + 0.005 buffer)

    def test_each_planned_move_pushes_our_world_and_the_command_restores_the_planners(self):
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        arm.travel(scene.overlook_joints(arm), "to food_detect_start")
        self.assertTrue(robot.world_pushed)
        self.assertEqual([b["name"] for b in robot.worlds[-1]["obstacles"]], [b["name"] for b in scene.world()["obstacles"]])
        self.assertEqual(robot.worlds[-1]["base_frame"], "arm_mount")
        scene.restore_world(robot)                      # the planner's own world back: /reset
        self.assertEqual((robot.worlds[-1], robot.world_pushed), ("reset", False))

    def test_the_utensil_is_kept_clear_of_our_boxes(self):
        """A path that brings a utensil sphere into a box, or nearer than the clearance, is refused; one that keeps
        clear passes. The geometry against a brute-force distance, a rotated box included."""
        R = g.rpy_matrix(0.0, 0.0, math.radians(30))
        boxes = (np.array([[0.4, 0.0, 0.1]]), np.array([R]), np.array([[0.05, 0.1, 0.02]]), ["box"])
        rng = np.random.default_rng(1)
        pts = rng.uniform(-0.3, 0.3, (200, 3)) + boxes[0][0]
        got = g.sphere_box_clearance(pts[:, None, :], np.zeros(1), *boxes[:3])[:, 0, 0]
        local = (pts - boxes[0][0]) @ R
        grid = np.stack(np.meshgrid(*[np.linspace(-h, h, 41) for h in boxes[2][0]], indexing="ij"), -1).reshape(-1, 3)
        brute = np.array([np.min(np.linalg.norm(grid - p, axis=1)) for p in local])
        inside = np.all(np.abs(local) <= boxes[2][0], axis=1)
        np.testing.assert_allclose(got[~inside], brute[~inside], atol=0.006)       # (the grid's spacing)
        self.assertTrue(np.all(got[inside] <= 0))

        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        q = scene.overlook_joints(arm)
        tip = arm.tip(q)
        spheres = scene.utensil_spheres(arm)
        clear = float(cfg.motion.free_space.utensil_clearance_m)
        half = np.array([0.02, 0.02, 0.02])
        start, goal = list(q), list(q)
        start[0] -= 20.0
        goal[0] += 20.0                                  # a swing about the base: the tip passes over q's tip at mid-path

        def probe(gap):                                  # a box under q's tip sphere, gap below it
            return (np.array([tip + np.array([0.0, 0.0, -(spheres[-1][3] + gap + half[2])])]), np.array([np.eye(3)]),
                    np.array([half]), ["probe"])
        for gap, refused in ((-0.01, True), (clear - 0.003, True), (clear + 0.005, False)):
            if refused:
                with self.assertRaisesRegex(MoveError, "within .* of probe"):
                    arm.check_path([start, q, goal], lambda x, y: -np.inf, None, spheres, probe(gap))
            else:
                arm.check_path([start, q, goal], lambda x, y: -np.inf, None, spheres, probe(gap))
        # a path that starts that near (or inside) may leave from there, but never go nearer
        arm.check_path([q, goal], lambda x, y: -np.inf, None, spheres, probe(clear - 0.005))
        with self.assertRaisesRegex(MoveError, "within .* of probe"):     # out and back over it: nearer than either end
            arm.check_path([goal, q, goal], lambda x, y: -np.inf, None, spheres, probe(clear - 0.005))

    def test_a_planned_path_past_our_joint_limits_is_refused(self):
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        q = scene.captured("home")
        past = list(q)
        past[3] = -arm.limits[3] - 1.0                                       # J4 a degree past the model's
        with self.assertRaisesRegex(MoveError, "past its limit"):
            arm.check_path([q, past, q], lambda x, y: -np.inf)

    def test_a_planned_path_that_tilts_too_far_is_refused_before_it_runs(self):
        """One rule for every move: the utensil at most max_tilt_deg (or a move's own cap) off vertical, or 5 deg past
        the move's own start or end -- a path that tips it further on the way is refused before anything moves."""
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        home = np.array(scene.captured("home"))
        path = [list(home), list(home + [0, 0, 0, 0, 0, 40.0, 0]), list(home)]         # tips the fork on the way
        worst = max(g.tilt_deg(arm.tool(q)[1]) for q in g.densify(path, 0.5))
        self.assertGreater(worst, float(cfg.motion.max_tilt_deg) + 5)
        with self.assertRaisesRegex(MoveError, "tilts the utensil"):
            arm.check_path(path, lambda x, y: -np.inf)
        arm.check_path(path, lambda x, y: -np.inf, tilt_cap=worst + 0.5)
        self.assertEqual(arm.tilt_limit([path[0], path[1]]), g.tilt_deg(arm.tool(path[1])[1]) + 5.0)   # its own end
        self.assertEqual(robot.moves, [])

    def test_the_utensil_checks_our_boxes_the_bowl_exact(self):
        sc = Scene()
        site(sc)
        centres, rots, half, names = scene.utensil_boxes()
        raw = {o["name"]: o for o in scene.world()["obstacles"]}
        self.assertEqual(sorted(names), sorted(n for n in raw if n not in ("pedestal", "table")))
        pad = float(cfg.motion.free_space.utensil_padding_m)
        for k, n in enumerate(names):
            np.testing.assert_allclose(half[k] * 2, np.asarray(raw[n]["dims"]) + (0 if n.startswith("bowl_") else 2 * pad))
            np.testing.assert_allclose(rots[k], g.rpy_matrix(*np.radians(raw[n]["rpy_deg"])))

    def test_the_utensil_spheres_are_the_old_stacks(self):
        """The old planner server's spoon_spheres (flange frame, from the pads at 0.17 m) as cuRobo used them -- on
        bracelet_link, so grown by the robot model's 5 mm collision_sphere_buffer -- in the tool frame."""
        robot = FakeRobot(Scene(), scene.captured("home"))
        arm = Arm(robot)
        tip_f, buf = np.asarray(arm.tip_tool) + [0, 0, 0.12], 0.005
        old = [list(np.array([0, 0, 0.17]) + s * (tip_f - [0, 0, 0.17]) - [0, 0, 0.12]) + [(0.012 if s < 0.8 else 0.02) + buf]
               for s in np.linspace(0, 1, 5)]
        np.testing.assert_allclose(scene.utensil_spheres(arm), old, atol=1e-12)
        np.testing.assert_allclose(scene.utensil_spheres(arm, food=True)[-1],
                                   list(tip_f - [0, 0, 0.01] - [0, 0, 0.12]) + [0.02 + buf], atol=1e-12)

    def test_a_planned_path_never_dips_the_utensil_into_the_bowl(self):
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        q = scene.overlook_joints(arm)
        T, R = arm.tool(q)
        low = arm.ik(T - np.array([0.0, 0.0, 0.17]), R, q)            # the tip down into the bowl's opening
        with self.assertRaisesRegex(MoveError, "under the floor"):
            arm.check_path([q, low, q], scene.tip_floor, None, scene.utensil_spheres(arm), scene.utensil_boxes())

    def test_tip_floor(self):
        sc = Scene()
        site(sc)
        e = float(cfg.scene_camera.error_m)
        self.assertAlmostEqual(scene.tip_floor(*sc.bowl_xy), sc.table_z + 0.06 + float(cfg.motion.free_space.min_tip_above_bowl_m) + e)
        self.assertAlmostEqual(scene.tip_floor(0.5, 0.5), sc.table_z + float(cfg.motion.free_space.min_tip_above_table_m) + e)
        self.assertEqual(scene.tip_floor(-0.5, 0.0), -np.inf)


class Refine(unittest.TestCase):
    def test_the_wrist_camera_takes_the_bowl_nearest_the_scene_cameras(self):
        """Two bowls in the wrist camera's view: the one nearer the scene camera's reading is taken, not the better-scored
        one further away; a single bowl however far off is taken; the scene camera's own reading is kept for the log;
        no bowl at all: refused."""
        from unittest import mock
        from rammp_feeding import scene_camera as sc_
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        e = float(cfg.scene_camera.error_m)
        near, far = [sc.bowl_xy[0] + 0.5 * e, sc.bowl_xy[1]], [sc.bowl_xy[0] + 3 * e, sc.bowl_xy[1]]
        m = np.zeros((480, 848), bool)
        robot.detect = lambda role, prompts, t: [Detection("bowl", 0.9, [0, 0, 1, 1], m), Detection("bowl", 0.5, [0, 0, 1, 1], m)]
        fits = iter([dict(bowl_xy=far, rim_radius_m=0.08, table_z=0.0), dict(bowl_xy=near, rim_radius_m=0.07, table_z=0.003)])
        with mock.patch.object(sc_, "bowl_at", lambda *a: next(fits)):
            st = sc_.refine(robot, arm)
        self.assertEqual((st["bowl_xy"], st["table_z"]), (near, 0.003))        # the wrist camera's table
        self.assertEqual(st["scene_bowl_xy"], list(sc.bowl_xy))
        site(sc)
        with mock.patch.object(sc_, "bowl_at", lambda *a: dict(bowl_xy=far, rim_radius_m=0.08, table_z=0.0)):
            self.assertEqual(sc_.refine(robot, arm)["bowl_xy"], far)            # 3 x error_m off: taken
        with mock.patch.object(sc_, "bowl_at", lambda *a: dict(rejected="nothing bowl-sized on the table")):
            with self.assertRaisesRegex(RuntimeError, "sees no bowl"):
                sc_.refine(robot, arm)


class Setup(unittest.TestCase):
    def test_setup_captures_each_pose_and_skips_what_it_is_told(self):
        """rammp setup home overlook deliver: each pose captured where the arm is when Enter is pressed, the delivery
        look checked for a face; s skips a step, and the last (home, then check) can be skipped too."""
        from fake_robot import FakeFace
        from rammp_feeding import setup_steps as su
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        keep = {n: (state_dir() / (n + ".json")).read_text() for n in ("home", "overlook", "deliver")}
        try:
            robot.raw = np.radians(np.asarray(scene.captured("deliver")) + [1.0, 0, 0, 0, 0, 0, 0])
            answers = iter(["s", "s", "", "s"])                     # home: skip; overlook: skip; deliver: go; end: skip
            self.assertEqual(su.run(robot, arm, ("home", "overlook", "deliver"), ask=lambda _: next(answers),
                                    face=FakeFace(robot)), 0)
            self.assertLess(g.max_step(scene.captured("deliver"), np.degrees(robot.raw)), 0.01)
            self.assertEqual((state_dir() / "home.json").read_text(), keep["home"])           # skipped: untouched
        finally:
            for n, t in keep.items():
                (state_dir() / (n + ".json")).write_text(t)

    def test_setup_writes_the_bowl_centre_in_place(self):
        from rammp_feeding import setup_steps as su
        p = state_dir() / "local.yaml"
        p.write_text("workspace:\n  table_extent_m: {x: [0.1, 0.7], y: [-0.3, 0.8]}\n  bowl_centre_m: [0.29, 0.16]   # old\n")
        try:
            self.assertTrue(su.write_bowl_centre([0.501, -0.089]))
            self.assertIn("bowl_centre_m: [0.501, -0.089]", p.read_text())
            self.assertIn("table_extent_m: {x: [0.1, 0.7]", p.read_text())
            p.write_text("utensil: {}\n")
            self.assertFalse(su.write_bowl_centre([0.5, 0.0]))
        finally:
            p.unlink()


class Cli(unittest.TestCase):
    def test_each_command_takes_only_its_own_stages(self):
        from rammp_feeding.cli import main
        for argv in (["acquire", "--stop-after", "hover1"], ["deliver", "--start-at", "scoop"],
                     ["home", "--stop-after", "over"], ["capture", "nowhere"]):
            with self.assertRaises(SystemExit, msg=" ".join(argv)):
                main(argv)


class Record(unittest.TestCase):
    def test_an_event_keeps_its_data_and_its_own_time(self):
        from rammp_feeding import record
        s = record.session("unit")
        record.event("located", t=1.0, bowl_xy=[0.4, 0.1])
        e = json.loads((s / "events.jsonl").read_text().splitlines()[-1])
        self.assertEqual((e["kind"], e["bowl_xy"]), ("located", [0.4, 0.1]))
        self.assertGreater(e["t"], 1.0)


class Profiles(unittest.TestCase):
    def test_every_camera_profile_has_its_roles_keys(self):
        import yaml
        for f in sorted((ROOT / "config/camera").glob("*.yaml")):
            c = yaml.safe_load(f.read_text())
            role = {"glass_to_optical_m", "mount"} if "mount" in c else {"link_frame", "error_m"}
            self.assertLessEqual(role | {"streams"}, set(c), f.name)
            self.assertLessEqual({"colour_topic", "depth_topic"}, set(c["streams"]), f.name)

    def test_the_arm_is_not_built_until_the_robots_measurements_are_in(self):
        from unittest import mock
        robot = FakeRobot(Scene(), scene.captured("home"))
        with mock.patch.dict(cfg["utensil"], {"glass_to_tip_m": None}):
            with self.assertRaisesRegex(RuntimeError, r"not measured on this robot yet: utensil\.glass_to_tip_m"):
                Arm(robot)
            arm = Arm(robot, measured=False)          # state, home --direct, calibrate-wrist: no tip model
            self.assertIsNone(arm.tip_tool)
            from rammp_feeding.cli import report
            report(arm, "now")

    def test_the_arm_is_not_built_without_the_sites_own_values(self):
        from unittest import mock
        robot = FakeRobot(Scene(), scene.captured("home"))
        for section, key in (("workspace", "table_extent_m"), ("workspace", "bowl_centre_m")):
            with mock.patch.dict(cfg[section], {key: None}):
                with self.assertRaisesRegex(RuntimeError, r"not measured on this robot yet: %s\.%s" % (section, key)):
                    Arm(robot)

    def test_the_measured_mount_is_taken_for_its_camera_only(self):
        from rammp_feeding import config as C
        d = Path(tempfile.mkdtemp(dir=WORK))
        (d / "local.yaml").write_text("profiles: {camera: realsense_d435i}\n")
        m = dict(camera_in_tool_m=[0.04, 0.07, -0.11], rotation_quat_xyzw=[0.0, 0.0, 1.0, 0.0])
        env = dict(os.environ)
        try:
            os.environ.pop("RAMMP_NO_LOCAL")
            os.environ["RAMMP_STATE_DIR"] = str(d)
            self.assertIn("camera.mount.camera_in_tool_m", C.unmeasured(C.load()))
            (d / "wrist_mount.json").write_text(json.dumps(dict(m, camera="realsense_d435i")))
            self.assertEqual(C.load().camera.mount.camera_in_tool_m, m["camera_in_tool_m"])
            (d / "wrist_mount.json").write_text(json.dumps(dict(m, camera="realsense_d405")))
            self.assertIsNone(C.load().camera.mount.camera_in_tool_m)
        finally:
            os.environ.clear()
            os.environ.update(env)


    def test_an_existing_calibration_entered_by_hand_is_taken(self):
        """The two files as docs/porting.md (6b) has a site write them, with only the keys it lists: the wrist camera's
        mount and the utensil's pixels are measured, and the scene camera's pose is the one published."""
        from unittest import mock
        from rammp_feeding import config as C, scene_camera as sc_
        d = Path(tempfile.mkdtemp(dir=WORK))
        (d / "local.yaml").write_text("profiles: {camera: realsense_d435i, utensil: fork}\nutensil: {glass_to_tip_m: 0.205}\n")
        mount = dict(camera="realsense_d435i", utensil="fork", camera_in_tool_m=[0.023, 0.05, -0.103],
                     rotation_quat_xyzw=[0.0, 0.0, 1.0, 0.0], tip_px=[438.0, 443.0], image_region_px=[300, 380, 640, 480])
        (d / "wrist_mount.json").write_text(json.dumps(mount))
        pose = dict(parent_frame="arm_mount", child_frame="scene_camera_link", xyz=[0.6, -0.5, 0.12],
                    quat_xyzw=[0.0, 0.0, 0.7071068, 0.7071068])
        (d / "scene_extrinsic.json").write_text(json.dumps(pose))
        env = dict(os.environ)
        try:
            os.environ.pop("RAMMP_NO_LOCAL")
            os.environ["RAMMP_STATE_DIR"] = str(d)
            c = C.load()
            self.assertEqual(C.unmeasured(c), [])
            self.assertEqual(c.camera.mount.camera_in_tool_m, mount["camera_in_tool_m"])
            self.assertEqual((c.utensil.tip_px, c.utensil.image_region_px), (mount["tip_px"], mount["image_region_px"]))
        finally:
            os.environ.clear()
            os.environ.update(env)
        with mock.patch.object(sc_, "state_dir", lambda: d):
            T, _ = sc_.extrinsic()
        np.testing.assert_allclose(T[:3, 3], pose["xyz"])
        np.testing.assert_allclose(T[:3, :3], g.rot("z", 90.0), atol=1e-6)


class WristCalibration(unittest.TestCase):
    def test_the_hand_eye_solve_finds_the_mount_from_the_calibrations_poses(self):
        """The board seen from rammp calibrate-wrist's own poses round its start over feeding_start, through a known
        mount, each board pose 0.5 mm and 0.1 deg off: the solve finds the mount again, and the board's place agrees."""
        from rammp_feeding import wrist_calibration as wc
        arm = Arm(FakeRobot(Scene(), scene.captured("home")))
        T, R = arm.tool(scene.captured("home"))
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = g.rodrigues([0.03, -0.02, math.pi]), [0.04, 0.07, -0.11]
        B = np.eye(4)
        B[:3, :3], B[:3, 3] = g.rodrigues([math.pi, 0.0, 0.3]), T + np.array([0.03, 0.02, -0.2])
        caps, rng = [], np.random.default_rng(1)          # its own noise, whatever ran before
        for Tk, Rk in wc.poses(T + np.array([0.0, 0.0, 0.1]), R):
            G = np.eye(4)
            G[:3, :3], G[:3, 3] = Rk, Tk
            Cb = np.linalg.inv(G @ X) @ B
            Cb[:3, :3] = g.rodrigues(rng.normal(0.0, math.radians(0.1), 3)) @ Cb[:3, :3]
            Cb[:3, 3] += rng.normal(0.0, 0.0005, 3)
            caps.append((Rk, Tk, Cb, None))                 # as the route records them: with the frame
        Xs, spread_m, _ = wc.solve(caps)
        self.assertLess(np.linalg.norm(Xs[:3, 3] - X[:3, 3]), float(cfg.perception.wrist_calibration.max_spread_m))
        self.assertLess(math.degrees(np.linalg.norm(g.rot_log(Xs[:3, :3] @ X[:3, :3].T))), 0.5)
        self.assertLess(spread_m, float(cfg.perception.wrist_calibration.max_spread_m))

    def test_the_utensil_is_the_still_part_of_the_frames(self):
        """A fork along the tool's axis (to 0.11 m out of the tool point), the same in every frame, over a background that
        changes from frame to frame: its tips and its region are found where it was drawn."""
        from PIL import Image, ImageDraw
        from rammp_feeding import wrist_calibration as wc
        from rammp_feeding.camera import CameraModel, Intrinsics
        k = Intrinsics(607.3, 607.4, 323.8, 244.7, 640, 480)
        cam = CameraModel(k, k, np.eye(4))
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = g.rodrigues([0.03, -0.02, math.pi]), [0.023, 0.050, -0.103]
        Xi = np.linalg.inv(X)
        def px(z):
            return cam.pixel(Xi[:3, :3] @ np.array([0.0, 0.0, z]) + Xi[:3, 3])
        rng = np.random.default_rng(2)
        frames = []
        for _ in range(7):
            im = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8).repeat(1, 0))
            ImageDraw.Draw(im).line([px(-0.03), px(0.11)], fill=(230, 230, 230), width=14)
            frames.append(np.asarray(im))
        tip, box = wc.fork_in_image(frames, X, cam)
        self.assertLess(np.hypot(*np.subtract(tip, px(0.11))), 8.0)
        u, v = px(0.05)
        self.assertTrue(box[0] <= u <= box[2] and box[1] <= v <= box[3])

    def test_the_fork_pass_hangs_the_tips_over_the_boards_centre(self):
        from rammp_feeding import wrist_calibration as wc
        WC = cfg.perception.wrist_calibration
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = g.rodrigues([0.0, 0.0, math.pi]), [0.02, 0.05, -0.10]
        B = np.eye(4)
        B[:3, :3], B[:3, 3] = g.rodrigues([0.0, 0.0, 0.5]), [0.05, -0.35, -0.02]
        R = g.rot("x", 180.0)                                     # the utensil straight down
        pts = wc.fork_poses(B, X, R)
        nx, ny = (int(n) for n in WC.squares)
        centre = B[:3, :3] @ [nx * float(WC.square_m) / 2, ny * float(WC.square_m) / 2, 0.0] + B[:3, 3]
        depth = -0.10 + float(cfg.utensil.camera_to_tip_m)          # how far along the tool axis the tips hang
        np.testing.assert_allclose(pts[0] + R @ [0, 0, depth], centre + [0, 0, float(WC.fork_height_m)], atol=1e-9)
        shifts = sorted(np.round(np.linalg.norm(p - pts[0]), 6) for p in pts[1:])
        self.assertEqual(shifts, [float(WC.fork_shift_m)] * 4)

    def test_no_board_beyond_the_tips_is_refused(self):
        """The fork drawn the same in every frame, the background changing except just beyond its tips (white table
        there, as when the board lies elsewhere): refused rather than a tip read off the still table."""
        from PIL import Image, ImageDraw
        from rammp_feeding import wrist_calibration as wc
        from rammp_feeding.camera import CameraModel, Intrinsics
        k = Intrinsics(607.3, 607.4, 323.8, 244.7, 640, 480)
        cam = CameraModel(k, k, np.eye(4))
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = g.rodrigues([0.03, -0.02, math.pi]), [0.023, 0.050, -0.103]
        Xi = np.linalg.inv(X)

        def px(z):
            return cam.pixel(Xi[:3, :3] @ np.array([0.0, 0.0, z]) + Xi[:3, 3])
        rng = np.random.default_rng(5)
        frames = []
        for _ in range(7):
            im = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
            d = ImageDraw.Draw(im)
            d.ellipse([px(0.13)[0] - 40, px(0.13)[1] - 40, px(0.13)[0] + 40, px(0.13)[1] + 40], fill=(240, 240, 240))
            d.line([px(-0.03), px(0.11)], fill=(230, 230, 230), width=14)
            frames.append(np.asarray(im))
        self.assertIsNone(wc.fork_in_image(frames, X, cam))

    def test_the_printed_board_is_found_at_its_size(self):
        """The board's own image, as a camera 0.4 m straight above it would see it: every corner found, 0.4 m away, and
        the depth check reads the print's scale; and the sheet to print is a PDF."""
        from rammp_feeding import wrist_calibration as wc
        from rammp_feeding.camera import CameraModel, Intrinsics
        nx, ny = (int(n) for n in cfg.perception.wrist_calibration.squares)
        px_per_m, margin = 4000.0, 40
        sq = float(cfg.perception.wrist_calibration.square_m)
        w, h = int(round(nx * sq * px_per_m)), int(round(ny * sq * px_per_m))
        img = wc.board().generateImage((w + 2 * margin, h + 2 * margin), marginSize=margin, borderBits=1)
        f = px_per_m * 0.4
        cam = CameraModel(*[Intrinsics(f, f, margin, margin, w + 2 * margin, h + 2 * margin)] * 2, np.eye(4))
        rgb = np.dstack([img] * 3)
        C, n = wc.board_pose(cam, rgb)
        self.assertEqual(n, (nx - 1) * (ny - 1))
        self.assertAlmostEqual(C[2, 3], 0.4, delta=0.001)
        flat = np.ones(img.shape, np.uint16)                 # the depth of the board's plane, 0.4 m away
        self.assertAlmostEqual(wc.depth_scale(cam, 400 * flat, rgb, C), 1.0, delta=0.005)
        self.assertAlmostEqual(wc.depth_scale(cam, 364 * flat, rgb, C), 0.91, delta=0.005)   # printed at 91 %
        p = WORK / "charuco_board.pdf"
        wc.board_pdf(p)
        self.assertTrue(p.read_bytes().startswith(b"%PDF"))


class SceneCalibration(unittest.TestCase):
    def test_the_scene_camera_through_the_board(self):
        """The board on the table, seen by the scene camera from a known pose and by the wrist camera from
        calibrate-wrist's poses (a known mount, each view 0.5 mm and 0.1 deg off): the scene camera's pose again."""
        from rammp_feeding import scene_camera as sc_
        from rammp_feeding import wrist_calibration as wc
        arm = Arm(FakeRobot(Scene(), scene.captured("home")))
        T, R = arm.tool(scene.captured("home"))
        X = np.eye(4)
        X[:3, :3], X[:3, 3] = g.rodrigues([0.03, -0.02, math.pi]), [0.023, 0.050, -0.103]
        B = np.eye(4)
        B[:3, :3], B[:3, 3] = g.rodrigues([0.0, 0.0, 0.4]), T + np.array([0.03, -0.12, -0.2])
        E = np.eye(4)                                       # base <- the scene camera's link frame
        E[:3, :3], E[:3, 3] = g.rodrigues([0.1, 0.2, 2.5]), [0.6, -0.5, 0.12]
        L = np.eye(4)                                       # its link <- its colour optical frame (its driver's)
        L[:3, :3], L[:3, 3] = g.rodrigues([-math.pi / 2, 0.0, 0.0]) @ g.rodrigues([0.0, -math.pi / 2, 0.0]), [0.0, 0.015, 0.0]
        caps, rng = [], np.random.default_rng(1)          # its own noise, whatever ran before
        for Tk, Rk in wc.poses(T + np.array([0.0, 0.0, 0.1]), R):
            G = np.eye(4)
            G[:3, :3], G[:3, 3] = Rk, Tk
            Cw = np.linalg.inv(G @ X) @ B
            Cw[:3, :3] = g.rodrigues(rng.normal(0.0, math.radians(0.1), 3)) @ Cw[:3, :3]
            Cw[:3, 3] += rng.normal(0.0, 0.0005, 3)
            caps.append((Rk, Tk, Cw, None))
        Bm, spread_m, _ = wc.board_in_base(caps, X)
        self.assertLess(spread_m, float(cfg.perception.wrist_calibration.max_spread_m))
        Es = sc_.camera_in_base(Bm, np.linalg.inv(E @ L) @ B, L)
        self.assertLess(np.linalg.norm(Es[:3, 3] - E[:3, 3]), float(cfg.perception.wrist_calibration.max_spread_m))
        self.assertLess(math.degrees(np.linalg.norm(g.rot_log(Es[:3, :3] @ E[:3, :3].T))), 0.5)



class Node(unittest.TestCase):
    """What the capability engine (rammp_feeding/engine.py) relies on, without ROS."""

    def test_with_the_record_off_nothing_is_written(self):
        from rammp_feeding import record
        from rammp_feeding.config import runs_dir
        was, before = cfg["record"]["enabled"], sorted(runs_dir().rglob("*"))
        cfg["record"]["enabled"] = False
        try:
            record.end_session()
            self.assertIsNone(record.session("off"))
            self.assertIsNone(record.bite())
            self.assertIsNone(record.stage("perceive"))
            self.assertIsNone(record.here())
            record.event("x", a=1)
            self.assertIsNone(record.save_arrays("y", a=[1.0]))
            record.save_json(record.stage("s"), "f.json", {})
            record.save_image(record.here(), "f.jpg", np.zeros((2, 2, 3), np.uint8))
            with self.assertRaises(FileNotFoundError):
                record.latest_bite()
            self.assertEqual(sorted(runs_dir().rglob("*")), before)
        finally:
            cfg["record"]["enabled"] = was
            record.end_session()

    def test_the_nodes_parameters_select_the_sites_profiles(self):
        from rammp_feeding import config
        env = dict(os.environ)
        try:
            config.apply_parameters(dict(state_dir=str(state_dir()), runs_dir=str(WORK / "runs"), record=True,
                                         camera_profile="oak_d_pro", scene_camera_profile="oak_d_pro_w",
                                         config="delivery.stop_m=0.09"))
            self.assertEqual((cfg.profiles.camera, cfg.profiles.scene_camera), ("oak_d_pro", "oak_d_pro_w"))
            self.assertEqual(cfg.camera.streams.depth_info_topic, "/wrist_camera/color/camera_info")
            self.assertEqual(cfg.scene_camera.streams.depth_topic, "/scene_camera/depth/image_rect_raw")
            self.assertEqual(cfg.scene_camera.pose_from, "platform")
            self.assertAlmostEqual(float(cfg.delivery.stop_m), 0.09)
            self.assertIs(cfg.record.enabled, True)
        finally:
            os.environ.clear()
            os.environ.update(env)
            config.reload()
        self.assertEqual(cfg.profiles.camera, "realsense_d405")

    def test_the_robots_own_tf_places_the_scene_camera(self):
        from rammp_feeding import scene_camera as sc_
        from rammp_feeding.config import Cfg

        class Tf:
            published = []

            def transform(self, target, source, wait_s=3.0):
                T = np.eye(4)
                T[:3, 3] = [0.1, -0.5, 0.4]
                return T if (target, source) == (str(cfg.arm.base_frame), "scene_camera") else None

            def publish_static(self, *a):
                self.published.append(a)
        was = sc_.SC
        sc_.SC = Cfg(dict(was, pose_from="platform", link_frame="scene_camera"))
        f = state_dir() / "scene_extrinsic.json"
        existed = f.exists()
        try:
            f.write_text(json.dumps(dict(xyz=[0, 0, 0], quat_xyzw=[0, 0, 0, 1])))
            T, source = sc_.extrinsic(Tf())
            self.assertEqual(list(T[:3, 3]), [0.1, -0.5, 0.4])
            self.assertIn("tf_static", source)
            sc_.publish(Tf())
            self.assertEqual(Tf.published, [], "our pose published over the robot's own")
            with self.assertRaises(RuntimeError):
                sc_.calibrate(Tf(), None)
        finally:
            sc_.SC = was
            if not existed:
                f.unlink()

    def test_the_planners_world_file_is_written_only_when_it_changes(self):
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        p = Path(str(cfg.planner.world_file)).expanduser()
        scene.push_world(robot)
        m0 = p.stat().st_mtime_ns
        time.sleep(0.02)
        scene.push_world(robot)
        self.assertEqual(p.stat().st_mtime_ns, m0, "rewritten unchanged")
        try:
            scene.measured_mouth([-0.6, 0.3, 0.4])
            scene.push_world(robot)
            self.assertNotEqual(p.stat().st_mtime_ns, m0, "not written when it changed")
        finally:
            scene.measured_mouth(None)
        self.assertEqual(len(robot.worlds), 3, "the planner not told every time")

    def test_a_camera_keeps_a_few_frames_undecoded_and_sees_a_cancel(self):
        from types import SimpleNamespace
        from rammp_feeding.ros import Camera

        def depth(v):
            return SimpleNamespace(encoding="16UC1", height=2, width=3, data=np.full(6, v, np.uint16).tobytes())
        cam, t0 = Camera("/c", "/d", keep=30), time.time()
        for v in range(40):
            cam.on_depth(depth(v + 1))
        self.assertEqual(len(cam.depth), 30)
        self.assertIsInstance(cam.depth[-1][1], SimpleNamespace, "decoded as it arrived")
        self.assertTrue((cam.depth_median(5, t0) == 13).all())
        polls = []

        def check():
            polls.append(1)
            if len(polls) > 2:
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            Camera("/c", "/d", check=check).frame(wait_s=5.0)

if __name__ == "__main__":
    unittest.main(verbosity=1)
