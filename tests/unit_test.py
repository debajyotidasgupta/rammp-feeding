"""Unit tests: no ROS, no arm, no GPU (python3 tests/unit_test.py)."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
WORK = Path(tempfile.mkdtemp(prefix="rammp_unit_"))
os.environ.update(RAMMP_NO_LOCAL="1", RAMMP_STATE_DIR=str(WORK / "state"),
                  RAMMP_SET="deployment.runs_dir=%s,planner.world_file=%s,planner.world_file_in_planner=" % (WORK / "runs", WORK / "state/world.yaml"))

import numpy as np  # noqa: E402

from fake_robot import FakeRobot, Scene  # noqa: E402
from rammp_feeding import geometry as g, scene  # noqa: E402
from rammp_feeding.arm import Arm, MoveError, TorqueGuard, chain, time_scale  # noqa: E402
from rammp_feeding.config import cfg, state_dir  # noqa: E402
from rammp_feeding.ros import Detection  # noqa: E402

RNG = np.random.default_rng(0)


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
    kin = chain()

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
    def test_time_scale_caps(self):
        path = [[0.0] * 7, [30.0] + [0] * 6, [60.0] + [0] * 6]
        dt = time_scale(path, 20.0, 72.0)
        self.assertGreaterEqual(sum(dt), 3.0)                    # 60 deg at <= 20 deg/s
        v = [30.0 / d for d in dt]
        self.assertTrue(all(x <= 20.0 + 1e-6 for x in v))

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
            robot.inject[0] = (0.5 if stops else 0.97, [step] + [0] * 6)   # half way: above the food; at the end: in it
            if stops:
                with self.assertRaises(MoveError):
                    plunge(arm, u, scoop_after=False)
            else:
                plunge(arm, u, scoop_after=False)
                self.assertLess(arm.tip()[2], tip[2] - 0.1)


class Restored(unittest.TestCase):
    """What the port keeps from the stack before it."""

    def _at_overlook(self):
        sc = Scene(floor_lift=0.007)
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        robot.raw = np.radians(scene.overlook_joints(arm))
        sc.piece_xy = tuple(arm.tip()[:2])
        return sc, robot, arm

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
        acquire(robot, arm, dict(tool_goal=list(arm.tool()[0])), Path(tempfile.mkdtemp()), stop_after="hover10")
        self.assertAlmostEqual(arm.tip()[2] - (sc.table_z + sc.floor_lift + sc.piece_height), 0.010, delta=0.002)
        rise(arm)
        self.assertAlmostEqual(arm.tip()[2], z0, delta=0.002)

    def test_direct_home(self):
        from rammp_feeding.cli import home
        sc, robot, arm = self._at_overlook()
        home(robot, arm, direct=True)
        self.assertLess(g.max_step(arm.q(), scene.captured("home")), 0.1)


class Delivery(unittest.TestCase):
    def _at_look(self):
        from fake_robot import FakeFace
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("deliver"))
        return sc, robot, Arm(robot), FakeFace(robot)

    def test_no_face_from_the_delivery_look_then_the_second(self):
        from rammp_feeding.deliver import present
        sc, robot, arm, face = self._at_look()
        look1, finds = arm.tool()[0], []
        face.find = lambda cam, s: finds.append(arm.tool()[0]) or len(finds) > 1    # no face from the first look
        self.assertEqual(present(robot, arm, face, Path(tempfile.mkdtemp())), 0)
        self.assertLess(np.linalg.norm(finds[1] - (look1 + np.array(cfg.delivery.second_look_m))), 0.002)
        mouth = np.array(sc.mouth) + [0, 0, float(cfg.delivery.aim_dz_m)]
        self.assertAlmostEqual(np.linalg.norm(arm.tip() - mouth), float(cfg.delivery.stop_m), delta=0.01)

    def test_no_face_from_either_look_is_refused(self):
        from rammp_feeding.deliver import present
        sc, robot, arm, face = self._at_look()
        face.find = lambda cam, s: False
        with self.assertRaises(MoveError):
            present(robot, arm, face, Path(tempfile.mkdtemp()))

    def test_the_user_in_the_planners_world_is_where_the_operator_seats_them(self):
        from rammp_feeding.deliver import present
        sc, robot, arm, face = self._at_look()
        sc.mouth = tuple(np.array(sc.mouth) + [0.05, -0.03, 0.02])     # measured elsewhere: the world stays as set up
        present(robot, arm, face, Path(tempfile.mkdtemp()))
        head = next(o for o in scene.world()["obstacles"] if o["name"] == "user_head")
        m = np.array(cfg.motion.free_space.user_mouth_m)
        self.assertLess(np.linalg.norm(np.array(head["position"]) - (m + [-0.09, 0.0, 0.05])), 1e-9)


class Perception(unittest.TestCase):
    def test_food_filters(self):
        from rammp_feeding.perception import food, mask_of
        rgb = np.full((480, 848, 3), (200, 170, 90), np.uint8)
        bowl = mask_of(Detection("b", 1.0, [300, 100, 600, 400]), rgb.shape[:2])
        dets = [Detection("f", 0.9, [400, 200, 440, 240]),          # kept
                Detection("f", 0.8, [402, 202, 441, 241]),          # the same piece again
                Detection("f", 0.7, [450, 380, 480, 420]),          # inside the utensil's region
                Detection("f", 0.6, [100, 100, 140, 140]),          # outside the bowl
                Detection("f", 0.5, [500, 200, 505, 260]),          # a sliver
                Detection("f", 0.4, [320, 120, 560, 380])]          # bowl-sized
        self.assertEqual([d.box for d in food(rgb, dets, bowl)], [[400, 200, 440, 240]])

    def test_under_tip_from_the_rendered_depth(self):
        from rammp_feeding.perception import under_tip
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        robot.raw = np.radians(scene.overlook_joints(arm))
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

    def test_each_planned_move_pushes_our_world_and_the_command_restores_the_planners(self):
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        arm.travel(scene.overlook_joints(arm), "to food_detect_start")
        self.assertTrue(robot.world_pushed)
        self.assertEqual([b["name"] for b in robot.worlds[-1]["obstacles"]], [b["name"] for b in scene.world()["obstacles"]])
        scene.restore_world(robot)                      # no `world` parameter: the config's
        self.assertEqual((robot.worlds[-1], robot.world_pushed), (str(cfg.planner.restore_world), False))
        robot.world_param = "world_empty.yaml"          # a planner run with one, as a profile sets it
        scene.restore_world(robot)
        self.assertEqual(robot.worlds[-1], "world_empty.yaml")

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
                    arm.check_path([start, q, goal], None, spheres, probe(gap))
            else:
                arm.check_path([start, q, goal], None, spheres, probe(gap))
        # a path that starts that near (or inside) may leave from there, but never go nearer
        arm.check_path([q, goal], None, spheres, probe(clear - 0.005))
        with self.assertRaisesRegex(MoveError, "within .* of probe"):     # out and back over it: nearer than either end
            arm.check_path([goal, q, goal], None, spheres, probe(clear - 0.005))

    def test_a_planned_path_that_tilts_too_far_is_refused_before_it_runs(self):
        sc = Scene()
        site(sc)
        robot = FakeRobot(sc, scene.captured("home"))
        arm = Arm(robot)
        path = [scene.captured("home"), scene.overlook_joints(arm)]
        worst = max(g.tilt_deg(arm.tool(q)[1]) for q in g.densify(path, 0.5))
        arm.check_path(path, None, max_tilt=worst + 0.5)
        with self.assertRaisesRegex(MoveError, "tilts the utensil"):
            arm.check_path(path, None, max_tilt=worst - 0.5)
        self.assertEqual(robot.moves, [])

    def test_the_utensil_checks_our_boxes_the_bowl_exact(self):
        sc = Scene()
        site(sc)
        centres, rots, half, names = scene.utensil_boxes()
        raw = {o["name"]: o for o in scene.world()["obstacles"]}
        self.assertEqual(sorted(names), sorted([n for n in raw if n != "pedestal"] + ["bowl_lid"]))
        raw["bowl_lid"] = dict(dims=list(half[names.index("bowl_lid")] * 2), rpy_deg=[0.0, 0.0, 0.0])
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
        with self.assertRaisesRegex(MoveError, "within .* of (bowl_lid|table)"):     # (the raised table spans under it)
            arm.check_path([q, low, q], None, scene.utensil_spheres(arm), scene.utensil_boxes())


class SceneCalibration(unittest.TestCase):
    def test_recovers_a_moved_camera(self):
        """Depth points on the arm's spheres (and the table), seen from a known pose; the fit from a guess 4 cm and
        6 deg off finds it again."""
        from rammp_feeding import scene_camera as sc_
        robot = FakeRobot(Scene(), scene.captured("home"))
        arm = Arm(robot)
        R_true, t_true = g.rodrigues([0.1, -0.2, 2.0]), np.array([0.2, 0.7, 0.4])
        caps = []
        for q in (scene.captured("overlook"), scene.captured("home")):
            S = sc_.spheres(arm, q)
            pts = []
            for c in S:
                d = RNG.normal(size=(60, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
                p = c[:3] + c[3] * d
                pts.append(p[(p - t_true) @ (c[:3] - t_true) < (c[:3] - t_true) @ (c[:3] - t_true)])   # the near side
            P = np.vstack(pts)
            caps.append(((P - t_true) @ R_true, S))                  # into the camera's frame
        R0 = R_true @ g.rodrigues([0.0, 0.0, np.radians(6)])
        R, t, rms, n = sc_._fit(caps, R0, t_true + np.array([0.03, -0.02, 0.02]), None)
        self.assertLess(np.linalg.norm(t - t_true), 0.005)
        self.assertLess(np.degrees(np.linalg.norm(g.rot_log(R @ R_true.T))), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
