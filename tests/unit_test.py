#!/usr/bin/env python3
"""Offline unit tests for the machine / camera / scene layer (no arm, no Jetson, no models):

    .venv/bin/python tests/unit_test.py          # (tests/offline.sh runs it with the trace and the site check)

  settings     root / name templates, RAMMP_ROOT, local.yaml picking profiles, a utensil's by_camera values
  camera       the mount read from a calibration file; any rotation through cam_to_tool / the tip model
  scene cam    locate_scene's chain on a ray-cast synthetic scene: the bowl centre and the table recovered
  preflight    the tool check and the other-clients check, against stub controller answers
  scene        the computed overlook, the position-aware tip floor (with the scene camera's error)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="unit_test_"))

# one hermetic configuration for the in-process tests (settings load once, at import)
SCENE_XYZ, SCENE_Q = [-0.06506, 0.64157, 0.35206], [0.044383, 0.020535, -0.265541, 0.962859]   # the site's calibration
os.environ.update(RAMMP_NO_LOCAL="1", RAMMP_CAMERA="realsense_d405", RAMMP_SCENE_CAMERA="orbbec_gemini336l",
                  RAMMP_SET=",".join([
                      "deployment.runs_dir=%s" % TMP,
                      "workspace.overlook_camera_height_m=0.35",
                      "scene_camera.extrinsic.xyz=%s" % json.dumps(SCENE_XYZ),
                      "scene_camera.extrinsic.quat_xyzw=%s" % json.dumps(SCENE_Q)]))

import numpy as np  # noqa: E402

from settings import cfg  # noqa: E402


def settings_get(key: str, **env) -> str:
    e = dict(os.environ, **env)
    for k in ("RAMMP_SET", "RAMMP_CAMERA", "RAMMP_SCENE_CAMERA", "RAMMP_NO_LOCAL"):
        if k not in env:
            e.pop(k, None)
    return subprocess.run([sys.executable, "-m", "settings", "get", key], cwd=ROOT, env=e, capture_output=True,
                          text=True, check=True).stdout.strip()


class Settings(unittest.TestCase):
    def test_root_and_name_templates(self):
        self.assertEqual(settings_get("deployment.venvs.detector", RAMMP_NO_LOCAL="1", RAMMP_ROOT="/opt/x"),
                         "/opt/x/venvs/la")
        self.assertEqual(settings_get("deployment.containers.driver", RAMMP_NO_LOCAL="1",
                                      RAMMP_SET="deployment.name=site"), "site_driver")

    def test_by_camera_applies_only_with_that_camera(self):
        import yaml
        cfgdir = Path(__file__).resolve().parent.parent / "config"
        fork = yaml.safe_load((cfgdir / "utensil" / "fork.yaml").read_text())
        d405 = yaml.safe_load((cfgdir / "camera" / "realsense_d405.yaml").read_text())
        want = fork["by_camera"]["realsense_d405"]["glass_to_tip_m"] + d405["glass_to_optical_m"]   # the D405's: ruler + its glass
        got = settings_get("utensil.camera_to_tip_m", RAMMP_NO_LOCAL="1", RAMMP_CAMERA="realsense_d405")
        self.assertAlmostEqual(float(got), want, places=6)
        got = settings_get("utensil.camera_to_tip_m", RAMMP_NO_LOCAL="1", RAMMP_CAMERA="kinova_wrist")
        self.assertAlmostEqual(float(got), fork["camera_to_tip_m"], places=6)                      # the built-in camera's

    def test_local_yaml_picks_profiles(self):
        conf = TMP / "config"
        subprocess.run(["cp", "-R", str(ROOT / "config"), str(conf)], check=True)
        (conf / "local.yaml").write_text("profiles: {camera: realsense_d405}\n")
        self.assertEqual(settings_get("camera.streams.kind", RAMMP_CONFIG=str(conf)), "ros2")
        self.assertEqual(settings_get("camera.streams.kind", RAMMP_CONFIG=str(conf), RAMMP_NO_LOCAL="1"), "kinova_vision")


class Camera(unittest.TestCase):
    def test_mount_file_and_rotation(self):
        f = TMP / "mount.yaml"
        f.write_text("parent_frame: end_effector_link\nmount_xyz: [0.0, 0.075, -0.005]\nmount_quat_xyzw: [0, 0, 1, 0]\n")
        out = subprocess.run([sys.executable, "-c", "from feeding.camera import CAM_IN_TOOL, CAM_ROT, DEFAULT_MOUNT;"
                              "print(list(CAM_IN_TOOL.round(4)), DEFAULT_MOUNT)"], cwd=ROOT, capture_output=True,
                             text=True, check=True,
                             env=dict(os.environ, RAMMP_SET="camera.mount.file=%s" % f, RAMMP_NO_LOCAL="1")).stdout
        self.assertIn("[0.0, 0.075, -0.125] True", out.replace("np.float64(", "").replace(")", ""))

    def test_any_rotation_goes_through_cam_to_tool(self):
        import feeding.camera as C
        with mock.patch.object(C, "CAM_ROT", np.eye(3)):
            self.assertTrue(np.allclose(C.cam_to_tool([0.01, 0.02, 0.3]), C.CAM_IN_TOOL + [0.01, 0.02, 0.3]))
        self.assertTrue(np.allclose(C.cam_to_tool([0.01, 0.02, 0.3]), C.CAM_IN_TOOL + [-0.01, -0.02, 0.3]))


class SceneCamera(unittest.TestCase):
    """A synthetic scene ray-cast into the Orbbec's depth image through the same extrinsic: table + a bowl."""

    def test_table_is_the_dominant_surface_and_a_blocked_bowl_is_refused(self):
        import feeding.locate_scene as L
        rng = np.random.default_rng(0)
        z = np.concatenate([rng.normal(-0.024, 0.002, 3000), rng.uniform(0.1, 0.6, 2500), np.full(500, -0.8)])
        xy = rng.uniform(0.1, 0.5, (len(z), 2))
        self.assertAlmostEqual(L.table_height(np.c_[xy, z], [0.3, 0.2]), -0.024, delta=0.002)   # not the arm, the floor
        # the table read sloped (a rotated extrinsic: +7 mm per 10 cm), most pixels on its high side: at the bowl
        xy = np.r_[rng.uniform([0.3, 0.3], [0.5, 0.5], (4000, 2)), rng.uniform([0.1, 0.0], [0.3, 0.2], (800, 2))]
        z = -0.030 + 0.07 * (xy[:, 0] - 0.3) + 0.07 * (xy[:, 1] - 0.2) + rng.normal(0, 0.001, len(xy))
        self.assertAlmostEqual(L.table_height(np.c_[xy, z], [0.3, 0.2]), -0.030, delta=0.002)
        n = 400                                         # a box full of an arm (0.3 m up), a few bowl points
        box_pts = np.column_stack([np.full(n, 0.29), np.full(n, 0.158), np.where(np.arange(n) < 300, 0.3, 0.03)])
        ring_pts = np.column_stack([rng.uniform(0.1, 0.5, 400), rng.uniform(0.0, 0.3, 400), np.full(400, -0.024)])
        uv = np.vstack([np.tile([50.0, 50.0], (n, 1)), np.tile([30.0, 65.0], (400, 1))])   # in the box; the ring
        m = L.measure([40, 40, 60, 60], uv, np.vstack([box_pts, ring_pts]))
        self.assertIn("in front of the bowl", m.get("rejected", ""))

    def test_bowl_and_table_recovered(self):
        import feeding.locate_scene as L
        table_z, top_z, r, c = -0.024, 0.046, 0.054, np.array([0.290, 0.158])
        d = cfg.scene_camera.depth
        R, t, _ = L.extrinsic()
        v, u = np.mgrid[0:int(d.height), 0:int(d.width)]
        ray_opt = np.stack([(u - float(d.cx)) / float(d.fx), (v - float(d.cy)) / float(d.fy), np.ones(u.shape)], -1)
        ray = ray_opt.reshape(-1, 3) @ L.OPT_TO_LINK.T @ R.T            # base-frame directions (depth along z_opt = 1)
        o = t
        depth = np.zeros(len(ray))
        s_tab = (table_z - o[2]) / ray[:, 2]
        hit = (ray[:, 2] < 0) & (s_tab > 0)
        depth[hit] = s_tab[hit]
        # the bowl's outer wall: a cylinder from the table to its rim
        a = (ray[:, :2] ** 2).sum(1)
        b = 2 * (ray[:, :2] * (o[:2] - c)).sum(1)
        cc = ((o[:2] - c) ** 2).sum() - r * r
        disc = b * b - 4 * a * cc
        ok = disc > 0
        s_w = np.full(len(ray), np.inf)
        s_w[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
        z_w = o[2] + s_w * ray[:, 2]
        wall = ok & (s_w > 0) & (z_w > table_z) & (z_w < top_z) & ((depth == 0) | (s_w < depth))
        depth[wall] = s_w[wall]
        # the rim's top face (a thin ring) and the floor inside
        s_top = (top_z - o[2]) / ray[:, 2]
        p_top = o[:2] + s_top[:, None] * ray[:, :2]
        rr = np.linalg.norm(p_top - c, axis=1)
        rim = (s_top > 0) & (rr < r) & (rr > r - 0.006) & ((depth == 0) | (s_top < depth))
        depth[rim] = s_top[rim]
        img = depth.reshape(u.shape)
        p_link, _ = L.depth_points(img)
        p_base = p_link @ R.T + t
        uv = L.project_colour(p_link)
        sel = np.linalg.norm(p_base[:, :2] - c, axis=1) < r + 0.003
        sel &= p_base[:, 2] > table_z + 0.005
        bx = [float(uv[sel, 0].min()), float(uv[sel, 1].min()), float(uv[sel, 0].max()), float(uv[sel, 1].max())]
        m = L.measure(bx, uv, p_base)
        self.assertNotIn("rejected", m, m.get("rejected"))
        self.assertLess(abs(m["table_z"] - table_z), 0.002)
        self.assertLess(np.linalg.norm(np.array(m["bowl_xy"]) - c), 0.012, m["bowl_xy"])
        self.assertLess(abs(m["bowl_top_z"] - top_z), 0.005)


class Preflight(unittest.TestCase):
    def test_tool_mismatch(self):
        from kinova.scripts import safe_move
        tool = SimpleNamespace(tool_transform=SimpleNamespace(x=0, y=0, z=0.2, theta_x=0, theta_y=0, theta_z=0),
                               tool_mass=1.025)
        stub = mock.MagicMock()
        stub.return_value.GetToolConfiguration.return_value = tool
        with mock.patch("kortex_api.autogen.client_stubs.ControlConfigClientRpc.ControlConfigClient", stub):
            bad = safe_move.tool_mismatch(SimpleNamespace(_tcp_router=object()))
            self.assertEqual(len(bad), 2, bad)                        # the frame and the mass
            with self.assertRaises(safe_move.MoveError):
                safe_move.check_tool(SimpleNamespace(_tcp_router=object()))
            tool.tool_transform.z, tool.tool_mass = 0.12, 0.831
            self.assertEqual(safe_move.tool_mismatch(SimpleNamespace(_tcp_router=object())), [])
        self.assertEqual(safe_move.tool_mismatch(SimpleNamespace()), [])      # the fake controller: ours

    def test_other_clients(self):
        from kinova.scripts import preflight as P

        def conn(*infos):
            sock = SimpleNamespace(getsockname=lambda: ("192.168.1.11", 4000))
            lst = SimpleNamespace(connection=[SimpleNamespace(connection_information=i) for i in infos])
            return SimpleNamespace(_tcp_transport=SimpleNamespace(sock=sock),
                                   _tcp_session=SimpleNamespace(GetConnections=lambda: lst))
        rep = P.Report()
        P.check_clients(conn("type=TCP;ip=127.0.0.1;port=1", "type=TCP;ip=192.168.1.11;port=4000"), rep)
        self.assertEqual(rep.rows[-1][0], "OK")
        P.check_clients(conn("type=TCP;ip=127.0.0.1;port=1", "type=TCP;ip=192.168.1.11;port=4000",
                             "type=TCP;ip=192.168.1.11;port=5216"), rep)
        self.assertEqual(rep.rows[-1][0], "FAIL")
        self.assertIn("192.168.1.11:5216", rep.rows[-1][2])


class PlungeFloor(unittest.TestCase):
    """A clear bowl: depth reads the table beneath it -- the plunge stops at the bowl's own floor (2026-10-02)."""

    def test_the_bowls_floor_bounds_a_reading_through_it(self):
        import kinova.scripts.acquire as A
        u = A.Under.from_dict(dict(top=0.417, gap_top=0.181, floor=0.445, gap_floor=0.209, height=0.028))
        tip_z = 0.455 - 0.236                                      # the camera's height - camera-to-tip
        floor = {"bowl.floor_above_table_m": 0.020}
        with mock.patch.object(A.scene, "table_height", lambda: 0.007), \
                mock.patch.object(A, "cfg", SimpleNamespace(get=lambda k, d=None: floor.get(k, d))):
            d = A._plunge_depth(u, 0.0, tip_z)
            self.assertAlmostEqual(tip_z - d.down, 0.027, places=6)          # at the floor, not 3 mm above the table
            floor["bowl.floor_above_table_m"] = None
            self.assertAlmostEqual(A._plunge_depth(u, 0.0, tip_z).down, 0.209, places=6)   # unset: as measured


class OpenMouthWait(unittest.TestCase):
    """The open-mouth hold over a far face's detection gaps: skipped up to open_mouth_gap_s, never a closed mouth."""

    def run_wait(self, frames):
        import feeding.detect_mouth_mp as M
        seq = iter(frames)                              # per 1/30 s frame: "open", "closed" or None (no face found)
        L_open = np.zeros((478, 2)); L_open[13] = [0, 0]; L_open[14] = [0, 30]; L_open[61] = [-50, 15]; L_open[291] = [50, 15]
        L_shut = L_open.copy(); L_shut[14] = [0, 1]
        state = {}

        def read():
            time.sleep(1 / 30)
            state["f"] = next(seq, "closed")
            return True, np.zeros((480, 848, 3), np.uint8), 0.0
        with mock.patch.object(M, "landmarks", lambda det, img: ({"open": L_open, "closed": L_shut}.get(state["f"]), 0.0)):
            return M._wait(None, read, TMP, timeout=float(len(frames)) / 30 + 0.5)

    def test_gaps_up_to_the_limit_are_skipped(self):
        # 0.4 s open with two 0.13 s detection gaps: counts
        self.assertEqual(self.run_wait(["open"] * 3 + [None] * 4 + ["open"] * 3 + [None] * 4 + ["open"] * 4), 0)

    def test_a_long_gap_or_a_closed_mouth_resets(self):
        self.assertEqual(self.run_wait(["open"] * 5 + [None] * 15 + ["open"] * 5 + ["closed"] * 10), 4)
        self.assertEqual(self.run_wait((["open"] * 4 + ["closed"]) * 4 + ["closed"] * 10), 4)


class SceneCalibration(unittest.TestCase):
    """The scene camera's pose from the arm's spheres: synthetic, the truth known."""

    def test_recovers_a_knocked_camera(self):
        import feeding.calibrate_scene as C
        rng = np.random.default_rng(1)
        R_true, t_true = C.rot(np.radians([4.0, 20.0, -105.0])), np.array([0.1, 0.7, 0.35])

        def capture(chain):                             # an arm pose: its spheres, and the camera's view of them
            S = np.array([[*(np.array(a) + (np.array(b) - np.array(a)) * f), 0.05 - 0.02 * f]
                          for a, b in zip(chain[:-1], chain[1:]) for f in (0.0, 0.33, 0.66)])
            pts = []
            for c in S:                                 # each sphere's half facing the camera
                v = rng.normal(size=(300, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
                pts.append(c[:3] + v[(v @ (t_true - c[:3])) > 0] * c[3])
            pts.append(np.c_[rng.uniform(0.1, 0.7, 4000), rng.uniform(-0.3, 0.6, 4000), np.full(4000, -0.03)])  # the table
            X = np.vstack(pts) + rng.normal(0, 0.002, (sum(len(p) for p in pts), 3))
            return (X - t_true) @ R_true, S             # the points in the camera's frame
        caps = [capture([[0, 0, 0.2], [0.05, 0.05, 0.35], [0.2, 0.08, 0.4], [0.35, 0.12, 0.38], [0.42, 0.2, 0.33], [0.43, 0.22, 0.25]]),
                capture([[0, 0, 0.2], [-0.05, 0.08, 0.38], [0.05, 0.25, 0.45], [0.15, 0.4, 0.35], [0.25, 0.45, 0.3], [0.32, 0.4, 0.22]])]
        R0, t0 = R_true @ C.rot(np.radians([3.0, -4.0, 4.0])), t_true + [0.04, -0.03, 0.02]   # knocked: 5 cm, 6 deg
        R, t, rms, n = C.best_fit(caps, R0, t0)
        self.assertLess(np.linalg.norm(t - t_true), 0.01)
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R) - 1) / 2, -1, 1))), 1.0)
        self.assertLess(rms, 0.004)


    def test_one_pose_and_the_table_recover_a_hard_knock(self):
        import feeding.calibrate_scene as C
        rng = np.random.default_rng(2)
        R_true, t_true = C.rot(np.radians([4.0, 20.0, -105.0])), np.array([0.1, 0.7, 0.35])
        chain = [[0, 0, 0.2], [0.05, 0.05, 0.35], [0.2, 0.08, 0.4], [0.35, 0.12, 0.38], [0.42, 0.2, 0.33], [0.43, 0.22, 0.25]]
        S = np.array([[*(np.array(a) + (np.array(b) - np.array(a)) * f), 0.05 - 0.02 * f]
                      for a, b in zip(chain[:-1], chain[1:]) for f in (0.0, 0.33, 0.66)])
        pts = []
        for c in S:
            v = rng.normal(size=(300, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
            pts.append(c[:3] + v[(v @ (t_true - c[:3])) > 0] * c[3])
        pts.append(np.c_[rng.uniform(0.1, 0.7, 4000), rng.uniform(-0.3, 0.6, 4000), np.full(4000, 0.007)])  # the table
        X = np.vstack(pts) + rng.normal(0, 0.002, (sum(len(p) for p in pts), 3))
        caps = [((X - t_true) @ R_true, S)]
        R0, t0 = R_true @ C.rot(np.radians([6.0, -8.0, 8.0])), t_true + [0.08, -0.06, 0.04]   # 11 cm, 13 deg
        R, t, rms, n = C.best_fit(caps, R0, t0, table_z=0.007)
        self.assertLess(np.linalg.norm(t - t_true), 0.01)
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(R_true.T @ R) - 1) / 2, -1, 1))), 1.0)


class Scene(unittest.TestCase):
    def setUp(self):
        (TMP / "scene_state.json").write_text(json.dumps(dict(t=__import__("time").time(), bowl_xy=[0.29, 0.158],
                                                              table_z=-0.024, bowl_top_z=0.046, rim_radius_m=0.054)))

    def test_computed_overlook_puts_the_aim_over_the_bowl_the_lens_at_its_height(self):
        from kinova import scene
        from kinova.frames import R_from_kortex
        from kinova.motion import tip_in_tool
        from feeding.camera import CAM_IN_TOOL
        from settings import cfg
        t = scene.overlook_pose()["tool_pose"]
        T = np.array([t["x_m"], t["y_m"], t["z_m"]])
        R = R_from_kortex(t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"])
        aim = T + R @ (tip_in_tool() if str(cfg.workspace.overlook_aim) == "tip" else CAM_IN_TOOL)
        self.assertTrue(np.allclose(aim[:2], [0.29, 0.158], atol=1e-9), aim)
        lens_z = (T + R @ CAM_IN_TOOL)[2]
        self.assertAlmostEqual(lens_z, -0.024 + float(cfg.workspace.overlook_camera_height_m), places=9)

    def test_tip_floor_bowl_table_and_beside_it(self):
        from kinova import scene
        e = 0.05                                             # scene_camera.extrinsic.error_m
        self.assertAlmostEqual(scene.tip_floor(0.29, 0.158), 0.046 + 0.015 + e)
        self.assertAlmostEqual(scene.tip_floor(0.29 + 0.054 + 0.039 + e, 0.158), 0.046 + 0.015 + e)
        self.assertAlmostEqual(scene.tip_floor(0.29 + 0.054 + 0.041 + e, 0.158), -0.024 + 0.05 + e)
        self.assertEqual(scene.tip_floor(0.058, -0.181), -np.inf)   # feeding_start: beside the table, below its top

    def test_world_grows_by_the_error(self):
        from kinova.curobo_world import world
        ws = dict(bowl_xy=[0.29, 0.158], table_z=-0.024, bowl_top_z=0.046, bowl_radius=0.054)
        a = {o["name"]: o for o in world(ws, [-0.277, 0.306, 0.397], chair=[])["obstacles"]}
        b = {o["name"]: o for o in world(dict(ws, error_m=0.05), [-0.277, 0.306, 0.397], chair=[])["obstacles"]}
        self.assertAlmostEqual(b["table"]["position"][2] - a["table"]["position"][2], 0.05)
        self.assertAlmostEqual(b["bowl_wall_0"]["dims"][0] - a["bowl_wall_0"]["dims"][0], 0.10)


class CuroboWorld(unittest.TestCase):
    """The planner's world and utensil model (kinova/curobo_world.py): plain data, no GPU."""

    def test_hollow_bowl_user_and_floats(self):
        from kinova.curobo_world import no_pad_names, world
        w = world(dict(bowl_xy=[0.29, 0.158], table_z=-0.024, bowl_top_z=0.046, bowl_radius=0.054),
                  [-0.277, 0.306, 0.397], chair=[])
        names = [o["name"] for o in w["obstacles"]]
        self.assertEqual(sum(n.startswith("bowl_wall_") for n in names), 12)
        self.assertIn("bowl_floor", names)
        self.assertNotIn("bowl", names)                      # no solid bowl: the utensil works over its opening
        head = next(o for o in w["obstacles"] if o["name"] == "user_head")
        self.assertAlmostEqual(head["position"][2], 0.447)
        self.assertTrue(all(isinstance(v, float) for o in w["obstacles"] for v in o["position"] + o["dims"]))
        self.assertEqual(set(no_pad_names(w)), {"pedestal", "bowl_floor"} | {"bowl_wall_%d" % k for k in range(12)})

    def test_spoon_spheres_end_at_the_tip(self):
        from kinova.curobo_world import TOOL_Z, robot_config, spoon_spheres
        tip = np.array([-0.0116, 0.0342, 0.135])
        sp = spoon_spheres(tip, food=False)
        c = np.array(sp[-1]["center"])                       # bracelet (x, -y, -0.061525 - z_flange)
        self.assertTrue(np.allclose([c[0], -c[1], -0.061525 - c[2]], tip + [0, 0, TOOL_Z]))
        self.assertEqual(len(spoon_spheres(tip, food=True)), len(sp) + 1)
        base = dict(robot_cfg=dict(kinematics=dict(collision_spheres=dict(bracelet_link=[]), self_collision_ignore={})))
        r = robot_config(base, tip, food=True)
        k = r["robot_cfg"]["kinematics"]
        self.assertEqual(len(k["collision_spheres"]["bracelet_link"]), len(sp) + 1)
        self.assertIn("left_inner_finger_pad", k["self_collision_ignore"]["bracelet_link"])
        self.assertEqual(base["robot_cfg"]["kinematics"]["collision_spheres"]["bracelet_link"], [])   # not mutated


if __name__ == "__main__":
    unittest.main(verbosity=2)
