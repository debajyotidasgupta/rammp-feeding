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
        self.assertEqual(settings_get("utensil.camera_to_tip_m", RAMMP_NO_LOCAL="1", RAMMP_CAMERA="realsense_d405"), "0.26")
        self.assertEqual(settings_get("utensil.camera_to_tip_m", RAMMP_NO_LOCAL="1", RAMMP_CAMERA="kinova_wrist"), "0.265")

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


class Scene(unittest.TestCase):
    def setUp(self):
        (TMP / "scene_state.json").write_text(json.dumps(dict(t=__import__("time").time(), bowl_xy=[0.29, 0.158],
                                                              table_z=-0.024, bowl_top_z=0.046, rim_radius_m=0.054)))

    def test_computed_overlook_puts_the_camera_over_the_bowl(self):
        from kinova import scene
        from kinova.frames import R_from_kortex
        from feeding.camera import CAM_IN_TOOL
        t = scene.overlook_pose()["tool_pose"]
        cam = np.array([t["x_m"], t["y_m"], t["z_m"]]) + R_from_kortex(t["theta_x_deg"], t["theta_y_deg"],
                                                                        t["theta_z_deg"]) @ CAM_IN_TOOL
        self.assertTrue(np.allclose(cam, [0.29, 0.158, -0.024 + 0.35], atol=1e-9), cam)

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
