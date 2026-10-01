#!/usr/bin/env python3
"""cuRobo, kept warm behind one HTTP port: collision-free paths for the free-space moves.

Runs where torch + cuRobo are (the planner container, `make planner`; on an x86 box docker/Dockerfile.planner_x86):

    python3 -m kinova.planner_server [--port 8769]

    GET  /ping                         -> ok (once both planners are warm)
    POST /plan  {"start": [7 deg], "goal": [7 deg], "carrying": bool}
                                       -> {"ok": true, "path": [[7 deg], ...], "dt": s, "planning_s": s}
                                          or {"ok": false, "status": ..., "error": ...}
    POST /world                        -> rebuilds the world from the latest workspace / mouth (below); {"ok": true}

It only plans; the arm side runs the path through its own guards (kinova/motion.run_path: the arm's speed limits,
the controller's validation, the torque watch). The world (kinova/curobo_world.py): the workspace
(kinova/workspace.py: the scene camera's bowl and table, grown by its error), the user's mouth (the last
delivery's, runs_dir/mouth_state.json, else motion.free_space.user_mouth_m), the wheelchair's boxes, the pedestal;
the utensil from the tip model, and a carried piece in the "carrying" planner.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

from kinova.curobo_world import no_pad_names, robot_config, world
from kinova.workspace import workspace
from settings import cfg

RUNS = Path(os.path.expanduser(str(cfg.deployment.runs_dir)))
LOCK = threading.Lock()


def mouth_now() -> list[float]:
    try:
        return [float(v) for v in json.loads((RUNS / "mouth_state.json").read_text())["mouth_base"]]
    except (OSError, ValueError, KeyError):
        return [float(v) for v in cfg.motion.free_space.user_mouth_m]


def tip_tool() -> np.ndarray:
    """The utensil tip in the tool frame -- kinova.motion.tip_in_tool's model, without its Kortex imports."""
    from feeding.camera import FXC, VP, cam_to_tool
    zt = float(cfg.utensil.camera_to_tip_m)
    d = (np.array(cfg.utensil.tip_px, float) - VP) * zt / FXC
    return cam_to_tool([d[0], d[1], zt]) + np.array(cfg.get("utensil.tine_offset_tool_m", [0.0, 0.0, 0.0]), float)


class Planners:
    """The two warm cuRobo planners: the utensil empty, and carrying a piece."""

    def __init__(self):
        import rammp_curobo
        self.cfg_dir = Path(rammp_curobo.__file__).resolve().parent / "configs"
        self.work = Path(tempfile.mkdtemp(prefix="planner_"))
        self.base_robot = yaml.safe_load((self.cfg_dir / "robot_gen3_2f85.yaml").read_text(encoding="utf-8"))
        self.base_plan = yaml.safe_load((self.cfg_dir / "gen3.yaml").read_text(encoding="utf-8"))
        self.by_load: dict[bool, object] = {}
        self.build()

    def build(self) -> None:
        from rammp_curobo import CuRoboPlanner
        w = world(workspace(), mouth_now(), table_extent=dict(cfg.workspace.table_extent_m))
        (self.work / "world.yaml").write_text(yaml.safe_dump(w))
        P = cfg.motion.free_space
        for food in (False, True):
            tag = "food" if food else "empty"
            (self.work / ("robot_%s.yaml" % tag)).write_text(yaml.safe_dump(robot_config(self.base_robot, tip_tool(), food)))
            plan = json.loads(json.dumps(self.base_plan))
            plan["robot"], plan["world"] = "robot_%s.yaml" % tag, "world.yaml"
            plan["planner"]["collision_activation_distance"] = float(P.activation_distance_m)
            plan["planner"]["world_padding"] = float(P.world_padding_m)
            plan["planner"]["no_pad_names"] = no_pad_names(w)
            path = self.work / ("plan_%s.yaml" % tag)
            path.write_text(yaml.safe_dump(plan))
            self.by_load[food] = CuRoboPlanner.from_config(str(path))
        self.world = w

    def refresh(self) -> int:
        """The latest workspace and mouth into both planners, in place (no re-warm)."""
        w = world(workspace(), mouth_now(), table_extent=dict(cfg.workspace.table_extent_m))
        for pl in self.by_load.values():
            pl.update_world(w["obstacles"])
        self.world = w
        return len(w["obstacles"])

    def plan(self, start_deg, goal_deg, carrying: bool) -> dict:
        pl = self.by_load[bool(carrying)]
        t0 = time.time()
        res = pl.plan_to_joints(list(np.radians(goal_deg)), list(np.radians(start_deg)))
        if not res.success:
            return dict(ok=False, status=str(res.status), error=str(res.error), planning_s=round(time.time() - t0, 3))
        tr = res.joint_traj
        path = np.degrees(np.unwrap(tr.positions, axis=0))
        return dict(ok=True, path=path.round(5).tolist(), dt=float(tr.dt), planning_s=round(time.time() - t0, 3),
                    goal_mismatch_deg=round(float(np.degrees(getattr(res, "goal_mismatch_rad", 0.0) or 0.0)), 4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(cfg.deployment.servers.planner_port))
    a = ap.parse_args()
    P = Planners()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/ping"):
                self._send(200, "ok"); return
            if self.path.startswith("/world"):
                self._send(200, P.world); return
            self._send(404, {})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n) or b"{}")
            with LOCK:
                if self.path.startswith("/plan"):
                    self._send(200, P.plan(q["start"], q["goal"], q.get("carrying", False))); return
                if self.path.startswith("/world"):
                    self._send(200, dict(ok=True, obstacles=P.refresh())); return
            self._send(404, {})

    print("planner server on :%d (cuRobo warm: empty + carrying; %d obstacles)" % (a.port, len(P.world["obstacles"])),
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
