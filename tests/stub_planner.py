"""A stand-in for kinova/planner_server.py where there is no GPU: the same HTTP API, a straight joint-space line.

    from stub_planner import serve
    serve(port, via)                # in a thread; /ping, /plan, /world as the real server

via(start, goal, carrying) -> [joints, ...]: the corners of the path (default: none, a straight joint line); the
stub joins them with straight joint lines. It knows no obstacles: it exercises the plumbing (the stages ask, check
our floor, run, settle), not collision avoidance -- tests/site_check.py --planner curobo does that, on a CUDA box.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def line(start, goal, step_deg: float = 1.0) -> list[list[float]]:
    a, b = np.asarray(start, float), np.asarray(goal, float)
    n = max(2, int(np.ceil(np.abs(b - a).max() / step_deg)) + 1)
    return [list(a + (b - a) * s) for s in np.linspace(0.0, 1.0, n)]


def serve(port: int, via=None) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send("ok")

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path.startswith("/plan"):
                corners = [q["start"]] + (list(via(q["start"], q["goal"], q.get("carrying", False))) if via else []) \
                    + [q["goal"]]
                for i in range(1, len(corners)):         # each corner the short way round from the one before
                    prev = np.asarray(corners[i - 1], float)
                    corners[i] = list(prev + (np.asarray(corners[i], float) - prev + 180.0) % 360.0 - 180.0)
                path = [p for a, b in zip(corners, corners[1:]) for p in line(a, b)[1:]]
                self._send(dict(ok=True, path=[corners[0]] + path, dt=0.05, planning_s=0.0))
            else:
                self._send(dict(ok=True, obstacles=0))

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
