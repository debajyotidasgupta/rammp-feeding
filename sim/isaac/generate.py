#!/usr/bin/env python3
"""One generation's bites, the stack's own code: site_check for every (strategy, scenario), in parallel.

    python3 sim/isaac/generate.py gen.json OUTDIR [--workers 48] [--servers 8]     # in planners.sif, with a GPU

gen.json: {"strategies": [{"id": ..., "set": {config key: value}}, ...],
           "scenarios":  [{"id": ..., "workspace": {...}, "mouth_m": [...], "food": {...}}, ...]}
Each pair becomes a site file (tests/sites/site_20260930.yaml's shape: the cameras, the workspace as the scene camera
reports it, the mouth, the food, and the strategy's config keys under `set`) and one tests/site_check.py run with
cuRobo, whose executed joint path is dumped to OUTDIR/dumps/<strategy>__<scenario>.json. cuRobo is kept warm: --servers
planner servers on the GPU, each with its own runs dir, shared round-robin by the workers (a worker holds its server
for the whole of one bite). OUTDIR/status.json: what ran, what the stack refused and why.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
PYPATH = os.pathsep.join([str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
BASE = yaml.safe_load((ROOT / "tests" / "sites" / "site_20260930.yaml").read_text())
PORT0 = 18700                                     # 9100 is node_exporter's on the cluster nodes


def site_file(strategy: dict, scenario: dict) -> dict:
    s = json.loads(json.dumps(BASE))
    s["set"] = dict(s.get("set", {}), **{k: v for k, v in strategy.get("set", {}).items() if v is not None})   # null: the default
    s["workspace"] = dict(scenario["workspace"])
    s["user"] = dict(s.get("user", {}), mouth_m=list(scenario["mouth_m"]))
    s["food"] = dict(scenario.get("food", s.get("food", {"height_m": 0.025, "size_m": 0.025})))
    return s


def start_servers(n: int, work: Path) -> list[tuple[int, Path]]:
    env = dict(os.environ, RAMMP_NO_LOCAL="1", RAMMP_CAMERA="realsense_d405", RAMMP_SCENE_CAMERA="orbbec_gemini336l",
               PYTHONPATH=PYPATH)
    out = []
    for k in range(n):
        port, runs = PORT0 + k, work / ("server_%02d" % k)
        runs.mkdir(parents=True, exist_ok=True)
        e = dict(env, RAMMP_SET="deployment.runs_dir=%s,deployment.servers.planner_port=%d" % (runs, port))
        log = open(runs / "server.log", "w")
        subprocess.Popen([sys.executable, "-m", "kinova.planner_server", "--port", str(port)], cwd=ROOT, env=e,
                         stdout=log, stderr=subprocess.STDOUT)
        out.append((port, runs))
    for port, _ in out:                                   # every server warm before the first bite
        for _ in range(900):
            try:
                if b"ok" in urllib.request.urlopen("http://127.0.0.1:%d/ping" % port, timeout=2).read():
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise SystemExit("planner server on :%d did not come up" % port)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--servers", type=int, default=8)
    a = ap.parse_args()
    g = json.loads(Path(a.gen).read_text())
    out = Path(a.out); (out / "dumps").mkdir(parents=True, exist_ok=True); (out / "sites").mkdir(exist_ok=True)
    t0 = time.time()
    servers = start_servers(a.servers, out / "work")
    print("%d planner servers warm in %.0f s" % (len(servers), time.time() - t0), flush=True)
    locks = [threading.Lock() for _ in servers]
    jobs = [(st, sc) for st in g["strategies"] for sc in g["scenarios"]]
    status, n_done = {}, [0]

    def run(k: int, st: dict, sc: dict) -> None:
        name = "%s__%s" % (st["id"], sc["id"])
        site = out / "sites" / (name + ".yaml")
        site.write_text(yaml.safe_dump(site_file(st, sc)))
        j = k % len(servers)
        port, runs = servers[j]
        work = out / "work" / ("w_%s" % name)
        env = dict(os.environ, SITE_WORK=str(work), SITE_RUNS=str(runs), PYTHONPATH=PYPATH)
        with locks[j]:                                     # the server's world is this bite's while it runs
            p = subprocess.run([sys.executable, "tests/site_check.py", str(site), "--planner", "server",
                                "--planner-port", str(port), "--dump", str(out / "dumps" / (name + ".json"))],
                               cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        lines = [l for l in p.stderr.splitlines() if "rc=" in l or "GOES" in l or "STOPPED" in l]
        status[name] = dict(ok=p.returncode == 0, why=next((l.strip() for l in lines if "rc=1" in l or "rc=9" in l), ""),
                            strategy=st["id"], scenario=sc["id"])
        n_done[0] += 1
        if n_done[0] % 10 == 0 or n_done[0] == len(jobs):
            print("  %d/%d bites (%d through) %.0f s" % (n_done[0], len(jobs), sum(v["ok"] for v in status.values()),
                                                        time.time() - t0), flush=True)

    with cf.ThreadPoolExecutor(max_workers=min(a.workers, len(servers) * 4)) as pool:
        list(pool.map(lambda kv: run(kv[0], *kv[1]), enumerate(jobs)))
    (out / "status.json").write_text(json.dumps(status, indent=1))
    print("DONE %d bites, %d went through, %.0f s" % (len(jobs), sum(v["ok"] for v in status.values()), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
