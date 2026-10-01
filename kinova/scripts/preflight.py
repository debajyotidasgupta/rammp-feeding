#!/usr/bin/env python3
"""Before any motion: is the arm (and what the stack needs) as the calibration assumes? Never moves the arm.

    python -m kinova.scripts.preflight          # make preflight; make feed runs it first
    python -m kinova.scripts.preflight --fix    # make preflight FIX=1: also set our tool and speed limits

Each check prints OK / WARN / FAIL; any FAIL exits 1 (make feed does not start):
  tool          the arm's tool frame + mass (ControlConfig) are ours: another tool shifts every cartesian target FAIL
  other clients nothing else connected to the arm (another driver, a script, the web app)             FAIL
  faults        the arm's fault banks are clear                                                       FAIL
  kinematics    at feeding_start: the tool pose the arm reports matches the one captured there. Differing
                at the same joints = the arm's kinematics or joint zeros changed: the captured poses are
                stale, re-capture                                                                     FAIL
  speed limits  the trajectory soft limits are ours (slower is safe, but every move takes longer)     WARN
  servers       the detector, face (with a live colour stream) and camera servers answer              WARN
--fix sets the tool and the limits (make tool APPLY=1 / make limits APPLY=1: the arm's values are saved first).
Where the utensil tip is in the image, and where the table and bowl are, are checked at food_detect_start by the
perception stages, not here.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request

import runlog
from kinova.connection import kinova_connection
from kinova.scripts.safe_move import fold, joints, tool_mismatch
from settings import cfg

log = runlog.get("preflight")
HOME = "calibration/home_pose.json"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, name: str, text: str) -> None:
        self.rows.append((level, name, text))
        log.info("  %-4s  %-13s %s", level, name, text)

    @property
    def failed(self) -> bool:
        return any(r[0] == "FAIL" for r in self.rows)


def check_tool(conn, rep: Report, fix: bool) -> None:
    bad = tool_mismatch(conn)
    if bad and fix:
        from kinova.scripts import arm_tool
        before = arm_tool.read(conn)
        saved = arm_tool.save_before(before)
        arm_tool.write(conn, {k: v for k, v in cfg.deployment.arm.tool.items() if k != "tolerance"})
        bad = tool_mismatch(conn)
        rep.add("OK" if not bad else "FAIL", "tool", "set ours (the arm's values saved to %s)" % saved)
        return
    rep.add("FAIL" if bad else "OK", "tool",
            "; ".join(bad) + " -- make tool APPLY=1 (or make preflight FIX=1)" if bad else
            "frame %s m, %.3f kg: ours" % (list(cfg.deployment.arm.tool.transform_m), float(cfg.deployment.arm.tool.mass_kg)))


def check_clients(conn, rep: Report) -> None:
    """Nobody else connected to the arm: two programs commanding one arm fight. The arm lists every session;
    ours is our socket, 127.0.0.1 the arm's own."""
    try:
        ours = "%s;port=%d" % conn._tcp_transport.sock.getsockname()
        listed = conn._tcp_session.GetConnections().connection
    except AttributeError:
        rep.add("OK", "other clients", "skipped (no session list)")
        return
    others = []
    for c in listed:
        info = dict(kv.split("=", 1) for kv in c.connection_information.split(";") if "=" in kv)
        if info.get("ip") == "127.0.0.1" or "ip=%s;port=%s" % (info.get("ip"), info.get("port")) == "ip=" + ours:
            continue
        others.append("%s:%s" % (info.get("ip", "?"), info.get("port", "?")))
    if others and cfg.preflight.get("allow_other_clients", False):
        rep.add("WARN", "other clients", "%d connected to the arm (%s) -- allowed by preflight.allow_other_clients"
                % (len(others), ", ".join(others)))
        return
    rep.add("FAIL" if others else "OK", "other clients",
            "%d other connection(s) to the arm: %s -- another driver, script or the Kinova web app: stop it before "
            "moving" % (len(others), ", ".join(others)) if others else "none: this stack alone is connected")


def check_faults(conn, rep: Report) -> None:
    b = conn.base_cyclic.RefreshFeedback().base
    ok = b.fault_bank_a == 0 and b.fault_bank_b == 0
    rep.add("OK" if ok else "FAIL", "faults", "clear" if ok else "a=0x%08X b=0x%08X -- clear them on the arm first"
            % (b.fault_bank_a, b.fault_bank_b))


def check_kinematics(conn, rep: Report) -> None:
    home = json.load(open(HOME))
    q = joints(conn)
    off = max(abs(fold(a - b)) for a, b in zip(q, home["joints_deg"]))
    if off > 0.5:
        rep.add("OK", "kinematics", "skipped: not at feeding_start (%.1f deg away) -- checked there" % off)
        return
    from kinova.frames import R_from_kortex
    b = conn.base_cyclic.RefreshFeedback().base
    t = home["tool_pose"]
    dp = 1000 * math.dist((b.tool_pose_x, b.tool_pose_y, b.tool_pose_z), (t["x_m"], t["y_m"], t["z_m"]))
    Ra = R_from_kortex(b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z)
    Rb = R_from_kortex(t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"])
    c = max(-1.0, min(1.0, (float((Ra.T @ Rb).trace()) - 1) / 2))
    dr = math.degrees(math.acos(c))
    lim = cfg.preflight
    ok = dp <= float(lim.kinematics_mm) and dr <= float(lim.kinematics_deg)
    rep.add("OK" if ok else "FAIL", "kinematics",
            "at feeding_start the arm reports the tool %.1f mm / %.2f deg from where it was captured (%s)%s"
            % (dp, dr, home.get("captured", "?"), "" if ok else
               " -- the arm's kinematics changed: re-capture the poses (make capture) before feeding"))


def check_limits(conn, rep: Report, fix: bool) -> None:
    from kinova.scripts import arm_limits
    want = cfg.deployment.arm.soft_limits.joint_speed_dps
    want = [float(x) for x in want] if isinstance(want, (list, tuple)) else [float(want)] * 7
    got = arm_limits.read(conn)["soft"].get("ANGULAR_TRAJECTORY", {}).get("joint_speed_dps", [])
    same = len(got) == 7 and max(abs(a - b) for a, b in zip(got, want)) < 0.5
    if not same and fix:
        arm_limits.apply(conn)
        got = arm_limits.read(conn)["soft"].get("ANGULAR_TRAJECTORY", {}).get("joint_speed_dps", [])
        same = len(got) == 7 and max(abs(a - b) for a, b in zip(got, want)) < 0.5
    rep.add("OK" if same else "WARN", "speed limits",
            "trajectory joint speed %s deg/s%s" % ([round(v) for v in got],
                                                   "" if same else " (ours %s: moves are capped there; make limits "
                                                   "APPLY=1)" % [round(v) for v in want]))


def _get(port: int, path: str) -> str | None:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=2) as r:
            return r.read().decode()
    except Exception:  # noqa: BLE001
        return None


def check_servers(rep: Report) -> None:
    s = cfg.deployment.servers
    down = [n for n, p in (("detector", s.detector_port), ("face", s.face_port), ("camera", s.camera_port))
            if (_get(int(p), "/ping") or "").strip() != "ok"]
    st = _get(int(s.face_port), "/stream")
    stream = json.loads(st) if st else {}
    age = stream.get("last_frame_age_s")
    live = bool(stream.get("connected")) and age is not None and age < 2.0
    if down:
        rep.add("WARN", "servers", "%s not answering -- make models" % ", ".join(down))
    else:
        rep.add("OK" if live else "WARN", "servers",
                "all up; colour stream %s" % ("live (%.2f s old)" % age if live else "NOT live %s -- make models"
                                              % stream))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="set our tool and speed limits (saving the arm's values)")
    a = ap.parse_args()
    runlog.setup("preflight")
    rep = Report()
    log.info("preflight (arm %s)", cfg.deployment.arm.ip)
    with kinova_connection() as conn:
        check_tool(conn, rep, a.fix)
        check_clients(conn, rep)
        check_faults(conn, rep)
        check_kinematics(conn, rep)
        check_limits(conn, rep, a.fix)
    check_servers(rep)
    runlog.event("preflight", rows=rep.rows)
    log.info("%s", "PREFLIGHT FAILED -- fix the FAIL lines first" if rep.failed else "preflight ok")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
