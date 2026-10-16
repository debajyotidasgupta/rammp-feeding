#!/usr/bin/env python3
"""The arm's kinematic limits: show them, or apply config/ (deployment.arm.soft_limits).

    python -m kinova.scripts.arm_limits            # show the hard limits and every mode's soft limits
    python -m kinova.scripts.arm_limits --apply    # set the configured joint-speed soft limits (make limits APPLY=1)
    python -m kinova.scripts.arm_limits --restore <runs_dir>/arm_limits_before_<time>.json   # put saved ones back

The soft limits live ON THE ARM (Kinova ControlConfig), not in this code: every trajectory the controller runs is
capped by them, and a joint move asked to go faster is refused (INVALID_DURATION). --apply / --restore first save
the values they replace to <runs_dir>/arm_limits_before_<time>.json (another user's settings can be put back).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import runlog
from kinova.connection import kinova_connection
from settings import cfg

log = runlog.get("arm_limits")


def _client(conn):
    from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
    return ControlConfigClient(conn._tcp_router)


def read(conn) -> dict:
    """{'hard': {...}, 'soft': {mode name: {...}}}: speeds deg/s, accelerations deg/s^2, twist m/s and deg/s."""
    from kortex_api.autogen.messages import ControlConfig_pb2 as CC
    cc = _client(conn)
    names = {v: k for k, v in CC.ControlMode.items()}
    h = cc.GetKinematicHardLimits()
    # the hard limits come back mixed: joint speeds in deg/s, accelerations in rad/s^2
    out = dict(hard=dict(joint_speed_dps=list(h.joint_speed_limits),
                         joint_accel_dps2=[math.degrees(v) for v in h.joint_acceleration_limits],
                         twist_linear_mps=h.twist_linear, twist_angular_dps=h.twist_angular), soft={})
    for m in cc.GetAllKinematicSoftLimits().kinematic_limits_list:
        if list(m.joint_speed_limits) or m.twist_linear:
            out["soft"][names.get(m.control_mode, str(m.control_mode))] = dict(
                joint_speed_dps=list(m.joint_speed_limits), joint_accel_dps2=list(m.joint_acceleration_limits),
                twist_linear_mps=m.twist_linear, twist_angular_dps=m.twist_angular)
    return out


def show(lim: dict) -> None:
    h = lim["hard"]
    log.info("  hard   joint speed %s deg/s, twist %.2f m/s",
             [round(v, 1) for v in h["joint_speed_dps"]], h["twist_linear_mps"])
    for name, s in lim["soft"].items():
        log.info("  soft   %-30s joint speed %s deg/s, accel %s deg/s^2, twist %.2f m/s",
                 name, [round(v, 1) for v in s["joint_speed_dps"]], [round(v, 1) for v in s["joint_accel_dps2"]],
                 s["twist_linear_mps"])


def save_before(lim: dict) -> Path:
    """The arm's limits, to <runs_dir>/arm_limits_before_<time>.json, before anything replaces them."""
    out = Path(str(cfg.deployment.runs_dir)).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    saved = out / time.strftime("arm_limits_before_%Y%m%d_%H%M%S.json")
    saved.write_text(json.dumps(dict(limits=lim, saved=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))
    return saved


def ours(before: dict) -> dict:
    """{mode: joint speeds deg/s} from deployment.arm.soft_limits, never above a joint's hard limit."""
    want = cfg.deployment.arm.soft_limits
    js = want.joint_speed_dps                                   # one value for every joint, or one per joint
    per = [float(x) for x in js] if isinstance(js, (list, tuple)) else None
    hard = before["hard"]["joint_speed_dps"]
    out = {}
    for mode in want.modes:
        n = len(before["soft"].get(mode, {}).get("joint_speed_dps", [])) or len(hard)
        out[mode] = [min(per[i] if per else float(js), hard[i]) for i in range(n)]
    return out


def write(conn, speeds: dict, before: dict) -> None:
    from kortex_api.autogen.messages import ControlConfig_pb2 as CC
    cc = _client(conn)
    for mode, v in speeds.items():
        msg = CC.JointSpeedSoftLimits()
        msg.control_mode = CC.ControlMode.Value(mode)
        msg.joint_speed_soft_limits.extend(v)
        cc.SetJointSpeedSoftLimits(msg)
        log.info("  set    %-30s joint speed %s -> %s deg/s", mode,
                 [round(x, 1) for x in before["soft"].get(mode, {}).get("joint_speed_dps", [])], v)


def apply(conn, restore: str | None = None) -> Path:
    """Set our joint-speed soft limits (or the ones saved in `restore`), saving the arm's first; the saved file."""
    before = read(conn)
    if restore:
        soft = json.loads(Path(restore).read_text())["limits"]["soft"]
        speeds = {m: list(v["joint_speed_dps"]) for m, v in soft.items() if m in cfg.deployment.arm.soft_limits.modes}
    else:
        speeds = ours(before)
    saved = save_before(before)
    log.info("  saved the arm's limits to %s (put back: make limits RESTORE=%s)", saved, saved)
    write(conn, speeds, before)
    runlog.event("soft_limits_set", speeds=speeds, before=before["soft"], saved=str(saved))
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="set the configured joint-speed soft limits")
    ap.add_argument("--restore", help="set the joint speeds saved in this file (an arm_limits_before_*.json)")
    a = ap.parse_args()
    runlog.setup("arm_limits")
    with kinova_connection() as conn:
        if a.apply or a.restore:
            apply(conn, a.restore)
        show(read(conn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
