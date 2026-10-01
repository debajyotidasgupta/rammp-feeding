#!/usr/bin/env python3
"""The arm's kinematic limits: show them, or apply config/ (deployment.arm.soft_limits).

    python -m kinova.scripts.arm_limits            # show the hard limits and every mode's soft limits
    python -m kinova.scripts.arm_limits --apply    # set the configured joint-speed soft limits (make limits)

The soft limits live ON THE ARM (Kinova ControlConfig), not in this code: every trajectory the controller runs is
capped by them, and a joint move asked to go faster is refused (INVALID_DURATION). --apply first saves the values
it replaces to the run record (runlog), and prints them, so they can be put back:

    RAMMP_SET="deployment.arm.soft_limits.joint_speed_dps=40" python -m kinova.scripts.arm_limits --apply
"""

from __future__ import annotations

import argparse
import math

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


def apply(conn) -> None:
    from kortex_api.autogen.messages import ControlConfig_pb2 as CC
    want = cfg.deployment.arm.soft_limits
    js = want.joint_speed_dps                                   # one value for every joint, or one per joint
    per = [float(x) for x in js] if isinstance(js, (list, tuple)) else None
    v = per if per else float(js)
    before = read(conn)
    runlog.save_json("soft_limits_before", before)
    hard = before["hard"]["joint_speed_dps"]
    cc = _client(conn)
    for mode in want.modes:
        n = len(before["soft"].get(mode, {}).get("joint_speed_dps", [])) or len(hard)
        speeds = [min(per[i] if per else v, hard[i]) for i in range(n)]   # never above a joint's hard limit
        msg = CC.JointSpeedSoftLimits()
        msg.control_mode = CC.ControlMode.Value(mode)
        msg.joint_speed_soft_limits.extend(speeds)
        cc.SetJointSpeedSoftLimits(msg)
        log.info("  set    %-30s joint speed %s -> %s deg/s", mode,
                 [round(x, 1) for x in before["soft"].get(mode, {}).get("joint_speed_dps", [])], speeds)
    runlog.event("soft_limits_set", modes=list(want.modes), joint_speed_dps=v, before=before["soft"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="set the configured joint-speed soft limits")
    a = ap.parse_args()
    runlog.setup("arm_limits")
    with kinova_connection() as conn:
        if a.apply:
            apply(conn)
        show(read(conn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
