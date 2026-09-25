"""Every value the arm-side modules use, as JSON -- run before and after a refactor and compare (no robot needed).

    PYTHONPATH=/work python3 tests/arm_values.py > values.json      # in the driver container
"""
import json
import sys
from unittest import mock

import numpy as np

out = {}
import kinova.scripts.safe_move as sm
out["safe_move"] = dict(max_tilt=sm.MAX_TILT_DEG, torque=sm.TORQUE_STEP_NM, no_progress=sm.NO_PROGRESS_S,
                        limits={str(k): v for k, v in sm.LIMITS.items()}, limit_margin=sm.LIMIT_MARGIN)
import kinova.scripts.sequence as sq
out["sequence"] = dict(lift_z=sq.LIFT_Z, near=sq.NEAR_DEG, lift_speed=sq.LIFT_SPEED, swing_speed=sq.SWING_SPEED)
import kinova.scripts.acquire as aq
names = ["MAX_OVER_M", "BACKOFF_FROM", "PUNCH_FLOOR_M", "PLUNGE_FAST", "PLUNGE_SLOW", "SLOW_ABOVE", "A_MAX", "SCOOP_TORQUE_NM"]
out["acquire"] = {n: getattr(aq, n) for n in names}
out["acquire"]["CAM_IN_TOOL"] = list(map(float, aq.CAM_IN_TOOL))
out["acquire"]["LIMITS"] = {str(k): v for k, v in aq.LIMITS.items()}
out["acquire"]["tip_in_tool"] = [round(float(x), 9) for x in aq.tip_in_tool()]
path = [[0.0] * 7]
for k in range(35):
    q = list(path[-1]); q[1] += 1.3; q[3] -= 0.8; path.append(q)
for k in range(60):
    q = list(path[-1]); q[0] -= 1.5; q[3] += 0.3; path.append(q)
out["acquire"]["time_scale"] = [round(float(x), 9) for x in aq._time_scale(path, 20.0)]
with mock.patch.object(sys, "argv", ["acquire", "pick"]):
    import argparse
    captured = {}
    orig = argparse.ArgumentParser.parse_args
    def grab(self, *a, **k):
        ns = orig(self, *a, **k); captured.update(vars(ns)); raise SystemExit(0)
    with mock.patch.object(argparse.ArgumentParser, "parse_args", grab):
        try:
            aq.main()
        except SystemExit:
            pass
out["acquire_args"] = {k: v for k, v in captured.items() if k != "step"}
import kinova.scripts.deliver as dl
names = ["LIFT_M", "RETRACT_TRIES", "SWING_DPS", "DETECT_SIDESTEP_M", "STOP_M", "AIM_DZ", "APPROACH_MPS", "MAX_APPROACH_M",
         "MAX_SHIFT_S", "NOSE_TO_MOUTH", "MOUTH_BEHIND_NOSE", "BITE_N", "BITE_HOLD_S", "RETRACT_MPS", "RETRACT_SWING_DPS",
         "RETRACT_DOWN_DPS", "RETRACT_HOME_DPS", "RETRACT_TORQUE_NM", "SWING_TORQUE_NM", "APPROACH_TORQUE_NM", "MIN_TIP_Z"]
out["deliver"] = {n: (list(getattr(dl, n)) if isinstance(getattr(dl, n), tuple) else getattr(dl, n)) for n in names}
captured.clear()
with mock.patch.object(sys, "argv", ["deliver", "bite"]):
    with mock.patch.object(argparse.ArgumentParser, "parse_args", grab):
        try:
            dl.main()
        except SystemExit:
            pass
out["deliver_args"] = {k: v for k, v in captured.items() if k != "step"}
import kinova.connection as cn
out["connection"] = dict(ip=cn.DEFAULT_IP, tcp=cn.TCP_PORT, udp=cn.UDP_PORT)
print(json.dumps(out, indent=1, sort_keys=True, default=float))
