"""Run every arm stage against the fake controller and write what it did: a behaviour trace.

    python tests/arm_trace.py OUT.json                 # record
    python tests/arm_trace.py OUT.json --compare BASE.json
    python tests/arm_trace.py /tmp/t.json --compare tests/golden/arm_trace.json.gz    # against the committed one

tests/golden/arm_trace.json.gz is the trace of the code as committed. Re-record it only for an INTENDED behaviour
change, and say which steps changed and why. History:
  2026-09-25  the readability pass: identical to the code before except `deliver sidestep`, which never ran
              (its --step clashed with the step argument).
  2026-09-25  packed food (acquisition.plunge.no_floor): step 24 (food "3 mm tall") now plunges 25 mm below
              its top instead of refusing; 21-23 read the arm pose before refusing; the packed_scenarios added.
  2026-09-25  a "floor" > 25 mm above the table is food: the scenarios' table set to 0.255 (just under their
              synthetic floor); a packed scenario with a high false floor added.
  2026-09-25  pick hands its floor reading to the plunge in memory: it no longer writes under.json (the shell's
              file; the container could not overwrite it) -- the pick steps' state files differ, nothing else.
  2026-09-25  motion 2.2x faster except the plunge (config): durations and speeds differ everywhere; with the old
              speeds pinned (RAMMP_SET) the trace was identical to the previous golden one -- the tool-frame tine
              offset, the depth retries and the wait record change nothing else.
  2026-09-25  transit moves near the arm's limits (65 deg/s, 0.22 m/s lift, 72 deg/s^2, joint-move pad 0.3 s):
              with the previous values pinned the trace was identical.
  2026-09-25  goto re-sends a joint move the controller refuses as too short at its stated minimum duration;
              the fake refuses like the arm (joint_limit_dps) -- two scenarios added, nothing else changed.
  2026-09-25  trajectories planned with the acceleration cap / motion.controller_accel_factor (1.6): with the
              factor pinned to 1.0 the trace was identical; validated on the arm (1.0 and 1.3 refused, 1.6 accepted).

No robot, no network. Each scenario is the real command line (the module's main()), run on one persisted fake
arm, like a session on the real one. The trace holds, per step: every controller request (IK, FK, validated and
executed trajectories, actions), the printed output, the exit code, and the state files the stages hand to each
other. Two traces that match mean the refactor changed nothing the arm or the operator would see.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

import fake_kortex as fk  # noqa: E402

fk.install_clock()
os.environ.setdefault("RAMMP_FOOD", "strawberry")      # the scenarios were recorded with the strawberry aim offset
os.environ.setdefault("RAMMP_UTENSIL", "fork_20260924")   # ... and the fork calibration of 2026-09-24

# The stages' state files go to a scratch directory, never the real /tmp: in the driver container those hold
# the live delivery state and the swing / retract plans.
WORK = Path(tempfile.mkdtemp(prefix="arm_trace_"))
STATE = WORK / "state"
STATE.mkdir()
os.environ["RAMMP_STATE_DIR"] = str(STATE)

from settings import cfg  # noqa: E402

STATE_FILES = [str(STATE / f) for f in ("deliver_state.json", "deliver_swing_plan.json", "deliver_retract_plan.json",
                                        "punch_done.json", "scoop_log.json", "under.json", "under_over.json")]


def pose_of(rec):
    t = rec["tool_pose"]
    return [t["x_m"], t["y_m"], t["z_m"], t["theta_x_deg"], t["theta_y_deg"], t["theta_z_deg"]]


CAL = ROOT / "calibration"
HOME = json.loads((CAL / "home_pose.json").read_text())
LIFT = json.loads((CAL / "lift_pose.json").read_text())
DETECT = cfg.overlook_pose()
SE = cfg.bowl.scoop_end
R_OVER = fk.R_rpy(*pose_of(DETECT)[3:])
RX90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
SCOOP_END_POSE = list(SE.tool_m) + fk.rpy_R(RX90 @ R_OVER)       # utensil level, pointing at the user (+Y)


def new_arm():
    arm = fk.FakeArm()
    arm.register(HOME["joints_deg"], pose_of(HOME))
    arm.register(LIFT["joints_deg"], pose_of(LIFT))
    arm.register(DETECT["joints_deg"], pose_of(DETECT))
    arm.register(SE.joints_deg, SCOOP_END_POSE)
    return arm


ARM = new_arm()


@contextlib.contextmanager
def _conn():
    yield fk.FakeConnection(ARM)


def depth_frame():
    """A synthetic overlook-from-OVER depth frame: floor at 410 mm, a 30 mm food piece under the tip line."""
    c, d = cfg.camera.colour, cfg.camera.depth
    vp, tip, zt = np.array(c.vertical_vp, float), np.array(cfg.utensil.tip_px, float), float(cfg.utensil.camera_to_tip_m)
    t = np.array(d.depth_to_colour_m, float)
    z = 0.38
    p = vp + (tip - vp) * zt / z
    X, Y, Z = (p[0] - c.cx) * z / c.fx - t[0], (p[1] - c.cy) * z / c.fy - t[1], z - t[2]
    ud, vd = d.fx * X / Z + d.cx, d.fy * Y / Z + d.cy
    yy, xx = np.mgrid[0:int(d.height), 0:int(d.width)]
    out = np.full((int(d.height), int(d.width)), 410.0 - 1000 * t[2])
    out[np.hypot(xx - ud, yy - vd) < 14] = 380.0 - 1000 * t[2]
    out[:6, :] = 0                                           # a band with no return
    return out.astype(np.uint16)


def fake_urlopen(url, timeout=None):
    b = io.BytesIO()
    np.save(b, depth_frame())
    return contextlib.closing(io.BytesIO(b.getvalue()))


def write(path, obj):
    Path(path).write_text(json.dumps(obj))


def face(nose_depth=0.44, mouth=True, stale=False, filled=False, name="face", mouth_depth=0.445):
    d = WORK / name
    d.mkdir(exist_ok=True)
    (d / "colour.jpg").write_bytes(b"jpg")
    f = dict(nose=dict(box_upright=[330, 475, 390, 525], depth_m=nose_depth))
    if mouth:
        f["mouth"] = dict(box_upright=[320, 530, 400, 570], depth_m=mouth_depth, cam_xyz=[0.004, -0.006, mouth_depth],
                          prompt="mouth", depth_filled=filled)
    (d / "face.json").write_text(json.dumps(f))
    now = fk.clock.time()
    os.utime(d / "colour.jpg", (now - 2, now - 2))
    os.utime(d / "face.json", (now - 3 if stale else now - 1,) * 2)
    return ["--face", str(d / "face.json"), "--colour", str(d / "colour.jpg")]


def clear_state():
    for f in STATE_FILES:
        Path(f).unlink(missing_ok=True)


def run(module, args, trace):
    mod = __import__(module, fromlist=["main"])
    ARM.calls = []
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", [module] + args), mock.patch.object(mod, "kinova_connection", _conn), \
            mock.patch("urllib.request.urlopen", fake_urlopen), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = mod.main()
        except SystemExit as e:
            rc = e.code
    files = {}
    for f in STATE_FILES:
        if Path(f).exists():
            name = "/tmp/" + Path(f).name              # traces name them as on the arm
            try:
                files[name] = fk.rounded(json.loads(Path(f).read_text()))
            except ValueError:
                files[name] = Path(f).read_text()
    step = dict(cmd=module.split(".")[-1] + " " + " ".join(a if not a.startswith(str(WORK)) else Path(a).name for a in args),
                rc=rc, stdout=out.getvalue().replace(str(STATE), "/tmp").splitlines(),
                stderr=[l for l in err.getvalue().replace(str(STATE), "/tmp").splitlines() if "Deprecat" not in l],
                calls=ARM.calls, files=files, end_joints=fk.rounded(ARM.q), end_pose=fk.rounded(ARM.pose))
    trace.append(step)
    print("  %-60s rc=%s  %d calls" % (step["cmd"][:60], rc, len(ARM.calls)), file=sys.__stderr__)


SEQ, ACQ, DEL = "kinova.scripts.sequence", "kinova.scripts.acquire", "kinova.scripts.deliver"


def target():
    p = pose_of(DETECT)
    write(STATE / "target.json", dict(tool_goal=[p[0] + 0.012, p[1] - 0.018, p[2]]))


def under(top_gap, floor_gap, **extra):
    write(STATE / "under.json", dict(top=0.265 + top_gap, gap_top=top_gap, floor=0.265 + floor_gap, gap_floor=floor_gap,
                                  height=floor_gap - top_gap, **extra))


def scenarios(trace):
    # the synthetic floor readings below (gap_floor 0.123 from a tip at z 0.392) put the bowl floor at z ~0.27: the
    # table just under it, as with the real bowl -- else the packed-food rule would read every one as food
    nodes = [n for n in (dict.get(cfg, "workspace", {}), dict.get(cfg["bowl"], "captured_at", {}))   # the real
             if "table_height_m" in n]                                                              # nodes
    real = [n["table_height_m"] for n in nodes]
    for n in nodes:
        n["table_height_m"] = 0.255
    try:
        _scenarios(trace)
    finally:
        for n, v in zip(nodes, real):
            n["table_height_m"] = v


def _scenarios(trace):
    clear_state()
    # --- stage 1 motions
    ARM.place(HOME["joints_deg"])
    run(SEQ, ["detect"], trace)
    run(SEQ, ["home"], trace)
    run(SEQ, ["detect"], trace)
    run(SEQ, ["home"], trace)                                  # from food_detect_start
    run(SEQ, ["home"], trace)                                  # already there: settle
    ARM.place(pose=[0.1, 0.1, 0.3, 180, 0, 90])
    run(SEQ, ["home"], trace)                                  # unknown pose: refuses

    # --- acquisition, the stepped route (aim checks)
    ARM.place(DETECT["joints_deg"]); target()
    run(ACQ, ["over"], trace)
    under(0.090, 0.121); run(ACQ, ["descend", "--to-above", "0.030"], trace)
    under(0.031, 0.062); run(ACQ, ["descend", "--to-above", "0.010"], trace)
    under(0.011, 0.042); run(ACQ, ["punch"], trace)
    run(ACQ, ["scoop", "--validate-only"], trace)
    run(ACQ, ["scoop"], trace)
    # refusals
    ARM.place(HOME["joints_deg"]); run(ACQ, ["over"], trace)                  # not at food_detect_start
    ARM.place(DETECT["joints_deg"]); write(STATE / "target.json", dict(tool_goal=[0.5, 0.5, pose_of(DETECT)[2]]))
    run(ACQ, ["over"], trace)                                                  # too far
    under(0.005, 0.03); run(ACQ, ["descend", "--to-above", "0.010"], trace)   # already above
    under(0.25, 0.3); run(ACQ, ["descend", "--to-above", "0.010"], trace)     # not an approach
    write(STATE / "under.json", dict(top=0.3, gap_top=0.05)); run(ACQ, ["punch"], trace)   # no floor

    # --- acquisition, the plunge
    ARM.place(DETECT["joints_deg"]); target(); run(ACQ, ["over"], trace)
    under(0.092, 0.123, depth_source="sensor"); run(ACQ, ["plunge", "--validate-only"], trace)
    run(ACQ, ["plunge"], trace)
    run(ACQ, ["scoop"], trace)
    for bad in (dict(filled_floor=0.8), dict(depth_source="da3-raw", filled_floor=0.1)):
        ARM.place(DETECT["joints_deg"]); under(0.092, 0.123, **bad); run(ACQ, ["plunge"], trace)
    ARM.place(DETECT["joints_deg"]); under(0.02, 0.04); run(ACQ, ["plunge"], trace)       # too short
    ARM.place(DETECT["joints_deg"]); under(0.092, 0.095); run(ACQ, ["plunge"], trace)     # food too thin
    # the controller refusing the first speed, then a torque step mid-plunge
    ARM.place(DETECT["joints_deg"]); under(0.092, 0.123); ARM.refuse_above_dps = 3.0
    run(ACQ, ["plunge", "--then-scoop", "--validate-only"], trace); ARM.refuse_above_dps = None
    t_hit = fk.clock.time() + 1.2
    ARM.torque = lambda t: [0, 9.0 if t > t_hit else 0.0, 0, 0, 0, 0, 0]
    run(ACQ, ["plunge"], trace); ARM.torque = lambda t: [0.0] * 7

    # --- one full bite: pick, then delivery
    clear_state(); ARM.place(DETECT["joints_deg"]); target()
    run(ACQ, ["pick"], trace)
    ARM.place(SE.joints_deg)                                   # the scoop ends on the calibrated scoop end
    run(DEL, ["swing", "--validate-only"], trace)
    run(DEL, ["swing"], trace)
    run(DEL, ["approach"] + face(stale=True, name="stale"), trace)               # older than its capture
    run(DEL, ["approach"] + face(mouth=False, name="nomouth"), trace)            # exit 3: sidestep
    before = (list(ARM.q), list(ARM.pose))
    run(DEL, ["sidestep", "--step", "0.05"], trace)          # as deliver.sh calls it
    ARM.place(*before)                                       # the next steps from the same pose either way
    run(DEL, ["approach"] + face(nose_depth=0.905, mouth_depth=0.921, name="stale_depth"), trace)   # exit 5
    run(DEL, ["approach", "--validate-only"] + face(name="v"), trace)
    run(DEL, ["approach"] + face(filled=True, name="filled"), trace)
    t0 = fk.clock.time()
    ARM.force = lambda t: [0.0, 0.0, 2.0 if t0 + 1.5 < t < t0 + 1.9 else 0.0]
    run(DEL, ["bite"], trace)
    ARM.force = lambda t: [0.0, 0.0, 0.0]
    run(DEL, ["bite", "--timeout", "2"], trace)                                   # exit 4
    run(DEL, ["retract", "--validate-only"], trace)
    run(DEL, ["retract"], trace)
    # --- the next bite: swing and retract plans come from the caches
    ARM.place(SE.joints_deg)
    run(DEL, ["swing"], trace)
    run(DEL, ["approach", "--allow-nose"] + face(mouth=False, name="nose"), trace)
    run(DEL, ["retract"], trace)
    # --- refusals
    run(DEL, ["retract"], trace)                                                  # no swing recorded
    ARM.place(DETECT["joints_deg"]); run(DEL, ["swing"], trace)                  # not pointing at the user
    ARM.place(pose=SCOOP_END_POSE[:1] + [0.2] + SCOOP_END_POSE[2:]); run(DEL, ["swing"], trace)   # outside the region


def packed_scenarios(trace):
    """Packed food: no bowl floor visible beside the target -- plunge a fixed depth below its top."""
    ws = cfg["workspace"]
    t0 = ws["table_height_m"]
    try:
        ARM.place(DETECT["joints_deg"]); write(STATE / "under.json", dict(top=0.39, gap_top=0.125))   # no floor at all
        run(ACQ, ["plunge", "--then-scoop"], trace)
        ws["table_height_m"] = 0.392 - 0.125 - 0.010 - 0.015            # the table 15 mm below the food's top + margin
        ARM.place(DETECT["joints_deg"]); under(0.125, 0.128)
        run(ACQ, ["plunge", "--validate-only"], trace)                    # shortened to the table bound
        ws["table_height_m"] = 0.392 - 0.125 - 0.010 - 0.005
        ARM.place(DETECT["joints_deg"]); under(0.125, 0.128)
        run(ACQ, ["plunge"], trace)                                       # the table too close: refused
        ws["table_height_m"] = t0
        ARM.place(DETECT["joints_deg"]); under(0.119, 0.132)                 # 13 mm "tall", the floor 40 mm up
        run(ACQ, ["plunge", "--validate-only"], trace)
        ARM.place(DETECT["joints_deg"]); under(0.125, 0.128, filled_top=0.9)
        run(ACQ, ["plunge"], trace)                                       # the top is the DA3 fallback: refused
    finally:
        ws["table_height_m"] = t0


def moved_scenarios(trace):
    """The bowl moved and the table raised: every bowl-dependent pose follows (kinova/scene.py)."""
    ws = cfg["workspace"]
    orig = (list(ws["bowl_centre_m"]), ws["table_height_m"])
    try:
        clear_state()
        ws["bowl_centre_m"] = [orig[0][0] + 0.020, orig[0][1] - 0.010]
        ws["table_height_m"] = orig[1] + 0.010
        ARM.place(HOME["joints_deg"])
        run(SEQ, ["detect"], trace)                      # to the shifted food_detect_start (IK, seeded)
        p = ARM.pose
        write(STATE / "target.json", dict(tool_goal=[p[0] + 0.012, p[1] - 0.018, p[2]]))
        run(ACQ, ["pick"], trace)                        # the scoop ends at the shifted scoop end
        run(DEL, ["swing"], trace)
        run(DEL, ["approach"] + face(name="moved"), trace)
        run(DEL, ["retract"], trace)                     # home to the shifted food_detect_start
        run(SEQ, ["home"], trace)
        ws["table_height_m"] = orig[1] + 0.200           # the lift would no longer clear the table
        ARM.place(HOME["joints_deg"])
        run(SEQ, ["detect"], trace)
        ws["table_height_m"] = orig[1]
        ws["bowl_centre_m"] = [orig[0][0] + 0.60, orig[0][1]]   # far away: IK fails in the real arm; here the
        ARM.ik_unreachable = lambda pose: pose[0] > 0.7          # fake is told so
        run(SEQ, ["detect"], trace)
    finally:
        ARM.ik_unreachable = lambda pose: False
        ws["bowl_centre_m"], ws["table_height_m"] = orig
        from kinova import scene
        scene._RESOLVED.clear()


def compare(a, b):
    bad = 0
    if len(a) != len(b):
        print("step count differs: %d vs %d" % (len(a), len(b)))
    for i, (x, y) in enumerate(zip(a, b)):
        for k in x:
            if x[k] != y.get(k):
                bad += 1
                print("step %d (%s): %s differs" % (i, x["cmd"], k))
                if k in ("stdout", "stderr"):
                    for l1, l2 in zip(x[k] + [""] * 99, y.get(k, []) + [""] * 99):
                        if l1 != l2:
                            print("   - %s\n   + %s" % (l1, l2))
                            break
                elif k == "calls":
                    for j, (c1, c2) in enumerate(zip(x[k], y[k])):
                        if c1 != c2:
                            print("   call %d: %s vs %s" % (j, json.dumps(c1)[:300], json.dumps(c2)[:300]))
                            break
                    else:
                        print("   %d vs %d calls" % (len(x[k]), len(y[k])))
    print("%d steps compared: %s" % (min(len(a), len(b)), "IDENTICAL" if bad == 0 and len(a) == len(b) else "%d differences" % bad))
    return bad == 0 and len(a) == len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--compare")
    a = ap.parse_args()
    trace = []
    scenarios(trace)
    if "workspace" in cfg:                     # (the code before kinova/scene.py has no workspace)
        moved_scenarios(trace)
    if "no_floor" in cfg.acquisition.plunge:
        packed_scenarios(trace)
    ARM.place(HOME["joints_deg"]); ARM.joint_limit_dps = 45.0       # the arm's joint limits slower than asked:
    run(SEQ, ["detect"], trace)                                      # each swing half re-sent at its minimum
    run(SEQ, ["home"], trace)
    ARM.joint_limit_dps = None
    clear_state()
    Path(a.out).write_text(json.dumps(trace, indent=0))
    print("%d steps -> %s" % (len(trace), a.out))
    if a.compare:
        import gzip
        raw = gzip.open(a.compare, "rt").read() if a.compare.endswith(".gz") else Path(a.compare).read_text()
        return 0 if compare(json.loads(raw), trace) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
