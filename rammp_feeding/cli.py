"""rammp -- the feeding module's one command (ros2 run rammp_feeding rammp <command>; under sheppy: rammp feed).

    rammp check                    the arm, the planner, SAM3 and the cameras answer; where the arm is. Never moves.
    rammp state                    where the arm is: joints, tool, utensil tip and tilt
    rammp home [--direct [--dps D]]   to feeding_start (a cuRobo path, from anywhere; --direct: one joint move, no
                                   planner -- only for a workspace someone has checked is clear)
    rammp detect [--plan]          to food_detect_start, over the bowl (a cuRobo path; --plan: plan only)
    rammp perceive                 at food_detect_start: the target piece (a new bite)
    rammp acquire [--stop-after over|hover30|hover10|plunge|punch] [--start-at over|punch|scoop]
    rammp rise                     after a hover: straight back up to the overlook height
    rammp deliver [--stop-after swing|present|bite] [--start-at swing|present|bite|retract] [--no-wait-open]
    rammp retract                  from the user back to food_detect_start
    rammp feed [--bites N]         the loop: bite after bite until the bowl is empty or the user declines
    rammp locate-scene             the bowl and the table from the scene camera (state_dir/scene_state.json)
    rammp calibrate-scene [--reset]   the scene camera's pose from the arm (one capture here; nothing moves)
    rammp capture home|overlook|deliver   the arm's joints now as feeding_start / food_detect_start's arm shape / the
                                   delivery look (state_dir)

Exit codes: 0 done; 1 stopped or refused (the arm stationary where it stopped); 3 the bowl is empty; 5 food left but
none pickable; 7 the user declined (retracted); 8 the presentation refused (retracted).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np

from rammp_feeding import record, scene
from rammp_feeding.config import cfg, state_dir

log = logging.getLogger("rammp")


def _robot():
    from rammp_feeding.ros import Robot
    return Robot()


def report(arm, what: str) -> None:
    from rammp_feeding.geometry import tilt_deg
    q = arm.q()
    T, R = arm.tool(q)
    log.info("  %-18s joints %s", what, " ".join("%7.2f" % v for v in q))
    log.info("  %-18s tool (%.4f %.4f %.4f), tip (%.4f %.4f %.4f), utensil %.1f deg off vertical", "", *T, *arm.tip(q),
             tilt_deg(R))
    record.event("at", name=what, joints=q, tool=T)


def detect(robot, arm, plan: bool = False) -> None:
    ws = scene.workspace()
    goal = scene.overlook_joints(arm)
    log.info("  the bowl at (%.3f, %.3f), table z %.3f (%s)", *ws["bowl_xy"], ws["table_z"], ws["source"])
    arm.travel(goal, "to food_detect_start", go=not plan)


def home(robot, arm, direct: bool = False, dps: float | None = None) -> None:
    """To feeding_start: a planned path; --direct: one joint move, no planner -- only for a workspace someone has
    checked is clear (the line can pass close to the table). The utensil's tilt only falls along it."""
    if direct:
        from rammp_feeding.geometry import tilt_deg
        arm.goto(scene.captured("home"), "direct to feeding_start", dps=dps or float(cfg.motion.direct_dps),
                 max_tilt=max(float(cfg.motion.max_tilt_deg), tilt_deg(arm.tool()[1]) + 3.0))
        return
    arm.travel(scene.captured("home"), "to feeding_start")


def perceive(robot, arm) -> int:
    from rammp_feeding.perception import NoTarget, perceive as run
    record.bite()
    try:
        run(robot, arm, record.stage("perceive"))
    except NoTarget as e:
        log.info("  %s", e)
        return e.rc
    return 0


def _target() -> dict:
    p = record.latest_bite() / "perceive" / "target.json"
    if not p.exists():
        raise FileNotFoundError("the latest bite (%s) has no target -- rammp perceive" % p.parent.parent)
    return json.loads(p.read_text())


def feed(robot, arm, face, bites: int) -> int:
    from rammp_feeding.acquire import acquire
    from rammp_feeding.deliver import deliver
    if cfg.profiles.scene_camera:
        from rammp_feeding.scene_camera import locate
        locate(robot, record.session("feed"))
    detect(robot, arm)
    n = 0
    while n < bites:
        t0 = time.time()
        log.info("\n== bite %d  %s", n + 1, time.strftime("%H:%M:%S"))
        rc = perceive(robot, arm)
        if rc == 3:
            log.info("the bowl is empty")
            break
        if rc != 0:
            log.info("food left but none pickable -- stopping")
            home(robot, arm)
            return rc
        acquire(robot, arm, _target(), record.stage("acquire"))
        rc = deliver(robot, arm, face, record.stage("deliver"))
        if rc in (7, 8):
            log.info("the user declined" if rc == 7 else "the presentation was refused -- stopping")
            home(robot, arm)
            return 0 if rc == 7 else rc
        n += 1
        log.info("== bite %d done in %.0f s", n, time.time() - t0)
        record.event("bite_done", n=n, seconds=round(time.time() - t0, 1))
    home(robot, arm)
    return 0


def check(robot, arm) -> int:
    """What the module needs from the stack, each as it is served. Never moves."""
    bad = 0

    def row(ok, name, text, warn=False):
        nonlocal bad
        bad += not ok and not warn
        log.info("  %-4s %-14s %s", "OK" if ok else "WARN" if warn else "FAIL", name, text)
    try:
        report(arm, "the arm")
        row(not robot.estopped, "arm", "e-stop ENGAGED" if robot.estopped else "joint states streaming, e-stop clear")
        row(not robot.arbitration or any(robot.token), "arbitration", "enforced, no token (RAMMP_ARM_TOKEN)"
            if robot.arbitration and not any(robot.token) else "enforced, token held" if robot.arbitration else "disabled")
    except Exception as e:  # noqa: BLE001
        row(False, "arm", str(e))
    row(robot.description is not None, "robot model", "/robot_description" if robot.description else
        "no /robot_description (arm_state_publisher): the bundled copy of the driver's", warn=True)
    for name, client in (("trajectories", robot._exec), ("planner", robot._plan)):
        row(client.wait_for_server(timeout_sec=2.0), name, "action server up")
    for name, client in (("planner world", robot._world), ("sam3", robot._detect)):
        row(client.wait_for_service(timeout_sec=2.0), name, "service up")
    try:
        row(True, "planner scene", "%s (ours only while we plan, then this again)"
            % (robot.planner_parameter("world") or "%s (its config's)" % cfg.planner.restore_world))
    except Exception as e:  # noqa: BLE001
        row(False, "planner scene", str(e))
    for role, cam in robot.cameras.items():
        try:
            cam.frame(wait_s=5.0)                  # (the first frames after discovery can take a few seconds)
            cam.depth_median(3, time.time() - 1.0, wait_s=5.0, min_frames=1)   # (streaming at all)
            m = robot.camera_model(role)
            row(True, role, "colour + depth streaming; camera_info %dx%d, depth -> colour from /tf_static"
                % (m.colour.width, m.colour.height))
        except Exception as e:  # noqa: BLE001
            row(False, role, str(e))
    if cfg.profiles.scene_camera:
        seg = robot.segmented_objects(5.0)
        row(seg is not None, "scene segmenter", "publishing (%d objects)" % len(seg) if seg is not None else
            "%s silent: the bowl falls back to SAM3" % cfg.ros.segmenter_topic, warn=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rammp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "state", "home", "detect", "perceive", "acquire", "rise", "deliver",
                                        "retract", "feed", "locate-scene", "calibrate-scene", "capture"])
    ap.add_argument("name", nargs="?", help="capture: home | overlook | deliver")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--stop-after", default="")
    ap.add_argument("--start-at", default="")
    ap.add_argument("--no-wait-open", action="store_true")
    ap.add_argument("--bites", type=int, default=99)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--dps", type=float, default=None)
    a, _ = ap.parse_known_args(argv)               # (ros2 run / launch append their own --ros-args)
    import signal

    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)           # docker stop / sheppy: as Ctrl-C, the move in flight cancelled
    record.setup_logging()
    record.session(a.command)
    from rammp_feeding.arm import Arm, MoveError
    robot = _robot()
    rc = 0
    try:
        arm = Arm(robot)                           # (the arm's model, the wrist camera's camera_info and tf)
        if cfg.profiles.scene_camera:
            from rammp_feeding.scene_camera import publish
            publish(robot)                         # our measured scene camera pose, on /tf_static
        c = a.command
        if c == "check":
            rc = check(robot, arm)
        elif c == "state":
            report(arm, "now")
        elif c == "home":
            home(robot, arm, a.direct, a.dps)
        elif c == "detect":
            detect(robot, arm, a.plan)
        elif c == "perceive":
            rc = perceive(robot, arm)
        elif c == "acquire":
            from rammp_feeding.acquire import acquire
            rc = acquire(robot, arm, _target(), record.stage("acquire"), a.stop_after, a.start_at or "over")
        elif c == "rise":
            from rammp_feeding.acquire import rise
            rise(arm)
        elif c in ("deliver", "retract", "feed"):
            from rammp_feeding.deliver import deliver
            from rammp_feeding.face import Face
            face = Face() if c != "retract" else None
            if c == "feed":
                record.session("feed")
                rc = feed(robot, arm, face, a.bites)
            else:
                rc = deliver(robot, arm, face, record.stage("deliver"), a.stop_after,
                             "retract" if c == "retract" else (a.start_at or "swing"), not a.no_wait_open)
        elif c == "locate-scene":
            from rammp_feeding.scene_camera import locate
            locate(robot, record.session("scene"))
        elif c == "calibrate-scene":
            from rammp_feeding.scene_camera import calibrate
            rc = 0 if calibrate(robot, arm, a.reset) else 1
        elif c == "capture":
            if a.name not in ("home", "overlook", "deliver"):
                raise SystemExit("rammp capture home|overlook|deliver")
            qs = []
            for _ in range(30):
                qs.append(arm.q())
                time.sleep(0.02)
            q = [round(float(v), 3) for v in np.median(qs, axis=0)]
            (state_dir() / (a.name + ".json")).write_text(json.dumps(dict(joints_deg=q, captured=time.strftime("%F %T")), indent=1))
            log.info("  %s <- %s", state_dir() / (a.name + ".json"), q)
        if c in ("home", "detect", "rise", "retract", "acquire", "deliver"):
            report(arm, "now")
    except (MoveError, RuntimeError, FileNotFoundError, ValueError) as e:
        log.error("\n  STOPPED: %s", e)
        record.event("stopped", command=a.command, reason=str(e))
        rc = 1
    except KeyboardInterrupt:
        log.error("\n  interrupted -- the move in flight was cancelled")
        rc = 1
    finally:
        if getattr(robot, "world_pushed", False):  # the shared planner's own world back, for its other clients
            try:
                scene.restore_world(robot)
            except Exception as e:  # noqa: BLE001
                log.error("  the planner's own world was not restored (%s): it still holds ours", e)
        robot.close()
    record.event("done", command=a.command, rc=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
