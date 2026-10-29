"""The wrist camera's mount -- its colour optical frame in the tool frame -- measured from a printed ChArUco board.

board      the sheet to print (rammp calibrate-wrist --board -> runs_dir/charuco_board.pdf, US Letter, at 100 %); its
           square side as printed is perception.wrist_calibration.square_m (measure it: printers scale)
calibrate  from feeding_start the arm rises and moves out from its base, then holds the board in view from a few poses
           pivoting on one point under the tool; at each, OpenCV finds the board in the colour image (its pose from the
           camera's camera_info), and the hand-eye solve (cv2.calibrateHandEye) gives the camera on the tool. At the
           first pose the board's distance by the camera's own depth must match its distance by the image (a print
           scaled from square_m puts the camera that much further or nearer). Accepted only when the board's place
           agrees across the poses: state_dir/wrist_mount.json, which the camera profile's mount takes (config.load);
           then back to feeding_start
views      the same route, shared (board_views): rammp calibrate-scene places the board in the base frame from it,
           through the mount measured here (scene_camera.calibrate)
"""

from __future__ import annotations

import json
import logging
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from rammp_feeding import record
from rammp_feeding.config import cfg, state_dir
from rammp_feeding.geometry import matrix_quat, max_step, rodrigues, rot_log

log = logging.getLogger("rammp")
WC = cfg.perception.wrist_calibration


def board():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, str(WC.dictionary)))
    sq = float(WC.square_m)
    b = cv2.aruco.CharucoBoard(tuple(int(n) for n in WC.squares), sq, sq * float(WC.marker_ratio), d)
    b.setLegacyPattern(bool(WC.legacy_pattern))
    return b


def board_pdf(path) -> None:
    """The board at its size, centred on a US Letter page at 300 dpi, with what it is printed under it."""
    dpi = 300
    nx, ny = (int(n) for n in WC.squares)
    w, h = (int(round(n * float(WC.square_m) / 0.0254 * dpi)) for n in (nx, ny))
    page = Image.new("L", (int(8.5 * dpi), int(11 * dpi)), 255)
    x0, y0 = (page.width - w) // 2, (page.height - h) // 2
    page.paste(Image.fromarray(board().generateImage((w, h), marginSize=0, borderBits=1)), (x0, y0))
    text = "ChArUco %dx%d %s, %.1f mm squares: print at 100%% (actual size), then measure across all %d squares" % (
        nx, ny, WC.dictionary, 1000 * float(WC.square_m), nx)
    note = Image.new("L", (int(ImageDraw.Draw(page).textlength(text)) + 4, 14), 255)
    ImageDraw.Draw(note).text((0, 0), text, fill=0)
    note = note.resize((note.width * 3, note.height * 3))
    page.paste(note, ((page.width - note.width) // 2, y0 + h + 40))
    page.save(path, "PDF", resolution=float(dpi))


def corners(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The board's corners found in one colour image: their pixels (N x 2) and their points on the board (N x 3, m)."""
    import cv2
    b = board()
    uv, ids, _, _ = cv2.aruco.CharucoDetector(b).detectBoard(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return np.zeros((0, 2)), np.zeros((0, 3))
    return uv.reshape(-1, 2).astype(float), np.asarray(b.getChessboardCorners(), float)[ids.ravel()]


def board_pose(cam, rgb: np.ndarray) -> tuple[np.ndarray | None, int]:
    """camera <- board (4 x 4) from one colour image (cam: the camera's model), and how many corners were found."""
    import cv2
    img, obj = corners(rgb)
    n = len(img)
    if n < int(WC.min_corners):
        return None, n
    c = cam.colour
    K = np.array([[c.fx, 0.0, c.cx], [0.0, c.fy, c.cy], [0.0, 0.0, 1.0]])
    ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if ok:
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, n
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    return T, n


def depth_scale(cam, depth_mm: np.ndarray, rgb: np.ndarray, C: np.ndarray) -> float | None:
    """The board's distance by the camera's depth over its distance by its image (square_m), median over its corners
    (each corner's 5 x 5 depth window): 1.0 when square_m is the printed size; None with too few corners in depth."""
    uv, obj = corners(rgb)
    z = (obj @ C[:3, :3].T + C[:3, 3])[:, 2]
    r = []
    for (u, v), zi in zip(uv, z):
        ud, vd = (int(round(x)) for x in cam.colour_to_depth(u, v, zi))
        w = depth_mm[max(0, vd - 2):vd + 3, max(0, ud - 2):ud + 3]
        w = w[w > 0]
        if w.size >= 5:
            r.append(cam.depth_to_colour(ud, vd, float(np.median(w)) / 1000.0)[2] / zi)
    return float(np.median(r)) if len(r) >= int(WC.min_corners) else None


def poses(T0, R0) -> list[tuple[np.ndarray, np.ndarray]]:
    """The tool poses: (T0, R0), then turned +-tilt_deg about the tool's x and y and +-turn_deg about its axis, each
    pivoting on the point pivot_m along the tool axis (on the board)."""
    p = np.asarray(T0) + np.asarray(R0)[:, 2] * float(WC.pivot_m)
    out = [(np.asarray(T0), np.asarray(R0))]
    for i, deg in ((0, WC.tilt_deg), (0, -WC.tilt_deg), (1, WC.tilt_deg), (1, -WC.tilt_deg), (2, WC.turn_deg),
                   (2, -WC.turn_deg)):
        Rw = rodrigues(np.asarray(R0)[:, i] * math.radians(float(deg)))
        out.append((p + Rw @ (np.asarray(T0) - p), Rw @ R0))
    return out


def board_in_base(caps, X) -> tuple[np.ndarray, float, float]:
    """base <- board (4 x 4) from the wrist camera's views (R tool, T tool, camera <- board) through its mount X (tool
    <- camera): their mean, and how far apart they come out (m, deg)."""
    B = []
    for Rt, Tt, C, *_ in caps:                          # (the route's captures carry their frame too)
        G = np.eye(4)
        G[:3, :3], G[:3, 3] = Rt, Tt
        B.append(G @ X @ C)
    U, _, Vt = np.linalg.svd(sum(b[:3, :3] for b in B))
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt, np.mean([b[:3, 3] for b in B], axis=0)
    spread_m = max(float(np.linalg.norm(b[:3, 3] - M[:3, 3])) for b in B)
    spread_deg = max(math.degrees(float(np.linalg.norm(rot_log(b[:3, :3] @ M[:3, :3].T)))) for b in B)
    return M, spread_m, spread_deg


def solve(caps) -> tuple[np.ndarray, float, float]:
    """The camera in the tool frame (4 x 4) from the wrist camera's views of the board, and how far apart the board's
    place in the base frame comes out across them (m, deg)."""
    import cv2
    R, t = cv2.calibrateHandEye([c[0] for c in caps], [c[1] for c in caps], [c[2][:3, :3] for c in caps],
                                [c[2][:3, 3] for c in caps], method=cv2.CALIB_HAND_EYE_PARK)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.ravel()
    return (X, *board_in_base(caps, X)[1:])


def fork_in_image(frames, X: np.ndarray, cam) -> tuple[list[float], list[int]] | None:
    """The utensil in the wrist image, from the route's frames (X: tool <- camera, cam: the camera's model): it stays put
    while the board behind it moves. The still pixels near the tool's axis as the camera sees it (to 0.25 m out of the
    tool point), the region of them from the image's edge; the middle of its far end along its own direction (the tines'
    tips: tip_px) and its box (image_region_px). None when that region runs to the end: no board behind the tips."""
    from scipy import ndimage
    g = np.stack([f.mean(2) for f in frames])
    Xi = np.linalg.inv(X)
    a, b = (np.array(cam.pixel(Xi[:3, :3] @ np.array([0.0, 0.0, z]) + Xi[:3, 3])) for z in (0.0, 0.25))
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L                                     # the tool's axis in the image, from the tool point outward
    vv, uu = np.mgrid[0:g.shape[1], 0:g.shape[2]]
    along = (uu - a[0]) * d[0] + (vv - a[1]) * d[1]
    near = (np.abs((uu - a[0]) * d[1] - (vv - a[1]) * d[0]) < float(WC.fork_corridor_px)) & (along <= L)
    lab, _ = ndimage.label((g.std(0) < float(WC.fork_still_std)) & near)
    edge = set(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
    if not edge:
        return None
    k = max(edge, key=lambda k: int((lab == k).sum()))
    v, u = np.nonzero(lab == k)
    if ((u - a[0]) * d[0] + (v - a[1]) * d[1]).max() > L - 2 * float(WC.tip_band_px):
        return None                                     # still all the way out: no board behind the tips to tell them by
    pts = np.c_[u, v].astype(float)
    e = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, -1]      # the utensil's own direction in the image ...
    s = (pts - pts.mean(0)) @ (e if e @ d > 0 else -e)                # ... outward: its tines' tips across its far end
    tips = s >= s.max() - float(WC.tip_band_px)
    m, H, W = int(WC.fork_margin_px), g.shape[1], g.shape[2]
    box = [max(0, int(u.min()) - m), max(0, int(v.min()) - m), min(W, int(u.max()) + m), min(H, int(v.max()) + m)]
    return [round(float(u[tips].mean()), 1), round(float(v[tips].mean()), 1)], box


def _path(arm, q0, T, R, label: str):
    from rammp_feeding.arm import MoveError
    T0, R0 = arm.tool(q0)
    w = rot_log(np.asarray(R) @ R0.T)
    n = max(4, int(math.ceil(max(np.linalg.norm(np.asarray(T) - T0) / 0.005, math.degrees(np.linalg.norm(w))))))
    path, why = arm.homotopy(q0, lambda s: (T0 + (np.asarray(T) - T0) * s, rodrigues(w * s) @ R0), n)
    if path is None:
        raise MoveError("%s: %s -- not moving" % (label, why))
    return path


def _route(arm) -> tuple[list, list]:
    """From feeding_start: the legs out to the poses over the board, and the way back (to feeding_start's joints)."""
    from rammp_feeding.arm import MoveError
    from rammp_feeding.scene import captured
    q = arm.q()
    if max_step(q, captured("home")) > 1.0:
        raise MoveError("the board's route starts at feeding_start (the arm is %.1f deg from it): rammp home --direct"
                        % max_step(q, captured("home")))
    T, R = arm.tool(q)
    up = T + np.array([0.0, 0.0, float(WC.rise_m)])
    start = up + np.array([T[0], T[1], 0.0]) / math.hypot(T[0], T[1]) * float(WC.out_m)   # away from the base's axis
    out = [(up, R, "up"), (start, R, "out from the base")]
    out += [(Tk, Rk, "pose %d" % k) for k, (Tk, Rk) in enumerate(poses(start, R))]
    return out, [(start, R, "back over the board"), (up, R, "in"), (None, None, "down to feeding_start")]


def plan_route(arm) -> None:
    """The route's tool poses, printed; nothing moves."""
    from rammp_feeding.geometry import tilt_deg
    from rammp_feeding.scene import captured
    legs, back = _route(arm)
    qk = arm.q()
    for Tg, Rg, label in legs + back:
        qk = captured("home") if Tg is None else _path(arm, qk, Tg, Rg, label)[-1]
        Tk, Rk = arm.tool(qk)
        log.info("    %-22s tool (%.3f, %.3f, %.3f), %.1f deg off vertical", label, *Tk, tilt_deg(Rk))


def board_views(robot, arm) -> list | None:
    """The route round the board and back to feeding_start: the wrist camera's view of it at each pose that saw it
    (R tool, T tool, camera <- board), or None (why logged; the arm back at feeding_start). At the first pose it waits
    for the board to be laid under the camera, and checks the print's scale by the camera's depth."""
    from rammp_feeding.scene import captured
    legs, back = _route(arm)
    cam, out = robot.camera_model("camera"), record.here()
    guard = float(cfg.guards.torque_step_nm.free_space)   # open air: the default (contact) limit trips on its acceleration

    def move(Tg, Rg, label):
        if Tg is None:
            arm.goto(captured("home"), label, dps=float(WC.speed_dps), torque_limit=guard)
        else:
            arm.run(_path(arm, arm.q(), Tg, Rg, label), label, dps=float(WC.speed_dps), torque_limit=guard)

    caps = []
    for Tg, Rg, label in legs:
        move(Tg, Rg, label)
        if not label.startswith("pose"):
            continue
        t_end = time.time() + (float(WC.wait_board_s) if label == "pose 0" else float(WC.settle_s))
        while True:
            time.sleep(float(WC.settle_s))
            _, rgb = robot.cameras["camera"].frame(after=time.time())
            C, n = board_pose(cam, rgb)
            if C is not None or time.time() > t_end:
                break
            if label == "pose 0" and t_end - time.time() > float(WC.wait_board_s) - 2 * float(WC.settle_s):
                log.info("    no board under the camera (%d corners): lay it under the utensil (waiting %.0f s)", n,
                         float(WC.wait_board_s))
        Image.fromarray(rgb).save(out / ("wrist_%s.jpg" % label.replace(" ", "")), quality=90)
        log.info("    %-22s %s", label, "the board, %d corners" % n if C is not None else "no board (%d corners)" % n)
        why = None
        if C is None and label == "pose 0":
            why = "no board under the camera in %.0f s" % float(WC.wait_board_s)
        elif label == "pose 0":
            D = robot.cameras["camera"].depth_median(10, time.time(), wait_s=5.0)
            k = depth_scale(cam, D, rgb, C)
            sq = 1000 * float(WC.square_m)
            if k is None:
                why = "no depth on the board's corners"
            elif abs(k - 1.0) > float(WC.max_scale_error):
                why = ("the board's squares read %.1f mm by the camera's depth, %.1f mm configured: measure the print "
                       "(perception.wrist_calibration.square_m)" % (sq * k, sq))
            else:
                log.info("    the board's squares read %.1f mm by the camera's depth (%.1f mm configured)", sq * k, sq)
        if why:
            for leg in back[1:]:
                move(*leg)
            log.info("  NOT measured: %s", why)
            return None
        if C is not None:
            Tt, Rt = arm.tool(arm.q())
            caps.append((Rt, Tt, C, rgb))
    for leg in back:
        move(*leg)
    if len(caps) < int(WC.min_poses):
        log.info("  NOT measured: the board in %d pose(s), %d needed", len(caps), int(WC.min_poses))
        return None
    return caps


def calibrate(robot, arm, plan: bool = False) -> dict | None:
    """The wrist camera's mount and the utensil's pixels in its image: the route round the board (laid under the
    utensil), the hand-eye solve, the still part of the frames; plan=True: the route, nothing moves."""
    if plan:
        plan_route(arm)
        return None
    caps = board_views(robot, arm)
    if caps is None:
        return None
    X, spread_m, spread_deg = solve(caps)
    log.info("  the camera in the tool frame: %s m; the board's place agrees across %d poses to %.1f mm, %.2f deg",
             np.round(X[:3, 3], 4).tolist(), len(caps), 1000 * spread_m, spread_deg)
    if spread_m > float(WC.max_spread_m) or spread_deg > float(WC.max_spread_deg):
        log.info("  NOT accepted (<= %.1f mm, <= %.2f deg)", 1000 * float(WC.max_spread_m), float(WC.max_spread_deg))
        return None
    fork = fork_in_image([c[3] for c in caps], X, robot.camera_model("camera"))
    if fork is None:
        log.info("  NOT accepted: the utensil not told apart in the frames -- lay the board under it, behind its tips")
        return None
    log.info("  the utensil in the image: its tips at (%.1f, %.1f) px, its region %s", *fork[0], fork[1])
    res = dict(camera=str(cfg.profiles.camera), utensil=str(cfg.profiles.utensil), tip_px=fork[0], image_region_px=fork[1],
               camera_in_tool_m=[round(float(v), 5) for v in X[:3, 3]],
               rotation_quat_xyzw=[round(v, 6) for v in matrix_quat(X[:3, :3])], spread_m=round(spread_m, 4),
               spread_deg=round(spread_deg, 3), poses=len(caps), square_m=float(WC.square_m), t=time.time())
    (state_dir() / "wrist_mount.json").write_text(json.dumps(res, indent=1))
    record.event("wrist_mount", **res)
    log.info("  accepted -> %s", state_dir() / "wrist_mount.json")
    return res
