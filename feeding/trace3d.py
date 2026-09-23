#!/usr/bin/env python3
"""Render the demo as a 3D trace, reconstructed from the logged JOINT angles.

Not from the logged tool pose -- from forward kinematics. That is the stronger
statement: it shows our kinematic model and the real arm agree, and it gives the
whole arm (elbow, wrist) rather than just the tool, which is what a whole-arm
clearance check needs.

Validated first: FK from the driver's joint angles reproduces its reported
/ee_state to 0.0 mm at `pinch_site`, and the Kortex tool frame lands 120.5 mm
beyond that -- matching the 120 mm tool transform the arm reports. So the
degrees->radians->model chain is correct end to end.

Drawn with PIL. matplotlib is not installed and is not worth adding for this.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import mujoco
from PIL import Image, ImageDraw

PHASE_COLOUR = {
    "DESCEND": (30, 90, 220), "SCOOP": (220, 40, 40), "ALIGN": (150, 150, 150),
    "TRANSPORT->PERSON": (230, 140, 0), "PRESENT": (140, 40, 190),
    "RETURN->BOWL": (0, 150, 120), "LIFT": (0, 170, 220),
    "HOLD@BOWL": (110, 110, 110), "HOLD": (110, 110, 110), "ADJUST": (190, 190, 190),
    "TRAVERSE": (190, 190, 190),
}


def load_joints(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    t = np.array([r["t"] for r in rows]); t -= t[0]
    Q = np.radians(np.array([[j["pos"] for j in r["joints"]] for r in rows]))
    Q = (Q + np.pi) % (2 * np.pi) - np.pi          # Kortex 0..360 -> (-pi, pi]
    return t, Q


def fk_chain(model_path, Q, stride=1):
    """Returns (tool_xyz, link_xyz) sampled every `stride` frames."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bracelet_link")
    links = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
             ("base_link", "half_arm_1_link", "forearm_link",
              "spherical_wrist_1_link", "bracelet_link")]
    tool, chain = [], []
    for i in range(0, len(Q), stride):
        d.qpos[:7] = Q[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R = d.xmat[bid].reshape(3, 3)
        # the arm's own tool frame: 120 mm beyond pinch_site along local z
        tool.append(d.site_xpos[pid] + R @ np.array([0, 0, -0.120]))
        chain.append(np.array([d.xpos[b] for b in links]))
    return np.array(tool), np.array(chain)


def project(P, azim, elev, w, h, pad=60):
    a, e = np.radians(azim), np.radians(elev)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]]) @ R
    V = P @ R.T
    u, v = V[:, 0], V[:, 2]
    def sc(x, lo, hi, a_, b_):
        return a_ + (b_ - a_) * (x - lo) / max(hi - lo, 1e-9)
    ulo, uhi, vlo, vhi = u.min(), u.max(), v.min(), v.max()
    sp = max(uhi - ulo, vhi - vlo)
    cu, cv = (ulo + uhi) / 2, (vlo + vhi) / 2
    X = sc(u, cu - sp / 2, cu + sp / 2, pad, w - pad)
    Y = sc(v, cv - sp / 2, cv + sp / 2, h - pad, pad)
    return np.column_stack([X, Y])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--model", default="sim/assets/kinova_gen3/gen3_fork.xml")
    ap.add_argument("--out", default="/tmp/trace3d.png")
    ap.add_argument("--from-t", type=float, default=0.0)
    ap.add_argument("--to-t", type=float, default=1e9)
    ap.add_argument("--stride", type=int, default=4)
    a = ap.parse_args()

    t, Q = load_joints(a.state)
    keep = (t >= a.from_t) & (t <= a.to_t)
    t, Q = t[keep], Q[keep]
    tool, chain = fk_chain(a.model, Q, a.stride)
    ts = t[::a.stride][:len(tool)]
    print("FK over %d samples (stride %d) -> %d points" % (len(Q), a.stride, len(tool)))
    print("tool path: x %.0f..%.0f  y %.0f..%.0f  z %.0f..%.0f mm"
          % (tool[:, 0].min()*1000, tool[:, 0].max()*1000,
             tool[:, 1].min()*1000, tool[:, 1].max()*1000,
             tool[:, 2].min()*1000, tool[:, 2].max()*1000))

    views = [("top  (x-y)  looking down", 0, 90),
             ("front (x-z)  looking along +y", 0, 0),
             ("iso", 35, 25)]
    W, H = 760, 660
    sheet = Image.new("RGB", (W * len(views), H + 40), (255, 255, 255))
    sd = ImageDraw.Draw(sheet)
    # Colour by PHASE, using the same segmentation as feeding.segment_demo, so
    # the picture and the numbers describe the same thing. Speed alone washes
    # out: most of this demo runs at 60-90 mm/s and everything reads "fast".
    from feeding.segment_demo import load as seg_load, segments as seg_segments, label as seg_label, BOWL_Y
    st, sP, sA, sF, sT = seg_load(a.state)
    segs, shz = seg_segments(st, sP)
    over = sP[:, 1] > BOWL_Y
    zf = np.full(len(st), np.nan)
    k = 0
    while k < len(st):
        if not over[k]:
            k += 1; continue
        mm = k
        while mm < len(st) and over[mm]:
            mm += 1
        zf[k:mm] = sP[k:mm, 2].min(); k = mm
    phase_of = np.array(["-"] * len(st), dtype=object)
    for i0, j0, mv in segs:
        phase_of[i0:j0] = seg_label(sP, sA, i0, j0, mv,
                                    zf[i0] if not np.isnan(zf[i0]) else None)
    # map the trace samples (already time-filtered and strided) onto phases
    phases = []
    for tv in ts:
        # `ts` already carries absolute times (the filter keeps t as-is), so
        # adding from_t here double-counted the offset and shifted every phase.
        idx = int(np.argmin(np.abs(st - tv)))
        phases.append(phase_of[idx])
    used = []
    for ph in phases:
        if ph not in used:
            used.append(ph)
    for k, (name, az, el) in enumerate(views):
        img = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        pts = project(tool, az, el, W, H)
        for i in range(1, len(pts)):
            c = PHASE_COLOUR.get(phases[i], (200, 200, 200))
            wdt = 4 if phases[i] in ("SCOOP", "DESCEND") else 2
            dr.line([tuple(pts[i-1]), tuple(pts[i])], fill=c, width=wdt)
        # draw the arm at the deepest point of the first descent
        low = int(np.argmin(tool[:, 2]))
        cp = project(chain[low], az, el, W, H)
        for i in range(1, len(cp)):
            dr.line([tuple(cp[i-1]), tuple(cp[i])], fill=(0, 0, 0), width=3)
        for p in cp:
            dr.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4], fill=(0, 0, 0))
        dr.ellipse([pts[low][0]-5, pts[low][1]-5, pts[low][0]+5, pts[low][1]+5],
                   fill=(255, 200, 0), outline=(0, 0, 0))
        dr.text((12, 12), name, fill=(0, 0, 0))
        # scale bar: 100 mm in this projection
        o = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        allp = np.vstack([tool, o])
        pp = project(allp, az, el, W, H)
        px = abs(pp[-1][0] - pp[-2][0]) or abs(pp[-1][1] - pp[-2][1])
        if px > 4:
            dr.line([(30, H - 26), (30 + px, H - 26)], fill=(0, 0, 0), width=3)
            dr.text((30, H - 44), "100 mm", fill=(0, 0, 0))
        # annotate the key events
        for lab, tgt in (("deepest", low),):
            dr.text((pts[tgt][0] + 9, pts[tgt][1] - 6), lab, fill=(150, 90, 0))
        sheet.paste(img, (k * W, 0))
    x = 10
    for ph in used:
        c = PHASE_COLOUR.get(ph, (200, 200, 200))
        sd.rectangle([x, H + 10, x + 16, H + 22], fill=c)
        sd.text((x + 20, H + 11), ph, fill=(0, 0, 0))
        x += 26 + 7 * len(ph)
    sd.text((10, H + 26), "black = arm linkage at the deepest tool point (FK from joints); "
                          "yellow = that point", fill=(0, 0, 0))
    sheet.save(a.out)
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
