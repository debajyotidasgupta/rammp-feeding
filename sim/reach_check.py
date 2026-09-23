#!/usr/bin/env python3
"""Can the fork tip reach the plate and the mouth from this mount? Damped
least-squares IK on the 7 arm joints, reporting convergence and clearance.

This validates the SCENE LAYOUT (mount position, table height, chair geometry)
before any skill code is written against it.
"""
from __future__ import annotations
import mujoco, numpy as np

RETRACT = np.array([0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633])


def ik(m, d, target, q0, iters=400, tol=1e-3):
    tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "arm_tcp")
    q = q0.copy()
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[:7] = q
        mujoco.mj_forward(m, d)
        err = target - d.site_xpos[tcp]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(m, d, jacp, None, tcp)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        q = np.clip(q + dq * 0.6, -2 * np.pi, 2 * np.pi)
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    return q, float(np.linalg.norm(target - d.site_xpos[tcp]))


def main() -> int:
    m = mujoco.MjModel.from_xml_path("sim/feeding_scene.xml")
    d = mujoco.MjData(m)
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    d.qpos[:7] = RETRACT; mujoco.mj_forward(m, d)
    mouth = d.site_xpos[sid("mouth")].copy()

    targets = {
        "above plate (+8cm)": np.array([0.55, -0.10, 0.86]),
        "food cube 0":        np.array([0.52, -0.13, 0.79]),
        "food cube 3":        np.array([0.59, -0.12, 0.79]),
        "mouth standoff":     mouth + np.array([0.06, 0.0, 0.0]),   # 6 cm in front
    }
    print("  target                 residual   reached (x,y,z)          verdict")
    for name, t in targets.items():
        q, res = ik(m, d, t, RETRACT)
        got = d.site_xpos[sid("arm_tcp")]
        ok = "OK" if res < 0.005 else ("marginal" if res < 0.02 else "UNREACHABLE")
        print("  %-21s %7.4f   %-24s %s" % (name, res, np.round(got, 3), ok))

    # collision sanity at the mouth standoff pose
    q, _ = ik(m, d, targets["mouth standoff"], RETRACT)
    d.qpos[:7] = q; mujoco.mj_forward(m, d)
    def gname(gid):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if n:
            return n
        # geoms inside attached sub-models are often unnamed: fall back to body
        b = m.geom_bodyid[gid]
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        return f"{bn}#g{gid}"

    print("  contacts at mouth standoff (dist<0 means penetrating):")
    if d.ncon == 0:
        print("    none")
    for i in range(d.ncon):
        c = d.contact[i]
        print("    %-34s %-34s dist=%+.4f" % (gname(c.geom1), gname(c.geom2), c.dist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
