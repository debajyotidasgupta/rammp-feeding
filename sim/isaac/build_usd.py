"""The replay's USD assets, once per checkout: the arm the planner plans for and the scene around it.

    /workspace/isaaclab/isaaclab.sh -p sim/isaac/build_usd.py --urdf .../kinova_gen3_7dof.urdf --meshes DIR --out DIR
        --headless

- gen3_2f85.usd: cuRobo's own Gen3 + Robotiq 2F-85 URDF (the model every plan is checked against), the gripper's
  mimic tags dropped (the replay sets each finger joint itself, closed on the utensil's handle);
- person.usd, wheelchair.usd, bowl.usd, spoon.usd: sim/isaac/prepare_assets.py's meshes, one Mesh prim per part,
  each with its own material (visual only: the replay's colliders are its own simple shapes).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--meshes", required=True)
ap.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)


def robot() -> None:
    src = Path(args.urdf)
    txt = src.read_text()
    txt = re.sub(r"<mimic[^>]*/>", "", txt)                       # every finger joint driven on its own
    txt = re.sub(r"<xacro:include[^>]*/>", "", txt)
    for j in ("left_outer_knuckle", "right_outer_knuckle", "left_inner_knuckle", "right_inner_knuckle",
              "left_inner_finger", "right_inner_finger"):          # the URDF fixes the fingers open: free them
        lo, hi = (-0.8, 0.0) if "inner_finger" in j else (0.0, 0.8)
        txt = re.sub(r'(<joint name="%s_joint") type="fixed">' % j,
                     r'\1 type="revolute"><limit effort="100" lower="%g" upper="%g" velocity="2.0"/>' % (lo, hi), txt)
    txt = txt.replace('filename="kortex_description', 'filename="%s/kortex_description' % src.parent)
    clean = out / "gen3_2f85.urdf"
    clean.write_text(txt)
    cfg = UrdfConverterCfg(asset_path=str(clean), usd_dir=str(out), usd_file_name="gen3_2f85.usd", fix_base=True,
                           merge_fixed_joints=False, self_collision=False, force_usd_conversion=True,
                           joint_drive=UrdfConverterCfg.JointDriveCfg(
                               gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=4000.0, damping=40.0)))
    print("robot:", UrdfConverter(cfg).usd_path, flush=True)


def material(stage, path: str, rgb, rough=0.5, metal=0.0):
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/S")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return m


MOVED = ("spoon", "person", "bowl")
FINISH = {"spoon": (0.25, 0.9), "bowl": (0.15, 0.0), "skin": (0.55, 0.0), "rail_l": (0.3, 0.8), "rail_r": (0.3, 0.8)}


def meshes(name: str) -> None:
    z = np.load(Path(args.meshes) / (name + ".npz"))
    stage = Usd.Stage.CreateNew(str(out / (name + ".usd")))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/" + name)
    stage.SetDefaultPrim(root.GetPrim())
    if name in MOVED:                          # posed every step by the replay: a kinematic body with no colliders
        rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.05)
    for part in [str(p) for p in z["names"]]:
        v, f, c = z[part + "_v"], z[part + "_f"], z[part + "_c"]
        m = UsdGeom.Mesh.Define(stage, "/%s/%s" % (name, part))
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray(f.reshape(-1).astype(np.int32).tolist()))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateDoubleSidedAttr(True)
        rough, metal = FINISH.get(part, FINISH.get(name, (0.6, 0.0)))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material(stage, "/%s/M_%s" % (name, part), c, rough, metal))
    stage.Save()
    print("%s.usd: %s" % (name, ", ".join(str(p) for p in z["names"])), flush=True)


robot()
for n in ("person", "wheelchair", "bowl", "spoon"):
    meshes(n)
print("USD-OK", flush=True)
import os  # noqa: E402
os._exit(0)
