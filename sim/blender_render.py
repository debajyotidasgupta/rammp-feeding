"""Path-traced frames of a twin export (sim/compare_planners.py --export): run inside Blender, ideally on a GPU.

    blender -b -P sim/blender_render.py -- EXPORT_DIR [--frames A:B] [--samples 64] [--still N]

EXPORT_DIR holds scene.json (geoms, materials, textures), meshes.npz (the MuJoCo meshes, geom frame) and
frames.npz (every frame's geom poses, visibility, utensil tip and camera). MuJoCo did all the posing; this only
builds the objects once, sets their poses per frame and draws them: Cycles, soft area lights, a room, depth of field.
Frames land in EXPORT_DIR/frames/NNNN.png (existing ones are skipped, so a run can be resumed).
"""

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Quaternion, Vector

PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 2, 3, 4, 5, 6, 7
METALS = {"chrome", "steel", "alu", "arm_grip_metal"}
TRAIL_S = 6.0


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = dict(dir=Path(a[0]), frames=None, samples=64, still=None)
    i = 1
    while i < len(a):
        if a[i] == "--frames":
            s, e = a[i + 1].split(":"); out["frames"] = (int(s), int(e)); i += 2
        elif a[i] == "--samples":
            out["samples"] = int(a[i + 1]); i += 2
        elif a[i] == "--still":
            out["still"] = int(a[i + 1]); i += 2
        else:
            raise SystemExit("unknown argument %s" % a[i])
    return out


def lin(c):
    return [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c[:3]]


# ------------------------------------------------------------------------------------------------ geometry
def mesh_from(name, v, f):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(v)); me.vertices.foreach_set("co", np.asarray(v, np.float32).ravel())
    me.loops.add(f.size); me.loops.foreach_set("vertex_index", np.asarray(f, np.int32).ravel())
    me.polygons.add(len(f)); me.polygons.foreach_set("loop_start", np.arange(0, f.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(f), bool))
    try:
        me.set_sharp_from_angle(angle=math.radians(38))
    except AttributeError:
        pass
    return me


def capsule(r, h, n=40, k=12):
    """A capsule along z: radius r, cylinder half length h."""
    rings = []
    for i in range(k + 1):                                  # the top hemisphere, pole to equator
        a = math.pi / 2 * (1 - i / k)
        rings.append((r * math.cos(a), h + r * math.sin(a)))
    for i in range(k + 1):
        a = -math.pi / 2 * i / k
        rings.append((r * math.cos(a), -h + r * math.sin(a)))
    v, f = [], []
    for rr, z in rings:
        for j in range(n):
            t = 2 * math.pi * j / n
            v.append((rr * math.cos(t), rr * math.sin(t), z))
    for i in range(len(rings) - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            c, d = a + n, b + n
            f += [(a, c, b), (b, c, d)]
    return np.array(v), np.array(f)


def primitive(name, g):
    t, s = g["type"], g["size"]
    bm = bmesh.new()
    if t == BOX:
        bmesh.ops.create_cube(bm, size=2.0)
        for vert in bm.verts:
            vert.co = Vector((vert.co.x * s[0], vert.co.y * s[1], vert.co.z * s[2]))
    elif t in (SPHERE, ELLIPSOID):
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
        sc = (s[0], s[0], s[0]) if t == SPHERE else s
        for vert in bm.verts:
            vert.co = Vector((vert.co.x * sc[0], vert.co.y * sc[1], vert.co.z * sc[2]))
    elif t == CYLINDER:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=56, radius1=s[0], radius2=s[0], depth=2 * s[1])
    elif t == CAPSULE:
        v, f = capsule(s[0], s[1])
        bm.free()
        return mesh_from(name, v, f)
    else:
        bm.free()
        return None
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    if t != BOX:
        me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
        try:
            me.set_sharp_from_angle(angle=math.radians(50))
        except AttributeError:
            pass
    return me


# ------------------------------------------------------------------------------------------------ materials
def principled(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    return mat, nt, b


def setin(b, key, v):
    if key in b.inputs:
        b.inputs[key].default_value = v


def image_tex(nt, b, path, scale, coords="Object", strength=None):
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.projection = "BOX"; tex.projection_blend = 0.25
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs[coords], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    return tex


def make_material(md, textures):
    name = md["name"]
    mat, nt, b = principled(name)
    rgba = md["rgba"]
    setin(b, "Base Color", (*lin(rgba), 1))
    rough = 0.12 + 0.7 * (1 - md["shininess"])
    setin(b, "Roughness", rough)
    setin(b, "Specular IOR Level", 0.25 + 0.5 * md["specular"])
    if name in METALS:
        setin(b, "Metallic", 1.0); setin(b, "Roughness", 0.18 if name != "alu" else 0.32)
    if name in ("arm", "arm_arm"):                          # Kinova's white satin shells
        setin(b, "Base Color", (0.80, 0.81, 0.83, 1)); setin(b, "Roughness", 0.32)
        setin(b, "Coat Weight", 0.35); setin(b, "Coat Roughness", 0.15)
    if name in ("arm_dark", "arm_arm_dark", "arm_grip_black", "rubber", "keys"):
        setin(b, "Roughness", 0.55)
    if name == "skin":
        setin(b, "Subsurface Weight", 0.25); setin(b, "Subsurface Radius", (0.9, 0.35, 0.2))
        setin(b, "Subsurface Scale", 0.006); setin(b, "Roughness", 0.48)
    if name == "hair":
        setin(b, "Base Color", (0.035, 0.022, 0.014, 1)); setin(b, "Roughness", 0.42); setin(b, "Coat Weight", 0.2)
    if name in ("shirt", "trousers", "cushion"):
        setin(b, "Roughness", 0.85); setin(b, "Sheen Weight", 0.15); setin(b, "Specular IOR Level", 0.2)
    if name in ("eyes", "sclera"):
        setin(b, "Roughness", 0.05); setin(b, "Coat Weight", 1.0)
    if name == "plastic_clear":
        setin(b, "Base Color", (0.92, 0.96, 1.0, 1)); setin(b, "Transmission Weight", 1.0)
        setin(b, "Roughness", 0.04); setin(b, "IOR", 1.46)
    if name == "mat_blue":
        setin(b, "Base Color", (*lin([0.10, 0.24, 0.78]), 1)); setin(b, "Roughness", 0.55)
    if name.startswith("food"):
        setin(b, "Subsurface Weight", 0.4); setin(b, "Subsurface Radius", (1.0, 0.6, 0.3))
        setin(b, "Subsurface Scale", 0.004); setin(b, "Roughness", 0.5)
    if name == "arm_fork":
        setin(b, "Base Color", (0.93, 0.93, 0.92, 1)); setin(b, "Roughness", 0.22)
        setin(b, "Subsurface Weight", 0.15)
    tex = md.get("texture")
    if tex == "wood":
        image_tex(nt, b, textures["wood"], 1.3); setin(b, "Roughness", 0.38)
        setin(b, "Coat Weight", 0.45); setin(b, "Coat Roughness", 0.08)
    elif tex == "concrete":
        image_tex(nt, b, textures["concrete"], 0.8); setin(b, "Roughness", 0.42)
    elif tex == "fabric":
        image_tex(nt, b, textures["fabric"], 12.0)
    elif tex == "screen":
        t = image_tex(nt, b, textures["screen"], 1.0, coords="Generated")
        nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
        setin(b, "Emission Strength", 2.2); setin(b, "Roughness", 0.06)
    return mat


def emissive(name, rgb, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*rgb, 1); e.inputs["Strength"].default_value = strength
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs["Emission"], o.inputs["Surface"])
    return mat


def plain(name, rgb, rough, **kw):
    mat, nt, b = principled(name)
    setin(b, "Base Color", (*lin(rgb), 1)); setin(b, "Roughness", rough)
    for k, v in kw.items():
        setin(b, k, v)
    return mat


# ------------------------------------------------------------------------------------------------ the set
def box_obj(name, centre, half, mat):
    me = primitive(name, dict(type=BOX, size=half))
    ob = bpy.data.objects.new(name, me)
    ob.location = centre
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def area(name, loc, target, power, size, rgb, size_y=None):
    li = bpy.data.lights.new(name, "AREA")
    li.energy, li.color = power, rgb
    li.shape = "RECTANGLE" if size_y else "SQUARE"
    li.size = size
    if size_y:
        li.size_y = size_y
    ob = bpy.data.objects.new(name, li)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.to_track_quat("-Z", "Y")
    bpy.context.scene.collection.objects.link(ob)
    return ob


def room(S, textures):
    """A warm, softly lit room round the MuJoCo set: floor, two walls, a window, a skirting, a pendant glow."""
    fz = S["floor_z"]
    floor = plain("room_floor", [0.42, 0.36, 0.30], 0.32, **{"Coat Weight": 0.25})
    _, nt, b = floor, floor.node_tree, floor.node_tree.nodes["Principled BSDF"]
    image_tex(nt, b, textures["wood"], 0.45)
    box_obj("floor", (0, 0, fz - 0.01), (8, 8, 0.01), floor)
    wall = plain("room_wall", [0.50, 0.48, 0.45], 0.9)
    box_obj("wall_back", (-1.45, 0.6, fz + 1.6), (0.02, 4, 1.6), wall)        # behind the user
    box_obj("wall_side", (1.0, 2.05, fz + 1.6), (4, 0.02, 1.6), wall)         # the far side
    skirting = plain("skirting", [0.95, 0.95, 0.94], 0.4)
    box_obj("skirt_back", (-1.42, 0.6, fz + 0.06), (0.012, 4, 0.06), skirting)
    box_obj("skirt_side", (1.0, 2.02, fz + 0.06), (4, 0.012, 0.06), skirting)
    # a window in the side wall: a bright panel behind mullions, and the daylight through it
    # a window in the back wall, beside and behind the user: a bright panel behind mullions, the daylight through it
    wy, wz = 1.15, fz + 1.45
    box_obj("window", (-1.425, wy, wz), (0.005, 0.7, 0.55), emissive("daylight", (0.80, 0.88, 1.0), 2.2))
    frame = plain("window_frame", [0.92, 0.92, 0.92], 0.35)
    for dy in (-0.7, 0.0, 0.7):
        box_obj("mullion", (-1.41, wy + dy, wz), (0.02, 0.02, 0.57), frame)
    for dz in (-0.55, 0.55):
        box_obj("transom", (-1.41, wy, wz + dz), (0.02, 0.72, 0.02), frame)
    area("window_light", (-1.35, wy, wz), (0.1, 0.2, S["table_z"] + 0.3), 300, 1.4, (0.80, 0.88, 1.0), size_y=1.1)
    # key: a big warm softbox, front right above the table; cool rim behind the user; soft fill from the camera side
    tz = S["table_z"]
    area("key", (1.35, -0.55, tz + 1.5), (0.0, 0.25, tz + 0.25), 330, 1.4, (1.0, 0.88, 0.74))
    area("rim", (-1.1, 1.0, tz + 1.35), (0.0, 0.2, tz + 0.45), 380, 0.7, (0.62, 0.78, 1.0))
    area("fill", (1.7, 1.0, tz + 0.6), (0.0, 0.25, tz + 0.3), 45, 2.0, (0.90, 0.93, 1.0))
    bowl = S["bowl"]
    area("pendant", (bowl[0], bowl[1], tz + 1.05), (bowl[0], bowl[1], tz), 45, 0.35, (1.0, 0.82, 0.62))
    w = bpy.data.worlds.new("world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.045, 0.05, 0.06, 1); bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = w


# ------------------------------------------------------------------------------------------------ main
def main():
    A = args()
    D = A["dir"]
    S = json.loads((D / "scene.json").read_text())
    M = np.load(D / "meshes.npz")
    Fr = np.load(D / "frames.npz")
    pos, quat, vis, tips, cams = Fr["pos"], Fr["quat"], Fr["vis"], Fr["tip"], Fr["cam"]

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sc = bpy.context.scene
    mats = {md["name"]: make_material(md, S["textures"]) for md in S["materials"]}
    shared = {}
    objs = []
    for j, g in enumerate(S["geoms"]):
        if g["type"] == PLANE or (g["material"] is None and g["rgba"][3] < 0.3):
            objs.append(None); continue
        if g["type"] == MESH:
            k = g["mesh"]
            if k not in shared:
                shared[k] = mesh_from("mesh%d" % k, M["v%d" % k], M["f%d" % k])
            me = shared[k]
        else:
            me = primitive(g["name"], g)
        if me is None:
            objs.append(None); continue
        ob = bpy.data.objects.new(g["name"], me)
        ob.rotation_mode = "QUATERNION"
        mat = mats.get(g["material"]) if g["material"] else None
        if mat is None:
            mat = plain("rgba_%d" % j, g["rgba"][:3], 0.5)
        ob.material_slots and None
        ob.data.materials.clear() if g["type"] != MESH else None
        if g["type"] == MESH:                               # a shared mesh: the material on the object
            if not me.materials:
                me.materials.append(None)
            ob.material_slots[0].link = "OBJECT"
            ob.material_slots[0].material = mat
        else:
            me.materials.append(mat)
        if g["type"] == BOX and min(g["size"]) > 0.004:
            bv = ob.modifiers.new("bevel", "BEVEL")
            bv.width, bv.segments, bv.limit_method = min(0.004, min(g["size"]) * 0.4), 3, "ANGLE"
        sc.collection.objects.link(ob)
        objs.append(ob)
    room(S, S["textures"])

    # the utensil tip's trail (a glowing tube over the last few seconds) and the tip itself
    cyan = emissive("trail", (0.15, 0.85, 1.0), 6.0)
    cu = bpy.data.curves.new("trail", "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = 0.0016; cu.bevel_resolution = 3
    trail = bpy.data.objects.new("trail", cu); trail.data.materials.append(cyan); sc.collection.objects.link(trail)
    tipb = bpy.data.objects.new("tip", primitive("tip", dict(type=SPHERE, size=[0.0042, 0, 0])))
    tipb.data.materials.append(emissive("tip", (0.3, 0.92, 1.0), 12.0)); sc.collection.objects.link(tipb)

    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"; cd.angle_y = math.radians(S["fovy"])
    cd.dof.use_dof = True; cd.dof.aperture_fstop = 4.0
    cam = bpy.data.objects.new("cam", cd); cam.rotation_mode = "QUATERNION"; sc.collection.objects.link(cam)
    sc.camera = cam
    fps = S["fps"]

    def pose(frame):
        k = min(max(frame - 1, 0), len(pos) - 1)
        for j, ob in enumerate(objs):
            if ob is None:
                continue
            ob.location = pos[k, j].tolist()
            ob.rotation_quaternion = Quaternion(quat[k, j].tolist())
            ob.hide_render = not bool(vis[k, j])
        look, dist, az, el = cams[k, :3], float(cams[k, 3]), math.radians(float(cams[k, 4])), math.radians(float(cams[k, 5]))
        f = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
        cam.location = Vector(look.tolist()) - dist * f
        cam.rotation_quaternion = f.to_track_quat("-Z", "Y")
        cd.dof.focus_distance = float(np.linalg.norm(tips[k] - np.array(cam.location)))
        tipb.location = tips[k].tolist()
        cu.splines.clear()
        k0 = max(0, k - int(TRAIL_S * fps))
        pts = [tips[i] for i in range(k0, k + 1)]
        keep = [pts[0]] if pts else []
        for p in pts[1:]:
            if np.linalg.norm(p - keep[-1]) > 0.002:
                keep.append(p)
        if len(keep) >= 2:
            sp = cu.splines.new("POLY"); sp.points.add(len(keep) - 1)
            for i, p in enumerate(keep):
                sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
        trail.hide_render = len(keep) < 2

    def handler(scene, *_):
        pose(scene.frame_current)
    bpy.app.handlers.frame_change_pre.append(handler)

    r = sc.render
    r.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for dv in prefs.devices:
        dv.use = dv.type == "OPTIX"
    sc.cycles.device = "GPU"
    sc.cycles.samples = A["samples"]; sc.cycles.use_adaptive_sampling = True; sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPTIX"
    except TypeError:
        pass
    sc.cycles.max_bounces, sc.cycles.transmission_bounces, sc.cycles.caustics_reflective = 6, 6, False
    sc.cycles.caustics_refractive = False
    r.use_persistent_data = True
    r.resolution_x, r.resolution_y, r.resolution_percentage = S["width"], S["height"], 100
    r.fps = fps
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    sc.view_settings.exposure = -0.2
    r.image_settings.file_format = "PNG"
    r.use_overwrite, r.use_placeholder = False, True
    (D / "frames").mkdir(exist_ok=True)
    if A["still"] is not None:
        sc.frame_set(A["still"]); pose(A["still"])
        r.filepath = str(D / ("still_%04d.png" % A["still"]))
        bpy.ops.render.render(write_still=True)
        return
    s, e = A["frames"] or (1, S["frames"])
    sc.frame_start, sc.frame_end = s, e
    r.filepath = str(D / "frames" / "####")
    bpy.ops.render.render(animation=True)


main()
