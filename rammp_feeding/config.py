"""Every tunable number and switch, from config/ -- one loader for the whole module.

    from rammp_feeding.config import cfg
    cfg.delivery.stop_m                     # attribute access
    cfg.get("guards.torque_step_nm.approach")
    cfg.utensil.camera_to_tip_m             # the selected profiles: cfg.camera / .scene_camera / .utensil / .bowl / .food

Layers, later wins: config/rammp.yaml < the selected profiles (config/<kind>/<name>.yaml) < the machine's own file
(<state_dir>/local.yaml: /config/local.yaml in the container) < the wrist camera's measured mount and the utensil's
pixels (<state_dir>/wrist_mount.json: rammp calibrate-wrist) < RAMMP_SET="a.b=1,c=[1, 2]" (one run, the tests; the
node's ROS parameters arrive this way, rammp_feeding/engine.py). A `profiles.<kind>` override selects that profile.
Units are in the key names (_m, _mps, _dps, _nm, _n, _s, _px).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml


def _root() -> Path:
    """Where config/ is: the source tree, or the installed package's share."""
    src = Path(__file__).resolve().parent.parent
    if (src / "config" / "rammp.yaml").exists():
        return src
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("rammp_feeding"))


ROOT = _root()
CONFIG = ROOT / "config"


class Cfg(dict):
    """A dict with attribute access and dotted lookup."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError("config has no '%s' (keys: %s)" % (k, ", ".join(self)))
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def get(self, dotted, default=KeyError):
        node = self
        for part in str(dotted).split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is KeyError:
                raise KeyError("config has no '%s'" % dotted)
            else:
                return default
        return Cfg(node) if isinstance(node, dict) and not isinstance(node, Cfg) else node

    def path(self, dotted) -> Path:
        """A path setting: ~ expanded, relative to the module's root."""
        p = Path(os.path.expanduser(str(self.get(dotted))))
        return p if p.is_absolute() else ROOT / p


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def _set(d, dotted, value):
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(d.get(p), dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value


def overrides(text: str) -> list[tuple[str, object]]:
    """RAMMP_SET's syntax: "a.b=1,c=[1, 2]" -> [(dotted key, YAML value)]."""
    out = []
    for item in filter(None, re.split(r",(?![^\[]*\])", text)):
        k, _, v = item.partition("=")
        out.append((k.strip(), yaml.safe_load(v)))
    return out


def load() -> Cfg:
    c = yaml.safe_load((CONFIG / "rammp.yaml").read_text())
    state = Path(os.path.expanduser(os.environ.get("RAMMP_STATE_DIR", c["deployment"]["state_dir"])))
    local_path = state / "local.yaml"
    local = {} if os.environ.get("RAMMP_NO_LOCAL") or not local_path.exists() else (yaml.safe_load(local_path.read_text()) or {})
    sets = overrides(os.environ.get("RAMMP_SET", ""))
    _merge(c["profiles"], local.get("profiles") or {})
    for k, v in sets:                   # a profile selected for this run loads in place of the machine's
        if k.startswith("profiles."):
            _set(c, k, v)
    for kind in ("camera", "scene_camera", "utensil", "bowl", "food"):
        folder = "camera" if kind == "scene_camera" else kind
        c[kind] = yaml.safe_load((CONFIG / folder / (str(c["profiles"][kind]) + ".yaml")).read_text())
    _merge(c, local)
    wm = state / "wrist_mount.json"     # rammp calibrate-wrist's measurement of this wrist camera's mount
    if wm.exists():
        m = json.loads(wm.read_text())
        if m.get("camera") == c["profiles"]["camera"]:
            c["camera"]["mount"].update(camera_in_tool_m=m["camera_in_tool_m"], rotation_quat_xyzw=m["rotation_quat_xyzw"])
            if m.get("utensil") == c["profiles"]["utensil"]:                     # the utensil's pixels, the same run
                c["utensil"].update(tip_px=m["tip_px"], image_region_px=m["image_region_px"])
    for k, v in sets:
        _set(c, k, v)
    u = c["utensil"]                    # a ruler reading from the camera's glass + the camera's own glass-to-optical
    if u.get("glass_to_tip_m") is not None:
        u["camera_to_tip_m"] = round(float(u["glass_to_tip_m"]) + float(c["camera"].get("glass_to_optical_m", 0.0)), 6)
    c["deployment"]["state_dir"] = str(state)
    return Cfg(c)


cfg = load()


# ROS parameter -> its default and the config key it sets ("" leaves config/ and the site's local.yaml as they are)
PARAMS = {
    "state_dir": ("/config", None),                          # the site's state folder (deployment.state_dir)
    "runs_dir": ("/runs", "deployment.runs_dir"),
    "record": (False, "record.enabled"),                     # the run record: a debug setting
    "camera_profile": ("", "profiles.camera"),               # config/camera/<name>.yaml: the wrist camera
    "scene_camera_profile": ("", "profiles.scene_camera"),   # config/camera/<name>.yaml: the scene camera
    "config": ("", None),                                    # any other key: "a.b=1,c=[1, 2]" (RAMMP_SET's syntax)
}


def apply_parameters(values: dict) -> None:
    """The parameters into RAMMP_STATE_DIR / RAMMP_SET, then config/ loaded again with them."""
    os.environ["RAMMP_STATE_DIR"] = str(values["state_dir"])
    sets = ["%s=%s" % (key, str(values[name]).lower() if isinstance(values[name], bool) else values[name])
            for name, (_, key) in PARAMS.items() if key and values[name] != ""]
    if values["config"]:
        sets.append(str(values["config"]))
    if os.environ.get("RAMMP_SET"):
        sets.insert(0, os.environ["RAMMP_SET"])
    os.environ["RAMMP_SET"] = ",".join(sets)
    reload()


def reload() -> Cfg:
    """Load again (after RAMMP_STATE_DIR / RAMMP_SET changed: the node's parameters), into the same `cfg` object. Modules
    that keep a section of it (`DL = cfg.delivery`) keep the old values: reload before importing them."""
    cfg.clear()
    cfg.update(load())
    return cfg

# The site's own values -- measured on the robot (the wrist camera's mount, the utensil against that camera) or set
# for the site (the table, where the bowl is laid): null until they are, and nothing that needs them builds the arm
# without them (docs/porting.md).
MEASURED = ("camera.mount.camera_in_tool_m", "camera.mount.rotation_quat_xyzw", "utensil.glass_to_tip_m",
            "utensil.tip_px", "utensil.image_region_px", "workspace.table_extent_m", "workspace.bowl_centre_m")


def unmeasured(c: Cfg | None = None) -> list[str]:
    """The MEASURED settings still null in this machine's config."""
    c = cfg if c is None else c
    return [k for k in MEASURED if c.get(k, None) is None]


def state_dir() -> Path:
    """The machine's own state (calibrations measured on site): mounted at /config in the container."""
    p = Path(cfg.deployment.state_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir() -> Path:
    p = Path(os.path.expanduser(str(cfg.deployment.runs_dir)))
    p.mkdir(parents=True, exist_ok=True)
    return p


CAPTURES = ("home", "overlook", "deliver")    # the site's arm poses: rammp capture <name>


def calibration(name: str) -> dict:
    """A captured pose (joints_deg): this site's capture, state_dir/<name>.json."""
    p = state_dir() / (name + ".json")
    if not p.exists():
        raise FileNotFoundError("no %s -- pose the arm and rammp capture %s (docs/porting.md)" % (p, name))
    return json.loads(p.read_text())


def dump() -> str:
    return yaml.safe_dump(json.loads(json.dumps(cfg)), sort_keys=False)
