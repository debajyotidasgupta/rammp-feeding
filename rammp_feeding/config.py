"""Every tunable number and switch, from config/ -- one loader for the whole module.

    from rammp_feeding.config import cfg
    cfg.delivery.stop_m                     # attribute access
    cfg.get("guards.torque_step_nm.approach")
    cfg.utensil.camera_to_tip_m             # the selected profiles: cfg.camera / .scene_camera / .utensil / .bowl / .food

Layers, later wins: config/rammp.yaml < the selected profiles (config/<kind>/<name>.yaml) < the machine's own file
(<state_dir>/local.yaml: /config/local.yaml in the container) < RAMMP_SET="a.b=1,c=[1, 2]" (one run; the tests).
Units are in the key names (_m, _mps, _dps, _nm, _n, _s, _px).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml


def _root() -> Path:
    """Where config/ and calibration/ are: the source tree, or the installed package's share."""
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
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load() -> Cfg:
    c = yaml.safe_load((CONFIG / "rammp.yaml").read_text())
    state = Path(os.path.expanduser(os.environ.get("RAMMP_STATE_DIR", c["deployment"]["state_dir"])))
    local_path = state / "local.yaml"
    local = {} if os.environ.get("RAMMP_NO_LOCAL") or not local_path.exists() else (yaml.safe_load(local_path.read_text()) or {})
    _merge(c["profiles"], local.get("profiles") or {})
    for kind in ("camera", "scene_camera", "utensil", "bowl", "food"):
        folder = "camera" if kind == "scene_camera" else kind
        c[kind] = yaml.safe_load((CONFIG / folder / (str(c["profiles"][kind]) + ".yaml")).read_text())
    _merge(c, local)
    for item in filter(None, re.split(r",(?![^\[]*\])", os.environ.get("RAMMP_SET", ""))):
        k, _, v = item.partition("=")
        _set(c, k.strip(), yaml.safe_load(v))
    u = c["utensil"]                    # a ruler reading from the camera's glass + the camera's own glass-to-optical
    if u.get("glass_to_tip_m") is not None:
        u["camera_to_tip_m"] = round(float(u["glass_to_tip_m"]) + float(c["camera"].get("glass_to_optical_m", 0.0)), 6)
    c["deployment"]["state_dir"] = str(state)
    return Cfg(c)


cfg = load()

# Measured on the robot (the wrist camera's mount, the utensil against that camera): null until they are, and nothing
# builds the arm without them.
MEASURED = ("camera.mount.camera_in_tool_m", "camera.mount.rotation_quat_xyzw", "utensil.glass_to_tip_m",
            "utensil.tip_px", "utensil.image_region_px")


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


def calibration(name: str) -> dict:
    """A captured pose (joints_deg): the machine's own capture (state_dir/<name>.json) if there is one, else the tree's."""
    for p in (state_dir() / (name + ".json"), ROOT / "calibration" / (name + ".json")):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError("no %s.json in %s or calibration/ -- rammp capture %s" % (name, state_dir(), name))


def dump() -> str:
    return yaml.safe_dump(json.loads(json.dumps(cfg)), sort_keys=False)
