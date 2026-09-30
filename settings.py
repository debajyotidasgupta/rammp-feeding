"""Every tunable number and switch, from config/ -- one loader for all programs and shell scripts.

    from settings import cfg
    cfg.delivery.stop_m                 # attribute access
    cfg.get("guards.torque_step_nm.approach")
    cfg.utensil.camera_to_tip_m         # the selected utensil profile (camera / utensil / bowl / food)
    cfg.overlook_pose()                 # the bowl's measured overlook pose (joints + tool pose)

    python3 -m settings get delivery.stop_m     # shell: prints the value (true -> 1, false -> empty)
    python3 -m settings dump                    # the merged configuration, as YAML

Layers, later wins: config/rammp.yaml < the selected profiles (config/camera|utensil|bowl|food/<name>.yaml, under
cfg.camera / cfg.utensil / cfg.bowl / cfg.food; a utensil's by_camera.<camera> section with that camera) <
config/local.yaml (per machine, not committed; it may also pick the profiles) < RAMMP_SET="a.b=1,c=2".
Strings may use {root}, {name} (deployment.root / .name: per machine) and {arm_ip}; RAMMP_ROOT overrides the root.
RAMMP_CONFIG points at another config directory. Profiles can be swapped per run: RAMMP_UTENSIL=spoon, RAMMP_FOOD=strawberry etc.
Lists work in RAMMP_SET: RAMMP_SET="workspace.bowl_centre_m=[0.28, 0.27],delivery.stop_m=0.06".
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG = Path(os.environ.get("RAMMP_CONFIG", ROOT / "config"))


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
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, dict) and part.isdigit() and int(part) in node:
                node = node[int(part)]
            elif default is KeyError:
                raise KeyError("config has no '%s'" % dotted)
            else:
                return default
        return Cfg(node) if isinstance(node, dict) and not isinstance(node, Cfg) else node

    def path(self, dotted):
        """A path setting: ~ expanded, relative paths taken from the repo root."""
        p = Path(os.path.expanduser(str(self.get(dotted))))
        return p if p.is_absolute() else ROOT / p

    def aim_offset(self):
        """(x, y) m added to a detected food centre in OVER: the utensil's tine offset (tool frame, turned into the
        base frame at the overlook orientation) + the food's aim offset (base frame)."""
        import math
        t = [float(v) for v in self.get("utensil.tine_offset_tool_m", [0.0, 0.0, 0.0])]
        p = self.overlook_pose()["tool_pose"]
        tx, ty, tz = (math.radians(p[k]) for k in ("theta_x_deg", "theta_y_deg", "theta_z_deg"))
        cx, sx, cy, sy, cz, sz = math.cos(tx), math.sin(tx), math.cos(ty), math.sin(ty), math.cos(tz), math.sin(tz)
        # rows x and y of R = Rz @ Ry @ Rx (Kortex's fixed-frame XYZ)
        rx = (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx)
        ry = (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx)
        u = (sum(a * b for a, b in zip(rx, t)), sum(a * b for a, b in zip(ry, t)))
        f = self.get("food.aim_offset_base_m", [0.0, 0.0])
        return [u[0] + float(f[0]), u[1] + float(f[1])]

    def overlook_pose(self):
        return json.loads(self.path("bowl.overlook_pose").read_text())


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
        d = d.setdefault(int(p) if p.isdigit() and int(p) in d else p, {})
    last = parts[-1]
    d[int(last) if last.isdigit() and int(last) in d else last] = value


def _fill(node, subs):
    if isinstance(node, dict):
        return {k: _fill(v, subs) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, subs) for v in node]
    if isinstance(node, str) and "{" in node:
        try:
            return node.format(**subs)
        except (KeyError, IndexError):
            return node
    return node


def load() -> Cfg:
    c = yaml.safe_load((CONFIG / "rammp.yaml").read_text())
    path = CONFIG / "local.yaml"                             # RAMMP_NO_LOCAL=1: ignore it (the tests: hermetic)
    use = path.exists() and not os.environ.get("RAMMP_NO_LOCAL")
    local = (yaml.safe_load(path.read_text()) or {}) if use else {}
    _merge(c["profiles"], local.get("profiles") or {})       # a machine may pick its own camera, utensil, ...
    for kind in ("camera", "utensil", "bowl", "food"):
        name = os.environ.get("RAMMP_" + kind.upper(), c["profiles"][kind])
        c["profiles"][kind] = name
        c[kind] = yaml.safe_load((CONFIG / kind / (name + ".yaml")).read_text())
    # a utensil's measurements against one camera (tip pixel, camera-to-tip, ...) live in its profile under
    # by_camera.<camera>, applied only with that camera selected (before local.yaml / RAMMP_SET, which still win)
    over = (c["utensil"].pop("by_camera", None) or {}).get(c["profiles"]["camera"])
    if over:
        _merge(c["utensil"], over)
    _merge(c, local)
    for item in filter(None, re.split(r",(?![^\[]*\])", os.environ.get("RAMMP_SET", ""))):   # commas in [..] kept
        k, _, v = item.partition("=")
        _set(c, k.strip(), yaml.safe_load(v))
    d = c["deployment"]
    if os.environ.get("RAMMP_ROOT"):
        d["root"] = os.environ["RAMMP_ROOT"]
    return Cfg(_fill(c, dict(arm_ip=d["arm"]["ip"], root=str(d.get("root", "~")).rstrip("/"),
                             name=d.get("name", "rammp"), camera_port=d["servers"]["camera_port"])))


cfg = load()


def _shell(v):
    if isinstance(v, bool):
        return "1" if v else ""
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    if isinstance(v, str) and v.startswith("~"):
        return os.path.expanduser(v)
    return str(v)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "get":
        print(_shell(cfg.get(sys.argv[2])))
    elif len(sys.argv) >= 2 and sys.argv[1] == "dump":
        print(yaml.safe_dump(json.loads(json.dumps(cfg)), sort_keys=False))
    else:
        print(__doc__); sys.exit(2)
