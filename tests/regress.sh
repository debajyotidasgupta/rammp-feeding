#!/usr/bin/env bash
# Offline regression: the perception / measurement / face / depth-fill outputs on saved runs, no arm.
#   tests/regress.sh base      # record a baseline
#   tests/regress.sh check     # run again and compare with the baseline (numbers must match)
# Runs ON THE JETSON (host), from the repo root.
set -uo pipefail
cd "$(dirname "$0")/.."
export RAMMP_FOOD=strawberry        # the saved runs are strawberries (the aim offset is per food) ...
export RAMMP_UTENSIL=fork_20260924  # ... taken with the fork calibration of that day
export RAMMP_CAMERA=kinova_wrist     # ... and the built-in camera (a machine may select another in local.yaml)
MODE="${1:-check}"; RUNS="$HOME/rammp_runs"; OUT="$RUNS/regress/$MODE"; rm -rf "$OUT"; mkdir -p "$OUT"
LA="$HOME/venvs/la/bin/python"; MP="$HOME/venvs/mp/bin/python"
PERCEIVE=(20260924_201906 20260924_202241 20260924_202544)
FACE=(deliver_20260924_175952 deliver_20260924_193435 deliver_20260924_194800)
for r in "${PERCEIVE[@]}"; do
  S="$RUNS/$r"; D="$OUT/$r"; mkdir -p "$D"
  DEP="$S/depth.npy"; [ -f "$S/depth_sensor.npy" ] && DEP="$S/depth_sensor.npy"
  cp "$DEP" "$D/depth.npy"; cp "$S/colour.jpg" "$S/detection.json" "$D/"
  PYTHONPATH=. "$LA" -m feeding.locate_target --det "$D/detection.json" --depth "$D/depth.npy" --image "$D/colour.jpg" \
    --tool 0.2205 0.2826 0.5343 --out "$D/target" > "$D/locate.txt" 2>/dev/null
  [ -f "$S/acq_over.npy" ] && PYTHONPATH=. "$LA" -m feeding.measure_under --depth "$S/acq_over.npy" --out "$D/under_over.json" --guess 0.41 >/dev/null 2>&1
  PYTHONPATH=. "$LA" -m feeding.measure_under --depth "$D/depth.npy" --target "$D/target.json" --out "$D/under_target.json" >/dev/null 2>&1
done
for r in "${FACE[@]}"; do
  S="$RUNS/$r"; D="$OUT/$r"; mkdir -p "$D"; cp "$S/colour.jpg" "$S/depth.npy" "$D/"
  PYTHONPATH=. "$MP" -m feeding.detect_mouth_mp "$D" > "$D/mp.txt" 2>/dev/null
done
echo "recorded: $OUT"
if [ "$MODE" = check ]; then
  PYTHONPATH=. "$LA" - "$RUNS/regress/base" "$OUT" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
base, new = Path(sys.argv[1]), Path(sys.argv[2]); bad = 0; n = 0
def cmp(a, b, path=""):
    global bad
    if isinstance(a, dict):
        for k in set(a) | set(b):
            if k in ("image", "ms"): continue          # paths and timings may differ
            if k not in a or k not in b: print("  MISSING %s%s" % (path, k)); bad += 1; continue
            cmp(a[k], b[k], path + k + ".")
    elif isinstance(a, list) and a and not isinstance(a[0], (int, float)):
        for i, (x, y) in enumerate(zip(a, b)): cmp(x, y, path + "%d." % i)
    elif isinstance(a, (int, float, list)) and a is not None and b is not None:
        if not np.allclose(np.array(a, float), np.array(b, float), atol=1e-6, equal_nan=True): print("  DIFF %s %s vs %s" % (path, a, b)); bad += 1
    elif a != b: print("  DIFF %s %r vs %r" % (path, a, b)); bad += 1
for f in sorted(base.rglob("*.json")):
    g = new / f.relative_to(base); n += 1
    if not g.exists(): print("MISSING", g); bad += 1; continue
    cmp(json.loads(f.read_text()), json.loads(g.read_text()), f.relative_to(base).as_posix() + ": ")
print("compared %d json outputs: %s" % (n, "IDENTICAL" if bad == 0 else "%d differences" % bad))
sys.exit(1 if bad else 0)
PY
fi
