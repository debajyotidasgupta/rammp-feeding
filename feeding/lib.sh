# Sourced by every stage script (stage1 / acquire / deliver / feed): config access, the containers, and the
# run record. Runs ON THE JETSON, from the repo root.
#
# The run record, one directory per feeding session and one per bite:
#
#   ~/rammp_runs/<YYYYmmdd_HHMMSS>_<what>/     a session (make feed, or a stage run on its own)
#     session.json        when, where (host, git commit, dirty files), profiles, command
#     config.yaml         the merged configuration it ran with
#     console.log         everything printed, each line timestamped (ms)
#     events.jsonl        stage start / end, exit codes, durations (plus each stage's own, below)
#     bowl.json           the bowl, detected once per session
#     bite_01/
#       perceive/         colour, depth (+ png), detection, target, events
#       acquire/          the floor re-measure, the scoop's food path, arm/ (plans, telemetry, events, log)
#       deliver/look_1/   each look at the face: colour, depth, face, the open-mouth wait
#       deliver/arm/      the swing / approach / bite force / retract records
#   ~/rammp_runs/latest -> the current bite, so `make acquire` after `make perceive` finds it
#
# `python -m feeding.review <session or bite>` prints the timeline and makes a contact sheet.

S() { python3 -m settings get "$1"; }     # a value from config/ (settings.py)
RUNS="$(S deployment.runs_dir)"
DRIVER="$(S deployment.containers.driver)"; VISION="$(S deployment.containers.vision)"
PY="$(S deployment.venvs.detector)/bin/python"   # LocateAnything, Depth Anything 3, the measurements
MPY="$(S deployment.venvs.face)/bin/python"      # MediaPipe, frame grabs
DA3SRC="$(S deployment.paths.da3_src)"            # Depth Anything 3's source (on PYTHONPATH for $PY)
_hf="$(S deployment.paths.hf_home)"; [ -n "$_hf" ] && export HF_HOME="$_hf"   # where the models are cached
FRAMES="$(S perception.depth_frames)"
export PYTHONUNBUFFERED=1                         # lines reach the console log when they are printed
ARM_REC=/tmp/rammp_rec                            # the arm's record, inside the driver container

# console -> terminal as is, and -> console.log with a timestamp per line
_tee_ts() { local l t; while IFS= read -r l || [ -n "$l" ]; do
  printf '%s\n' "$l"; t=$EPOCHREALTIME; printf '%(%H:%M:%S)T.%.3s %s\n' "${t%.*}" "${t#*.}" "$l" >> "$1"; done; }

# ev KIND [json fields, comma-first]  ->  the session's events.jsonl
ev() { [ -n "${RAMMP_SESSION:-}" ] || return 0
  printf '{"t": %s, "stage": "%s", "kind": "%s"%s}\n' "$EPOCHREALTIME" "${STAGE:-shell}" "$1" "${2:-}" \
    >> "$RAMMP_SESSION/events.jsonl"; }

# session_begin WHAT: start a session unless one is active (exported to everything run from here)
session_begin() {
  if [ -z "${RAMMP_SESSION:-}" ]; then
    export RAMMP_SESSION="$RUNS/$(date +%Y%m%d_%H%M%S)_$1"
    mkdir -p "$RAMMP_SESSION"
    python3 -m settings dump > "$RAMMP_SESSION/config.yaml" 2>/dev/null || true
    python3 - "$RAMMP_SESSION/session.json" "$1" "$0 $*" <<'PY' 2>/dev/null || true
import json, socket, subprocess, sys, time
from settings import cfg
def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""
json.dump(dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), what=sys.argv[2], command=sys.argv[3],
               host=socket.gethostname(), commit=git("rev-parse", "--short", "HEAD"),
               dirty=[l[3:] for l in git("status", "--porcelain").splitlines()],
               profiles=dict(cfg.profiles), workspace=dict(cfg.workspace)), open(sys.argv[1], "w"), indent=1)
PY
    ev session_start ", \"what\": \"$1\""
  fi
  if [ -z "${RAMMP_TEE:-}" ]; then               # the outermost script records the console, once
    export RAMMP_TEE=1
    exec > >(_tee_ts "$RAMMP_SESSION/console.log") 2>&1
  fi
}

# bite_begin: the next bite directory of this session
bite_begin() {
  local n; n=$(find "$RAMMP_SESSION" -maxdepth 1 -name 'bite_*' | wc -l)
  export RAMMP_BITE="$RAMMP_SESSION/bite_$(printf %02d $((n + 1)))"
  mkdir -p "$RAMMP_BITE"; ln -sfn "$RAMMP_BITE" "$RUNS/latest"
}

# bite_attach: the bite this stage belongs to -- the active one, else the latest (a stage run on its own)
bite_attach() {
  if [ -z "${RAMMP_BITE:-}" ]; then
    [ -e "$RUNS/latest" ] || { echo "no bite yet -- run make perceive first"; exit 3; }
    export RAMMP_BITE="$(readlink -f "$RUNS/latest")"
    export RAMMP_SESSION="$(dirname "$RAMMP_BITE")"
  fi
}

# stage_begin NAME: this script's stage directory, its start event, and its end event on exit
stage_begin() {
  STAGE="$1"; STAGE_DIR="$RAMMP_BITE/$1"; mkdir -p "$STAGE_DIR"; _T0=$EPOCHREALTIME
  export RAMMP_RECORD="$STAGE_DIR"            # host-side python records here (runlog.py)
  ev stage_start ", \"bite\": \"$(basename "$RAMMP_BITE")\""
  trap '_rc=$?; arm_pull; ev stage_end ", \"bite\": \"$(basename "$RAMMP_BITE")\", \"rc\": $_rc, \"seconds\": $(python3 -c "print(round($EPOCHREALTIME - $_T0, 2))")"' EXIT
}

# the arm command for a kinova.scripts module, recording into the driver container
arm_cmd() { echo "docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work -e RAMMP_RECORD=$ARM_REC -w /work $DRIVER python3 -u -m kinova.scripts.$1"; }

# arm_pull: copy the arm's record out of the driver container into this stage's arm/ (host-owned files)
arm_pull() {
  [ -n "${STAGE_DIR:-}" ] || return 0
  docker exec "$DRIVER" test -d "$ARM_REC" 2>/dev/null || return 0
  local tmp="$STAGE_DIR/arm/.pull"; rm -rf "$tmp"; mkdir -p "$tmp"
  docker cp "$DRIVER:$ARM_REC/." "$tmp/" >/dev/null 2>&1 || true
  docker exec "$DRIVER" rm -rf "$ARM_REC" >/dev/null 2>&1 || true
  local f; for f in log.txt events.jsonl; do            # a stage run twice appends, never overwrites
    [ -f "$tmp/$f" ] && cat "$tmp/$f" >> "$STAGE_DIR/arm/$f" && rm -f "$tmp/$f"; done
  mv "$tmp"/* "$STAGE_DIR/arm/" 2>/dev/null || true; rm -rf "$tmp"
}
