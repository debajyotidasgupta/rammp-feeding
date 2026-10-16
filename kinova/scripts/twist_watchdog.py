#!/usr/bin/env python3
"""Stops the arm when a velocity-commanding loop stops beating -- a process of its own, with its own session.

    python -m kinova.scripts.twist_watchdog SECONDS      # started by deliver.track; one line on stdin per beat

A twist command (Base.SendTwistCommand) has no duration in this Kortex API: the arm keeps that velocity until
told otherwise. So the loop that sends them beats here every cycle; a silence over SECONDS -- the loop hung, crashed
or was killed (its end of the pipe closes) -- sends Base.Stop from this session. "done" ends the watch.
"""

from __future__ import annotations

import select
import sys

from kinova.connection import kinova_connection


def main() -> int:
    limit = float(sys.argv[1])
    with kinova_connection() as conn:
        print("ready", flush=True)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], limit)
            line = sys.stdin.readline() if ready else None
            if line is None or line == "":                    # silence, or the loop's process is gone
                conn.base.Stop()
                print("watchdog: %s -- the arm stopped" % ("no beat in %.2f s" % limit if line is None else "loop gone"),
                      file=sys.stderr, flush=True)
                return 1
            if line.strip() == "done":
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
