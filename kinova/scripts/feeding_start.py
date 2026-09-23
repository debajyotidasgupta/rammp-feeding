#!/usr/bin/env python3
"""`feeding start` -- bring the arm home to the feeding start pose.

    python -m kinova.scripts.feeding_start      # same as `make start`

A thin entry point for kinova.scripts.sequence home: from food_detect_start (or
the top of the lift) it retraces the recorded outbound path -- swing back via
the midpoint, one smooth vertical descent, then a joint snap onto the pose. From
anywhere it does not recognise it refuses rather than improvising a path.

It deliberately no longer does a direct joint move of its own: that version
used a per-joint SPEED constraint (see safe_move for why that is wrong) and no
motion guard, and a direct move from the overlook can drag the utensil across
the table edge, because feeding_start sits below table height.
"""

from kinova.scripts.sequence import main as _seq_main
import sys

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "home"] + sys.argv[1:]
    raise SystemExit(_seq_main())
