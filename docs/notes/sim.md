# The GPU strategy search (sim/isaac/), 2026-10-02

Goal: many bite strategies on many scenarios at once -- the stack's own code plans each bite with cuRobo, Isaac Lab
replays it with food physics on thousands of arms -- to find the quickest bite that stays safe and keeps the food.

## The pipeline (one generation)

1. `sim/isaac/search.py init|score`: a cross-entropy search. A strategy = config keys: continuous ones (the overlook
   height and offset, the plunge's speeds and slow zone, the scoop's speed / rot_exp / backoff_from, the swing's
   lift, the approach and back-off speeds) and options (food_detect_start's arm shape `overlook_seed_delta_deg`,
   `delivery.swing_standoff_m`, `delivery.retract.back_off_m`). Scenarios: the bowl in the workable region, the
   table at the site's height or lower (feeding_start is captured for it: a higher table puts its first waypoints
   under the utensil's floor and every strategy is refused), the mouth around the site's. Score = delivered share
   - 0.01 x bite s - 0.02 x tip RMS jerk (m/s^3); a tip closer than 7 cm to the mouth scores -1.
2. `sim/isaac/generate.py`: every (strategy, scenario) as a `tests/site_check.py --planner server` run against N warm
   planner servers (ports 18700+; 9100 is node_exporter's on the cluster nodes). Null keys are left out (the
   config's default; a "None" string crashed 48 bites the first time).
3. `sim/isaac/make_batch.py`: the bites that went through -> one path per candidate (PCHIP-interpolated at 120 Hz)
   x 32 food draws, and per-bite metrics (duration, tip jerk and peak acceleration while carrying, carry tilt).
4. `sim/isaac/feed_replay.py` (Isaac Lab 2.3.2): the replay, split by rows over the GPUs; `--video` films the grid
   and one arm close up; `sim/isaac/compose.py` makes the film; `search.py score` refits and writes the next
   generation.

Assets: `sim/isaac/prepare_assets.py` (the Mac: decimated wheelchair, SMPL-X person, bowl, spoon -> npz) and
`sim/isaac/build_usd.py` (in Isaac: those as USD, and cuRobo's Gen3 + 2F-85 URDF via the URDF importer). The
importer hangs on the 2F-85's COLLADA meshes: convert them to OBJ first (trimesh, in the file's mm -- the URDF scales
by 0.001) and point the URDF at the OBJs.

## Generation 0 (24 strategies x 8 scenarios, cuRobo bites only)

192 bites in ~4 min on node01 (3 planner servers, 15 workers); 83 went through. Refusals: no continuous scoop path
48, the scoop's back-off over its 300 mm limit 48 (three bowl positions -- x 0.48; (0.33, 0.34); y -0.015 -- refused
even with the current config: that limit, not the planner, bounds where the bowl can be), no swing path 13.
Motion time: the current config 37.0 s; the best 29.6 s (-20 %: the swing ending 12 cm short of the mouth, another
overlook arm shape, a faster plunge). The near-mouth swings carry with more tip jerk (7-10 m/s^3 against 5.5).
The Isaac replay of this generation was stopped before it ran (2026-10-02: the arm test came first).

Where a bite's time goes (cuRobo, the site's geometry): the plunge 9.0 s, retract 5.9 s (a straight back-off and a
planned return), scoop 4.7 s, approach 4.0 s, home 3.9 s, detect 3.7 s, swing 2.5 s. cuRobo plans in ~65 ms on a
3070, ~130 ms on the site Orin; free-space moves already run at the arm's speed and acceleration limits.

## The cluster (Unicorn, bhattacharjee queue)

Everything under node-local `/scratch/dd752` (node03: the Isaac image, assets, USD; node01: the planners' env, the
search). Images: `isaac-lab-2.3.2.sif` (docker://nvcr.io/nvidia/isaac-lab:2.3.2); the planners are
`pip --target /scratch/dd752/pyplanners` inside `pytorch:2.7.1-cuda12.8` (no fakeroot for a definition-file build):
cuRobo checked out at v0.7.8 (d64c4b0), RAMMP-CuRobo's core, mujoco, kortex_api, imageio-ffmpeg. Isaac in apptainer
needs writable kit data / cache / logs binds, the host's Vulkan ICD and EGL vendor json, and the libnvidia-*
libraries apptainer misses (allocator, ngx, present, vksc-core, wayland-client, sandboxutils) bound into
/.singularity.d/libs. Compute nodes cannot ssh each other: files move through the login, streamed.
