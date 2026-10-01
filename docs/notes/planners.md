# Three planners on the bite's free-space moves (2026-10-01)

`tests/planner_compare.py` (cascadilla, image `cornell/blast-eval:fd12217`, `docker/Dockerfile.blast_eval`): the
2026-09-30 site's bite at the 0.45 m overlook. Scripted = commit 41d4a88 (the last with it), run by its own
site_check; cuRobo = the stack now; BLAST = Dynamium Lab's B-spline trajectory optimizer (ICRA 2026, the Kinova
team's ask, commit fd12217), planning the same start / goal joints in the same world (the table and bowl grown by
the scene camera's 5 cm, the user, the wheelchair, the pedestal) with the same utensil and piece. All three judged
alike: MuJoCo's Gen3 for the utensil tip, cuRobo's collision check in the TRUE world, executed / time-optimal
durations under the arm's soft limits (61 deg/s, 45 deg/s^2 as the controller measures it). Raw rows:
planner_compare_20261001.json; the film: out/sim/three_planners.mp4 (MuJoCo, `sim.compare_planners` with 3 dumps).

| leg | planner | moves | time s | furthest-joint travel | in collision | tip peak above table | tip's least clearance | utensil tilt deg | planning |
|---|---|---|---|---|---|---|---|---|---|
| to food_detect_start | scripted | 3 | 3.23 | 154 deg | 0 | 0.218 m | 0.124 m | 0.5-1.6 | - |
| | cuRobo | 1 | 3.71 | 126 | 0 | 0.190 | 0.103 | 0.2-2.0 | 65 ms (GPU) |
| | BLAST | 1 | 3.58 | 126 | 0 | 0.238 | 0.116 | 0.5-**31.9** | 752 ms (CPU) |
| swing to the user (piece) | scripted | 1 | 6.72 | 115 | 0 | 0.353 | 0.177 | 89.4 | - |
| | cuRobo | 1 | 2.53 | 64 | 0 | 0.352 | 0.138 | 89.4-93.6 | 65 ms |
| | BLAST | 1 | 2.51 | 65 | 0 | 0.353 | 0.177 | 81.0-89.4 | 1969 ms |
| back to food_detect_start | scripted | 4 | 8.98 | 241 | 0 | 0.353 | 0.105 | 0.6-89.4 | - |
| | cuRobo | 1 | 3.06 | 86 | 0 | 0.353 | 0.120 | 0.6-89.4 | 66 ms |
| | BLAST | 1 | 2.90 | 86 | 0 | 0.353 | 0.103 | 0.1-89.4 | 814 ms |
| home | scripted | 4 | 4.41 | 155 | 0 | 0.220 | 0.120 | 0.5-1.5 | - |
| | cuRobo | 1 | 3.89 | 126 | 0 | 0.190 | 0.104 | 0.6-1.0 | 65 ms |
| | BLAST | 1 | 3.57 | 126 | 0 | **0.394** | 0.120 | 0.2-**46.7** | 871 ms |
| **the four legs** | scripted | 12 | **23.3** | | 0 | | | | |
| | cuRobo | 4 | **13.2** | | 0 | | | | |
| | BLAST | 4 | **12.6** | | 0 | | | | |

Reliability, 10 plans per leg, one attempt each: cuRobo 40/40 (65-66 ms median); BLAST 39/40 (one home failed;
median 830-1150 ms, 660-3690 ms).

**Reading it**
- cuRobo and BLAST both halve the scripted moves' time (23.3 s -> 13.2 / 12.6 s) and moves (12 -> 4); none of the
  44 judged paths touched the true scene.
- BLAST is time-optimal under the limits: 4 % faster than cuRobo's path re-timed by our run_path. But it has no
  orientation or height preference: the empty utensil swung up to 47 deg off vertical and its tip peaked 39 cm above
  the table (cuRobo: 1-2 deg, 19 cm -- the user asked for a lower arm); carrying the piece it rolled the level
  utensil by up to 8 deg (cuRobo: 4 deg). A tilt / height cost would have to be added for feeding.
- cuRobo plans in ~65 ms on a GPU; BLAST in ~0.8-2 s on 16 x86 cores.
- **BLAST cannot run on the Jetson as it is**: it requires AVX2 + FMA (x86) and VCL (x86-only vector library); its
  CUDA kernels (constraint evaluation) are not wired into the build or the optimizer. An ARM port (NEON / SIMDe) is
  the prerequisite for deployment.
- For the Kinova team, found while integrating: (1) `blast/manipulator/Kinova_Gen3.hpp`'s third link offset puts
  the lateral 6.4 mm on the wrong side ((0, -0.0064, -0.2104) vs the URDF's (0, +0.006375, -0.21038)): 13 mm at the
  flange; with the URDF's offsets BLAST's FK matches MuJoCo's Gen3 to 0.05 mm; (2) `optimize` segfaults with
  `external_collisions` on and an empty world; (3) `MAX_CAPSULES` is 7 at compile time: more capsules overflow
  silently (an "LLVM ERROR: out of memory" here) -- a gripper + utensil + carried piece leave 3 for the arm.
- Integration in this tree: tests/planner_compare.py builds BLAST's Gen3 (URDF offsets, the arm's soft limits,
  arm capsules + the 2F-85 + the utensil as capsules on the bracelet; the carried piece checked against the world
  only), converts kinova/curobo_world.py's boxes, and plans stop-to-stop with random restarts.
