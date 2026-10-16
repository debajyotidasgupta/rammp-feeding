# Design notes: what was tried, measured and decided

The code and the config say what the stack does now; these say why, and what came before. Dated; nothing here is
an instruction unless the README or CLAUDE.md repeats it.

| file | about |
|---|---|
| history.md | the README's record until 2026-10-01: the demos, acquisition and delivery as built, perception studies, poses |
| motion.md | the scripted free-space moves before cuRobo, and the cuRobo evaluation |
| config.md | where config/rammp.yaml's numbers and the profiles' calibrations came from |
| arm.md | kinova/ and the arm-side scripts: incidents, measurements, removed code |
| perception.md | feeding/ and sim/: prompts, filters, depth, the face, the twin |
| scene_error_study_20261001.json | tests/scene_error_study.py's raw rows (the overlook height and the scene camera's margin) |
| planners.md, planner_compare_20261001.json | scripted / cuRobo / BLAST on the same moves; findings for BLAST's authors |
| sim.md | the GPU strategy search (sim/isaac/): the pipeline, generation 0's findings, the cluster setup |
