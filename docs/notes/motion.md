# Motion: the free-space moves before cuRobo (until 2026-10-01)

Since 2026-10-01 every free-space move (make detect / make start, the swing to the user, the retract's return) is
one cuRobo path (kinova/curobo_client.travel). Until then they were scripted; the reasoning, kept because the
constraints it encoded still hold (the planner's world and our tip floor now carry them):

- **feeding_start is below the table, parked beside it.** A straight joint move from there to the overlook dragged
  the utensil across the table edge, so `make detect` went straight up first (one smooth Cartesian lift, not a
  staircase) to tool z 0.545, then swung. The lift stalled at ~0.565 m, where J6 pinned at its -120.3 limit and any
  correction pushing J6 further was silently ignored; 0.545 kept margin. The user found that peak unnatural
  (2026-09-30): the computed overlook then lowered it, and cuRobo's paths now peak 22-25 cm above the table.
- **The swing went via its midpoint.** J1 turned ~92 deg and J7 ~104 deg; two half-moves made the direction of
  travel explicit instead of trusting the controller to pick the short way on a continuous joint (the short way for
  J1 went round the back, away from the table).
- **The utensil stayed down throughout:** every waypoint had J2+J4+J6 ~ -180, and a joint interpolation between two
  such poses keeps the sum between them. safe_move still enforces the tilt limit on joint moves.
- **home was a retrace** of the recorded lift in reverse, refusing to start from anywhere it did not recognise
  (`motion.known_pose_deg`). With cuRobo, home plans from anywhere.
- **The delivery swing** (built with the user one leg at a time, 2026-09-24): from the scoop end, lift 175 mm (at
  +60 mm it touched the bowl and the user's head sat at the top of the image), then J1 -90 toward the user WHILE
  retracting the utensil along its own axis (in parallel, never one then the other), as far as the arm allowed
  (120 mm), then 50 mm to the user's right. The utensil stayed level, food side up: a 90 deg camera roll was tried
  and rejected (it turns the food sideways for the bite); the image is rolled in software instead. The 7th DoF was
  steered away from the J2/J4/J6 limits (without it the swing only fitted 60 mm of retraction). That geometry still
  defines where the swing ENDS (deliver.swing_goal); cuRobo plans the way there.
- **The swing was validated only for scoop ends on the base's -Y side** (y -0.15..0: bowl.swing_start_y_range_m):
  from a bowl in front of the user the same turn about the base would have swept the utensil through the face.
  cuRobo has the user in its world instead.
- **The retract** backed off along the utensil (never stopped on a torque step: moving away is the safe direction,
  and stopping could leave the food in the mouth), reversed the swing, went down to the scoop end + 80 mm steering
  the arm back into the scoop's own shape, then the joint move to food_detect_start. Unsteered, that joint move
  dipped the tip to the bowl floor (0.222 m); steered, never below ~0.340 m. Only the back-off survives.
- **Plan caches**: the scoop always ended on the calibrated scoop end, so the swing and the retract were the same
  joint paths every time, planned once (~10 s of stillness) and reused. Only the swing's goal is cached now.
- **The time scaling** bounds every joint's speed AND acceleration from rest to rest: an earlier leading-joint
  profile let J1 go from ~0 (the lift) to full speed (the swing) in one segment, 628 deg/s^2, and the controller
  refused the lift+swing at every speed down to 7 deg/s.
- **Braking reads as contact**: at 61-65 deg/s the stop at the end of a move stepped J2 -7.8 -> -4.8 N.m
  (2026-09-24) and -6.6 -> -3.4 N.m (2026-09-25) with nothing touched; hence 6 N.m for these moves, and
  `delivery.retract.home_arrived_deg`: a move that tripped on braking within 0.2 deg of its goal had arrived.

## cuRobo (2026-10-01)

- Evaluated on cascadilla (RTX 3070) with RAMMP-CuRobo on cuRobo v0.7.8: the site bite in 9 moves / 33 s of motion
  against the scripted 18 moves / 43 s. The bowl must be HOLLOW in its world (a solid box puts the overlook, the
  utensil over its opening, inside it); 0.35 m overlooks sat within cuRobo's 0.03 activation distance of the rim
  (0.01 used). PoseCostMetric holds are invalid in v0.7.8, so paths are unconstrained in orientation: the tilt
  stayed under 3.4 deg on the detect path.
- With the scene camera's 5 cm inflation, the raised table at x 0.08 put the parked arm's own base in collision:
  the inflated table stops short of the base by the error (kinova/curobo_world.py).
- The wheelchair (sim/assets/wheelchair/wheelchair_boxes.yaml): 20 boxes from ramms-sim's Unreal model, its height
  rough (the user: "rough height is enough"); x / y likely right.
