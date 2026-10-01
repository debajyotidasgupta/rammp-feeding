# Perception and sim: design notes

History and rationale moved out of the code comments. Grouped by file; the code says what it does now, this says
why it came to be that way.

## feeding/colour_stream.py (config perception.colour_stream)

- Why a resident reader: opening the RTSP stream for each use cost ~1.6 s a look.
- 2026-09-25: the colour stream stopped delivering for over a minute while the RTSP server still answered. A
  per-use open cannot tell that from a slow start; a resident reader logs it (colour_stream.jsonl gap/lost events).

## feeding/depth_fill.py

- 2026-09-23: Depth Anything 3's own metric scale made berries 2-3x too tall and invented a 39 mm table tilt --
  so DA3 is fitted to the sensor (sensor = s * da3 + b), never trusted on its own scale.
- One global fit could not follow DA3's local distortions: a 60x60 px hole came out 14 mm off (52 mm worst). Hence
  the local correction (the fit's residual on surrounding sensor pixels spread into the hole, Gaussian sigmas
  6 / 20 / 60 px).

## feeding/detect_mouth.py

- Depth is the 20th percentile (nearest surface), not the median: a median once read the mouth 67 mm behind the
  nose, off the face's edge.
- The "mouth" query has returned a full-frame box with the mouth in plain view; hence the plausibility check and
  the "lips" retry.

## feeding/detect_mouth_mp.py (config delivery.open_mouth_ratio)

- MediaPipe ~25 ms a frame on the Jetson CPU vs LocateAnything's 30-40 s.
- On 8 saved delivery frames MediaPipe found the face every time, including the two where LocateAnything's
  "mouth" box was the whole frame; where both worked they agreed to 2-10 px.
- mouth_open (inner-lip gap / mouth width): a closed mouth read <= 0.02 on every saved frame -- the basis for
  open_mouth_ratio.

## feeding/detector_server.py

- 2026-09-24 (user: "keep it on"): loading LocateAnything took 13.6 s on EVERY `make perceive`, i.e. every bite.
  Depth Anything 3 takes 10.4 s to load and 0.8 s for its first inference, then ~0.1 s. Hence one resident server
  for both (8.5 GB of the Jetson's 61 GB together), warmed up before it answers.

## feeding/grab_frame.py

- 2026-09-24 (user: "why is the detection taking so much time"): the colour frame used to be taken by recording
  the RTSP stream through gst-launch until a 22 s timeout killed it, keeping the last frame -- 22 s on every
  perceive. Replaced by the resident stream's /frame, else an OpenCV open that skips the first 8 decoded frames.

## feeding/locate_anything.py

- The INT4-via-TensorRT-LLM route in the original architecture (Sec. 7, written against triphammer's 12 GB card)
  does not exist for this model; BF16 (7.7 GB) on the 61 GB Orin is fine -- quantisation would only be a latency
  optimisation. (docs/feeding-architecture.md, which this referenced, no longer exists.)
- Without the <ref> walk every detection shared one label; the arm has to know which box is food and which the bowl.
- dedupe: a single query for "face" came back with 341 boxes (~2000 tokens, the generation cap), nearly all
  duplicates of a handful of real detections -- box counts were meaningless and latency was spent emitting
  redundant coordinates.
- _stop_on_repeat: 80 boxes for 4 berries; ~4 of 5.6 s spent emitting near-copies.
- detect_labelled: a comma-joined category string made the model fall back to regions -- 9 boxes, 6 junk, vs 4
  correct with "</c>". Asked for "food pieces</c>bowl</c>fork" it emitted the bowl ~60 times, exhausted
  max_new_tokens and never reached "fork"; dedupe hides repeats but cannot recover a starved category.
- Per-category dedupe (parse_labelled): a berry inside the bowl overlaps the bowl box almost completely; a global
  NMS would drop one of them.

## feeding/locate_scene.py (config perception.scene.*)

- detect_width: at 1280x720 the detector took ~5 s a prompt on the site Orin; detection runs on a downscaled copy.
- 2026-09-30: a lidded box 1 m away at the edge of the table won the single-best-box "bowl" call. Hence every
  candidate is placed in the base frame, filtered by reach (max_reach_m) and table, and the one nearest the last
  reading (else workspace.bowl_centre_m) wins.
- The centre is a circle through the rim (top 15 mm): from a side view the middle of the points' extent leans
  toward the camera, because only the near wall is visible.

## feeding/locate_target.py

- Sensor depth inside the bowl (early measurement): table ~480 mm, rim ~440 mm (40 mm proud), floor ~472 mm, each
  berry a 30 mm bump. Blob segmentation was tried first and dropped: the concave floor rises toward the rim, so
  every threshold merged a different slice of wall into a different berry and centroids wandered by tens of px;
  without restricting to inside the rim the rim itself looked like food.
- Registration from three berries: measured cx 657 +- 9, cy 226 +- 3.5 px against factory 621 / 238.
- DA3 invented a ~45 mm tilt across a table seen straight down and exaggerated local relief ~4x -- hence the
  scale + shift + plane fit. Scored against physical checks it overstated berry relief 2-3x raw and flattened it to
  9-14 mm fused; the sensor got 31-32 mm. So --da3 is a cross-check only.
- Z_tip was 260 mm for the knife (the first utensil).
- Measured tip pixel (utensil.tip_px): a fork's head is wide and curled; the top-centre of its box was ~10 mm off
  where it meets the food.
- top_surface_aim (perception.aim_top_surface), 2026-09-25: a merged box's centre sent the fork into the gap
  between a piece and a grape.
- food_left (perception.food_left_*), 2026-09-25: a pile of grapes and slices came back as one bowl-sized box
  (rightly dropped) and the loop ended "empty" with food left. Now empty means nothing stands above the floor.

## feeding/measure_under.py

- Every approach on the day it was written re-measured at a ~30 mm hover and found the food 4.7-6.7 mm closer
  than the overlook reading (150 mm further away) predicted -- why it is read close.
- Tip pixel, not the utensil box's centre: ~10 mm off for a fork.

## feeding/perceive.py (config perception.*, features.food_detector, food.*)

- One query per category: "food pieces</c>bowl</c>spoon" got stuck repeating one category's box until
  max_new_tokens; which one repeated was not stable (the bowl on one frame, the spoon the next, then the food), so
  reordering only moved the casualty. Separate queries cost a few seconds and remove the failure.
- The single-instance bowl query was 2.4 s and within ~12 px of the rim. Caching the bowl box per session saves
  1.8 s a bite.
- The utensil is not detected for use: a 'spoon' query once returned the whole bowl box, which then suppressed
  every berry. The cross-check query cost ~2 s a bite, so it runs only on request (features.utensil_cross_check).
- Sliver filter (min_box_side_px, max_box_aspect): a 10x310 px sliver came back as 'food pieces' alongside three
  correct berries.
- Inside-utensil filter (inside_utensil_max): the spoon's white rounded head was returned as a fourth food piece
  at (660,466)-(756,620), squarely inside the spoon; the utensil is in every frame from this pose.
- Food detector on the bowl crop in "fast" mode with stop-on-repeat: on 8 saved frames, the same berries as the
  full frame in hybrid mode at 128 tokens, 5.6 s -> 1.0 s. At 512 tokens the model emitted 80 boxes.
- Bowl squaring: box measured 549x644 against the true 549x544 (utensil merged at the bottom), dragging the centre down.
- food_inside_bowl_min: a 47x108 px object at the image edge came back as 'food pieces' with three real berries.
  food_max_frac_of_bowl: a 599x484 px 'food pieces' box once covered most of a 585x432 bowl.
- food.fallback_prompts, 2026-09-25: a pile was drawn as one bowl-sized box (rightly dropped), leaving nothing.
- excluded_colour (food.exclude_grapes), 2026-09-25: a brightness test let a lit red grape through; the green
  channel test does not.

## feeding/run_depth_anything3.py

- process_res: upstream's default 504 turns 1280x720 into ~504x284, throwing away exactly the resolution needed.
- Was fitted to the sensor by depth_fusion.py (gone); depth_fill.py does it now.
- aarch64 install: upstream pins xformers, open3d, pycolmap, moviepy etc.; none are needed for inference and
  xformers has no aarch64 wheel -- so --no-deps, plus einops/omegaconf/safetensors/imageio/addict/evo/scipy, and
  the export helpers' imports made lazy (now docker/patches/da3_optional_exports.patch, setup_jetson.sh).

## feeding/acquire.sh

- The acquisition sequence was frozen 2026-09-23.
- Depth re-read from where the arm is: at a 30 mm hover the food was 4.7-6.7 mm closer than the overlook reading
  predicted, every time on that day.

## feeding/deliver.sh, feeding/feed.sh

- Both built 2026-09-24 (with the user).
- deliver.sh's capture_depth comment said "20-frame median"; the count is perception.depth_frames (15).

## feeding/stage1.sh

- locate_target's image->motion map (+X_base moves the scene down, +Y_base right, at f/Z) was measured by probes
  at food_detect_start's orientation.
- The colour capture used to record the stream until a 22 s timeout and keep the last frame (see grab_frame.py).
