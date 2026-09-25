# One entry point, identical on every machine. Architecture is detected;
# nothing here is host-specific. Arm commands run on the jetson.
#
#   make status    what is running, is the arm reachable
#   make state     where is the ARM, right now
#
#   make start     back to feeding_start  (retraces from food_detect_start)
#   make detect    feeding_start -> food_detect_start  (smooth lift, then swing)
#   make perceive  from food_detect_start: capture, detect, depth, target
#   make over      move so the utensil is over the target food
#   make acquire   over -> approach -> punch -> scoop (STOP_AFTER=over|hover30|hover10|punch, START_AT=...)
#   make deliver   swing -> present -> bite -> retract (STOP_AFTER=swing|present|bite, START_AT=..., VALIDATE=1)
#   make retract   back from the user to food_detect_start
#   make feed      the loop: bite after bite until the bowl is empty or the user declines (BITES=1: one)
#
#   make review    the latest session's timeline, numbers and contact sheets (RUN=<session or bite dir>)
#   make locate-bowl   where the bowl and table are, from the latest perceive (to set config workspace:)
#   make limits    the arm's own speed limits (APPLY=1: set them from config deployment.arm.soft_limits)
#
#   make capture   re-measure feeding_start from the parked arm (rarely: see README)
#   make sdk / images / depth   build and run the containers
SHELL      := /bin/bash
# deployment values come from config/ (settings.py); ARM_IP=... on the command line still wins
SET        := python3 -m settings get
ARM_IP     ?= $(shell $(SET) deployment.arm.ip)
DRIVER     := $(shell $(SET) deployment.containers.driver)
VISION     := $(shell $(SET) deployment.containers.vision)
DETECT_PY  := $(shell $(SET) deployment.venvs.detector)/bin/python
FACE_PY    := $(shell $(SET) deployment.venvs.face)/bin/python
RUNS       := $(shell $(SET) deployment.runs_dir)
DET_PORT   := $(shell $(SET) deployment.servers.detector_port)
FACE_PORT  := $(shell $(SET) deployment.servers.face_port)
DEPTH_PORT := $(shell $(SET) deployment.servers.depth_port)
DEPTH_TOPIC:= $(shell $(SET) camera.streams.depth_topics)
MODELS_WAIT:= $(shell $(SET) deployment.servers.startup_timeout_s)
# RTSP transport for the depth stream. The session drops periodically either way
# (an upstream timeout, not a network fault -- the link measures 0% loss), but
# measured over ~250s: udp drops every ~62s with a fixed 6s recovery; tcp drops
# less often (63s, 82s, 94s) and recovers in under 1s. During a drop the topic
# goes SILENT -- vision.cpp:355 returns without publishing, it does not repeat
# the last frame -- so a consumer sees a gap, not a frozen image. Either way a
# shorter outage is better. Hence tcp; override with RTSP_PROTOCOLS=udp.
RTSP_PROTOCOLS ?= tcp
SUBMODULE  := third_party/kinova-gen3-ros2
ARCHTAG    := $(shell uname -m | sed 's/arm64/aarch64/; s/x86_64/x86-64/; s/amd64/x86-64/')
SDK        := $(HOME)/kortex_api_2.8.0_$(ARCHTAG)
# Arm commands run inside the driver container, which is where the KORTEX SDK
# lives. It mounts this tree at /work, so an edit here is live immediately.
# PYTHONDONTWRITEBYTECODE because the container runs as root: without it every
# invocation leaves root-owned __pycache__ in the synced tree, which then blocks
# sync.sh --delete and needs sudo to clear.
INARM      := docker exec -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work -w /work $(DRIVER) python3 -u -m

.PHONY: sdk kortex vision images depth clean-depth status state start detect perceive over acquire deliver retract feed detector models models-stop capture review locate-bowl limits

# --- the arm ----------------------------------------------------------------
# Every motion goes through kinova/scripts/safe_move.py: utensil kept pointing
# down, joint torque watched for contact, joint limits checked, and the
# controller's own reason reported if it refuses a move.

state:
	@$(INARM) kinova.scripts.arm_state

start:
	@$(INARM) kinova.scripts.sequence home

detect:
	@$(INARM) kinova.scripts.sequence detect

# Runs on the jetson HOST (it drives both containers and the GPU venv). Results
# go to the run record (~/rammp_runs/<session>/bite_NN/perceive/), outside the synced tree. STAGE1_DA3=1 adds
# the optional Depth Anything 3 cross-check.
perceive:
	@feeding/stage1.sh

# Keep every model loaded and warm for the whole session: LocateAnything + Depth Anything 3 (GPU, one
# server, 8.5 GB) and the MediaPipe face landmarker (CPU, its own venv). Each runs a warm-up inference before
# it answers; this returns once both do. make feed runs it first.
models:
	@if curl -s -m 2 http://127.0.0.1:$(DET_PORT)/ping | grep -q ok; then echo "detector + depth server already up"; \
	else PYTHONPATH=.:$$HOME/da3src/src nohup $(DETECT_PY) -m feeding.detector_server > $(RUNS)/detector.log 2>&1 & \
	  echo "starting detector + depth server (log: $(RUNS)/detector.log)"; fi
	@if curl -s -m 2 http://127.0.0.1:$(FACE_PORT)/ping | grep -q ok; then echo "face server already up"; \
	else PYTHONPATH=. nohup $(FACE_PY) -m feeding.detect_mouth_mp --serve > $(RUNS)/face_server.log 2>&1 & \
	  echo "starting face server (log: $(RUNS)/face_server.log)"; fi
	@if curl -s -m 2 http://127.0.0.1:$(DEPTH_PORT)/ping | grep -q ok; then echo "depth server already up"; \
	else docker exec $(VISION) pkill -9 -f "[d]epth_server.py" >/dev/null 2>&1; \
	  docker cp feeding/depth_server.py $(VISION):/tmp/depth_server.py >/dev/null && \
	  docker exec -d $(VISION) bash -lc "source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; cd /tmp && python3 depth_server.py --port $(DEPTH_PORT) --topic $(DEPTH_TOPIC) > /tmp/depth_server.log 2>&1" && \
	  echo "starting depth server (in $(VISION))"; fi
	@for i in $$(seq 1 $(MODELS_WAIT)); do curl -s -m 2 http://127.0.0.1:$(DET_PORT)/ping | grep -q ok && curl -s -m 2 http://127.0.0.1:$(FACE_PORT)/ping | grep -q ok && curl -s -m 2 http://127.0.0.1:$(DEPTH_PORT)/ping | grep -q ok && { echo "all models loaded and warm, depth streaming ($${i} s)"; exit 0; }; sleep 1; done; echo "models did not come up -- see $(RUNS)/*.log"; exit 1

detector: models

models-stop:
	@pkill -f "[f]eeding.detector_server" && echo "detector + depth server stopped" || true
	@pkill -f "[d]etect_mouth_mp --serve" && echo "face server stopped" || true
	@docker exec $(VISION) pkill -9 -f "[d]epth_server.py" && echo "depth server stopped" || true

# Acquisition, one verified step at a time. Uses the current bite ($(RUNS)/latest: the latest make perceive).
over:
	@STOP_AFTER=over feeding/acquire.sh

# Food acquisition from food_detect_start: OVER, down to 30 mm, re-measure, down
# to 10 mm, re-measure, PUNCH to bowl floor + 7 mm, SCOOP. Uses the latest
# perceive run. Stop partway with e.g. `make acquire STOP_AFTER=hover10`, and
# resume from there with `make acquire START_AT=punch`.
acquire:
	@STOP_AFTER=$(STOP_AFTER) START_AT=$(or $(START_AT),over) feeding/acquire.sh

# Food delivery, from the end of the scoop: lift and swing to the user (retracting the utensil as it
# turns), present 80 mm short of the mouth, wait for the bite (1 N), retract to food_detect_start.
deliver:
	@STOP_AFTER=$(STOP_AFTER) START_AT=$(or $(START_AT),swing) VALIDATE=$(VALIDATE) $(if $(filter undefined,$(origin WAIT_OPEN)),,WAIT_OPEN=$(WAIT_OPEN)) feeding/deliver.sh

retract:
	@START_AT=retract feeding/deliver.sh

# The whole loop from feeding_start, bite after bite, until the bowl is empty or the user does not open
# their mouth; then back to feeding_start. BITES=1 for a single test run.
feed:
	@BITES=$(or $(BITES),99) feeding/feed.sh

# The run record (feeding/lib.sh): every session in $(RUNS)/<date>_<what>/, the current bite at $(RUNS)/latest.
# The arm's kinematic soft limits (on the arm): show, or apply config deployment.arm.soft_limits
limits:
	@$(INARM) kinova.scripts.arm_limits $(if $(APPLY),--apply,)

review:
	@PYTHONPATH=. $(FACE_PY) -m feeding.review "$(or $(RUN),$$(dirname $$(readlink -f $(RUNS)/latest)))" --write

locate-bowl:
	@PYTHONPATH=. $(DETECT_PY) -m feeding.locate_bowl "$(or $(RUN),$$(readlink -f $(RUNS)/latest)/perceive)"

capture:
	@$(INARM) kinova.scripts.capture_home --out /work/calibration/home_pose.json
	@echo
	@echo "NOTE: that wrote into the synced tree. Pull it back to the Mac before"
	@echo "      the next 'sync.sh', which runs with --delete. Then restore J1 = -90"
	@echo "      and J3 = 0 exactly in the file -- see README."

# --- build ------------------------------------------------------------------

sdk:
	./setup_kortex_sdk.sh

kortex:
	@test -d "$(SDK)" || { echo "no SDK at $(SDK) -- run 'make sdk'"; exit 1; }
	$(MAKE) -C $(SUBMODULE) build-real KORTEX_SRC=$(SDK)

vision:
	docker build -f docker/Dockerfile.vision -t rammp/kortex-vision:humble docker/

images: kortex vision

# This arm's depth sensor reports fx=fy=334.810; the package's shipped default
# is 360.013 -- 7.53% off, a scale error on every deprojected point worth about
# 8 mm at the plate edge. calibration/depth_480x270.ini holds the device's own
# values (see calibration/README.md).
#
# CAVEAT, measured: passing depth_camera_info_url does NOT give you a populated
# /camera/depth/camera_info -- kinova_vision publishes it with width=height=0
# when the URL is custom. The image topic is fine (30.0 Hz measured). So a
# consumer must read the .ini directly rather than subscribing to camera_info.
# The flag is kept because it stops the WRONG intrinsics being advertised, not
# because it makes the right ones available.
depth:
	-docker rm -f kinova_vision 2>/dev/null
	docker run -d --name kinova_vision --network host --ipc host --restart unless-stopped \
	  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp -e ROS_DOMAIN_ID=0 \
	  -v $(CURDIR)/calibration:/calib:ro \
	  --entrypoint bash rammp/kortex-vision:humble -lc \
	  'source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
	   exec ros2 launch kinova_vision kinova_vision.launch.py \
	     device:=$(ARM_IP) launch_color:=false launch_depth:=true \
	     depth_camera_info_url:=file:///calib/depth_480x270.ini \
	     depth_rtsp_element_config:="depth protocols=$(RTSP_PROTOCOLS) latency=30"'
	@docker exec -d kinova_vision bash -lc 'sleep 31; source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash; \
	   exec ros2 launch kinova_vision kinova_vision.launch.py camera:=camera2 \
	     device:=$(ARM_IP) launch_color:=false launch_depth:=true \
	     depth_camera_info_url:=file:///calib/depth_480x270.ini \
	     depth_rtsp_element_config:="depth protocols=$(RTSP_PROTOCOLS) latency=30" > /tmp/camera2.log 2>&1'
	@echo "depth nodes started: /camera/depth/image_raw now, /camera2/depth/image_raw in 31 s (the arm ends each"
	@echo "RTSP session every ~62 s; staggered, one of the two is always streaming)"

clean-depth:
	-docker rm -f kinova_vision

status:
	@echo "arch      : $$(uname -m)"
	@echo "sdk       : $(SDK) $$([ -d "$(SDK)" ] && echo '(present)' || echo '(MISSING)')"
	@echo "submodule : $$(git submodule status $(SUBMODULE) 2>/dev/null | tr -s ' ')"
	@echo "containers:"; docker ps --format '  {{.Names}} {{.Image}} ({{.Status}})' 2>/dev/null || echo "  none"
	@echo "arm       : $$(timeout 3 bash -c '(exec 3<>/dev/tcp/$(ARM_IP)/10000)' 2>/dev/null && echo 'TCP 10000 open' || echo unreachable)"
