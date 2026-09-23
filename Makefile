# One entry point, identical on every machine. Architecture is detected;
# nothing here is host-specific.
#
#   make status    where is everything
#   make state     where is the ARM, right now
#   make start     go to the feeding start pose
#   make lift      raise the tool vertically to the reach limit (stage 1)
#   make capture   re-measure the feeding start pose from the parked arm
#
#   make sdk       fetch the KORTEX SDK for this arch
#   make images    build the real-arm image, then the vision image on top
#   make depth     run the depth node (container)
SHELL      := /bin/bash
ARM_IP     ?= 192.168.2.67
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
INARM      := docker exec -e PYTHONDONTWRITEBYTECODE=1 tcp_rec python3 -u -m

.PHONY: sdk kortex vision images depth clean-depth status state start lift capture

# --- the arm ----------------------------------------------------------------

state:
	@$(INARM) kinova.scripts.arm_state

start:
	@$(INARM) kinova.scripts.feeding_start $(if $(SPEED),--speed $(SPEED),)

# Stage 1: raise the tool on a pure vertical, then swing the base round to face
# the table. Vertical FIRST so nothing is swept on the way round.
lift:
	@$(INARM) kinova.scripts.lift_vertical --to-limit

capture:
	@$(INARM) kinova.scripts.capture_home --out /work/calibration/home_pose.json
	@echo
	@echo "NOTE: that wrote into the synced tree. Pull it back to the Mac before"
	@echo "      the next 'sync.sh', which runs with --delete."

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
	@echo "depth node started; topic /camera/depth/image_raw (per-unit calibration)"

clean-depth:
	-docker rm -f kinova_vision

status:
	@echo "arch      : $$(uname -m)"
	@echo "sdk       : $(SDK) $$([ -d "$(SDK)" ] && echo '(present)' || echo '(MISSING)')"
	@echo "submodule : $$(git submodule status $(SUBMODULE) 2>/dev/null | tr -s ' ')"
	@echo "containers:"; docker ps --format '  {{.Names}} {{.Image}} ({{.Status}})' 2>/dev/null || echo "  none"
	@echo "arm       : $$(timeout 3 bash -c '(exec 3<>/dev/tcp/$(ARM_IP)/10000)' 2>/dev/null && echo 'TCP 10000 open' || echo unreachable)"
