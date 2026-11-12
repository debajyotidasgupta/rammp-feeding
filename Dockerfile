# syntax=docker/dockerfile:1
# rammp_feeding as a RAMMP module image: the feeding engine (capability_engine.CapabilityEngine) and the rammp CLI. CPU
# only: SAM3 and cuRobo are their own nodes, MediaPipe runs on the CPU.
# The base carries the arm's, the common, the perception and the orchestrator interfaces (rammp-interfaces-ros2
# v2.0.0), the same as kinova-gen3-ros2 2.0.0's, the arm image rammp-deployments runs.
FROM ghcr.io/rammp-org/rammp-base:2.0.0-jp6

# numpy's BLAS on one thread: the depth medians and the path checks otherwise take every core of the Jetson
ENV OPENBLAS_NUM_THREADS=1
WORKDIR /module_ws

# capability_engine's self.arm plans and checks poses on a PyBullet model; PyPI ships no arm64 wheel, so it compiles
# here (the version capability_engine's goldens were recorded with). Its own layer, before the sources.
RUN pip3 install --no-cache-dir pybullet==3.2.7

# MediaPipe's face landmarker (CPU), and its model; OpenCV (contrib: ArUco's CharucoDetector, >= 4.7) for the ChArUco
# calibrations, pinned to what mediapipe 0.10.14 resolves with numpy < 2
RUN pip3 install --no-cache-dir "mediapipe==0.10.14" "numpy<2" "opencv-contrib-python==4.11.0.86" \
    && mkdir -p /opt/models && curl -fsSL -o /opt/models/face_landmarker.task \
       https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# The planner's interfaces, which the base does not carry, pinned to the planner rammp-deployments runs; and the
# engine contract (capability_engine), pinned to a commit of rammp-kitchen-capability-engines. (OpenCV and MediaPipe
# need libGL and GLib, below; the rest comes from rosdep.)
ARG CUROBO_REF=v2.0.1
ARG ENGINES_SHA=c3989c6854f34535013ea1b1e8adf9185f7149c8
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/refs/tags/${CUROBO_REF}.tar.gz /tmp/curobo.tar.gz
ADD https://github.com/rammp-org/rammp-kitchen-capability-engines/archive/${ENGINES_SHA}.tar.gz /tmp/engines.tar.gz
RUN mkdir -p src \
    && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' \
    && tar -xzf /tmp/engines.tar.gz -C src --strip-components=1 --wildcards '*/capability_engine' \
    && rm /tmp/curobo.tar.gz /tmp/engines.tar.gz

COPY . /module_ws/src/rammp_feeding/
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
    && rm -rf /module_ws/build /module_ws/log

# `ros2 launch` on Humble has no SIGTERM handler: as PID 1 it would ignore docker stop's SIGTERM until SIGKILL. On SIGINT
# it stops its node, which cancels the action in flight and lets the arm settle first.
STOPSIGNAL SIGINT
CMD ["ros2", "launch", "rammp_feeding", "feeding.launch.py"]
