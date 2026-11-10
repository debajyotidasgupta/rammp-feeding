# syntax=docker/dockerfile:1
# rammp_feeding as a RAMMP module image (RAMMP-docker's templates/module.Dockerfile). CPU only: SAM3 and cuRobo are
# their own nodes, MediaPipe runs on the CPU.
# The base carries the arm's, the common and the perception interfaces (rammp-interfaces-ros2 v2.0.0), the same as
# kinova-gen3-ros2 2.0.0's, the arm image rammp-deployments runs.
FROM ghcr.io/rammp-org/rammp-base:2.0.0-jp6

# The planner's interfaces, which the base does not carry, pinned to the planner rammp-deployments runs.
ARG CUROBO_REF=v2.0.1
WORKDIR /module_ws
RUN git clone -q https://github.com/rammp-org/RAMMP-CuRobo.git /tmp/curobo && git -C /tmp/curobo checkout -q ${CUROBO_REF} \
    && mkdir -p src && cp -r /tmp/curobo/rammp_curobo_interfaces src/ && rm -rf /tmp/curobo

# MediaPipe's face landmarker (CPU), and its model; OpenCV (contrib: ArUco's CharucoDetector, >= 4.7) for the ChArUco
# calibrations, pinned to what mediapipe 0.10.14 resolves with numpy < 2
RUN pip3 install --no-cache-dir "mediapipe==0.10.14" "numpy<2" "opencv-contrib-python==4.11.0.86" \
    && mkdir -p /opt/models && curl -fsSL -o /opt/models/face_landmarker.task \
       https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

COPY . /module_ws/src/rammp_feeding/
# (the base's rosdep is not initialised: the few system packages by name)
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3-scipy python3-pil libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
    && rm -rf /module_ws/build /module_ws/log

CMD ["ros2", "launch", "rammp_feeding", "feed.launch.py"]
