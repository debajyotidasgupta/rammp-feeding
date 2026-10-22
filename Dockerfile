# syntax=docker/dockerfile:1
# rammp_feeding as a RAMMP module image (RAMMP-docker's templates/module.Dockerfile). CPU only: SAM3 and cuRobo are
# their own nodes, MediaPipe runs on the CPU.
# The base carries the arm's and the common interfaces (rammp-interfaces-ros2 v1.1.0), the same as kinova-gen3-ros2
# 1.1.0's, the arm image rammp-deployments runs.
FROM ghcr.io/rammp-org/rammp-base:1.1.0-jp6

# The two interface packages this module calls that the base does not carry yet, pinned to what the nodes serving them
# were built against: the planner's (RAMMP-CuRobo v1.1.0, the image rammp-deployments runs) and the perception
# nodes' (rammp-interfaces-ros2 2bf67d4, #16: DetectedObject.size; dev, until its next tag). A different perception
# definition misreads every detection after the first: DetectedObject is an array element of ObjectDetections.
ARG CUROBO_REF=v1.1.0
ARG INTERFACES_REF=2bf67d4
WORKDIR /module_ws
RUN git clone -q https://github.com/rammp-org/RAMMP-CuRobo.git /tmp/curobo && git -C /tmp/curobo checkout -q ${CUROBO_REF} \
    && mkdir -p src && cp -r /tmp/curobo/rammp_curobo_interfaces src/ && rm -rf /tmp/curobo \
    && git clone -q https://github.com/rammp-org/rammp-interfaces-ros2.git /tmp/ifc && git -C /tmp/ifc checkout -q ${INTERFACES_REF} \
    && cp -r /tmp/ifc/rammp_perception_interfaces src/ && rm -rf /tmp/ifc

# MediaPipe's face landmarker (CPU), and its model
RUN pip3 install --no-cache-dir "mediapipe==0.10.14" "numpy<2" \
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
