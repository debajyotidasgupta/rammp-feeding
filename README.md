# rammp — Kinova Gen3 setup

Control stack for the Kinova arm at `192.168.2.67`.

## The arm

Read off the hardware itself via `check_connection`:

| | |
|---|---|
| Model | Gen3 **7-DoF**, `L53K` (kinematics `L53-CAB26-7FR2VI0S0000`) |
| Serial / part | `WO527059-0` / `KR10944` |
| Firmware | `5.2.8.0` (bootloader `51.2.2.0`) |
| Wrist | spherical |
| End effector | Robotiq 2F-85 gripper |
| Vision | L53 2D/3D module |
| Base | L53 fixed |

Network surface: **TCP 10000** (Kortex request/response), **UDP 10001** (real-time
cyclic), TCP 22 (SSH), TCP 80 (Kinova Web App). Default credentials `admin`/`admin`.

## Layout

```
kinova/                        native Python client (kortex_api 2.8.0)
third_party/kinova-gen3-ros2/  submodule -> rammp-org/kinova-gen3-ros2 @ 4aa7e5e
docker/                        stock Kinova ros2_kortex image (amd64, reference only)
docs/feeding-architecture.md   how the feeding system is meant to work
docs/camera-calibration.md     what needs calibrating, and what does not
docs/perception-quantization.md measured INT8/INT4/FP8 on the Orin
docs/demo-analysis.md          what the teleop demos measured
docs/skill-sequence.md         the bite cycle with measured parameters
docs/prior-art.md              ideas from empriselab/feeding-deployment, and what does not transfer
sim/                           MuJoCo feeding cell -- see sim/README.md
feeding/                       algorithm code shared by sim and (eventually) ROS 2
```

Clone with `--recurse-submodules`, or run `git submodule update --init` after.

## Simulation

`sim/` is a MuJoCo feeding cell: the Gen3 with a fork, a seated SMPL-X user who
can lean, a plate, and the arm's three camera streams. It exists to develop and
fault-inject the skill state machine and the safety monitor without a person in
the chair (`docs/feeding-architecture.md` §10). It needs no GPU and no arm.

```
python sim/selftest.py             25 checks, prints measured numbers
python sim/demo_meal.py --bites 4  the whole loop, end to end
python sim/validate_bite.py        how small a bite the torque residual can see
```

Findings so far — a tip-only safety check missing a 411 N collision, IK
redundancy putting the elbow inside the user, safe endpoints with an unsafe path
between them — are in `sim/README.md`.

Runs on the Mac only at the moment: triphammer is Python 3.8 on Ubuntu 20.04 and
needs an interpreter upgrade first (admin, like `nvidia-container-toolkit`).

## Python (native, on the Mac)

```bash
./setup_python.sh
./.venv/bin/python -m kinova.scripts.check_connection            # read-only
./.venv/bin/python -m kinova.scripts.check_connection --real-time # also UDP :10001
```

`kinova/connection.py` gives you a context manager that owns transports, routers
and sessions:

```python
from kinova.connection import kinova_connection

with kinova_connection() as conn:
    fb = conn.base_cyclic.RefreshFeedback()
    print([a.position for a in fb.actuators])
```

Override the target with `KINOVA_IP` / `KINOVA_USERNAME` / `KINOVA_PASSWORD`, or
pass a `ConnectionConfig`. Pass `real_time=True` to also open UDP :10001 and get
`conn.base_cyclic_rt`.

### Why kortex_api 2.8.0 specifically

`kortex_api` is not on PyPI — it comes from Kinova's Artifactory, pinned by URL in
`requirements.txt`. The version is load-bearing:

- **3.x** dropped `TCPTransport` for MQTT. This arm answers on TCP :10000, so 3.x
  cannot talk to it at all.
- **2.6.0** has TCP but pins `protobuf==3.5.1`, which has no wheel for Python ≥3.9
  and fails to build on 3.11.
- **2.8.0** has TCP *and* allows `protobuf<=3.20`, which installs on Python 3.11.

Verified working on Python 3.11 / macOS arm64.

## ROS 2 — use the RAMMP stack, not the image I built

**`rammp-cuda:1.0.0-jp6` is not the arm image.** It is arm64 ROS 2 Humble +
CUDA 12.6 + PyTorch + Cyclone DDS + the RAMMP *interface definitions*
(`rammp_arm_interfaces`, `rammp_common_interfaces`) — `JointImpedanceGains`,
`ListControllers`, `GoToEEPose`, `AcquireControl`, `EStop` and friends. Those are
message/action IDL only: every library in its overlay is `rosidl_generator_*` /
`typesupport_*`, and it contains **no node executables**, no `ros2_control`, no
`kortex` package. Verified on the exact digest `bd38c4…`.

The implementation lives in two public repos:

| Repo | What it is |
|---|---|
| [`rammp-org/kinova-gen3-driver`](https://github.com/rammp-org/kinova-gen3-driver) | C++ low-level core. Control modes: **Cartesian impedance** (task-space, null-space posture), **joint impedance** (in-loop diff IK), joint torque (gravity comp = torque mode with no feedforward), joint position, joint velocity. Plus a 2F-85 `GripperController` and a Supervisor/Arbiter/StreamingSession interface tier. Pinocchio for dynamics; KORTEX confined to `Transport`. |
| [`rammp-org/kinova-gen3-ros2`](https://github.com/rammp-org/kinova-gen3-ros2) | ROS 2 Humble front-end implementing those ports. Builds `FROM ros:humble`, vcs-imports the driver via `kinova_gen3.repos`, ships its own `Dockerfile`, `Makefile` and `deploy/compose.yaml`. |

Neither image is published to ghcr — build locally:

```sh
git submodule update --init --recursive     # vendored at third_party/
cd third_party/kinova-gen3-ros2
make build      # sim image, no KORTEX, no robot
make sim        # run the node against the sim transport
make stage-kortex && make real   # KORTEX-enabled, against the real arm
```

`make stage-kortex` copies the **aarch64 KORTEX SDK** into `docker/vendor/`
(gitignored, so it must be fetched). The Dockerfile pins
`KORTEX_SDK_DIR=kortex_api_2.8.0_aarch64` — the same 2.8.0 line the Python venv
here uses, for the same TCP-transport reason.

### rammp-cuda tags — `:humble` is a trap

`:humble` is **not** a rolling "latest Humble" tag. It is a stale alias for the
release candidate:

| Tag | Digest | Built | Overlay |
|---|---|---|---|
| `:humble` | `a5364324…` | 2026-09-01 | `arm_interfaces`, `rammp_prototype_interfaces` |
| `:1.0.0-rc1-jp6` | `a5364324…` | 2026-09-01 | *(identical image to `:humble`)* |
| `:1.0.0-jp6` | `bd38c47d…` | 2026-09-10 | `rammp_arm_interfaces`, `rammp_common_interfaces` |

The interface packages were **renamed and redesigned** between them — zero
package-name overlap. The rc1 set (`SetMode`, `SetSpeedPreset`, `ReachPreset`,
`CurbTraverse`, …) has no impedance, streaming, arbitration or gripper messages
at all. Anything built against `:humble` will not link against the 1.0.0
platform. If you use this image, use `:1.0.0-jp6`.

### But for the arm, you need neither

`kinova_gen3_ros2/package.xml` depends on **`kinova_gen3_interfaces`**, its own
package — an exact superset of `rammp_arm_interfaces` + `rammp_common_interfaces`
(same 21 definitions, plus `WrenchSetpoint.msg`). The front-end is self-contained
and builds `FROM ros:humble`. `rammp_arm_interfaces` inside `rammp-cuda` is the
platform-level mirror for *other* RAMMP nodes that talk to the arm node — the
consumer side, not the driver.

So: `rammp-cuda` is the base for RAMMP nodes needing CUDA + PyTorch (perception,
policies). It is not, and is not meant to be, the arm image.

### What will and won't run on this Mac

`deploy/compose.yaml` wants `network_mode: host`, `ipc: host`,
`cap_add: [SYS_NICE]` and `ulimits: {rtprio: 99, memlock: -1}`, and the driver's
own getting-started says it targets an **aarch64 Jetson with a PREEMPT_RT
kernel**. Docker Desktop on macOS runs a LinuxKit VM with no RT kernel, and host
networking there is the VM's network, not your LAN.

- **Works on the Mac:** `make build` / `make sim` — arm64 native, sim transport,
  the unit/RT-safety tests. Good for development.
- **Needs the Jetson:** real-robot 1 kHz control, and the `curobo_planner`
  companion (`rammp-curobo:jp6`, which needs `runtime: nvidia`). Without that
  planner, three of the four action servers accept goals and never succeed.

Set `KINOVA_ARM_IP=192.168.2.67` — the compose default is `192.168.1.10`.

### The amd64 image in `docker/`

`rammp/kortex-ros2:humble` (built and verified here) is the stock Kinova
`ros2_kortex` driver + MoveIt, pinned to `linux/amd64` because Kinova ships those
Debians for amd64 only. It is **not** the RAMMP stack and does not speak
`rammp_arm_interfaces`. Keep it only as a vendor-stock reference for MoveIt
planning; prefer the RAMMP stack above for anything real.

## Hosts and where the arm actually lives

The arm is on a private wired LAN, `192.168.2.0/24`. **`jetson` is the driver
host** — it is the only machine that is both on that LAN and running a
real-time kernel.

| Host | Arch | Kernel | Reaches 192.168.2.67 | Role |
|---|---|---|---|---|
| `jetson` (AGX Orin, `192.168.2.188` wired) | aarch64 | **6.8.12-1021-rt-tegra, `CONFIG_PREEMPT_RT=y`** | **yes** | **runs the driver** |
| `triphammer` (`192.168.2.136` wifi) | x86_64 | 5.15.0-139-generic | yes | RTX 4070 Ti, 12 GB — perception, later |
| MacBook | arm64 | Darwin | **no longer** | development + simulation |
| `cascadilla`, `buttermilk` | x86_64 | 6.8.0 PREEMPT_DYNAMIC | no | campus net, `128.84.114.x/23` |

The Mac was on the arm's LAN earlier and is not any more; do not assume it can
reach the arm. Everything in `sim/` runs there regardless, because it needs
neither the arm nor a GPU.

### Why the Jetson, specifically

Measured, not assumed:

- `CONFIG_PREEMPT_RT=y`, `/sys/kernel/realtime` = 1, `SCHED_FIFO` 1–99 available.
- It already boots with **`isolcpus=11 nohz_full=11 rcu_nocbs=11`**, which is
  exactly what the driver's `RT_CORE ?= 11` expects. (`nproc` reports 11, not
  12, precisely *because* core 11 is out of the default affinity mask — the
  machine has 12 cores, 0–11.)
- Verified in a live run: the node's RT thread comes up `SCHED_FIFO` priority
  **80 pinned to cpu 11**, while its other 29 threads keep the full mask.
- `NV Power Mode: MAXN`.

`triphammer` has the GPU but a stock preemption kernel, sits on wifi, and runs
Python 3.8 on Ubuntu 20.04. It is the right box for perception and the wrong one
for a 1 kHz cyclic loop.

### Building on the Jetson

Native arm64 throughout — no emulation. `ros:humble` publishes a `linux/arm64`
image, Kinova publishes an aarch64 KORTEX SDK, and the `Makefile` detects the
arch on its own:

```
ssh jetson
cd ~/rammp && make sdk && make kortex
```

Verify it is a real KORTEX build with `nm -C`, never `strings`: only the mangled
symbol exists, so `strings | grep KortexTransport::connect` returns nothing on a
perfectly good image.

```
docker run --rm --entrypoint bash kinova-gen3-ros2:kortex -lc \
  'nm -C /ros2_ws/install/kinova_gen3_ros2/lib/kinova_gen3_ros2/kinova_gen3_node \
   | grep -c "KortexTransport::connect"'      # 5 on a KORTEX build, 0 otherwise
```

**The gripper is holding a fork.** It must not be commanded open — a dropped
fork is a falling-object problem, and the 2F-85 never opens during a meal
anyway. Treat `/setpoint/gripper` as off-limits while the fork is mounted;
`/gripper_state` is read-only and safe (it reads `position ~0.947, present: true`).

## Real-arm (KORTEX) build

Staged and built on this Mac — arm64 throughout, no emulation:

```sh
# SDK extracted to ~/kortex_api_2.8.0_aarch64 (the Makefile's default KORTEX_SRC)
cd third_party/kinova-gen3-ros2
make build-real            # -> kinova-gen3-ros2:kortex
```

The SDK comes from the same Artifactory tree as the Python wheel:
`.../kortex/API/2.8.0/linux_aarch64_gcc_7.4.zip`. It stages into
`docker/vendor/`, which the submodule already gitignores, so staging does not
dirty the submodule.

**Verifying the link.** The runbook says to `strings` the binary for
`KortexTransport::connect` — that does not work, because only the mangled symbol
is present. Use `nm -C` instead:

| | `:humble` (sim) | `:kortex` (real) |
|---|---|---|
| binary | 4.46 MB | 12.68 MB |
| `nm -C \| grep -i kortex` | 0 | 22 |
| marker string | `built without KORTEX; use --sim` | Kinova 2.8.0 build paths |

Not yet run against the arm. `make real IP=192.168.2.67` is the attended step —
and note this Mac has no PREEMPT_RT kernel, so treat the 1 kHz loop as
unvalidated here.

## The depth feed drops every ~62s — root cause and what we do about it

**Not the camera restarting, and not the network.** The link measures 0% packet
loss over 50 pings. The vision module runs a GStreamer RTSP server which
advertises a **30 second session timeout** in its SETUP reply:

```
Session: <id>;timeout=30
```

A client must refresh inside that window. GStreamer's `rtspsrc` refreshes too
slowly and the session expires at roughly 2x the advertised timeout — hence the
~62s drop. Proven with `kinova/scripts/rtsp_probe.py`, which holds a session by
hand and refreshes it with `GET_PARAMETER`:

| client | keepalive | result |
|---|---|---|
| `rtspsrc` (kinova_vision) | its own cadence | drops at ~62s, every time |
| `rtsp_probe.py` | every 10s | **130s, 12 keepalives, zero drops** |

```sh
python -m kinova.scripts.rtsp_probe                  # report the server timeout
python -m kinova.scripts.rtsp_probe --hold 130       # endurance test
```

The real fix is upstream (rtspsrc's keepalive cadence, or ros2_kortex_vision
configuring it). The probe is the reproducer to attach to that report. What we
own here:

1. **TCP transport by default** (`RTSP_PROTOCOLS ?= tcp` in the Makefile).
   Measured over ~250s: UDP drops every ~62s with a fixed 6s recovery; TCP drops
   less often (63s, 82s, 94s) and recovers in under 1s.
2. **A staleness guard** in `docker/depth_stream.py`: it warns on stderr on any
   >2s gap and on runs of byte-identical frames. A stalled feed looks exactly
   like a stationary scene, and we once recorded a 70s "video" of one frozen
   frame without noticing.
3. **Verify before trusting a recording.** Compare frames, not the publish rate.

**Note on `max_pub_rate`:** it does *not* republish stale frames. `Vision::publish()`
(vision.cpp:355) returns false when `gst_app_sink_pull_sample` yields nothing, logs
"Could not get frame", and publishes nothing — the topic goes quiet during a drop.

## Operational gotchas (from the RAMMP docs + the sim run)

Docs: https://rammp-org.github.io/kinova-gen3-ros2/ — `interface/`,
`troubleshooting/`, `getting-started/`, `guide-goto-actions/`.

**Verified here in sim.** `/list_controllers` returns 7 of 8 available:
`joint_position`, `joint_impedance`, `ee_pose_impedance`, `ee_pose_position`,
`joint_torque`, `joint_velocity`, `ee_twist`. The 8th, `cartesian_impedance`
(channels `pose` + `wrench`), reports `available=False`.

The docs say only that `/setpoint/wrench` is "declared but no controller
consumes this yet". The actual reason is in `stream_server.cpp:49`: that row is
`core_backed=false` because the core has no `kEeWrench` setpoint kind, and it
needs two channels. `test/conformance/checks_streaming.py` *asserts* it stays
unavailable — so this is deliberate, not a regression. Task-space compliance is
still reachable via `ee_pose_impedance` / `GoToEEPose`.

**Gotchas that will bite on the real arm:**

- **Zero velocity does not hold position.** "Zero velocity means stop driving,
  not stay put" — the shoulder creeps ~0.038 rad/s. Use `ee_pose_position` to
  hold a pose.
- **`/acquire_control` seizes.** It succeeds *even if another owner holds the
  arm*, halting their motion. It is not a polite lock.
- **E-stop is asymmetric.** Engaging is never age-checked; a *clear* is rejected
  if older than `estop_clear_max_age_s` (default 1.0 s).
- **`/joint_states`, `/ee_state`, `/gripper_state` are best-effort QoS.** A
  default (reliable) subscriber sees silence and looks like a dead publisher.
  Use `ros2 topic echo --qos-reliability best_effort /joint_states`.
- **First setpoints get lost.** Create the publisher and let discovery settle
  *before* calling `/open_stream`, or the deadline expires and the stream closes.
- **Gripper fields are not sticky.** Core reads position, speed and force every
  call — a message carrying only `position` commands the other two to defaults.
- **Gripper effort reads inverted.** Low effort (~0.05) means a sustained grip;
  current spikes during closure then settles. High effort is not success.
- **Keep `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` everywhere.** Mixing Cyclone and
  FastRTPS gives partial discovery, which looks like a flaky node.
- **No planner = goals hang.** `go_to_ee_pose`, `go_to_joint_config` and
  `go_to_preset` are clients of `/rammp_curobo/plan_to_*`. Without the
  `rammp-curobo` container they are accepted and never complete.
  `execute_joint_trajectory` is the one tier that does not need it.

**Makefile vs docs quickstart.** The Makefile header says "Run these ON the
Jetson (arm64)" and pins `--cpu 11` to abra's `isolcpus` core. The docs'
quickstart is a plain `docker run --rm -it --network host --ipc host` with no
`--cpu`. On this Mac the plain form is the better fit — core 11 here is an
ordinary scheduled core, so the pin is meaningless.

## Safety

`check_connection` is strictly read-only. Anything that commands motion should
confirm `ARMSTATE_SERVOING_READY` and clear fault banks first — both are reported
by the connection check.
