# rammp — robot-assisted feeding, Kinova Gen3

Starting again from a known state. One arm, one fork, one command.

The arm is a **Kinova Gen3 7-DoF** on a wheelchair mount at the user's right,
holding a **fixed inline fork** in a Robotiq 2F-85. There is **no wrist
force/torque sensor** — all force sensing is joint torque. Compute is a **Jetson
AGX Orin** running the driver in a container.

## The command

```
make start            # go to the feeding start pose
make state            # where is the arm right now
make status           # what is running, is the arm reachable
```

`feeding start` is the only motion command that exists. Every feeding sequence
begins and ends at that pose. It is a direct, speed-limited joint move: on
deployment the arm always stops near the pose already, so there is no path
planning and none is needed.

## The feeding start pose

Measured on this installation and checked in as `calibration/home_pose.json`.

| | |
|---|---|
| joints (deg) | `-89.998, 8.074, -0.003, -139.195, 0.447, -40.078, -174.305` |
| tool point | `(0.0247, -0.2238, 0.2121)` m in the arm base frame |
| fork direction | `(-0.005, -0.164, -0.986)` — **80.6° below horizontal** |
| joint 1 holding torque | 0.291 N·m |

It is safe at rest by construction, not by monitoring: **joint 1 at −90°** swings
the whole arm clockwise, clear of the user on their right, and the fork points at
the floor. Nothing has to be running for that to hold.

**Joint 3 is squared to 0** so the folded forearm lies in the joint-1 plane
instead of splaying out of it. That is not cosmetic — it dropped joint 1's
holding torque from 0.858 to 0.291 N·m by taking the off-axis moment off the
base joint. A large joint-1 holding torque is a hint the pose has drifted.

Re-measure it with `make capture` after any re-park, then pull the file back to
the Mac before the next `sync.sh` (which runs `--delete`).

## Two facts that constrain everything

**1. The arm reports a configured TOOL frame, not the flange.** There is a
120 mm z-offset and 0.831 kg registered (`GetToolConfiguration`). The fork adds
more: camera-to-tine-tip measures ~105 mm by hand. So the reported `tool_pose`
lands within a few cm of the real tine tip — good enough to stand off by, not
good enough for fine work. Do not read `tool_pose` as the flange or as the tip.

**2. The user-facing camera cannot drive motion.** It carries ~100 mm of error,
and at deployment it mounts on the **wheelchair armrest** — so its transform to
the arm base is not fixed and cannot be calibrated, because there is no constant
to calibrate. It can hold an *abort* authority (presence, and the frame-to-frame
*change* in mouth position, which is far more trustworthy than the absolute).
It cannot hold a *setpoint* authority at any precision.

## Layout

```
kinova/
  connection.py            session, routers, teardown
  scripts/
    feeding_start.py       THE command: go to the start pose
    arm_state.py           where is the arm, --watch to follow it
    capture_home.py        re-measure the start pose from the parked arm
    check_connection.py    identity, firmware, live joint state
calibration/
  home_pose.json           the start pose, measured
  depth_480x270.ini        this camera's own depth intrinsics, not the default
docker/                    driver and vision images
third_party/               kinova-gen3-ros2 (submodule)
```

The gripper is **never commanded** anywhere in this tree. It is holding the
fork; opening it drops the fork. `arm_state` reports gripper position and that
is the extent of it.

## Machines

| host | role |
|---|---|
| jetson | AGX Orin, arm64, runs the driver container `tcp_rec`; talks to the arm |
| triphammer | x86-64, RTX 4070 Ti, GPU work; no PREEMPT_RT, on wifi |
| the Mac | source of truth for this tree; `./sync.sh [host]` pushes with `--delete` |

Arm is at **192.168.2.67**. Edit on the Mac, never on a replica.

## History

Everything before the reset — the MuJoCo testbed, the perception stack, the
skill work, the architecture and poster documents — is in the first commit and
nothing else. It was deleted rather than carried because the setup it assumed
is not the setup being built. `git show a85881d` for what it established.
