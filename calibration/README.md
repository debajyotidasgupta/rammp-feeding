# calibration/

Per-arm files the config points at (camera intrinsics live in `config/camera/*.yaml`, not here).

| file | what | written by |
|---|---|---|
| `home_pose.json` | **feeding_start**: joints, tool pose, the arm's tool configuration | `make capture` (`kinova/scripts/capture_home.py`) |
| `food_detect_pose.json` | **food_detect_start** (the bowl's overlook, `bowl.overlook_pose`): joints + tool pose | `capture_home.py --label food_detect_start --out calibration/food_detect_pose.json` |
| `depth_480x270.ini` | this unit's depth intrinsics as the device reports them (VisionConfig, device 9), in kinova_vision's camera_info format | by hand, from the device |

## Depth intrinsics

`kinova_vision` publishes `camera_info` from generic files in the package unless given
`depth_camera_info_url`; `make depth` passes `depth_480x270.ini`. The package default's
depth focal length is 7.5% off this unit's (360.013 vs 334.810 px), a scale error that
grows off-axis (~8 mm at the plate edge and at face range). The colour intrinsics match
the package; the colour principal point used is measured (`config/camera/kinova_wrist.yaml`).

Depth -> colour extrinsic (factory): t = [-0.02706, -0.00997, -0.00471] m, R = identity.
Anything that samples depth at a colour pixel must account for this 29 mm baseline.
