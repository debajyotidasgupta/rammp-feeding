# Per-unit camera calibration for this arm

Captured from the device itself with `kinova/scripts/camera_intrinsics.py`
(`VisionConfig.GetIntrinsicParametersProfile`, vision device 9), 2026-09-22.

## Why these exist

`kinova_vision` publishes `camera_info` from generic files inside the package,
not from the arm, unless you pass `color_camera_info_url` /
`depth_camera_info_url`. For this unit those two sources disagree:

| | device | package default | |
|---|---|---|---|
| depth fx, fy | **334.810** | 360.013 | shipped is **+7.53%** |
| depth cx | **238.536** | 243.872 | +5.34 px |
| depth cy | **139.694** | 137.922 | −1.77 px |
| colour (all resolutions) | — | — | identical to 3 dp |

So the **depth** stream was being deprojected with the wrong focal length. The
error is a scale, so it grows off-axis: ≈2 mm at the centre of the plate, ≈8 mm
at the plate edge and at face range. That is the size of a food cube.

The **colour** intrinsics match the shipped file exactly at every resolution, so
there is nothing to override — but see `docs/camera-calibration.md` for why the
16:9 colour profiles still should not be trusted: the device reports a principal
point that is not centred (cy/h = 0.331 at 720p, 0.090 at 1080p) and the 1080p
value is not geometrically consistent with any crop of a 4:3 sensor.

## Depth → colour extrinsics (factory, from the device)

```
t = [-0.02706  -0.00997  -0.00471] m      |t| = 29.2 mm
R = identity
```

A 29 mm baseline with no rotation. Anything that samples the depth image at a
pixel found in the colour image must account for it.
